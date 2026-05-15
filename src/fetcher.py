import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    id: str
    title: str
    author: str
    url: str
    feed_name: str
    feed_type: str
    language: str
    published: datetime
    description: str
    audio_url: Optional[str] = None
    youtube_video_id: Optional[str] = None
    transcript_url: Optional[str] = None
    transcript_type: Optional[str] = None


_YT_VIDEO_RE = re.compile(r'(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})')
_YT_FEED_RE = re.compile(r'youtube\.com/feeds/videos\.xml')
_HEBREW_RE = re.compile(r'[֐-׿]')
_ALPHA_RE = re.compile(r'[a-zA-Z֐-׿]')


def _detect_feed_type(url: str) -> str:
    if _YT_FEED_RE.search(url):
        return "youtube_rss"
    if "open.spotify.com/show/" in url:
        return "spotify"
    return "rss"


def _detect_language(feed_obj, entries: list) -> str:
    lang = (feed_obj.get("language") or "").lower().strip()
    if lang:
        if lang.startswith(("he", "iw")):
            return "he"
        if len(lang) >= 2:
            return lang[:2]
    sample = " ".join([
        feed_obj.get("title", ""),
        feed_obj.get("subtitle", ""),
        (entries[0].get("title", "") if entries else ""),
        (entries[0].get("summary", "")[:300] if entries else ""),
    ])
    alpha = _ALPHA_RE.findall(sample)
    if alpha and sum(1 for c in alpha if _HEBREW_RE.match(c)) / len(alpha) > 0.25:
        return "he"
    return "en"


def _parse_dt(entry) -> datetime:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if t:
        return datetime(*t[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _extract_episode_id(entry, feed_type: str) -> str:
    if feed_type == "youtube_rss":
        vid = getattr(entry, "yt_videoid", None)
        if vid:
            return f"yt:video:{vid}"
    eid = entry.get("id") or entry.get("link") or entry.get("title", "")
    return eid


def _extract_youtube_video_id(entry, feed_type: str) -> Optional[str]:
    if feed_type == "youtube_rss":
        vid = getattr(entry, "yt_videoid", None)
        if vid:
            return vid
    for link in entry.get("links", []):
        m = _YT_VIDEO_RE.search(link.get("href", ""))
        if m:
            return m.group(1)
    m = _YT_VIDEO_RE.search(entry.get("link", ""))
    if m:
        return m.group(1)
    return None


def _extract_audio_url(entry, feed_base: str) -> Optional[str]:
    for enc in entry.get("enclosures", []):
        if enc.get("type", "").startswith("audio/"):
            href = enc.get("href") or enc.get("url", "")
            if href:
                return urljoin(feed_base, href)
    return None


def _build_episode_url(entry, feed_url: str, feed_base: str, audio_url: Optional[str]) -> str:
    ep_link = entry.get("link", "")

    for generic in (feed_url, feed_base):
        if generic and ep_link.rstrip("/") == generic.rstrip("/"):
            return audio_url or ep_link

    if ep_link:
        parsed = urlparse(ep_link)
        if not parsed.path or parsed.path == "/":
            return audio_url or ep_link

    return ep_link or audio_url or feed_url


def _extract_transcript_url(entry):
    for link in entry.get("links", []):
        if link.get("rel") == "transcript":
            return link.get("href"), link.get("type")
    pt = entry.get("podcast_transcript")
    if pt:
        url = pt.get("url") or pt.get("href")
        return url, pt.get("type")
    return None, None


def _extract_author(entry, feed) -> str:
    author = entry.get("author") or entry.get("author_detail", {}).get("name", "")
    if not author:
        author = feed.get("author") or feed.get("title", "")
    return author.strip()


def _extract_description(entry) -> str:
    summary = entry.get("summary") or ""
    content_list = entry.get("content", [])
    if content_list:
        content = max(content_list, key=lambda c: len(c.get("value", "")))
        if len(content.get("value", "")) > len(summary):
            summary = content["value"]
    return summary


def fetch_feed(feed_config: dict) -> list[Episode]:
    url = feed_config["url"]
    name = feed_config["name"]
    feed_type = feed_config.get("type") or _detect_feed_type(url)

    logger.info(f"Fetching: {name} — {url}")
    try:
        parsed = feedparser.parse(url, agent="Mozilla/5.0 (compatible; PodcastSummarizer/1.0)")
    except Exception as e:
        logger.error(f"Failed to fetch {name}: {e}")
        return []

    if parsed.bozo and not parsed.entries:
        logger.warning(f"Feed parse error for {name}: {parsed.bozo_exception}")
        return []

    language = feed_config.get("language") or _detect_language(parsed.feed, parsed.entries)
    feed_base = parsed.feed.get("link", url)
    episodes = []

    for entry in parsed.entries:
        try:
            ep_id = _extract_episode_id(entry, feed_type)
            title = entry.get("title", "(no title)").strip()
            author = _extract_author(entry, parsed.feed)
            published = _parse_dt(entry)
            description = _extract_description(entry)
            audio_url = _extract_audio_url(entry, feed_base)
            youtube_id = _extract_youtube_video_id(entry, feed_type)
            transcript_url, transcript_type = _extract_transcript_url(entry)
            ep_url = _build_episode_url(entry, url, feed_base, audio_url)

            episodes.append(Episode(
                id=ep_id,
                title=title,
                author=author,
                url=ep_url,
                feed_name=name,
                feed_type=feed_type,
                language=language,
                published=published,
                description=description,
                audio_url=audio_url,
                youtube_video_id=youtube_id,
                transcript_url=transcript_url,
                transcript_type=transcript_type,
            ))
        except Exception as e:
            logger.warning(f"Skipping entry in {name}: {e}")

    episodes.sort(key=lambda e: e.published, reverse=True)
    logger.info(f"  -> {len(episodes)} episodes found in {name}")
    return episodes


def get_recent_episodes(feed_configs: list, hours: int = 168) -> list[Episode]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []
    for cfg in feed_configs:
        feed_type = cfg.get("type") or _detect_feed_type(cfg["url"])
        if feed_type == "spotify":
            logger.warning(f"Skipping {cfg['name']}: type=spotify requires Spotify API (not available)")
            continue
        episodes = fetch_feed(cfg)
        recent = [e for e in episodes if e.published >= cutoff]
        result.extend(recent)
    result.sort(key=lambda e: e.published, reverse=True)
    return result
