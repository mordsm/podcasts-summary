import re
import os
import logging
import tempfile
import ipaddress
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _is_safe_url(url: str) -> bool:
    """Block SSRF targets: non-http(s) schemes, private/loopback/link-local IPs, cloud metadata."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = (p.hostname or "").lower()
        if not host:
            return False
        if host in {"localhost", "metadata.google.internal", "169.254.169.254"}:
            return False
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return False
        except ValueError:
            pass  # hostname, not a bare IP — pass through
        return True
    except Exception:
        return False


def _safe_get_text(url: str, timeout: int = 30, headers: dict = None) -> str:
    """GET with SSRF check and 500 MB download cap. Returns decoded text."""
    if not _is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL: {url}")
    r = requests.get(url, timeout=timeout, stream=True,
                     headers=headers or {})
    r.raise_for_status()
    chunks = []
    total = 0
    for chunk in r.iter_content(65536):
        total += len(chunk)
        if total > _MAX_DOWNLOAD_BYTES:
            r.close()
            raise ValueError(f"Response exceeded 500 MB cap: {url}")
        chunks.append(chunk)
    raw = b"".join(chunks)
    encoding = r.encoding or r.apparent_encoding or "utf-8"
    return raw.decode(encoding, errors="replace")


@dataclass
class TranscriptResult:
    text: str
    method: str
    language: str
    word_count: int


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def strip_vtt(vtt: str) -> str:
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    return " ".join(lines)


# ── Method 1: RSS <podcast:transcript> tag ────────────────────────────────────

def try_rss_transcript(episode) -> Optional[TranscriptResult]:
    if not episode.transcript_url:
        return None
    try:
        text = _safe_get_text(episode.transcript_url, timeout=30)
        mime = (episode.transcript_type or "").lower()
        if "vtt" in mime or "srt" in mime:
            text = strip_vtt(text)
        elif "html" in mime:
            text = strip_html(text)
        text = text.strip()
        if len(text) < 100:
            return None
        return TranscriptResult(text, "rss_tag", "auto", len(text.split()))
    except Exception as e:
        logger.debug(f"RSS transcript tag failed for {episode.title}: {e}")
        return None


# ── Method 2: YouTube captions ────────────────────────────────────────────────

def _lang_priority(language: str) -> list:
    if language == "he":
        return ["he", "iw", "en"]
    return ["en", "he", "iw"]


def try_youtube_captions(video_id: str, language: str) -> Optional[TranscriptResult]:
    # Strategy A: youtube-transcript-api (no download needed)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        langs = _lang_priority(language)

        for manual in (True, False):
            for lang in langs:
                try:
                    t = (tlist.find_manually_created_transcript([lang]) if manual
                         else tlist.find_generated_transcript([lang]))
                    snippets = t.fetch()
                    text = " ".join(s.text for s in snippets).replace("\n", " ")
                    text = re.sub(r"\s+", " ", text).strip()
                    return TranscriptResult(text, "youtube_captions", t.language_code, len(text.split()))
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"youtube-transcript-api failed for {video_id}: {e}")

    # Strategy B: yt-dlp --write-auto-sub fallback
    try:
        import yt_dlp
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tmpl = os.path.join(tmpdir, "%(id)s")
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["he", "iw", "en"],
                "subtitlesformat": "vtt",
                "outtmpl": out_tmpl,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            for f in os.listdir(tmpdir):
                if f.endswith(".vtt"):
                    with open(os.path.join(tmpdir, f), encoding="utf-8") as vf:
                        text = strip_vtt(vf.read()).strip()
                    if len(text) > 100:
                        lang = f.split(".")[-2] if "." in f else "auto"
                        return TranscriptResult(text, "youtube_captions_yt_dlp", lang, len(text.split()))
    except Exception as e:
        logger.debug(f"yt-dlp captions failed for {video_id}: {e}")

    return None


# ── Method 3: Fetch episode web page for richer show notes ───────────────────

_AUDIO_EXT_RE = re.compile(r'\.(mp3|m4a|ogg|opus|aac|wav|flac)(\?.*)?$', re.IGNORECASE)
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form"}


def try_page_content(episode, min_length: int) -> Optional[TranscriptResult]:
    """Fetch the episode's web page and extract body text.

    Useful when the RSS description is short but the episode page has full
    show notes (e.g. Reversim, where the page has 3× more content than RSS).
    Skips audio URLs and YouTube watch pages (handled by other methods).
    """
    url = episode.url or ""
    if not url:
        return None
    if _AUDIO_EXT_RE.search(url):
        return None
    if "youtube.com/watch" in url or "youtu.be/" in url:
        return None

    try:
        page_text = _safe_get_text(
            url, timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"},
        )
        soup = BeautifulSoup(page_text, "lxml")
        for tag in soup.find_all(_SKIP_TAGS):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.debug(f"Page fetch failed for {episode.title}: {e}")
        return None

    # Only use if meaningfully longer than the RSS description
    rss_len = len((episode.description or "").split())
    page_len = len(text.split())
    if page_len < min_length or page_len <= rss_len:
        return None

    return TranscriptResult(
        f"[SHOW NOTES — PAGE]\n{text}",
        "page_content",
        episode.language,
        page_len,
    )


# ── Method 4: Long description as show notes ──────────────────────────────────

def try_description(episode, min_length: int) -> Optional[TranscriptResult]:
    raw = episode.description or ""
    text = strip_html(raw).strip()
    if len(text) < min_length:
        return None
    return TranscriptResult(
        f"[SHOW NOTES]\n{text}",
        "description",
        episode.language,
        len(text.split()),
    )


# ── Method 4: Whisper (speech-to-text fallback) ───────────────────────────────

def try_whisper(episode, model_size: str = "small") -> Optional[TranscriptResult]:
    try:
        import yt_dlp
        from faster_whisper import WhisperModel
    except ImportError as e:
        logger.warning(f"Whisper deps not installed ({e}), skipping")
        return None

    if not episode.audio_url and not episode.youtube_video_id:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tmpl = os.path.join(tmpdir, "audio.%(ext)s")
        url = (f"https://www.youtube.com/watch?v={episode.youtube_video_id}"
               if episode.youtube_video_id else episode.audio_url)

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": out_tmpl,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            return None

        audio_files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not audio_files:
            return None

        audio_path = os.path.join(tmpdir, audio_files[0])
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            lang = episode.language if episode.language in ("he", "en") else None
            segments, info = model.transcribe(
                audio_path, language=lang, vad_filter=True, beam_size=5,
                condition_on_previous_text=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return TranscriptResult(text, "whisper", info.language, len(text.split()))
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return None


# ── Method 0: Cached transcript from previous run ─────────────────────────────

def try_cached_transcript(episode, transcripts_dir) -> Optional[TranscriptResult]:
    """Load a previously saved transcript file to avoid re-running Whisper."""
    from pathlib import Path
    transcripts_dir = Path(transcripts_dir)
    safe_name = re.sub(r'[^\w\- ]', '_', f"{episode.feed_name} — {episode.title}")[:80]
    cache_path = transcripts_dir / f"{safe_name}.txt"
    if not str(cache_path.resolve()).startswith(str(transcripts_dir.resolve())):
        logger.warning(f"Path traversal blocked for cache read: {safe_name!r}")
        return None
    if not cache_path.exists():
        return None
    try:
        content = cache_path.read_text(encoding="utf-8")
        if "--- TRANSCRIPT ---" not in content:
            return None
        header, _, text = content.partition("--- TRANSCRIPT ---")
        text = text.strip()
        if not text:
            return None
        lang = "auto"
        method = "cached"
        for line in header.splitlines():
            if line.startswith("Language:"):
                lang = line.split(":", 1)[1].strip()
            elif line.startswith("Method:"):
                raw = line.split(":", 1)[1].strip()
                # Avoid double-prefixing if already cached
                method = raw if raw.startswith("cached") else "cached_" + raw
        return TranscriptResult(text, method, lang, len(text.split()))
    except Exception as e:
        logger.debug(f"Cache read failed for {episode.title}: {e}")
        return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_transcript(episode, settings: dict, whisper_count: int = 0,
                   skip_whisper: bool = False,
                   enforce_whisper: bool = False,
                   transcripts_dir=None) -> Optional[TranscriptResult]:
    if transcripts_dir:
        result = try_cached_transcript(episode, transcripts_dir)
        if result:
            logger.info(f"  Transcript via cache ({result.word_count} words)")
            return result

    if not enforce_whisper:
        result = try_rss_transcript(episode)
        if result:
            logger.info(f"  Transcript via rss_tag ({result.word_count} words)")
            return result

        if episode.youtube_video_id:
            result = try_youtube_captions(episode.youtube_video_id, episode.language)
            if result:
                logger.info(f"  Transcript via youtube_captions ({result.word_count} words)")
                return result

        result = try_page_content(episode, settings.get("description_min_length", 1500))
        if result:
            logger.info(f"  Transcript via page_content ({result.word_count} words)")
            return result

        result = try_description(episode, settings.get("description_min_length", 1500))
        if result:
            logger.info(f"  Transcript via description ({result.word_count} words)")
            return result

        # Last resort before Whisper: accept very short descriptions (≥50 words) rather than
        # downloading audio — useful for YouTube videos blocked by bot detection
        result = try_description(episode, 50)
        if result:
            logger.info(f"  Transcript via short description fallback ({result.word_count} words)")
            return result
    else:
        logger.info("  enforce_whisper: skipping non-whisper methods")

    if skip_whisper:
        logger.info("  Whisper skipped (test mode)")
        return None

    max_w = settings.get("max_whisper_per_run", 2)
    if whisper_count >= max_w:
        logger.info(f"  Whisper limit reached ({whisper_count}/{max_w})")
        return None

    result = try_whisper(episode, settings.get("whisper_model", "small"))
    if result:
        logger.info(f"  Transcript via whisper ({result.word_count} words)")
    return result
