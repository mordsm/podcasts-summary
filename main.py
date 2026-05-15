"""
Production entry point for the podcast/YouTube summarization pipeline.

Usage:
    python main.py           # normal run: episodes from last 7 days, skip seen
    python main.py --test    # test run: 3 smallest episodes (1 YouTube, 1 RSS-Spotify, 1 other RSS)
"""
import sys
import io
import re
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SEEN_PATH = DATA_DIR / "seen.json"
RESULTS_PATH = ROOT / "results.txt.md"
CONFIG_PATH = ROOT / "config" / "feeds.yaml"
DEBUG_DIR = DATA_DIR / "transcripts"

MAX_SEEN_ENTRIES = 1000


# ── Config & State ────────────────────────────────────────────────────────────

def load_config() -> tuple[list, dict]:
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["feeds"], config["settings"]


def load_seen() -> dict:
    if SEEN_PATH.exists():
        with open(SEEN_PATH, encoding="utf-8-sig") as f:
            return json.load(f)
    return {"version": 1, "entries": {}}


def save_seen(seen: dict):
    entries = seen["entries"]
    if len(entries) > MAX_SEEN_ENTRIES:
        sorted_items = sorted(entries.items(), key=lambda x: x[1])
        seen["entries"] = dict(sorted_items[-MAX_SEEN_ENTRIES:])
    DATA_DIR.mkdir(exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, ensure_ascii=False, indent=2)


def mark_seen(seen: dict, episode_id: str):
    seen["entries"][episode_id] = datetime.now(timezone.utc).isoformat()


def is_seen(seen: dict, episode_id: str) -> bool:
    return episode_id in seen["entries"]


# ── Test mode episode selection ───────────────────────────────────────────────

def _enclosure_size(episode) -> int:
    """Return RSS enclosure byte length if available, else sys.maxsize."""
    import sys as _sys
    url = episode.audio_url or ""
    if not url:
        return _sys.maxsize
    # feedparser stores enclosure length in the feed entry; we don't have direct
    # access here, so fall back to sys.maxsize for YouTube / unknown sources.
    return _sys.maxsize


def select_test_episodes(feed_configs: list) -> list:
    """
    Fetch one recent episode per feed across all feeds, then pick:
      - 1 from youtube_rss feeds
      - 1 from rss feeds that have a spotify_url field  (counts as "spotify")
      - 1 from rss feeds without spotify_url            (other RSS)
    Within each bucket, prefer the episode whose audio enclosure is smallest.
    Returns up to 3 episodes total.
    """
    from src.fetcher import fetch_feed

    youtube_bucket = []
    spotify_rss_bucket = []
    other_rss_bucket = []

    for cfg in feed_configs:
        feed_type = cfg.get("type", "rss")
        if feed_type == "spotify":
            continue
        try:
            episodes = fetch_feed(cfg)
        except Exception as e:
            logger.warning(f"Test fetch failed for {cfg['name']}: {e}")
            continue
        if not episodes:
            continue
        ep = episodes[0]

        if feed_type == "youtube_rss":
            youtube_bucket.append(ep)
        elif cfg.get("spotify_url"):
            spotify_rss_bucket.append(ep)
        else:
            other_rss_bucket.append(ep)

    def pick_smallest(bucket):
        return min(bucket, key=_enclosure_size, default=None)

    selected = []
    for bucket in (youtube_bucket, spotify_rss_bucket, other_rss_bucket):
        ep = pick_smallest(bucket)
        if ep:
            selected.append(ep)
    return selected[:3]


# ── Output helpers ────────────────────────────────────────────────────────────

def append_result(text: str):
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("----\n")
        f.write(text)
        f.write("\n")


# ── Telegram stub ─────────────────────────────────────────────────────────────

def send_telegram(text: str):
    logger.info("TODO Telegram: " + text[:100])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Podcast/YouTube summarization pipeline")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process 3 smallest episodes (1 per feed type), ignore 7d window")
    parser.add_argument("--feed", type=str, default=None,
                        help="Filter feeds by name substring (case-insensitive)")
    args = parser.parse_args()

    feed_configs, settings = load_config()
    if args.feed:
        feed_configs = [f for f in feed_configs if args.feed.lower() in f["name"].lower()]
        if not feed_configs:
            logger.error(f"No feed matched: {args.feed!r}")
            return
        logger.info(f"Feed filter: {args.feed!r} → {len(feed_configs)} feed(s)")
    seen = load_seen()

    # ── Collect episodes ──
    if args.test:
        logger.info("Test mode: selecting 3 smallest episodes across feed types")
        # Only wipe results in pure test mode (no feed filter); with --feed, always append
        if not args.feed and RESULTS_PATH.exists():
            RESULTS_PATH.unlink()
        episodes = select_test_episodes(feed_configs)
        logger.info(f"Test episodes selected: {len(episodes)}")
    else:
        from src.fetcher import get_recent_episodes
        hours = settings.get("hours_lookback", 168)
        logger.info(f"Normal mode: fetching episodes from last {hours}h")
        all_recent = get_recent_episodes(feed_configs, hours=hours)
        episodes = [e for e in all_recent if not is_seen(seen, e.id)]
        logger.info(f"New episodes after seen filter: {len(episodes)}")

    if not episodes:
        logger.info("No new episodes to process.")
        return

    from src.transcript import get_transcript
    from src.summarize import summarize_episode

    max_whisper = settings.get("max_whisper_per_run", 1)
    whisper_count = 0

    for episode in episodes:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Processing: [{episode.feed_type}] {episode.feed_name} — {episode.title}")
        logger.info(f"  Published: {episode.published.strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"  URL: {episode.url}")

        # In test mode, don't apply the whisper budget to get_transcript so all 3 run
        transcript = get_transcript(episode, settings,
                                    whisper_count=0 if args.test else whisper_count,
                                    transcripts_dir=DEBUG_DIR)

        if transcript is None:
            logger.warning("  No transcript found — skipping episode")
            mark_seen(seen, episode.id)
            save_seen(seen)
            continue

        if transcript.method == "whisper":
            whisper_count += 1
            logger.info(f"  Whisper used ({whisper_count}/{max_whisper})")

        logger.info(f"  Transcript: {transcript.method} ({transcript.word_count} words, lang={transcript.language})")

        # Save transcript to debug file
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^\w\- ]', '_', f"{episode.feed_name} — {episode.title}")[:80]
        debug_path = DEBUG_DIR / f"{safe_name}.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"Feed: {episode.feed_name}\n")
            f.write(f"Episode: {episode.title}\n")
            f.write(f"Method: {transcript.method}\n")
            f.write(f"Language: {transcript.language}\n")
            f.write(f"Words: {transcript.word_count}\n")
            f.write(f"URL: {episode.url}\n")
            f.write("\n--- TRANSCRIPT ---\n\n")
            f.write(transcript.text)
        logger.info(f"  Transcript saved to {debug_path.name}")

        try:
            summary = summarize_episode(episode, transcript, settings)
        except Exception as e:
            logger.error(f"  Summarization failed: {e}")
            mark_seen(seen, episode.id)
            save_seen(seen)
            continue

        append_result(summary)
        mark_seen(seen, episode.id)
        save_seen(seen)
        send_telegram(f"{episode.feed_name} — {episode.title}")
        logger.info("  Done.")

        # Stop if whisper budget is exhausted (production only; test mode processes all 3)
        if not args.test and whisper_count >= max_whisper:
            remaining = episodes[episodes.index(episode) + 1:]
            needs_whisper = [
                e for e in remaining
                if not e.transcript_url and not e.youtube_video_id
            ]
            if needs_whisper:
                logger.info(
                    f"Whisper limit reached ({whisper_count}/{max_whisper}). "
                    f"{len(needs_whisper)} episodes deferred to next cron run."
                )
                break

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
