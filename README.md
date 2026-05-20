# Podcast & YouTube Summarizer

An automated pipeline that monitors Hebrew and English podcast RSS feeds and YouTube channels, fetches new episodes, extracts transcripts, and generates detailed bilingual summaries (Hebrew + English). Runs entirely on GitHub Actions — no local setup, no external paid APIs.
Results are delivered automatically to a configured Telegram channel.
---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Output Format](#output-format)
- [Transcript Extraction Methods](#transcript-extraction-methods)
- [Summarization Pipeline](#summarization-pipeline)
- [Configuration](#configuration)
- [GitHub Actions Setup](#github-actions-setup)
- [Running Manually](#running-manually)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [State Management](#state-management)

---

## How It Works

```
Every hour (GitHub Actions cron)
        │
        ▼
  Fetch all RSS / YouTube feeds
        │
        ▼
  Filter: skip already-seen episodes
        │
        ▼
  For each new episode:
    ├─ Try to get transcript (7 methods, cheapest first)
    ├─ Summarize with GitHub Models (gpt-4o / gpt-4o-mini)
    ├─ Format bilingual output (Hebrew + English)
    ├─ Validate and resolve all links
    └─ Send to Telegram
        │
        ▼
  git commit & push state back to repo
```

The pipeline runs on a free GitHub-hosted Ubuntu runner. All state is stored in the repository itself — no database, no external services.

---

## Features

- **Fully automated** — GitHub Actions cron fires every hour, processes new episodes, and commits results back
- **Bilingual output** — Long Hebrew summary (800–1200 words) + concise English summary (200–300 words)
- **7 transcript methods** — Tries every available source before falling back to Whisper audio transcription
- **Transcript caching** — Whisper results are saved to `data/transcripts/` and re-used on subsequent runs, avoiding costly re-transcription. Files older than 30 days are deleted automatically on each run.
- **Whisper budget** — Only 1 audio transcription per run to stay within GitHub Actions runner time limits; remaining episodes are deferred to the next cron run
- **Smart link handling** — Dead links are dropped, `example.com` removed, and URL shorteners (`bit.ly`, `t.co`, etc.) resolved to their final destination
- **Feed filtering** — Run on a specific feed by name via `workflow_dispatch` input
- **Test mode** — Process one small episode per feed type to verify the pipeline without long Whisper jobs
- **Telegram delivery** — Each new summary is sent automatically to a Telegram channel; supports chunked messages for long summaries and respects rate limits
- **Resend history** — Re-send all existing `results.txt.md` entries to Telegram via a single `workflow_dispatch` toggle (requires `--write-results` to have been used previously)
- **No external paid APIs** — Uses GitHub's free Models API (`MODELS_TOKEN`) and falls back to local BART + Helsinki models if unavailable
- **SSRF protection** — All outbound HTTP requests validate the target URL against a blocklist of private/loopback/link-local IPs and cloud metadata endpoints before fetching
- **Download size cap** — RSS transcript and episode-page fetches are capped at 500 MB; link-liveness checks are capped at 1 MB

---

## Output Format

By default, summaries are sent to Telegram only. To also write them to `results.txt.md`, pass `--write-results` when running the pipeline. Each episode produces one block:

```markdown
----
## Chapter Name : <episode title>

**Podcast:** <feed name>
**Author:** <author>
**Date:** <episode publish date> UTC
**Generated:** <summary generation time> UTC
**Link:** <episode URL>

---

**Hebrew Summary:**
<detailed Hebrew summary — 800–1200 words, bold section headers, bullet points>

**English Summary:**
<concise English summary — 200–300 words>

**Original description:**
<first 600 characters of the RSS description>

**Links mentioned:**
• [Page Title](https://resolved-url.com)
• ...

---
*Pipeline:*
  • Transcript: <method> (<N> words, lang=<lang>) — <audio analysis note>
  • Summary: GitHub Models gpt-4o-mini (he+en)
```

The **Pipeline** section shows:
- Which transcript method was used and word count
- Whether the **full audio file was transcribed** (Whisper) or show notes / captions were used instead
- Which summarization model ran

---

## Transcript Extraction Methods

Methods are tried in order from cheapest (no download) to most expensive (full audio):

| # | Method | Description |
|---|--------|-------------|
| 0 | **Cache** | Loads a previously saved transcript from `data/transcripts/` — skips all other methods |
| 1 | **RSS `<podcast:transcript>` tag** | Parses a transcript URL embedded in the RSS feed (VTT, SRT, or HTML formats) |
| 2 | **YouTube captions** | Uses `youtube-transcript-api` to fetch manual or auto-generated captions; falls back to `yt-dlp --write-auto-sub` |
| 3 | **Episode web page** | Fetches the episode's URL and extracts body text — useful for shows like Reversim where the web page has 3× more content than the RSS description |
| 4 | **RSS description** | Uses the RSS `<description>` field if it is ≥1500 words (likely full show notes) |
| 5 | **Short description fallback** | Accepts any description ≥50 words as a last resort before audio download — helps when YouTube bot-detection blocks yt-dlp on CI runners |
| 6 | **Whisper** | Downloads audio via `yt-dlp`, transcribes with `faster-whisper` (small model, CPU, int8). Limited to `max_whisper_per_run` per run (default: 1) |

Language priority: Hebrew episodes prefer `he/iw` captions first, then `en`. English episodes prefer `en` first.

---

## Summarization Pipeline

### Primary: GitHub Models (free API)

Requires a GitHub Personal Access Token with Models API access stored as `MODELS_TOKEN` secret.

- Tries **gpt-4o** first (2,000 word input limit to stay under free-tier TPM)
- Falls back to **gpt-4o-mini** (6,000 word input limit)
- Prompts the model to produce a structured Hebrew summary with bold headers and bullet points, preserving all English tech terms, product names, and URLs
- Returns both Hebrew (800–1200 words) and English (200–300 words) summaries

### Fallback: BART + Helsinki (local models)

Used when no API token is available. Runs entirely on CPU inside the GitHub Actions runner.

**Hebrew episode pipeline:**
1. Extractive pre-summary (if >1,500 words) to reduce translation cost
2. Translate Hebrew → English (`Helsinki-NLP/opus-mt-tc-big-he-en`)
3. Summarize with BART (`facebook/bart-large-cnn`) in 800-word chunks
4. Translate English summary → Hebrew (`Helsinki-NLP/opus-mt-en-he`)

**English episode pipeline:**
1. Extractive pre-summary (if >4,000 words)
2. Summarize with BART in 800-word chunks
3. Translate English summary → Hebrew

### Extractive fallback

If both API and local models fail, a simple sentence-extraction summary is used as a last resort.

---

## Configuration

All feeds and settings live in `config/feeds.yaml`.

### Settings

```yaml
settings:
  hours_lookback: 168          # Look back N hours for new episodes (default: 7 days)
  description_min_length: 1500 # Min word count to treat RSS description as transcript
  whisper_model: small         # faster-whisper model size (tiny/base/small/medium/large)
  max_whisper_per_run: 1       # Max Whisper jobs per cron run (defers the rest)
  bart_chunk_words: 800        # BART input chunk size in words
  extractive_max_sentences: 15 # Sentences to keep in extractive fallback
```

### Adding a Feed

```yaml
feeds:
  - name: My Podcast
    url: https://example.com/feed.xml
    # optional — used only as metadata, not for fetching:
    spotify_url: https://open.spotify.com/show/...
```

To force Whisper transcription for a specific feed (skipping captions/description methods):
```yaml
  - name: My Channel
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC...
    enforce_whisper: true   # always use Whisper, skip captions/description
```

For YouTube channels, use the channel RSS URL:
```yaml
  - name: My Channel
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC...
```

Language is auto-detected from feed metadata and Hebrew character ratio. Override is not needed in most cases.

---

## GitHub Actions Setup

### Required Secrets

| Secret | Description |
|--------|-------------|
| `MODELS_TOKEN` | GitHub Personal Access Token with Models API access. Create at: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained → Add `models:read` permission. **Do not use the default `GITHUB_TOKEN`** — it doesn't have Models API access. |
| `TELEGRAM_BOT_TOKEN` | Token for your Telegram bot (from [@BotFather](https://t.me/BotFather)). Optional — if not set, Telegram delivery is silently skipped. |
| `TELEGRAM_CHAT_ID` | Target channel or chat ID (e.g. `@MyChannel` or a numeric ID). The bot must be added as an **admin** of the channel. |

### Workflow Triggers

**Automatic (cron):** Fires every hour at `:00 UTC`. Processes all unseen episodes from the last 7 days.

**Manual (`workflow_dispatch`):** Go to the repo → Actions → Podcast Summary → Run workflow.

| Input | Description |
|-------|-------------|
| `feed` | Optional substring to filter by feed name (e.g. `רברס` or `Creative Channel`) |
| `test` | If checked, processes only 1 small episode per feed type (YouTube / Spotify-RSS / other RSS) — fast verification without triggering Whisper |
| `resend_history` | If checked, re-sends every entry already in `results.txt.md` to Telegram (requires `--write-results` to have been used previously) |

### First Run

GitHub may delay scheduled workflow runs for newly created repositories by several hours (known behavior). To verify the workflow works, use manual `workflow_dispatch` first.

---

## Running Manually

The pipeline is designed to run on GitHub Actions, not locally. To trigger a run without waiting for the cron:

1. Go to the repository on GitHub
2. Click **Actions** → **Podcast Summary** → **Run workflow**
3. Optionally fill in `feed` (e.g. `בזמן שעבדתם`) and check `test` for a quick run
4. Click **Run workflow**

Results are sent to Telegram after the run. To also write them to `results.txt.md`, add `--write-results` to the workflow inputs.

---

## Project Structure

```
podcasts-summary/
├── main.py                     # Pipeline orchestrator — entry point
├── requirements.txt            # Python dependencies
├── config/
│   └── feeds.yaml              # Feed list + pipeline settings
├── src/
│   ├── fetcher.py              # RSS/YouTube feed parsing, Episode dataclass
│   ├── transcript.py           # All transcript extraction methods
│   └── summarize.py            # Summarization, formatting, link handling
├── data/
│   ├── seen.json               # Tracks processed episode IDs (max 1000 entries)
│   └── transcripts/            # Cached transcript files (one .txt per episode)
└── .github/
    └── workflows/
        └── summarize.yml       # GitHub Actions workflow definition
```

### Key Files

**`data/seen.json`** — JSON object mapping episode IDs to the ISO timestamp when they were processed. Prevents re-processing. Capped at 1,000 entries (oldest are pruned first).

**`data/transcripts/<name>.txt`** — Cached transcript files. Format:
```
Feed: <feed name>
Episode: <episode title>
Method: <whisper|youtube_captions|...>
Language: <he|en|auto>
Words: <count>
URL: <episode URL>

--- TRANSCRIPT ---

<full transcript text>
```

**`results.txt.md`** — Optional append-only output file. Only written when `--write-results` flag is passed. In test mode with `--feed`, new entries are appended. In bare `--test` mode (no feed filter), the file is cleared first.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.30.0,<2.0.0 | GitHub Models API client |
| `feedparser` | 6.0.11 | RSS/Atom feed parsing |
| `beautifulsoup4` | 4.12.3 | HTML parsing for page content extraction |
| `lxml` | 5.3.0 | Fast HTML/XML parser backend |
| `requests` | 2.32.3 | HTTP client for RSS, transcripts, link checking |
| `PyYAML` | 6.0.2 | Config file parsing |
| `youtube-transcript-api` | 1.2.4 | YouTube caption fetching (no download) |
| `yt-dlp` | 2026.3.17 | Audio download and YouTube subtitle fallback |
| `faster-whisper` | 1.0.3 | CPU-optimized Whisper transcription |
| `transformers` | 4.57.6 | BART summarization + Helsinki translation models |
| `torch` | 2.12.0 | PyTorch backend for local models |
| `sentencepiece` | 0.2.0 | Tokenizer for Helsinki translation models |
| `sacremoses` | 0.1.1 | Text normalization for translation |

---

## Security

### SSRF protection

All outbound HTTP requests made from RSS feed content (transcript URLs, episode page URLs, and extracted links) are validated before fetching. The `_is_safe_url()` check blocks:

- Non-`http`/`https` schemes
- `localhost`, `127.x.x.x`, and all loopback addresses
- `169.254.169.254` and `metadata.google.internal` (cloud instance metadata endpoints)
- All RFC-1918 private ranges (`10.x`, `172.16–31.x`, `192.168.x`) and link-local addresses

### Download size limits

Fetches from untrusted RSS sources are capped to prevent memory exhaustion:

| Fetch type | Cap |
|---|---|
| RSS transcript tag, episode web page | 500 MB |
| Link liveness / title check | 1 MB |

### Path traversal protection

Transcript filenames are derived from RSS feed and episode titles. After sanitizing the name, the resolved path is asserted to remain inside `data/transcripts/` before any read or write.

---

## State Management

The pipeline uses two files for state — no database required:

- **`data/seen.json`** — Which episodes have been processed (persisted in git after every run)
- **`data/transcripts/`** — Full transcript text for each processed episode (persisted in git, used as a cache to avoid re-running Whisper). Files are automatically deleted after 30 days based on their first git commit date.

Both files are committed back to `master` by the workflow after every run, so state survives across cron invocations.

To reprocess an episode, delete its entry from `seen.json` and its file from `data/transcripts/`.
