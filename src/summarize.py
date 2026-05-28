import re
import html
import logging
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")

_LINK_CHECK_MAX_BYTES = 1 * 1024 * 1024  # 1 MB — enough to find a <title> tag


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
            pass
        return True
    except Exception:
        return False

_HEBREW_RE = re.compile(r"[֐-׿]")
_AUDIO_EXT_RE = re.compile(r'\.(mp3|m4a|ogg|opus|aac|wav|flac)(\?.*)?$', re.IGNORECASE)
_TITLE_RE = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
_SHORTENER_RE = re.compile(
    r'^https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|buff\.ly|'
    r'rebrand\.ly|short\.io|tiny\.cc|is\.gd|cutt\.ly|rb\.gy)/',
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(r'^https?://(?:[^/]*\.)?example\.com', re.IGNORECASE)


def _extract_urls(text: str) -> list:
    return list(dict.fromkeys(_URL_RE.findall(text)))


def _resolve_and_check(url: str) -> tuple[str, str] | None:
    """Fetch url, follow redirects, check liveness. Returns (final_url, title) or None if dead."""
    if _AUDIO_EXT_RE.search(url):
        return None
    if _EXAMPLE_RE.match(url):
        return None
    if not _is_safe_url(url):
        return None
    try:
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; PodcastSummarizer/1.0)"},
                         allow_redirects=True, stream=True)
        if r.status_code >= 400:
            r.close()
            return None
        final_url = r.url
        # Read only enough bytes to find the <title> tag
        chunks = []
        total = 0
        for chunk in r.iter_content(4096):
            chunks.append(chunk)
            total += len(chunk)
            if total >= _LINK_CHECK_MAX_BYTES:
                break
        r.close()
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        title = ""
        m = _TITLE_RE.search(raw[:4096])
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            title = re.sub(r"<[^>]+>", "", title).strip()
            title = html.unescape(title)
            title = title[:120]
        return (final_url, title)
    except Exception:
        return None


def _enrich_urls(urls: list) -> list[tuple[str, str]]:
    """Return list of (final_url, title) for live URLs only, fetched in parallel.
    Dead links, example.com URLs, and audio files are dropped.
    Shortener URLs are resolved to their final destination."""
    results: dict[str, tuple[str, str] | None] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        future_to_url = {ex.submit(_resolve_and_check, u): u for u in urls}
        try:
            for future in as_completed(future_to_url, timeout=20):
                orig_url = future_to_url[future]
                try:
                    results[orig_url] = future.result()
                except Exception:
                    results[orig_url] = None
        except Exception:
            pass
    # Preserve original order, drop dead links
    return [(final_url, title) for u in urls
            if (r := results.get(u)) is not None
            for final_url, title in [r]]


_TIMESTAMP_RE = re.compile(
    r'\[?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]?'   # [00:51], [1:04:30], 00:51
    r'|\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)'     # (00:51)
)


def _clean_text(text: str, strip_urls: bool = False) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[SHOW NOTES[^\]]*\]", "", text)
    text = _TIMESTAMP_RE.sub(" ", text)
    if strip_urls:
        text = _URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_mostly_hebrew(text: str) -> bool:
    if not text:
        return False
    letters = re.findall(r"[a-zA-Z֐-׿]", text)
    if not letters:
        return False
    return sum(1 for c in letters if _HEBREW_RE.match(c)) / len(letters) > 0.5


def _extractive_summary(text: str, max_sentences: int = 15, max_chars: int = 5000) -> str:
    clean = _clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return clean[:max_chars]
    if len(sentences) <= max_sentences:
        result = " ".join(sentences)
    else:
        head = sentences[: max_sentences - 2]
        tail = sentences[-2:]
        result = " ".join(head) + " [...] " + " ".join(tail)
    return result[:max_chars]


# ── BART + Helsinki models (requires torch + transformers) ─────────────────────

def _bart_summarize(text: str, settings: dict) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    chunk_size = settings.get("bart_chunk_words", 800)
    words = text.split()
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

    chunk_summaries = []
    for chunk in chunks[:8]:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=500, min_length=80, num_beams=4)
        chunk_summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    combined = " ".join(chunk_summaries)
    if len(chunk_summaries) > 1:
        inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=800, min_length=150, num_beams=4)
        combined = tokenizer.decode(ids[0], skip_special_tokens=True)
    return combined


def _translate_he_to_en(text: str) -> str:
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()

    words = text.split()
    chunks = [" ".join(words[i: i + 400]) for i in range(0, min(len(words), 3200), 400)]
    parts = []
    for c in chunks:
        inputs = tokenizer(c, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(**inputs, max_length=512)
        parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts)


def _translate_en_to_he(text: str) -> str:
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    model_name = "Helsinki-NLP/opus-mt-en-he"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()

    # Translate sentence-by-sentence to avoid per-sequence length truncation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text]
    parts = []
    for s in sentences:
        inputs = tokenizer([s], return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        parts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts)


_PRE_EXTRACT_HE_WORDS = 1500   # max words to translate (he→en)
_PRE_EXTRACT_EN_WORDS = 4000   # max words to feed into BART

_GITHUB_MODELS_PROMPT = """\
You are summarizing a Hebrew podcast episode. Write TWO detailed summaries.

IMPORTANT RULES:
- Keep ALL English tech terms as-is (product names, company names, tools, frameworks, acronyms like AI, AGI, SaaS, API, etc.)
- Preserve ALL URLs and links mentioned anywhere in the transcript or description
- Hebrew summary must be LONG and DETAILED (800-1200 words) — cover every topic discussed
- Use bold section headers (**כותרת**) and bullet points
- Include all numbers, statistics, names, and specific claims made
- Do NOT skip any technological, business, or product topics

1. Hebrew summary — structured with bold headers and bullets. Cover EVERY subject: technology topics, business models, products, companies, people mentioned, arguments made, predictions, and all links/resources. 800-1200 words.

2. English summary — 200-300 words, same structure, key points only.

Respond EXACTLY in this format (no extra text before or after):
HEBREW_SUMMARY:
<Hebrew text>

ENGLISH_SUMMARY:
<English text>

Episode: {title}
Podcast: {feed_name}

Transcript:
{transcript}"""


_MODEL_WORD_LIMITS = {
    "gpt-4o": 2000,       # ~3k tokens input, stays under 8k TPM with 4k output
    "gpt-4o-mini": 4000,  # ~6k tokens input, conservative to avoid context refusals
}

_REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "too long",
    "falls outside",
    "cannot process",
    "could you provide",
    "please provide",
    "exceeds",
)


def _is_refusal(text: str) -> bool:
    """Return True if the model returned an apology/refusal instead of a summary."""
    lower = text.lower()
    return (
        "HEBREW_SUMMARY:" not in text
        and any(phrase in lower for phrase in _REFUSAL_PHRASES)
    )


def _summarize_with_github_models(episode, text: str, github_token: str) -> tuple:
    """Returns (hebrew_summary, english_summary, steps) using GitHub Models free API."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token,
    )

    result = ""
    used_model = "gpt-4o-mini"
    last_exc = None

    # Try gpt-4o first, fall back to gpt-4o-mini; each model gets its own word limit.
    # For refusals (text too long), retry with progressively smaller chunks.
    # For API exceptions, skip to the next model.
    for model in ("gpt-4o", "gpt-4o-mini"):
        word_limit = _MODEL_WORD_LIMITS[model]
        words = text.split()
        got_result = False

        for attempt, limit in enumerate([word_limit, word_limit // 2, word_limit // 4]):
            truncated = " ".join(words[:limit]) if len(words) > limit else text
            prompt = _GITHUB_MODELS_PROMPT.format(
                title=episode.title,
                feed_name=episode.feed_name,
                transcript=truncated,
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                )
                candidate = response.choices[0].message.content or ""
                if _is_refusal(candidate):
                    logger.warning(
                        f"  GitHub Models {model} refused (attempt {attempt + 1}, "
                        f"{len(truncated.split())} words) — retrying with fewer words"
                    )
                    continue  # try smaller chunk for same model
                logger.info(f"  GitHub Models: used {model} ({len(truncated.split())} words)")
                result = candidate
                used_model = model
                got_result = True
                break
            except Exception as e:
                logger.warning(f"  GitHub Models {model} failed: {type(e).__name__}: {e}")
                last_exc = e
                break  # API error — skip remaining chunk sizes, try next model

        if got_result:
            break  # done

    if not result:
        if last_exc:
            raise last_exc
        raise RuntimeError("All GitHub Models attempts failed")

    hebrew_summary = ""
    english_summary = ""
    if "HEBREW_SUMMARY:" in result and "ENGLISH_SUMMARY:" in result:
        hebrew_summary = result.split("HEBREW_SUMMARY:")[1].split("ENGLISH_SUMMARY:")[0].strip()
        english_summary = result.split("ENGLISH_SUMMARY:")[1].strip()
    else:
        hebrew_summary = result

    return hebrew_summary, english_summary, [f"Summary: GitHub Models {used_model} (he+en)"]


def _summarize_with_models(episode, transcript_text: str, lang: str, settings: dict) -> tuple:
    """Returns (hebrew_summary, english_summary, pipeline_steps_list).
    Uses GitHub Models (free, GITHUB_TOKEN) if available, else BART+Helsinki fallback."""
    import os
    github_token = os.environ.get("MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        text = _clean_text(transcript_text, strip_urls=False)
        return _summarize_with_github_models(episode, text, github_token)

    # ── Fallback: BART + Helsinki (no API key available) ──────────────────────
    steps = []
    text = _clean_text(transcript_text, strip_urls=True)

    if lang in ("he", "iw"):
        pre_words = len(text.split())
        en_input = text
        if pre_words > _PRE_EXTRACT_HE_WORDS:
            en_input = _extractive_summary(text, max_sentences=40,
                                           max_chars=_PRE_EXTRACT_HE_WORDS * 7)
            steps.append(f"Pre-extract for translation: {pre_words}→{len(en_input.split())} words")
        en_text = _translate_he_to_en(en_input)
        steps.append("Translate: he→en (Helsinki opus-mt-tc-big-he-en)")
        n_chunks = max(1, len(en_text.split()) // settings.get("bart_chunk_words", 800))
        english_summary = _bart_summarize(en_text, settings)
        steps.append(f"English summary: BART facebook/bart-large-cnn ({n_chunks} chunks)")
        hebrew_summary = _translate_en_to_he(english_summary)
        steps.append("Hebrew summary: BART → translate en→he (Helsinki opus-mt-en-he)")
        return hebrew_summary, english_summary, steps

    else:
        pre_words = len(text.split())
        if pre_words > _PRE_EXTRACT_EN_WORDS:
            text = _extractive_summary(text, max_sentences=80,
                                       max_chars=_PRE_EXTRACT_EN_WORDS * 6)
            steps.append(f"Pre-extract: {pre_words}→{len(text.split())} words")
        n_chunks = max(1, len(text.split()) // settings.get("bart_chunk_words", 800))
        english_summary = _bart_summarize(text, settings)
        steps.append(f"English summary: BART facebook/bart-large-cnn ({n_chunks} chunks)")
        hebrew_summary = _translate_en_to_he(english_summary)
        steps.append("Translate: en→he (Helsinki opus-mt-en-he)")
        return hebrew_summary, english_summary, steps


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_output(episode, hebrew_summary: str, english_summary: str,
                   urls: list, pipeline_steps: list) -> tuple[str, str]:
    """Returns (full_text, telegram_text). full_text goes to results.txt.md;
    telegram_text omits English summary and original description."""
    from datetime import datetime, timezone

    # Clear English if it came out as Hebrew (extractive/model error)
    if english_summary and _is_mostly_hebrew(english_summary):
        english_summary = ""

    desc_clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", episode.description or "")).strip()

    url_block = ""
    if urls:
        enriched = _enrich_urls(urls[:20])
        lines = []
        _SKIP_TITLES = {"privacy faq", "privacy policy", "just a moment..."}
        for u, title in enriched:
            if title.lower() in _SKIP_TITLES:
                continue
            lines.append(f"• [{title}]({u})" if title else f"• {u}")
        url_block = "\n\n**Links mentioned:**\n" + "\n".join(lines)

    steps_block = "\n".join(f"  • {s}" for s in pipeline_steps)
    date_str = episode.published.strftime("%d/%m/%Y %H:%M") + " UTC"
    generated_str = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M") + " UTC"
    desc_block = f"\n**Original description:**  \n{desc_clean[:600]}" if desc_clean else ""

    he_block = f"**Hebrew Summary:**  \n{hebrew_summary}\n\n" if hebrew_summary else ""
    en_block = f"**English Summary:**  \n{english_summary}\n\n" if english_summary else ""

    header = (
        f"## {episode.url}  \n\n"
        f"**{episode.feed_name}** / {episode.author}  \n"
        f"**{episode.title}**  \n"
        f"{date_str}  ( Generated: {generated_str}  )\n"
        f"\n---\n\n"
    )
    footer = (
        f"{url_block}\n\n"
        f"---\n"
        f"*Pipeline:*\n{steps_block}\n"
    )

    full_text = header + he_block + en_block + desc_block + "\n" + footer
    telegram_text = header + he_block + footer
    return full_text, telegram_text


# ── Public API ────────────────────────────────────────────────────────────────

def summarize_episode(episode, transcript, settings: dict) -> tuple[str, str]:
    lang = transcript.language or episode.language
    raw_text = transcript.text
    urls = _extract_urls(raw_text) + _extract_urls(episode.description or "")
    urls = list(dict.fromkeys(urls))

    method = transcript.method
    if "whisper" in method:
        audio_note = "Full audio file transcribed (Whisper)"
    elif method.startswith("youtube_captions"):
        audio_note = "No audio download — YouTube captions used"
    elif method == "rss_tag":
        audio_note = "No audio download — transcript from RSS feed"
    else:
        audio_note = "No audio download — summary based on show notes / description only"

    transcript_step = f"Transcript: {method} ({transcript.word_count} words, lang={lang}) — {audio_note}"
    if getattr(transcript, "attempted", []):
        transcript_step += f"\n  • Tried and failed: {', '.join(transcript.attempted)}"
    pipeline_steps = [transcript_step]

    try:
        hebrew_summary, english_summary, model_steps = _summarize_with_models(
            episode, raw_text, lang, settings)
        pipeline_steps.extend(model_steps)
    except Exception as e:
        logger.warning(f"Model pipeline unavailable ({type(e).__name__}: {e}), using extractive fallback")
        max_sent = settings.get("extractive_max_sentences", 15)
        extracted = _extractive_summary(_clean_text(raw_text, strip_urls=True), max_sentences=max_sent, max_chars=5000)
        if lang in ("he", "iw"):
            hebrew_summary = f"[Extractive summary]\n\n{extracted}"
            english_summary = ""
        else:
            hebrew_summary = ""
            english_summary = f"[Extractive summary]\n\n{extracted}"
        pipeline_steps.append(f"Summary: extractive ({max_sent} sentences, BART unavailable: {type(e).__name__})")
        pipeline_steps.append("תרגום: — (לא בוצע)")

    return _format_output(episode, hebrew_summary, english_summary, urls, pipeline_steps)
