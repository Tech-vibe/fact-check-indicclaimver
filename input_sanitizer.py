"""
input_sanitizer.py
==================
Pre-processing module for the IndicClaimVerifier LLM pipeline.

Purpose
-------
Sanitizes raw claim items BEFORE they are fed to the LLM, fixing the two
categories of JSON parse errors observed in production:

  Error 1 | "Expecting ',' delimiter"  (e.g. claim S2/T/EN/1006)
  -----------------------------------------------------------------
  Root cause : Evidence texts scraped from the web contain literal
               backslash-quote sequences (\") which the model reproduces
               verbatim inside its own JSON output. A bare backslash
               followed by a quote breaks Python's JSON parser.
  Fix        : Replace every \" and \\ in the evidence with safe
               equivalents before the text ever reaches the model.

  Error 2 | "Unterminated string starting at"  (e.g. S2/T/EN/1009/1010)
  -----------------------------------------------------------------------
  Root cause : Some evidence fields are 5,000-10,000+ characters long.
               The total prompt size causes the model to hit MAX_TOKENS
               before it can close the JSON strings in its response.
  Fix        : Truncate each evidence field to EVIDENCE_MAX_CHARS so the
               model always has enough token budget to finish its output.

Architecture
------------
PreprocessingPipeline runs sanitize_item() in a background daemon thread,
staying PREFETCH_BUFFER_SIZE items ahead of the main LLM thread via a
bounded queue.

  Timeline example (prefetch = 3):

    Background thread  ->  sanitize[2], sanitize[3], sanitize[4], blocks
    Main thread        ->  LLM(item[0]),      LLM(item[1]),      ...

  The sanitizer costs only a few milliseconds per item (pure CPU string
  work), so it always finishes well before the LLM call completes (5-30s).
  The critical path latency is UNCHANGED.

Public API
----------
  sanitize_text(text, max_chars, field_name) -> str
      Clean a single text value.

  sanitize_item(item) -> dict
      Return a sanitized shallow copy of one claim dict.

  PreprocessingPipeline(items, prefetch)
      Iterable that yields sanitized items in order, pre-processed in the
      background. Call .start() before iterating.
"""

import re
import queue
import threading
import logging
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ---- Module-level probe for indic-nlp-library --------------------------------
#
# Import the normalizer factory once at startup. If the library is not
# installed, _INDIC_NLP_AVAILABLE stays False and the normalizer step is
# silently skipped inside _normalize_indic_unicode().

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory as _IndicNormalizerFactory
    _indic_normalizer_factory = _IndicNormalizerFactory()
    _INDIC_NLP_AVAILABLE = True
    logger.info("[SANITIZER] indic-nlp-library loaded — Indic Unicode normalization active.")
except Exception as _indic_exc:
    _indic_normalizer_factory = None
    _INDIC_NLP_AVAILABLE = False
    logger.info("[SANITIZER] indic-nlp-library not available (%s). Using unicodedata NFC fallback.", _indic_exc)

# Per-language normalizer cache so we don't reconstruct on every call.
# Key: indicnlp language code string ('bn', 'hi', etc.)
_normalizer_cache: dict = {}

# ---- Tuning constants -------------------------------------------------------

# Max characters per evidence field after sanitization.
# Budget math (assuming 2048-token context window):
#   - System prompt:         ~300 tokens
#   - User prompt structure: ~200 tokens
#   - Claim text:            ~  50 tokens
#   - 3x evidence at 600 chars each (3.5 chars/tok): ~514 tokens
#   - Total input:           ~1064 tokens
#   - Remaining for output:    984 tokens  (well above MAX_TOKENS=800)
# Keeping this at 600 ensures the model ALWAYS has room to finish its JSON.
EVIDENCE_MAX_CHARS = 600

# The claim (Text field) is always short, but guard against edge cases.
CLAIM_MAX_CHARS = 500

# How many pre-processed items to buffer ahead of the LLM thread.
PREFETCH_BUFFER_SIZE = 3

# ---- Compiled regex patterns ------------------------------------------------

# C0/C1 control characters EXCEPT \n (0x0A) and \t (0x09).
# Invisible characters that silently corrupt a JSON string if echoed by LLM.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# ---- Unicode normalization map ----------------------------------------------

# Typographic quotes and dashes are common in scraped legal text.
# Replace with ASCII equivalents so the model cannot produce bad escapes.
_UNICODE_NORMALIZE_MAP = str.maketrans({
    '\u2018': "'",    # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",    # RIGHT SINGLE QUOTATION MARK
    '\u201a': "'",    # SINGLE LOW-9 QUOTATION MARK
    '\u201b': "'",    # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    '\u201c': '"',    # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',    # RIGHT DOUBLE QUOTATION MARK
    '\u201e': '"',    # DOUBLE LOW-9 QUOTATION MARK
    '\u201f': '"',    # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    '\u2032': "'",    # PRIME
    '\u2033': '"',    # DOUBLE PRIME
    '\u2013': '-',    # EN DASH
    '\u2014': '-',    # EM DASH
    '\u00ad': '',     # SOFT HYPHEN (invisible, causes JSON issues)
    '\u200b': '',     # ZERO WIDTH SPACE
    '\u200c': '',     # ZERO WIDTH NON-JOINER
    '\u200d': '',     # ZERO WIDTH JOINER
    '\ufeff': '',     # BOM
})


# ---- Low-level text fixers --------------------------------------------------

def _fix_backslash_quotes(text: str) -> str:
    """
    Replaces backslash-escaped quotes and unescaped double-quotes with single-quotes
    so that embedded context sources inside LLM prompts never contain double-quotes.
    This prevents the LLM from echoing unescaped double quotes inside its JSON string fields.
    """
    text = text.replace('\\\\', ' ')   # double backslash -> space
    text = text.replace('\\"', "'")    # backslash-quote  -> single quote
    text = text.replace('"', "'")      # raw double-quote -> single quote
    text = text.replace('\\', ' ')     # remaining backslash -> space
    return text


def _remove_control_chars(text: str) -> str:
    """Remove C0/C1 control characters (except newline and tab)."""
    return _CONTROL_CHAR_RE.sub(' ', text)


def _normalize_unicode(text: str) -> str:
    """Replace typographic/Unicode characters with ASCII equivalents."""
    return text.translate(_UNICODE_NORMALIZE_MAP)


def _normalize_whitespace(text: str) -> str:
    """
    Replace ALL whitespace variants with a single space and strip.

    Crucially, newlines (\n) and tabs (\t) are converted to spaces here,
    NOT kept. This prevents the LLM from echoing a literal newline or tab
    inside its JSON output string, which would cause an
    'Invalid control character' JSON parse error.
    """
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)   # all whitespace -> space
    text = re.sub(r' {2,}', ' ', text)              # collapse runs of spaces
    return text.strip()


def _truncate(text: str, max_chars: int, field_name: str) -> str:
    """
    Truncate text to at most max_chars characters, breaking on a word
    boundary where possible. Appends an ellipsis to indicate truncation.
    """
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars].rsplit(' ', 1)
    short = cut[0] if len(cut) > 1 else text[:max_chars]
    result = short.rstrip() + '...'

    logger.debug(
        "[SANITIZER] '%s' truncated: %d -> %d chars",
        field_name, len(text), len(result)
    )
    return result


def _clean_web_scraped_boilerplate(text: str) -> str:
    """
    Strips scraped website header/footer navigation bars, site menus, and social share links
    (e.g., 'প্রচ্ছদ জাতীয় রাজনীতি সারাদেশ আন্তর্জাতিক অর্থনীতি খেলা ক্রিকেট বিনোদন...').
    Finds the main article body text so evidence passed to the LLM contains actual facts,
    not site navigation menus.
    """
    nav_pattern = r'(?:প্রচ্ছদ|জাতীয়|রাজনীতি|সারাদেশ|আন্তর্জাতিক|অর্থনীতি|খেলা|ক্রিকেট|বিনোদন|আইন|আদালত|ধর্ম|লাইফস্টাইল|অন্যান্য|খুলনা|চট্টগ্রাম|ঢাকা|বরিশাল|ময়মনসিংহ|রংপুর|রাজশাহী|সিলেট|অপরাধ|শিক্ষা|প্রবাস|তারকা|বিজ্ঞান|প্রযুক্তি|হোম|সংবাদ|বিশেষ)\s*'
    text = re.sub(r'(' + nav_pattern + r'){3,}', ' ', text)

    text = re.sub(r'\b(Home|News|Politics|Sports|Entertainment|Lifestyle|Business|Opinion|Tech|Science|World|India|National|State|City|Photos|Videos|Share|Tweet|Facebook|Twitter|WhatsApp|Telegram|Subscribe|Privacy Policy|Terms of Use)\b\s*', ' ', text, flags=re.IGNORECASE)

    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _remove_mojibake_artifacts(text: str) -> str:
    """
    Removes Mojibake, multi-byte UTF-8 decoding corruption, and repetitive
    Latin-1 byte sequences (e.g. 'à à à à à ²à à à à à à à ¾...') caused by
    web scrapers reading Indic UTF-8 bytes as Windows-1252 / ISO-8859-1.
    
    Stripping these sequences pulls the actual informative body text forward
    before character/token truncation occurs.
    """
    # Pattern for repeated latin-1 artifact chars and replacement chars: à, ¾, ², ¹, ª, â, µ, , etc.
    mojibake_re = re.compile(r'(?:[\u00e0\u00be\u00b2\u00b9\u00aa\u00e2\u00b5\ufffd\xa0]\s*|[\.,]\s*){3,}')
    text = mojibake_re.sub(' ', text)
    text = re.sub(r'[,\.\s]{3,}', ' ', text)
    return text


def _normalize_indic_unicode(text: str, lang_code: str = 'hi') -> str:
    """
    Normalize Indic Unicode codepoints to NFC canonical form so that two
    visually-identical strings with different byte representations (e.g.
    precomposed vs decomposed vowel matras in Bengali/Hindi) are treated
    as identical by downstream text matchers and the LLM tokenizer.

    Uses IndicNormalizerFactory from indic-nlp-library when available,
    which handles script-specific edge cases (nukta, anusvara, chandrabindu,
    half-consonant forms) for 12+ languages.

    Fallback (always safe): Python's built-in unicodedata.normalize('NFC', text)
    which handles standard Unicode composition but lacks Indic-specific rules.

    Args:
        text:      Input text string.
        lang_code: ISO 639-1 code for the script ('bn', 'hi', 'ta', 'te',
                   'kn', 'ml', 'gu', 'pa'). Defaults to 'hi' (Devanagari).
                   Ignored when using the NFC fallback.

    Returns:
        Normalized text string.
    """
    if _INDIC_NLP_AVAILABLE:
        try:
            # Retrieve or build a cached normalizer for this language
            if lang_code not in _normalizer_cache:
                _normalizer_cache[lang_code] = _indic_normalizer_factory.get_normalizer(lang_code)
            normalizer = _normalizer_cache[lang_code]
            return normalizer.normalize(text)
        except Exception as exc:
            logger.debug("[SANITIZER] indic-nlp normalize error for lang='%s' (%s). Using NFC.", lang_code, exc)

    # NFC fallback: standard Unicode canonical decomposition + recomposition
    return unicodedata.normalize('NFC', text)


# ---- Public sanitize API ----------------------------------------------------

def sanitize_text(text: Any, max_chars: int, field_name: str, lang_code: str = 'hi') -> str:
    """
    Apply the full sanitization pipeline to a single text field.

    Pipeline (order is significant):
      1. Coerce to str (handles None and non-string values).
      2. Fix backslash-quote sequences  -> eliminates JSON-breaking escapes.
      3. Remove C0/C1 control characters.
      4. Normalize Unicode typographic characters -> ASCII equivalents.
      4b. Normalize Indic Unicode codepoints to NFC canonical form
          (via indic-nlp-library if available, unicodedata.normalize NFC otherwise).
      5. Remove Mojibake and UTF-8 byte decoding artifacts (à à à à à ²...).
      6. Strip web-scraped site navigation menus / headers.
      7. Normalize whitespace.
      8. Truncate to max_chars            -> prevents token overflow.

    Args:
        text:       Raw input value from the JSON record.
        max_chars:  Maximum allowed character count after sanitization.
        field_name: Label used in log/debug messages only.
        lang_code:  ISO 639-1 language code used by indic-nlp-library normalizer
                    (e.g. 'bn', 'hi', 'ta'). Defaults to 'hi'. Ignored when
                    indic-nlp-library is not installed.

    Returns:
        A clean, safe string ready to embed in the LLM prompt.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ''

    text = _fix_backslash_quotes(text)
    text = _remove_control_chars(text)
    text = _normalize_unicode(text)
    text = _normalize_indic_unicode(text, lang_code=lang_code)   # step 4b
    text = _remove_mojibake_artifacts(text)
    text = _clean_web_scraped_boilerplate(text)
    text = _normalize_whitespace(text)
    text = _truncate(text, max_chars, field_name)
    return text


def sanitize_item(item: dict) -> dict:
    """
    Sanitize all text fields in a single claim item dict.

    Returns a *new* dict (shallow copy) with sanitized values.
    The original dict is never modified — safe to call from a background
    thread without locking.

    Fields sanitized:
        'Text'              -> up to CLAIM_MAX_CHARS characters
        'Evidence'          -> up to EVIDENCE_MAX_CHARS * 3 characters

    Any other fields (e.g. 'ID') are copied through unchanged.

    Args:
        item: A raw claim dict loaded from topk_output.json.

    Returns:
        A sanitized claim dict.
    """
    sanitized = dict(item)   # shallow copy

    claim_val = sanitized.get('Text') if sanitized.get('Text') is not None else sanitized.get('claim', '')
    sanitized['Text'] = sanitize_text(
        claim_val,
        CLAIM_MAX_CHARS,
        'Text'
    )

    for k1, k2 in (('Evidence', 'evidence'),):
        raw = sanitized.get(k1) if sanitized.get(k1) is not None else sanitized.get(k2, '')
        if raw:
            sanitized[k1] = sanitize_text(raw, EVIDENCE_MAX_CHARS * 3, k1)

    return sanitized


# ---- PreprocessingPipeline --------------------------------------------------

class PreprocessingPipeline:
    """
    Runs sanitize_item() in a background daemon thread, staying
    PREFETCH_BUFFER_SIZE items ahead of the consumer (LLM) thread.

    Design goals
    ------------
    - Zero latency impact: sanitizer runs during the LLM's inference time
      (~5-30 s per claim); sanitization itself takes < 5 ms.
    - Bounded memory: queue has a hard max so pre-processor cannot race ahead.
    - Crash-safe: if sanitize_item() throws unexpectedly, the raw item is
      passed through unchanged so the pipeline never stalls.
    - Daemon thread: will not prevent the process from exiting on Ctrl+C.

    Usage
    -----
        pipeline = PreprocessingPipeline(pending_claims)
        pipeline.start()

        for sanitized_item in pipeline:     # blocking get from queue
            result = process_claim(sanitized_item)

        # No explicit join() needed; daemon thread exits with the process.

    Thread Safety
    -------------
        Producer (background thread) : puts to queue only.
        Consumer (main thread)       : gets from queue only.
        queue.Queue is thread-safe; no additional locking is required.
    """

    _SENTINEL = object()   # signals end-of-stream; cannot be confused with data

    def __init__(self, items: list, prefetch: int = PREFETCH_BUFFER_SIZE):
        """
        Args:
            items:    Ordered list of raw claim dicts to pre-process.
            prefetch: Max number of pre-processed items to hold in the queue.
        """
        self._items  = items
        self._queue  = queue.Queue(maxsize=prefetch)
        self._thread = threading.Thread(
            target=self._worker,
            name='SanitizerThread',
            daemon=True
        )

    def _worker(self) -> None:
        """Background thread: sanitize each item and push it to the queue."""
        for item in self._items:
            try:
                sanitized = sanitize_item(item)
            except Exception as exc:
                logger.warning(
                    "[SANITIZER] Unexpected error for ID=%s (%s). "
                    "Passing raw item through.",
                    item.get('ID', '?'), exc
                )
                sanitized = item

            # Blocks if the queue is full — this is intentional and prevents
            # the pre-processor from running far ahead of the LLM thread.
            self._queue.put(sanitized)

        self._queue.put(self._SENTINEL)   # signal end-of-stream

    def start(self) -> None:
        """Start the background sanitizer thread."""
        self._thread.start()
        logger.info(
            "[SANITIZER] Pre-processing pipeline started "
            "(prefetch buffer = %d item(s)).",
            self._queue.maxsize
        )

    def __iter__(self):
        """
        Yield sanitized items one by one, blocking until each is ready.
        Returns when the sentinel is encountered (end of input list).
        """
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            yield item
