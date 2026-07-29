import json
import sys

from ddgs import DDGS
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from language.detector import detect_language
from sources.wikipedia import fetch_from_wikipedia
from sources.websearch import fetch_from_web, fetch_page_text
from cleaner.cleaner import clean_documents


# fallback language used when detection fails outright
DEFAULT_LANG = "en"

# Indian fact checking sites
# specifically designed to verify social media claims
FACTCHECK_SITES = [
    "boomlive.in",
    "altnews.in",
    "factchecker.in",
    "vishvasnews.com",
    "newschecker.in",
    "indiatoday.in/fact-check",
    "thelogicalindian.com/fact-check",
    "ndtv.com/fact-check",
]


def load_claims(input_path: str) -> list:
    """
    Loads the input JSON file containing claims.
    Expected shape: a list of {ID, Text} objects.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_field(item: dict, *keys, default=""):
    """
    Fetches the first matching key from a dict, trying multiple
    possible key names/cases.
    """
    for key in keys:
        if key in item:
            return item[key]
    return default


def translate_codemix_to_english(text: str) -> str:
    """
    Translates Romanized Hindi/Urdu codemix to English.
    Forces source language as Hindi so Google Translate
    correctly converts Roman Hindi to English.
    """
    try:
        result = GoogleTranslator(
            source="hi", target="en"
        ).translate(text)
        print(f"[CM] Translated to English: {result[:80]}")
        return result
    except Exception as e:
        print(f"[CM] Translation failed: {e}. Using original.")
        return text


def fetch_factcheck_results(query: str) -> list:
    """
    Searches Indian fact checking websites with the query.
    These sites specifically verify Indian social media claims.
    Returns list of document strings.
    """
    documents = []

    for site in FACTCHECK_SITES:
        site_query = f"{query} site:{site}"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    site_query,
                    region="in-en",
                    max_results=2
                ))

            for result in results:
                url = result.get("href", "")
                if not url:
                    continue

                text = fetch_page_text(url)
                if text:
                    documents.append(text[:50000])
                    print(f"[FACTCHECK] Fetched: {url[:60]}")

        except Exception as e:
            print(f"[FACTCHECK] Failed for {site}: {e}")
            continue

    print(f"[FACTCHECK] Total docs from fact checkers: {len(documents)}")
    return documents


def process_claim(item: dict, force_lang: str = None) -> dict:
    """
    Runs the full retrieval pipeline for a single claim.
    For codemix — translates to English then searches
    fact checkers + Wikipedia + web.
    For other languages — normal pipeline.
    Returns {id, text, documents}.
    """
    claim_id   = get_field(item, "ID", "id", "claim_id")
    claim_text = get_field(item, "Text", "text", "claim", "Claim")

    if not claim_text:
        print(f"[MAIN] ID:{claim_id} — Empty claim text. Skipping.")
        return None

    # ── CODEMIX PIPELINE ──────────────────────────────────────────
    if force_lang == "cm":
        print(f"[MAIN] ID:{claim_id} — Codemix claim detected.")
        print(f"[MAIN] Using translate → factcheck → Wikipedia → web strategy.")

        all_docs = []

        # Step 1 — translate codemix to English
        english_claim = translate_codemix_to_english(claim_text)

        # Step 2 — search fact checker sites with English query
        # most important for social media claims
        print(f"[MAIN] Step 1: Searching fact checker sites...")
        factcheck_docs = fetch_factcheck_results(english_claim)
        all_docs.extend(factcheck_docs)

        # Step 3 — search Wikipedia with English query
        print(f"[MAIN] Step 2: Searching Wikipedia...")
        wiki_docs = fetch_from_wikipedia(english_claim, "en")
        all_docs.extend(wiki_docs)

        # Step 4 — search web with English query
        print(f"[MAIN] Step 3: Searching web...")
        web_docs = fetch_from_web(english_claim, "en")
        all_docs.extend(web_docs)

        # Step 5 — if still not enough docs
        # retry with original Roman text
        if len(all_docs) < 3:
            print(f"[MAIN] Low docs — retrying with original text...")
            wiki_retry = fetch_from_wikipedia(claim_text, "en")
            web_retry  = fetch_from_web(claim_text, "en")
            all_docs.extend(wiki_retry)
            all_docs.extend(web_retry)

    # ── NORMAL PIPELINE ───────────────────────────────────────────
    else:
        # use forced language or auto detect
        if force_lang:
            lang_code = force_lang
            print(f"[MAIN] ID:{claim_id} — Using forced lang: {lang_code}")
        else:
            lang_code = detect_language(claim_text, claim_id=claim_id)
            if lang_code is None:
                print(f"[MAIN] ID:{claim_id} — Language undetected. "
                      f"Falling back to '{DEFAULT_LANG}'.")
                lang_code = DEFAULT_LANG

        # retrieve from both sources
        wiki_docs = fetch_from_wikipedia(claim_text, lang_code)
        web_docs  = fetch_from_web(claim_text, lang_code)
        all_docs  = wiki_docs + web_docs

        # retry with first 5 keywords if no documents found
        if len(all_docs) == 0:
            print(f"[MAIN] ID:{claim_id} — No documents found. "
                  f"Retrying with keywords...")
            keywords = " ".join(claim_text.split()[:5])
            print(f"[MAIN] Retry query: {keywords}")
            wiki_retry = fetch_from_wikipedia(keywords, lang_code)
            web_retry  = fetch_from_web(keywords, lang_code)
            all_docs   = wiki_retry + web_retry
            print(f"[MAIN] ID:{claim_id} — Retry count: {len(all_docs)}")

    # ── CLEAN AND DEDUPLICATE ─────────────────────────────────────
    documents = clean_documents(all_docs)

    print(f"[MAIN] ID:{claim_id} — Final document count: {len(documents)}")

    return {
        "id":        claim_id,
        "text":      claim_text,
        "documents": documents,
    }


def main(input_path: str, output_path: str, force_lang: str = None):
    claims = load_claims(input_path)
    print(f"[MAIN] Loaded {len(claims)} claims from {input_path}")

    if force_lang:
        print(f"[MAIN] Language override: all claims → '{force_lang}'")

    results = []

    for i, item in enumerate(claims, start=1):
        print(f"\n[MAIN] Processing claim {i}/{len(claims)}")
        record = process_claim(item, force_lang=force_lang)
        if record is not None:
            results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[MAIN] Done. Wrote {len(results)} records to {output_path}")


if __name__ == "__main__":
    # usage:
    # python main.py input.json output.json        ← auto detect
    # python main.py input.json output.json hi     ← force Hindi
    # python main.py input.json output.json cm     ← force codemix
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_json> <output_json> [language]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    force_lang  = sys.argv[3] if len(sys.argv) == 4 else None

    main(input_file, output_file, force_lang)