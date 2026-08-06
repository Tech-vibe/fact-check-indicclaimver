import os
import sys
import json
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

# Indian fact checking, legal news, mainstream media, and social verification sites
# specifically designed to verify social media, legal, and Indic language claims
FACTCHECK_SITES = [
    # Fact-checking & Verification
    "boomlive.in",
    "altnews.in",
    "factchecker.in",
    "vishvasnews.com",
    "newschecker.in",
    "thelogicalindian.com/fact-check",
    "facebook.com",

    # Regional & Mainstream Indian News
    "india.com",
    "timesofindia.indiatimes.com",
    "sangbadpratidin.in",
    "anandabazar.com",
    "kolkata24x7.in",
    "bengali.abplive.com",
    "eisamay.com",
    "tv9bangla.com",
    "jansatta.com",
    "ndtv.com",
    "bhaskar.com",
    "indiatv.in",
    "indiatvnews.com",
    "thehindu.com",
    "ddnews.gov.in",
    "indianexpress.com",
    "dnaindia.com",
    "indiatoday.in",
    "aninews.in",
    "deccanchronicle.com",

    # Legal, Judicial & Official Portals
    "livelaw.in",
    "legalservicesindia.com",
    "indialegallive.com",
    "freelaw.in",
    "legaleagleweb.com",
    "scobserver.in",
    "24law.in",
    "lawbeat.in",
    "verdictum.in",
    "scconline.com",
    "primelegal.in",
    "sechimachal.hp.gov.in",
    "landconflictwatch.org",
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


def save_output_atomic(target_path: Path, data: list):
    """
    Atomically writes data as JSON to target_path using a temporary .tmp file swap.
    Guarantees downstream listeners (like Ranker) always read clean, valid JSON.
    """
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, target_path)


def resolve_default_paths():
    """
    Resolves default input and output paths for the Indic_Claim_Ver pipeline architecture.
    Input:  Indic_Claim_Ver/Input/ (finds claims.json or first .json in Input/)
    Output: Indic_Claim_Ver/Output/Retrieved_Output/retrieved_documents.json
    """
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent

    # Input directory / file resolution
    input_dir = parent_dir / "Input"
    default_input = input_dir / "claims.json"
    if not default_input.exists() and input_dir.exists():
        json_files = list(input_dir.glob("*.json"))
        if json_files:
            default_input = json_files[0]
        else:
            default_input = input_dir

    # Output directory / file resolution
    output_dir = parent_dir / "Output" / "Retrieved_Output"
    default_output = output_dir / "retrieved_documents.json"

    return str(default_input), str(default_output)


def main(input_path: str = None, output_path: str = None, force_lang: str = None):
    default_in, default_out = resolve_default_paths()
    input_path  = input_path or default_in
    output_path = output_path or default_out

    claims = load_claims(input_path)
    print(f"[MAIN] Loaded {len(claims)} claims from {input_path}")

    if force_lang:
        print(f"[MAIN] Language override: all claims → '{force_lang}'")

    # If output_path is a directory or path without file extension, write retrieved_documents.json inside it
    out_obj = Path(output_path)
    if out_obj.is_dir() or not out_obj.suffix:
        out_obj.mkdir(parents=True, exist_ok=True)
        target_file = out_obj / "retrieved_documents.json"
    else:
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        target_file = out_obj

    # Load existing results to allow seamless resuming
    results = []
    processed_ids = set()
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results = existing
                    for item in results:
                        item_id = item.get("id") or item.get("ID")
                        if item_id:
                            processed_ids.add(item_id)
            if results:
                print(f"[MAIN] Resuming from existing output ({len(results)} claims already completed).")
        except Exception:
            pass

    for i, item in enumerate(claims, start=1):
        claim_id = get_field(item, "ID", "id", "claim_id")
        if claim_id and claim_id in processed_ids:
            continue

        print(f"\n[MAIN] Processing claim {i}/{len(claims)} (ID: {claim_id})")
        record = process_claim(item, force_lang=force_lang)
        if record is not None:
            results.append(record)
            if claim_id:
                processed_ids.add(claim_id)
            # Atomically save JSON after EVERY SINGLE CLAIM so Ranker streams immediately!
            save_output_atomic(target_file, results)

    print(f"\n[MAIN] Done. Wrote {len(results)} records to {target_file}")


if __name__ == "__main__":
    # Optional CLI Usage:
    # python main.py                                      ← default pipeline paths
    # python main.py input.json output.json               ← auto detect with custom paths
    # python main.py input.json output.json hi            ← force Hindi
    # python main.py input.json output.json cm            ← force codemix
    default_in, default_out = resolve_default_paths()

    input_file  = sys.argv[1] if len(sys.argv) > 1 else default_in
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_out
    force_lang  = sys.argv[3] if len(sys.argv) > 3 else None

    main(input_file, output_file, force_lang)