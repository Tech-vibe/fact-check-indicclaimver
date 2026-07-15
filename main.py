import json
import sys

from language.detector import detect_language
from sources.wikipedia import fetch_from_wikipedia
from sources.websearch import fetch_from_web
from cleaner.cleaner import clean_documents


# fallback language used when detection fails outright
DEFAULT_LANG = "en"


def load_claims(input_path: str) -> list:
    """
    Loads the input JSON file containing claims.
    Expected shape: a list of {id/ID, claim, label} objects.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_field(item: dict, *keys, default=""):
    """
    Fetches the first matching key from a dict, trying multiple
    possible key names/cases — input JSON field naming isn't
    guaranteed to be consistent (e.g. "id" vs "ID", "claim" vs "Claim").
    """
    for key in keys:
        if key in item:
            return item[key]
    return default


def process_claim(item: dict) -> dict:
    """
    Runs the full retrieval pipeline for a single claim:
    detect language -> fetch from wikipedia -> fetch from web
    -> clean/dedupe -> return final record with just
    {id, text, documents}.
    """
    claim_id = get_field(item, "ID", "id", "claim_id")
    claim_text = get_field(item, "Text", "text", "claim", "Claim")

    if not claim_text:
        print(f"[MAIN] ID:{claim_id} — Empty claim text. Skipping.")
        return None

    # detect language; fall back to English if detection fails
    # rather than dropping the claim entirely, so every claim in
    # the input still gets an attempt at retrieval
    lang_code = detect_language(claim_text, claim_id=claim_id)
    if lang_code is None:
        print(f"[MAIN] ID:{claim_id} — Language undetected. "
              f"Falling back to '{DEFAULT_LANG}'.")
        lang_code = DEFAULT_LANG

    # retrieve from both sources
    wiki_docs = fetch_from_wikipedia(claim_text, lang_code)
    web_docs = fetch_from_web(claim_text, lang_code)

    # combine, then dedupe cross-source duplicates (e.g. same
    # wikipedia article surfacing from both fetchers)
    all_docs = wiki_docs + web_docs
    documents = clean_documents(all_docs)

    print(f"[MAIN] ID:{claim_id} — Final document count: {len(documents)}")

    return {
        "id": claim_id,
        "text": claim_text,
        "documents": documents,
    }


def main(input_path: str, output_path: str):
    claims = load_claims(input_path)
    print(f"[MAIN] Loaded {len(claims)} claims from {input_path}")

    results = []

    for i, item in enumerate(claims, start=1):
        print(f"\n[MAIN] Processing claim {i}/{len(claims)}")
        record = process_claim(item)
        if record is not None:
            results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[MAIN] Done. Wrote {len(results)} records to {output_path}")


if __name__ == "__main__":
    # usage: python main.py input.json output.json
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_json> <output_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)