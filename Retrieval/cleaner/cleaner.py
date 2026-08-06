import re


def clean_text(text: str) -> str:
    """
    Cleans a single document string for safe JSON storage.
    Removes control characters, normalizes whitespace,
    and removes URLs.
    """
    # collapse multiple spaces and newlines into single space
    text = re.sub(r'\s+', ' ', text)

    # remove URLs — they add noise not evidence
    text = re.sub(r'http\S+', '', text)

    # remove control characters that break JSON
    # these are invisible characters that cause
    # the unclosed quotes issue you saw
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # remove non-readable characters but keep
    # Indian language unicode ranges
    text = re.sub(r'[^\w\s\u0900-\u097F\u0D00-\u0D7F'
                  r'\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF'
                  r'\u0A00-\u0A7F\u0980-\u09FF\u0600-\u06FF.,!?\'\"()-]',
                  ' ', text)

    # final whitespace cleanup
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def normalize_text_for_dedup(text: str) -> str:
    """
    Normalizes text for fuzzy deduplication by lowercasing and removing
    non-alphanumeric / non-Indic characters.
    """
    text = text.lower()
    text = re.sub(r'[^\w\u0900-\u097F\u0980-\u09FF\u0600-\u06FF]', '', text)
    return text


def deduplicate_documents(documents: list) -> list:
    """
    Removes duplicate and near-duplicate documents retrieved across sites.
    Uses normalized prefix fingerprinting and fuzzy SequenceMatcher overlap.
    """
    import difflib

    unique_documents = []
    seen_fingerprints = []

    for doc in documents:
        if not doc.strip():
            continue

        norm_doc = normalize_text_for_dedup(doc)
        if len(norm_doc) < 50:
            continue

        # Extract 400-char normalized fingerprint
        fingerprint = norm_doc[:400]

        is_dup = False
        for seen_fp in seen_fingerprints:
            # Check exact substring containment or high fuzzy similarity
            if fingerprint == seen_fp[:len(fingerprint)] or seen_fp == fingerprint[:len(seen_fp)]:
                is_dup = True
                break
            if difflib.SequenceMatcher(None, fingerprint, seen_fp).ratio() >= 0.75:
                is_dup = True
                break

        if not is_dup:
            seen_fingerprints.append(fingerprint)
            unique_documents.append(doc)

    return unique_documents




ADULT_KEYWORDS = [
    "porn", "xxx", "nude", "naked", "sex video",
    "chudai", "adult content", "18+",
]

def is_adult_content(text: str) -> bool:
    """
    Returns True if document contains adult content.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ADULT_KEYWORDS)

def clean_documents(documents: list) -> list:
    """
    Entry point: cleans each document then deduplicates.
    Filters out adult content.
    """
    # filter adult content first
    filtered = [doc for doc in documents if not is_adult_content(doc)]

    if len(filtered) < len(documents):
        print(f"[CLEAN] Filtered {len(documents) - len(filtered)} adult content documents")

    # clean each document
    cleaned = [clean_text(doc) for doc in filtered if doc.strip()]

    # deduplicate
    return deduplicate_documents(cleaned)