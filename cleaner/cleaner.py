def deduplicate_documents(documents: list) -> list:
    """
    Removes duplicate/near-duplicate documents that may appear in
    both the wikipedia and websearch results (e.g. the same
    Wikipedia article fetched once via wikipedia.py and again via
    websearch.py surfacing it as a search result).

    Uses the first 300 characters as a lightweight fingerprint —
    two documents starting identically are almost certainly the
    same source, so this is enough without needing full hashing
    or similarity comparison.
    """
    seen_signatures = set()
    unique_documents = []

    for doc in documents:
        signature = doc[:300]

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        unique_documents.append(doc)

    return unique_documents


def clean_documents(documents: list) -> list:
    """
    Entry point: takes the combined raw document list from
    wikipedia.py + websearch.py and removes cross-source duplicates.
    """
    return deduplicate_documents(documents)