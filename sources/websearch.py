import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from language.mapper import get_ddgo_region


# domains that render content via JavaScript — requests+BeautifulSoup
# can't see their real content, only an empty shell/boilerplate
SKIP_DOMAINS = [
    "youtube.com", "youtu.be",
    "facebook.com", "instagram.com",
    "twitter.com", "x.com", "tiktok.com",
]

# max characters kept per fetched document — avoids huge PDFs/pages
# from blowing up memory or downstream processing
MAX_DOC_CHARS = 50000

# words shorter than this are ignored when checking relevance
# (skips small connector words that don't carry real meaning)
MIN_KEYWORD_LEN = 3


def get_claim_keywords(claim: str) -> list:
    """
    Splits the claim into significant words for a quick relevance
    check. Works for both English and Indic scripts since we just
    split on whitespace — no language-specific tokenization needed
    for this simple check.
    """
    words = claim.split()
    return [w.strip() for w in words if len(w.strip()) >= MIN_KEYWORD_LEN]


def is_relevant(result: dict, keywords: list) -> bool:
    """
    Cheap relevance check using the title + snippet DuckDuckGo
    already returns with each result (no extra fetch needed).
    Returns True if at least one claim keyword appears in the
    title or body — used to filter out off-topic results before
    spending a network request fetching the full page.
    """
    if not keywords:
        return True  # nothing to check against, don't block anything

    title = result.get("title", "")
    body = result.get("body", "")
    combined = (title + " " + body)

    for keyword in keywords:
        if keyword in combined:
            return True

    return False


def fetch_page_text(url: str) -> str:
    """
    Fetches full text content from a given URL.
    Removes HTML tags and returns clean readable text.
    """
    try:
        # some websites block bots
        # adding a User-Agent header makes us look like a real browser
        headers = {"User-Agent": "Mozilla/5.0"}

        # fetch the page — timeout=10 means
        # if page takes more than 10 seconds give up
        response = requests.get(url, timeout=10, headers=headers)

        # 200 means success
        # anything else means something went wrong
        if response.status_code != 200:
            print(f"[WEB] Failed to fetch {url[:60]} "
                  f"— status: {response.status_code}")
            return ""

        # parse the raw HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # remove all junk tags that aren't readable content
        # script → javascript code
        # style  → css styling
        # nav    → navigation menus
        # footer → page footer
        # header → page header
        # aside  → sidebars
        for tag in soup(["script", "style", "nav",
                         "footer", "header", "aside"]):
            tag.decompose()   # decompose = delete this tag completely

        # extract all remaining text
        # separator=" " puts a space between each text block
        # strip=True removes extra whitespace
        text = soup.get_text(separator=" ", strip=True)

        # only return if text has meaningful content
        if len(text) > 100:
            return text
        return ""

    except Exception as e:
        print(f"[WEB] Page fetch error for {url[:60]}: {e}")
        return ""


def fetch_from_web(claim: str, lang_code: str) -> list:
    """
    Searches DuckDuckGo in the correct language for the claim.
    Runs both a general search and a news-focused search to
    surface both encyclopedic and recent news coverage.
    Fetches full text of top results.
    Returns list of document strings (deduplicated by URL).
    """
    # get correct DuckDuckGo region for this language
    # e.g. "hi" → "in-hi"
    region = get_ddgo_region(lang_code)

    documents = []
    seen_urls = set()
    keywords = get_claim_keywords(claim)

    # two searches: general (broad coverage) and news-focused
    # (recent articles) — combining both gives better coverage
    # for claims that are either encyclopedic facts or news events
    search_queries = [
        claim,
        claim + " news",
    ]

    for query in search_queries:
        print(f"[WEB] Searching DuckDuckGo (region:{region}) for: {query[:50]}")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region=region,
                    max_results=8
                ))

            print(f"[WEB] Found {len(results)} results")

            # loop through each search result
            for result in results:
                url = result.get("href", "")

                # skip if no URL in result
                if not url:
                    continue

                # skip duplicate URLs across the two searches
                if url in seen_urls:
                    print(f"[WEB] Already fetched, skipping duplicate: {url[:60]}")
                    continue

                # skip known JS-rendered / non-article sites — these
                # won't give us readable text since their real content
                # loads via JavaScript, which requests+BeautifulSoup
                # never sees
                if any(domain in url for domain in SKIP_DOMAINS):
                    print(f"[WEB] Skipping non-article URL: {url[:60]}")
                    continue

                # skip results whose title/snippet don't mention any
                # claim keyword — cheap check using data DuckDuckGo
                # already gave us, avoids fetching off-topic pages
                # (e.g. "news" queries drifting to unrelated articles)
                if not is_relevant(result, keywords):
                    print(f"[WEB] Skipping irrelevant result: {url[:60]}")
                    continue

                print(f"[WEB] Fetching: {url[:60]}")

                # fetch full page text from this URL
                text = fetch_page_text(url)

                if text:
                    # cap document length to avoid huge PDFs/pages
                    # blowing up downstream processing
                    documents.append(text[:MAX_DOC_CHARS])
                    seen_urls.add(url)
                    print(f"[WEB] Fetched ({len(text)} chars)")
                else:
                    print(f"[WEB] Empty or failed. Skipping.")

        except Exception as e:
            print(f"[WEB] Search failed for query '{query[:50]}': {e}")

    print(f"[WEB] Total documents fetched: {len(documents)}")
    return documents