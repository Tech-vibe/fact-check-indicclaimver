import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from language.mapper import get_wiki_lang


def fetch_from_wikipedia(claim: str, lang_code: str) -> list:
    """
    Searches Wikipedia in the correct language for the claim.
    Returns list of full article texts.
    """

    # get the correct wikipedia language prefix
    # e.g. "hi" → hi.wikipedia.org
    wiki_lang = get_wiki_lang(lang_code)

    # tell wikipedia library which language site to use
    wikipedia.set_lang(wiki_lang)

    print(f"[WIKI] Searching {wiki_lang}.wikipedia.org for: {claim[:50]}")

    documents = []

    try:
        # search for top 3 matching article titles
        search_results = wikipedia.search(claim, results=3)

        if not search_results:
            print(f"[WIKI] No results found for: {claim[:50]}")
            return []

        print(f"[WIKI] Found titles: {search_results}")

        for title in search_results:
            try:
                # fetch full article, auto_suggest=False means
                # use exact title dont let wikipedia redirect
                page = wikipedia.page(title, auto_suggest=False)

                # skip stub articles under 100 characters
                if page.content and len(page.content.strip()) > 100:
                    documents.append(page.content)
                    print(f"[WIKI] Fetched: '{title}' "
                          f"({len(page.content)} chars)")

            except DisambiguationError as e:
                # title points to multiple articles
                # try first suggestion
                print(f"[WIKI] Disambiguation for '{title}' "
                      f"— trying '{e.options[0]}'")
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    if page.content and len(page.content.strip()) > 100:
                        documents.append(page.content)
                        print(f"[WIKI] Fetched disambiguated: '{e.options[0]}'")
                except:
                    print(f"[WIKI] Disambiguation fetch failed. Skipping.")
                    continue

            except PageError:
                # wikipedia returned title but page doesnt exist
                print(f"[WIKI] Page not found for '{title}'. Skipping.")
                continue

            except Exception as e:
                # any other unexpected error
                print(f"[WIKI] Unexpected error for '{title}': {e}")
                continue

    except Exception as e:
        print(f"[WIKI] Search failed: {e}")

    print(f"[WIKI] Total documents fetched: {len(documents)}")
    return documents