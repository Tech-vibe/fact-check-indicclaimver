import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
from language.mapper import get_wiki_lang


def _fetch_wiki(claim: str, wiki_lang: str) -> list:
    """
    Internal function — searches one Wikipedia language site.
    Returns list of full article texts.
    """
    # switch Wikipedia to correct language
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
                # fetch full article
                # auto_suggest=False means use exact title
                # dont let wikipedia redirect to different article
                page = wikipedia.page(title, auto_suggest=False)

                # skip stub articles under 100 characters
                if page.content and len(page.content.strip()) > 100:
                    documents.append(page.content)
                    print(f"[WIKI] Fetched: '{title}' "
                          f"({len(page.content)} chars)")

            except DisambiguationError as e:
                # title points to multiple articles
                # try first suggestion from disambiguation list
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

    print(f"[WIKI] Total documents fetched from {wiki_lang}: {len(documents)}")
    return documents


def fetch_from_wikipedia(claim: str, lang_code: str) -> list:
    """
    Searches Wikipedia in the correct language for the claim.
    For codemix claims — searches both English and Hindi Wikipedia
    since codemix is Romanized Hindi written in English script.
    Returns list of full article texts.
    """
    if lang_code == "cm":
        # codemix → search both English and Hindi Wikipedia
        # combines results from both for maximum coverage
        print(f"[WIKI] Codemix claim — searching en + hi Wikipedia")
        en_docs = _fetch_wiki(claim, "en")
        hi_docs = _fetch_wiki(claim, "hi")
        return en_docs + hi_docs

    else:
        # all other languages → search correct language Wikipedia
        wiki_lang = get_wiki_lang(lang_code)
        return _fetch_wiki(claim, wiki_lang)