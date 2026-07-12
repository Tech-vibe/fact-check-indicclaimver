DDGO_region_map = {
    "en" : "in-en",
    "hi" : "in-hi",
    "bn" : "in-bn",
    "mr" : "in-mr",
}


WIKI_lang_map = {
    "en" : "en",
    "hi" : "hi",
    "bn" : "bn",
    "mr" : "mr",
}


def get_ddgo_region(lang_code: str) -> str:
    if lang_code not in DDGO_region_map:
        print(f"[WARN] mapper.py — Unknown lang code: {lang_code}. "
              f"This should not happen. Defaulting to in-en.")
    return DDGO_region_map.get(lang_code, "in-en")

def get_wiki_lang(lang_code: str) -> str:
    if lang_code not in WIKI_lang_map:
        print(f"[WARN] mapper.py — Unknown lang code: {lang_code}. "
              f"This should not happen. Defaulting to en.")
    return WIKI_lang_map.get(lang_code, "en")