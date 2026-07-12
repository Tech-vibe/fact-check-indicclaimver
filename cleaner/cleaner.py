from langdetect import detect, LangDetectException
import os

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/detection_failures.log"

def log_failure(claim: str, detected_lang: str, reason: str):
    """
    Writes failed detections to a log file.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"REASON   : {reason}\n")
        f.write(f"DETECTED : {detected_lang}\n")
        f.write(f"CLAIM    : {claim}\n")
        f.write(f"{'-'*50}\n")

def detect_language(claim: str, claim_id: str = "unknown") -> str:
    try:
        lang = detect(claim)
        
        if lang in SUPPORTED_LANGUAGES:
            print(f"[LANG] detected: {lang} ({SUPPORTED_LANGUAGES[lang]})")
            return lang
        else:
            print(f"[WARN] ID:{claim_id} — Unsupported language '{lang}'. Defaulting to 'en'")
            log_failure(claim, lang, "Unsupported language")
            return "en"
    except LangDetectException:
        print(f"[WARN] ID:{claim_id} — Detection failed. Defaulting to 'en'")
        log_failure(claim, "unknown", "LangDetect exception")
        return "en"