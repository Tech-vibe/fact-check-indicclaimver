from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory
import os

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "mr": "Marathi",
}

MIN_CONF = 0.3
MIN_GAP = 0.2

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/detection_failures.log"

def log_failure(claim: str, detected_lang: str,claim_id: str, reason: str, ):
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"ID       : {claim_id}\n")
        f.write(f"REASON   : {reason}\n")
        f.write(f"DETECTED : {detected_lang}\n")
        f.write(f"CLAIM    : {claim}\n")
        f.write(f"{'-'*50}\n")

def detect_language(claim: str, claim_id: str = "unknown") -> str | None:
    try:
        
        detected = detect_langs(claim)

        supported_found = []

        for d in detected:
            if str(d.lang) in SUPPORTED_LANGUAGES:
                supported_found.append((str(d.lang), round(d.prob, 2)))
        
        if not supported_found:
            all_detected = str(detected)
            print(f"[WARN] ID:{claim_id} — No supported language found. "
            f"Detected: {all_detected}. Skipping.")

            log_failure(claim,all_detected,claim_id,"No supported language found")
            return None
        best_lang, best_conf = supported_found[0]


        if best_conf < MIN_CONF:
            print(f"[LOW CONF] ID:{claim_id} — "
                  f"Low confidence detection: {best_lang}:{best_conf}. "
                  f"Flagging for review but using {best_lang}.")
            log_failure(
                claim,
                f"{best_lang}:{best_conf}",
                claim_id,
                f"Low confidence below {MIN_CONF}"
            )
            return best_lang


        if len(supported_found) > 1:
            second_lang, second_conf = supported_found[1]
            gap = round(best_conf - second_conf, 2)

            if gap < MIN_GAP:
                print(f"[LOW CONF] ID:{claim_id} — "
                      f"Close detection: {best_lang}:{best_conf} vs "
                      f"{second_lang}:{second_conf}. "
                      f"Using {best_lang} but flagging for review.")
                log_failure(
                    claim,
                    f"{best_lang}:{best_conf} vs {second_lang}:{second_conf}",
                    claim_id,
                    "Low confidence — close detection between two supported languages"
                )

                return best_lang
            #remember,best_lang is returned but still flaged for review
        
        print(f"[LANG] ID:{claim_id} — Detected: {best_lang} "
              f"({SUPPORTED_LANGUAGES[best_lang]}) "
              f"confidence: {best_conf}")
        return best_lang
    except LangDetectException:
        print(f"[WARN] ID:{claim_id} — Detection failed. Skipping")
        log_failure(claim, "unknown",claim_id, "LangDetect exception")
        return None