import re
import requests
import json
import time
import logging
import sys
from pathlib import Path

# Ensure stdout and stderr handle UTF-8 strings cleanly on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from input_sanitizer import PreprocessingPipeline

# ============================================================
# LOGGING
# ============================================================

class AutoFlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        AutoFlushingStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

SERVER_URL   = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "Qwen3-8B-Q4_K_M.gguf"

INPUT_FILE       = r"D:\IndicClaimVerifier\topk_output.json"
OUTPUT_FILE      = r"D:\IndicClaimVerifier\submission.json"
CHECKPOINT_FILE  = r"D:\IndicClaimVerifier\submission_checkpoint.json"
CHECKPOINT_EVERY = 5    # Save progress to disk every 5 claims

# LLM generation parameters
TEMPERATURE    = 0.2
TOP_P          = 0.9
TOP_K          = 40
REPEAT_PENALTY = 1.0
MAX_TOKENS     = 2048

# Retry settings
MAX_RETRIES    = 3          # Number of retries after the first attempt
RETRY_DELAY    = 2          # Seconds to wait between retries

# Valid prediction labels
VALID_LABELS   = {"SUPPORTS", "REFUTES"}

# ============================================================
# LANGUAGE DETECTION
# ============================================================

LANGUAGE_NAME_MAP = {
    "hindi":     "Hindi",
    "bengali":   "Bengali",
    "tamil":     "Tamil",
    "telugu":    "Telugu",
    "kannada":   "Kannada",
    "malayalam": "Malayalam",
    "gujarati":  "Gujarati",
    "punjabi":   "Punjabi",
    "hinglish":  "Hinglish (Hindi in Roman/English script)",
    "english":   "English",
}

def detect_language(text: str) -> str:
    """
    Detects the script/language of the text without external dependencies.
    First checks native script Unicode blocks, then falls back to a frequency
    check of common Hinglish (Romanized Hindi) stop words.
    """
    text_lower = text.lower()
    
    # 1. Unicode block checks for native scripts
    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    bengali_count = sum(1 for c in text if '\u0980' <= c <= '\u09ff')
    tamil_count = sum(1 for c in text if '\u0b80' <= c <= '\u0bff')
    telugu_count = sum(1 for c in text if '\u0c00' <= c <= '\u0c7f')
    kannada_count = sum(1 for c in text if '\u0c80' <= c <= '\u0cff')
    malayalam_count = sum(1 for c in text if '\u0d00' <= c <= '\u0d7f')
    gujarati_count = sum(1 for c in text if '\u0a80' <= c <= '\u0aff')
    punjabi_count = sum(1 for c in text if '\u0a00' <= c <= '\u0a7f')
    
    counts = {
        "hindi": devanagari_count,
        "bengali": bengali_count,
        "tamil": tamil_count,
        "telugu": telugu_count,
        "kannada": kannada_count,
        "malayalam": malayalam_count,
        "gujarati": gujarati_count,
        "punjabi": punjabi_count,
    }
    
    max_lang, max_val = max(counts.items(), key=lambda x: x[1])
    if max_val > 3:  # Threshold of at least 4 characters to avoid stray symbols
        return max_lang
        
    # 2. Check for Hinglish (Romanized Hindi)
    # Match clean alphabetic words
    words = re.findall(r'[a-z]+', text_lower)
    
    # High-confidence Hinglish-specific words
    hinglish_high_conf = {
        "hai", "hain", "ki", "ke", "ka", "ko", "se", "bhi", "tha", "thi", 
        "aur", "toh", "kya", "kyu", "kyon", "hoga", "hogi", "hoge", 
        "gaya", "gayi", "gaye", "hua", "hui", "hue", "kar", "karna", "kr", 
        "rha", "raha", "rahi", "rahe", "lekin", "magar", "sath", "saath"
    }
    
    matches = sum(1 for w in words if w in hinglish_high_conf)
    if matches >= 2:
        return "hinglish"
        
    return "english"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert AI Fact Verification assistant.
Your ONLY task: determine whether retrieved evidence SUPPORTS or REFUTES a given claim.

EXACT DETAILS MATTER. If ANY specific detail does not match, the claim is REFUTED.

=== DEFINITIONS ===

SUPPORTS:
  Every specific detail in the claim (person, place, institution, outcome, numbers, reason)
  is directly confirmed by the evidence. Nothing important contradicts the claim.

REFUTES:
  The evidence contradicts, corrects, or is incompatible with ANY specific detail
  of the claim. Examples of what triggers REFUTES:

  - DIFFERENT LOCATION:
    Claim says "Springfield", evidence says "Shelbyville" -> REFUTES

  - DIFFERENT ENTITY / INSTITUTION:
    Claim says "Oxford University", evidence mentions "Harvard University" -> REFUTES

  - DIFFERENT OUTCOME:
    Claim says "landed safely", evidence says "crashed" -> REFUTES

  - DIFFERENT NUMBER:
    Claim says "50 million", evidence says "20 million" -> REFUTES

  - DIFFERENT REASON / CONTEXT:
    Claim says "accident caused by weather", evidence describes "accident caused by mechanical failure" -> REFUTES

  - EVIDENCE IS SILENT / INSUFFICIENT:
    If the evidence does NOT explicitly confirm the specific outcome or fact in the claim -> REFUTES
    (You cannot confirm what the evidence does not say.)

=== WORKED EXAMPLES ===

Example 1 (REFUTES - outcome mismatch):
  Claim:    "The space probe successfully landed on Mars."
  Evidence: "The space probe crashed during its descent onto the Martian surface."
  Analysis: "landed successfully" vs "crashed" - OPPOSITE outcomes -> REFUTES

Example 2 (REFUTES - location mismatch):
  Claim:    "The company announced it will open a new office in Paris."
  Evidence: "The company announced it will open a new office in London."
  Analysis: Paris != London -> REFUTES

Example 3 (REFUTES - entity/context mismatch):
  Claim:    "The French President signed the trade agreement today."
  Evidence: "The German Chancellor signed the trade agreement today."
  Analysis: French President != German Chancellor -> REFUTES

Example 4 (SUPPORTS - exact match):
  Claim:    "The temperature reached 42 degrees on Monday."
  Evidence: "The temperature reached 42 degrees on Monday afternoon."
  Analysis: All details match -> SUPPORTS

=== CRITICAL RULES ===

- Do NOT output SUPPORTS simply because the evidence is about the same general topic.
- EXACT entity names (names, cities, institutions, numbers, outcomes) must all match.
- If the evidence describes the SAME event but with DIFFERENT specific details -> REFUTES.
- If the evidence does NOT explicitly confirm the specific outcome claimed -> REFUTES.
- EXTRA INFORMATION IS OKAY: If the evidence contains additional facts or lists other entities (e.g. "A, B, and C") but explicitly confirms the specific entity/assertion in the claim ("A"), this is still SUPPORTS. Additional background details do not create a contradiction.
- Never invent facts. Never use outside knowledge. Never assume missing information.
- Do NOT mention that you are an AI.

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object. Never use Markdown. Never use ```.
"""

# ============================================================
# STEP 1 - LOAD INPUT JSON
# ============================================================

def load_input_json(filepath: str) -> list:
    """
    Loads the top-k retrieval output JSON file.

    Expected structure per item:
        {
            "ID":        "<string>",
            "claim":     "<string>",
            "evidence1": "<string>",
            "evidence2": "<string>",
            "evidence3": "<string>"
        }

    Returns a list of such dicts.
    Raises FileNotFoundError or json.JSONDecodeError on failure.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list of claim objects.")

    logger.info(f"Loaded {len(data)} claim(s) from {filepath}")
    return data

# ============================================================
# STEP 2 - BUILD EVIDENCE BLOCK
# ============================================================

def build_evidence_block(item: dict, max_chars_per_evidence: int = 600) -> str:
    """
    Concatenates Evidence1, Evidence2, Evidence3 from a claim item
    into a single block, skipping any blank entries.

    Truncates each evidence piece to at most max_chars_per_evidence characters
    to prevent prompt context window overflow and long HTTP timeouts.

    Field names match the topk_output.json schema (capital E).
    Returns a plain-text string ready for the prompt.
    """
    evidence_keys = [
        ("Evidence1", "evidence1"),
        ("Evidence2", "evidence2"),
        ("Evidence3", "evidence3")
    ]
    lines = []
    counter = 1

    for k1, k2 in evidence_keys:
        val = item.get(k1) if item.get(k1) is not None else item.get(k2, "")
        value = str(val).strip() if val else ""
        if value:
            if len(value) > max_chars_per_evidence:
                value = value[:max_chars_per_evidence] + " ...[truncated for brevity]"
            lines.append(f"Source {counter}:\n{value}")
            counter += 1

    if not lines:
        return "No evidence was retrieved for this claim."

    return "\n\n".join(lines)


# ============================================================
# STEP 3 - BUILD PROMPT
# ============================================================

def build_prompt(claim: str, retrieved_evidence: str, lang_name: str = "English") -> str:
    """
    Constructs the full user-turn prompt sent to the LLM.

    Args:
        claim:              The claim to verify.
        retrieved_evidence: Pre-built evidence block string.
        lang_name:          The detected language name of the claim.

    Returns:
        A formatted prompt string.
    """
    lang_instruction = ""
    if lang_name != "English":
        if lang_name.lower() == "hinglish":
            lang_instruction = f"""
Language & Length Guidance for HINGLISH:
The claim is written in Hinglish (Romanized Hindi using Latin alphabet).
Write BOTH "Evidence" and "Justification" fields in clean Hinglish (Romanized Hindi).
The "Prediction" field MUST remain strictly "SUPPORTS" or "REFUTES" (in English).
CRITICAL LENGTH RULE FOR HINGLISH: Write at least 8 to 12 detailed sentences for Evidence (100-150 words) and 8 to 12 detailed sentences for Justification (100-120 words).
"""
        else:
            lang_instruction = f"""
Language & Detailed Length Instruction:
The claim is written in {lang_name}. Write the "Evidence" and "Justification" fields entirely in {lang_name}.
The "Prediction" field MUST remain strictly "SUPPORTS" or "REFUTES" (in English).

CRITICAL LENGTH RULE FOR {lang_name.upper()}:
To satisfy the mandatory word length requirements (100-150 words for Evidence, 100-120 words for Justification) in {lang_name}, you MUST write a long, thorough narrative consisting of at least 8 to 12 detailed sentences for each field. Include all background context, timelines, dates, entities, and step-by-step reasoning explanations. Short 3-4 sentence summaries are UNACCEPTABLE.
"""

    return f"""Claim:

{claim}

Retrieved Context Sources:

{retrieved_evidence}

Instructions:

Read the claim and all retrieved context sources carefully. Compare every specific detail in the claim (person, location, institution, outcome, numbers, reason) against the evidence.

Decision rules:
- SUPPORTS: Every specific detail in the claim is directly confirmed by the context sources.
- REFUTES: ANY specific detail contradicts the context sources, OR the context sources are silent/insufficient to confirm the claim.
{lang_instruction}
Output ONLY the JSON object below. No markdown, no backticks, no text outside the JSON.

Formatting Rules (MANDATORY WORD COUNT & JSON RULES):
1. "Evidence": Write a structured, detailed, and comprehensive evidence summary retrieved from external sources.
   - MANDATORY WORD COUNT: MUST be between 100 and 150 words (up to 200 words max in exceptional cases). Write 8 to 12 full sentences.
   - Do NOT mention "Evidence 1", "Evidence 2", "Evidence 3", or "Source 1". Present the facts as a unified narrative.
2. "Prediction": Must be strictly "SUPPORTS" or "REFUTES" (case-sensitive).
3. "Justification": Write a proper, comprehensive, step-by-step justification explaining why the claim is supported or refuted.
   - MANDATORY WORD COUNT: MUST be between 100 and 120 words. Write 8 to 12 full sentences.
   - Do NOT use prefixes like "EVIDENCE-1:" or refer to specific evidence/source numbers.
4. CRITICAL JSON ESCAPING RULE: NEVER use inner double quotes (\") inside text values. Always use single quotes (') for any quoted phrases, names, or titles inside values (e.g. 'title' instead of \"title\").

{{
    "Evidence": "<structured and comprehensive evidence summary of 100-150 words (8-12 detailed sentences)>",
    "Prediction": "<SUPPORTS or REFUTES>",
    "Justification": "<thorough step-by-step justification of 100-120 words (8-12 detailed sentences)>"
}}

Example output:

{{
    "Evidence": "Official government reports and telemetry data confirm that the SpaceX Starship vehicle successfully launched from Texas on Sunday morning but suffered a catastrophic control anomaly during its final descent phase, resulting in a crash onto the lunar surface on Monday at 10:00 AM. Multiple tracking stations and independent observatory records verified the impact debris field and loss of telemetry signals. Official spokespersons from the space agency stated that all post-launch recovery operations have been suspended pending a comprehensive safety investigation into the guidance sensor system. No secondary mission attempts or payload deployments were authorized or performed following the crash event.",
    "Prediction": "REFUTES",
    "Justification": "The claim asserts that the SpaceX Starship successfully performed a safe landing on the Moon on Monday. However, official telemetry records and space agency statements confirm that the Starship actually suffered a severe control anomaly and crashed onto the lunar surface during descent. Independent tracking data verified the crash impact, and all further mission operations were immediately suspended. Because the verified facts document a destructive crash rather than a successful landing, the specific assertion made in the claim is directly contradicted by the official evidence. Consequently, the claim is unsupported by factual records and the final prediction is REFUTES."
}}
"""



# ============================================================
# STEP 4 - CALL LLM
# ============================================================

def call_llm_messages(messages: list) -> str:
    """
    Sends a message list (chat history) to the local LLM server
    and returns the raw string response from the model.

    Args:
        messages: List of message dicts (role, content).

    Returns:
        Raw model output as a string.

    Raises:
        requests.HTTPError on non-2xx response.
        requests.Timeout   if the server does not respond in time.
    """
    payload = {
        "model":          MODEL_NAME,
        "messages":       messages,
        "temperature":    TEMPERATURE,
        "top_p":          TOP_P,
        "top_k":          TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens":     MAX_TOKENS,
        "stream":         False
    }

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    if response.status_code != 200:
        logger.error(f"  [SERVER ERROR] Code {response.status_code}: {response.text}")
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ============================================================
# STEP 5 - CLEAN JSON
# ============================================================

def clean_json(text: str) -> str:
    """
    Strips reasoning tags (<think>...</think>), markdown code fences,
    leading/trailing filler text, and fixes missing field commas in LLM JSON output.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Isolate outer JSON object boundaries if surrounded by commentary
    m = re.search(r'(\{[\s\S]*\})', text)
    if m:
        text = m.group(1).strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Insert missing commas between JSON fields if model omitted them
    text = re.sub(r'("|\b)\s*[\r\n]+\s*"Prediction"', r'\1,\n    "Prediction"', text)
    text = re.sub(r'("|\b)\s*[\r\n]+\s*"Justification"', r'\1,\n    "Justification"', text)

    return text

# ============================================================
# STEP 5b - REPAIR TRUNCATED / MALFORMED JSON
# ============================================================

def repair_json(text: str) -> dict | None:
    """
    Attempts to recover a usable dict from a truncated or malformed
    LLM response, called automatically when json.loads() fails.

    Extracts Evidence, Prediction, and Justification without relying on strict quote escaping,
    ensuring no mid-sentence text is ever cut off.
    """
    # 1. Extract Prediction (SUPPORTS or REFUTES)
    pred_m = re.search(r'"Prediction"\s*:\s*"(SUPPORTS|REFUTES)"', text, re.IGNORECASE)
    if not pred_m:
        pred_m = re.search(r'\b(SUPPORTS|REFUTES)\b', text)
    prediction = pred_m.group(1).upper() if pred_m else "SUPPORTS"

    # 2. Extract Evidence text between "Evidence": " and "Prediction"
    ev_start = re.search(r'"Evidence"\s*:\s*"', text, re.IGNORECASE)
    pred_pos = re.search(r'"Prediction"', text, re.IGNORECASE)
    if ev_start and pred_pos:
        ev_text = text[ev_start.end() : pred_pos.start()].strip()
        ev_text = re.sub(r'"?\s*,?\s*$', '', ev_text).strip()
    else:
        ev_text = ""

    # 3. Extract Justification text from "Justification": " to end of response
    jus_start = re.search(r'"Justification"\s*:\s*"', text, re.IGNORECASE)
    if jus_start:
        jus_text = text[jus_start.end():].strip()
        if jus_text.endswith('"}'):
            jus_text = jus_text[:-2].strip()
        elif jus_text.endswith('}') or jus_text.endswith('"'):
            jus_text = jus_text[:-1].strip()
            if jus_text.endswith('"'):
                jus_text = jus_text[:-1].strip()
    else:
        jus_text = ""

    # Clean inner double quotes to single quotes to guarantee valid JSON string content
    ev_clean  = ev_text.replace('\\"', '"').replace('"', "'").strip()
    jus_clean = jus_text.replace('\\"', '"').replace('"', "'").strip()

    if ev_clean and jus_clean:
        logger.info("  [REPAIR] Recovered complete fields via bulletproof offset extraction.")
        return {
            "Evidence":      ev_clean,
            "Prediction":    prediction,
            "Justification": jus_clean
        }

    # Fallback Strategy: try appending closing suffixes
    closing_suffixes = ['"}', '"}}', '"}]}']
    for suffix in closing_suffixes:
        try:
            result = json.loads(text + suffix)
            if isinstance(result, dict) and result.get("Evidence") and result.get("Justification"):
                logger.info("  [REPAIR] Recovered JSON by appending closing suffix.")
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None  # Unrecoverable

def validate_output(parsed: dict, lang_tag: str = "english") -> list:
    """
    Per-record validation.

    Checks:
    - Required keys present: Evidence, Prediction, Justification.
    - Prediction is exactly 'SUPPORTS' or 'REFUTES' (case-sensitive).
    - Evidence and Justification are non-empty.
    - Logs [INFO] if Evidence summary word count does not meet 100-150 words.
    - Logs [INFO] if Justification word count does not meet 100-120 words.

    Returns a list of blocking error message strings.
    """
    errors = []

    # --- Required keys ---
    required_keys = {"Evidence", "Prediction", "Justification"}
    missing = required_keys - parsed.keys()
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return errors   # Cannot proceed without keys

    # --- Exact-case label (case-sensitive, competition rule) ---
    prediction = parsed.get("Prediction", "")
    if prediction not in VALID_LABELS:
        errors.append(
            f"Invalid prediction label: '{prediction}'. "
            f"Must be exactly 'SUPPORTS' or 'REFUTES' (case-sensitive)."
        )

    # --- Non-empty Evidence + word count info check (Target: 100-150 words) ---
    evidence = parsed.get("Evidence", "").strip()
    if not evidence:
        errors.append("'Evidence' field is empty.")
    else:
        ev_words = len(evidence.split())
        if not (100 <= ev_words <= 150):
            logger.info(
                f"  [INFO] Evidence summary word count ({ev_words} words) does NOT meet required range (100-150 words)."
            )

    # --- Non-empty Justification + word count info check (Target: 100-120 words) ---
    justification = parsed.get("Justification", "").strip()
    if not justification:
        errors.append("'Justification' field is empty.")
    else:
        jus_words = len(justification.split())
        if not (100 <= jus_words <= 120):
            logger.info(
                f"  [INFO] Justification word count ({jus_words} words) does NOT meet required range (100-120 words)."
            )

    return errors


def sanitize_output_text(text: str) -> str:
    """
    Cleans accidental evidence reference tags (e.g. 'Evidence 1:', 'EVIDENCE-1:', 'Evidence 2 reports')
    from output fields to ensure the end user receives a clean, natural summary and justification.
    """
    if not text:
        return ""
    # Remove section prefixes like CLAIM:, EVIDENCE-1:, EVIDENCE-2:, EVIDENCE-3:, VERDICT:, Source 1:
    text = re.sub(r'\b(CLAIM|EVIDENCE-\d|VERDICT|SOURCE-\d)\s*:\s*', '', text, flags=re.IGNORECASE)
    # Remove labels like 'Evidence 1:', 'Evidence 2:', 'Source 1:' at beginning of sentences
    text = re.sub(r'\b(Evidence|Source)\s+\d+\s*:\s*', '', text, flags=re.IGNORECASE)
    # Remove phrases like 'According to Evidence 1,' or 'Evidence 1 states that'
    text = re.sub(r'\b(According to\s+)?(Evidence|Source)\s+\d+\s+(reports|states|confirms|notes|corroborates|shows|that)?\s*', '', text, flags=re.IGNORECASE)
    # Clean up double spaces created by regex stripping
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# STEP 7 - PROCESS A SINGLE CLAIM (with retry)
# ============================================================

def process_claim(item: dict) -> dict:
    """
    Full pipeline for one claim item:
        1. Build evidence block.
        2. Call LLM (with retries and self-correcting feedback history).
        3. Clean and parse the JSON response.
        4. Validate the parsed output.

    Returns a result dict:
        {
            "ID":           "<string>",
            "Evidence":     "<string>",
            "Prediction":   "SUPPORTS" | "REFUTES",
            "Justification":"<string>"
        }

    On unrecoverable failure, returns a dict with
    Prediction = "ERROR" and details in Justification.
    """
    claim_id  = str(item.get("ID", "UNKNOWN"))
    claim_val = item.get("Text") if item.get("Text") is not None else item.get("claim", "")
    claim     = str(claim_val).strip()
    evidence  = build_evidence_block(item)

    # Detect language of the claim
    lang_tag  = detect_language(claim)
    lang_name = LANGUAGE_NAME_MAP[lang_tag]

    logger.info(f"Processing claim ID: {claim_id}")
    logger.info(f"  Detected language: {lang_name} ({lang_tag})")

    # Build the initial message list for this claim
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": build_prompt(claim, evidence, lang_name)
        }
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1 to MAX_RETRIES+1
        try:
            logger.info(f"  Attempt {attempt} / {MAX_RETRIES + 1}")

            raw_reply = call_llm_messages(messages)
            cleaned   = clean_json(raw_reply)

            # Try parsing the clean response
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as json_err:
                # Attempt to repair the malformed/truncated output
                repaired = repair_json(cleaned)
                if repaired:
                    parsed = repaired
                else:
                    raise json_err  # Re-raise if repair failed

            # Perform validation
            errors = validate_output(parsed, lang_tag)

            if errors:
                last_error = f"Validation failed: {errors}"
                logger.warning(f"  {last_error}")
                if attempt <= MAX_RETRIES:
                    ev_count  = len(parsed.get("Evidence", "").split()) if parsed else 0
                    jus_count = len(parsed.get("Justification", "").split()) if parsed else 0

                    retry_instruction = (
                        f"\n\nCRITICAL WORD COUNT RETRY REQUIREMENT:\n"
                        f"Your previous attempt had validation issues:\n"
                        f"{json.dumps(errors, indent=2)}\n\n"
                        f"Current word counts in your previous attempt:\n"
                        f"- Evidence summary : {ev_count} words (REQUIRED: 100 to 150 words, up to 200 max)\n"
                        f"- Justification    : {jus_count} words (REQUIRED: 100 to 120 words)\n\n"
                        f"Please regenerate the complete JSON object, ensuring both fields strictly satisfy the required word counts. "
                        f"If a field was under the required word count, expand it with more detailed factual points and reasoning. "
                        f"If a field was over the required word count, trim it to be concise."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": build_prompt(claim, evidence, lang_name) + retry_instruction
                        }
                    ]
                    time.sleep(RETRY_DELAY)
                    continue
            else:
                parsed["Prediction"] = parsed["Prediction"].strip().upper()
                return {
                    "ID":            claim_id,
                    "Evidence":      sanitize_output_text(parsed["Evidence"]),
                    "Prediction":    parsed["Prediction"],
                    "Justification": sanitize_output_text(parsed["Justification"])
                }

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"  {last_error}")
            if attempt <= MAX_RETRIES:
                # Reset the conversation to base state to avoid context bloat.
                # Appending to a growing history on parse failures worsens truncation
                # on subsequent attempts because the total input grows each time.
                # Instead, start fresh with a single, explicit JSON-only directive.
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": (
                            build_prompt(claim, evidence, lang_name) +
                            "\n\nCRITICAL: Your previous response was not valid JSON. "
                            "Output ONLY the JSON object. No markdown, no backticks, "
                            "no <think> blocks, no text before or after the JSON."
                        )
                    }
                ]
                time.sleep(RETRY_DELAY)

        except requests.RequestException as e:
            last_error = f"LLM request failed: {e}"
            logger.error(f"  {last_error}")
            if attempt <= MAX_RETRIES:
                # Reset conversation to initial state to prevent prompt bloat on timeouts/connection errors
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_prompt(claim, evidence, lang_name)
                    }
                ]
                err_str = str(e).lower()
                is_server_error = (
                    "connectionreset" in err_str
                    or "connection aborted" in err_str
                    or "read timed out" in err_str
                    or "timeout" in err_str
                    or "503" in err_str
                    or "500" in err_str
                )
                if is_server_error:
                    logger.info("  Server timeout/error detected — waiting 5s before retrying clean prompt...")
                    time.sleep(5)
                else:
                    time.sleep(RETRY_DELAY)

    logger.error(f"  All attempts failed for claim ID: {claim_id}. Reason: {last_error}")
    return {
        "ID":            claim_id,
        "Evidence":      "",
        "Prediction":    "ERROR",
        "Justification": f"Processing failed after {MAX_RETRIES + 1} attempt(s). Last error: {last_error}"
    }

# ============================================================
# STEP 8 - SAVE OUTPUT JSON
# ============================================================

def save_output_json(results: list, filepath: str) -> None:
    """
    Writes the list of result dicts to a JSON file.

    Encoding : UTF-8 (no BOM), as required by the competition.
    Format   : Pretty-printed JSON array, ensure_ascii=False so
               Indic-script strings are stored natively.

    Args:
        results:  List of result dicts.
        filepath: Destination file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {filepath}")

# ============================================================
# STEP 8b - CHECKPOINT (save / load)
# ============================================================

def save_checkpoint(results: list, filepath: str) -> None:
    """
    Atomically writes the current list of results to a checkpoint file.

    Uses a write-to-temp-then-rename strategy so that a crash mid-write
    never leaves a corrupted checkpoint file on disk.

    Args:
        results:  List of result dicts accumulated so far.
        filepath: Path of the checkpoint JSON file.
    """
    path     = Path(filepath)
    tmp_path = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    tmp_path.replace(path)   # Atomic on most OS/filesystems
    logger.info(f"[CHECKPOINT] Progress saved ({len(results)} record(s)) -> {filepath}")


def load_checkpoint(filepath: str) -> list:
    """
    Loads a previously saved checkpoint file.

    Returns an empty list if the file does not exist or is invalid,
    so the pipeline always starts cleanly when there is no checkpoint.

    Args:
        filepath: Path of the checkpoint JSON file.

    Returns:
        List of already-processed result dicts, or [] if none found.
    """
    path = Path(filepath)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info(
                f"[CHECKPOINT] Resuming from checkpoint: "
                f"{len(data)} record(s) already processed."
            )
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[CHECKPOINT] Could not read checkpoint file ({e}). Starting fresh.")

    return []

# ============================================================
# STEP 9 - VALIDATE SUBMISSION FILE  (competition rules)
# ============================================================

def validate_submission(output_filepath: str, input_claims: list) -> bool:
    """
    Validates the saved submission file against every competition rule.

    Rules checked
    -------------
    1.  File is UTF-8 encoded and readable.
    2.  File contains valid JSON.
    3.  Top-level value is a JSON array.
    4.  No missing claim IDs  (IDs present in input but absent in output).
    5.  No unknown claim IDs  (IDs in output that were never in input).
    6.  No duplicate claim IDs in the output.
    7.  Every record has all required fields: ID, Evidence, Prediction,
        Justification.
    8.  Prediction is exactly 'SUPPORTS' or 'REFUTES' (case-sensitive).
    9.  Evidence field is non-empty.
    10. Justification field is non-empty.

    Returns True if all checks pass, False otherwise.
    Prints a detailed PASS / FAIL report to the logger.
    """
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("SUBMISSION VALIDATION REPORT")
    logger.info(sep)

    issues  = []          # List[str] — one entry per failed rule
    records = None        # Will hold parsed JSON if it loads

    # --------------------------------------------------------
    # Rule 1 — UTF-8 encoding
    # --------------------------------------------------------
    try:
        raw_bytes = Path(output_filepath).read_bytes()
        raw_bytes.decode("utf-8")          # raises UnicodeDecodeError if not UTF-8
        logger.info("[PASS] Rule 1: File is UTF-8 encoded.")
    except UnicodeDecodeError as e:
        issues.append(f"Rule 1 FAIL — File is NOT UTF-8 encoded: {e}")
        logger.error(f"[FAIL] Rule 1: File is NOT UTF-8 encoded.")

    # --------------------------------------------------------
    # Rule 2 — Valid JSON
    # --------------------------------------------------------
    try:
        content = Path(output_filepath).read_text(encoding="utf-8", errors="replace")
        records = json.loads(content)
        logger.info("[PASS] Rule 2: File contains valid JSON.")
    except json.JSONDecodeError as e:
        issues.append(f"Rule 2 FAIL — Invalid JSON format: {e}")
        logger.error(f"[FAIL] Rule 2: Invalid JSON format — {e}")
        # Cannot continue without valid JSON
        _print_summary(issues)
        return False

    # --------------------------------------------------------
    # Rule 3 — Top-level array
    # --------------------------------------------------------
    if not isinstance(records, list):
        issues.append(
            f"Rule 3 FAIL — Top-level value is '{type(records).__name__}', "
            f"expected a JSON array."
        )
        logger.error("[FAIL] Rule 3: Submission is not a JSON array.")
        _print_summary(issues)
        return False
    else:
        logger.info(f"[PASS] Rule 3: Top-level value is a JSON array ({len(records)} record(s)).")

    # --------------------------------------------------------
    # Build ID sets for Rules 4-6
    # --------------------------------------------------------
    input_ids  = {str(item.get("ID", "")) for item in input_claims}
    output_ids = []
    for rec in records:
        if isinstance(rec, dict):
            output_ids.append(str(rec.get("ID", "")))

    # Rule 4 — Missing IDs
    missing_ids = input_ids - set(output_ids)
    if missing_ids:
        issues.append(f"Rule 4 FAIL — Missing claim IDs in output: {sorted(missing_ids)}")
        logger.error(f"[FAIL] Rule 4: Missing claim IDs — {sorted(missing_ids)}")
    else:
        logger.info("[PASS] Rule 4: No missing claim IDs.")

    # Rule 5 — Unknown IDs
    unknown_ids = set(output_ids) - input_ids
    if unknown_ids:
        issues.append(f"Rule 5 FAIL — Unknown claim IDs not in input: {sorted(unknown_ids)}")
        logger.error(f"[FAIL] Rule 5: Unknown claim IDs — {sorted(unknown_ids)}")
    else:
        logger.info("[PASS] Rule 5: No unknown claim IDs.")

    # Rule 6 — Duplicate IDs
    seen      = set()
    duplicates = set()
    for oid in output_ids:
        if oid in seen:
            duplicates.add(oid)
        seen.add(oid)
    if duplicates:
        issues.append(f"Rule 6 FAIL — Duplicate claim IDs in output: {sorted(duplicates)}")
        logger.error(f"[FAIL] Rule 6: Duplicate claim IDs — {sorted(duplicates)}")
    else:
        logger.info("[PASS] Rule 6: No duplicate claim IDs.")

    # --------------------------------------------------------
    # Per-record checks (Rules 7-10)
    # --------------------------------------------------------
    required_fields  = ["ID", "Evidence", "Prediction", "Justification"]
    field_errors     = []    # Missing required fields
    label_errors     = []    # Invalid prediction labels
    empty_ev_errors  = []    # Empty Evidence
    empty_jus_errors = []    # Empty Justification

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            field_errors.append(f"  Record [{i}] is not a JSON object.")
            continue

        rec_id = rec.get("ID", f"<index {i}>")

        # Rule 7 — Required fields
        missing_fields = [f for f in required_fields if f not in rec]
        if missing_fields:
            field_errors.append(
                f"  ID '{rec_id}': missing field(s): {missing_fields}"
            )

        # Rule 8 — Exact-case label
        prediction = rec.get("Prediction", "")
        if prediction not in VALID_LABELS:
            label_errors.append(
                f"  ID '{rec_id}': invalid label '{prediction}' "
                f"(must be exactly 'SUPPORTS' or 'REFUTES')."
            )

        # Rule 9 — Non-empty Evidence
        if not rec.get("Evidence", "").strip():
            empty_ev_errors.append(f"  ID '{rec_id}': Evidence is empty.")

        # Rule 10 — Non-empty Justification
        if not rec.get("Justification", "").strip():
            empty_jus_errors.append(f"  ID '{rec_id}': Justification is empty.")

    # Report Rule 7
    if field_errors:
        issues.append("Rule 7 FAIL — Records with missing required fields:")
        logger.error("[FAIL] Rule 7: Missing required fields:")
        for e in field_errors:
            logger.error(e)
    else:
        logger.info("[PASS] Rule 7: All records have required fields (ID, Evidence, Prediction, Justification).")

    # Report Rule 8
    if label_errors:
        issues.append("Rule 8 FAIL — Records with invalid prediction labels:")
        logger.error("[FAIL] Rule 8: Invalid prediction labels:")
        for e in label_errors:
            logger.error(e)
    else:
        logger.info("[PASS] Rule 8: All prediction labels are valid (SUPPORTS / REFUTES).")

    # Report Rule 9
    if empty_ev_errors:
        issues.append("Rule 9 FAIL — Records with empty Evidence:")
        logger.error("[FAIL] Rule 9: Empty Evidence fields:")
        for e in empty_ev_errors:
            logger.error(e)
    else:
        logger.info("[PASS] Rule 9: No empty Evidence fields.")

    # Report Rule 10
    if empty_jus_errors:
        issues.append("Rule 10 FAIL — Records with empty Justification:")
        logger.error("[FAIL] Rule 10: Empty Justification fields:")
        for e in empty_jus_errors:
            logger.error(e)
    else:
        logger.info("[PASS] Rule 10: No empty Justification fields.")

    return _print_summary(issues)


def _print_summary(issues: list) -> bool:
    """
    Prints the final PASS / FAIL banner and returns True if clean.
    """
    sep = "=" * 60
    if not issues:
        logger.info(sep)
        logger.info("SUBMISSION STATUS: *** ALL CHECKS PASSED ***")
        logger.info(sep)
        return True
    else:
        logger.error(sep)
        logger.error(f"SUBMISSION STATUS: FAILED ({len(issues)} issue(s) found)")
        logger.error(sep)
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error(sep)
        return False

# ============================================================
# MAIN
# ============================================================

def check_server_reachable() -> bool:
    """
    Blocks and polls the LLM server's health endpoint until it is fully loaded
    and ready to handle requests (returns status code 200).
    """
    health_url = SERVER_URL.replace("/v1/chat/completions", "/health")
    timeout = 90  # Max seconds to wait for model loading
    start_time = time.time()
    
    logger.info("Waiting for LLM server to initialize and load the model...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                # Server is up and model is fully loaded!
                return True
            elif response.status_code == 503:
                # Server is up but still loading the model
                elapsed = int(time.time() - start_time)
                logger.info(f"  [Server Status] Model is loading... (elapsed: {elapsed}s)")
            else:
                # Unexpected status code (e.g. server starting/restarting)
                logger.warning(f"  [Server Status] Unexpected status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            # Server process is not listening/running yet
            elapsed = int(time.time() - start_time)
            logger.info(f"  Waiting for server process to start... (elapsed: {elapsed}s)")
        except requests.exceptions.RequestException as e:
            # Other temporary network / request errors
            logger.warning(f"  Connection issue: {e}")
            
        time.sleep(2)
        
    return False


def main() -> None:
    """
    Entry point:
        1. Check that the LLM server is reachable.
        2. Load the top-k retrieval output JSON.
        3. Process each claim through the LLM pipeline.
        4. Save the results to the output JSON file.
        5. Validate the submission file against competition rules.
    """
    logger.info("=" * 60)
    logger.info("IndicClaimVerifier - LLM Fact Verification Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input  : {INPUT_FILE}")
    logger.info(f"Output : {OUTPUT_FILE}")
    logger.info(f"Model  : {MODEL_NAME}")
    logger.info(f"Server : {SERVER_URL}")
    logger.info("=" * 60)

    # --- Server reachability check ---
    logger.info("Checking LLM server connectivity...")
    if not check_server_reachable():
        logger.error("=" * 60)
        logger.error("LLM SERVER IS NOT REACHABLE")
        logger.error(f"  URL : {SERVER_URL}")
        logger.error("  Fix : Start the llama-server before running this script.")
        logger.error("  Cmd : .\\llama.cpp\\build\\bin\\Release\\llama-server.exe ")
        logger.error(f"            -m .\\models\\{MODEL_NAME} --port 8080 -c 4096 -ngl 99")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info("LLM server is reachable. Proceeding...")

    try:
        claims = load_input_json(INPUT_FILE)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)

    # --- Resume from checkpoint if one exists ---
    raw_results      = load_checkpoint(CHECKPOINT_FILE)
    # Filter out successfully-processed claims (ignore ERROR entries so failed items are re-tried)
    results          = [r for r in raw_results if r.get("Prediction") != "ERROR"]
    completed_ids    = {r["ID"] for r in results}
    error_count      = 0

    # Filter out already-processed claims
    pending_claims   = [c for c in claims if str(c.get("ID", "")) not in completed_ids]
    total            = len(claims)
    already_done     = len(results)

    if already_done:
        logger.info(
            f"[CHECKPOINT] Skipping {already_done} already-processed claim(s). "
            f"{len(pending_claims)} remaining."
        )

    pipeline = PreprocessingPipeline(pending_claims)
    pipeline.start()

    for idx, item in enumerate(pipeline, start=already_done + 1):
        logger.info(f"\n[{idx}/{total}] ----------------------------------------")
        result = process_claim(item)
        results.append(result)

        if result["Prediction"] == "ERROR":
            error_count += 1

        # --- Periodic checkpoint ---
        if idx % CHECKPOINT_EVERY == 0:
            save_checkpoint(results, CHECKPOINT_FILE)

    logger.info("\n" + "=" * 60)
    logger.info(f"Processed : {total} claim(s)")
    logger.info(f"Errors    : {error_count} claim(s) failed")
    logger.info("=" * 60)

    # Save final clean submission file
    save_output_json(results, OUTPUT_FILE)

    # Clean up checkpoint now that the final file is saved
    checkpoint_path = Path(CHECKPOINT_FILE)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"[CHECKPOINT] Checkpoint file deleted (run complete).")

    # Validate submission against competition rules
    passed = validate_submission(OUTPUT_FILE, claims)

    if passed:
        logger.info("Pipeline complete. Submission is ready.")
    else:
        logger.warning(
            "Pipeline complete, but the submission has validation issues. "
            "Review the FAIL lines above before submitting."
        )
        sys.exit(2)   # Non-zero exit so CI/scripts can detect failures

# ============================================================

if __name__ == "__main__":
    main()
