import os
# Prevents the OpenMP duplicate library crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import types
import unicodedata
import torch
from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
from transformers.tokenization_utils_base import BatchEncoding

# =========================================================
# SECTION 1: NOISE FILTER  (5-Layer Enhanced, Indic-Aware)
# =========================================================

# Tier 1: obvious boilerplate keyword patterns
_NOISE_KEYWORDS = re.compile(
    r'(please enable javascript|refresh the page|you are being redirected|'
    r'subscribe now|sign up|sign in|log in|login|create an account|'
    r'free trial|try for free|upgrade to|ad-free|cookie|privacy policy|'
    r'terms (of|and) (use|service|conditions)|all rights reserved|'
    r'copyright \d{4}|\u00a9|click here|read more|learn more|'
    r'follow us|share this|newsletter|notification|push notification|'
    r'javascript (is )?disabled|browser (is )?outdated|update your browser|'
    r'frequently asked questions|get started|connect gmail|'
    r'welcome to our website|love finds its voice|'
    r'built with|powered by|back to top|skip to content)',
    re.IGNORECASE
)

# Tier 2: structural web-page noise (dates, bylines, breadcrumbs, captions, etc.)
_NOISE_STRUCTURAL = re.compile(
    r'(?:'
    r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[,\s]+\d{4}\b|'
    r'\b\d{1,2}:\d{2}\s*(?:am|pm|ist|gmt|utc)\b|'
    r'\b(?:updated|published|last updated|posted)\s+(?:on|at)?\s*\d|'
    r'\b(?:by|written by|reported by|edited by)\s+[A-Z][a-z]+\s+[A-Z][a-z]+|'
    r'\b(?:image|photo|video|picture|caption|fig|figure)\s*[:|]|'
    r'\b(?:also read|also watch|also see|read also|watch also|related|recommended)\s*[:|]|'
    r'(?:\w+\s*\|\s*){2,}|'
    r'\d{4}\s+\w+[.,]\s*(?:all rights|pvt|ltd|inc|llc)\b|'
    r'\b(?:share|tweet|whatsapp|telegram)\s+(?:on|via|to)\b'
    r')',
    re.IGNORECASE
)


def is_noisy_sentence(sentence: str) -> bool:
    """
    Returns True if the sentence is web boilerplate / noise.

    Five layers:
      1. Too short  (< 6 words)
      2. Keyword blacklist  (ads, JS warnings, subscriptions)
      3. Structural patterns  (dates, bylines, breadcrumbs, captions)
      4. Character quality  (Unicode L/M < 50% -> timestamps / URL noise; Indic-safe)
      5. Predominantly ALL-CAPS  (navigation headers / shouted text)
    """
    words = sentence.split()

    if len(words) < 6:
        return True
    if _NOISE_KEYWORDS.search(sentence):
        return True
    if _NOISE_STRUCTURAL.search(sentence):
        return True

    # Layer 4: Unicode category 'L' (Letter) and 'M' (Mark / Indic matras)
    no_space = sentence.replace(' ', '')
    if no_space:
        lm_chars = sum(1 for c in sentence if unicodedata.category(c).startswith(('L', 'M')))
        if lm_chars / len(no_space) < 0.50:
            return True

    # Layer 5: Predominantly ALL-CAPS words (> 50% of words)
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    if words and upper_words / len(words) > 0.50:
        return True

    return False


# =========================================================
# SECTION 2: SENTENCE SPLITTER  (noise-aware)
# =========================================================

def split_into_sentences(text: str) -> list:
    """
    Splits text into clean sentences handling:
      - English (.  !  ?)
      - Indic danda (\u0964)
      - Newlines (\\n+)
    Every candidate is run through the noise filter.
    """
    if not text:
        return []
    raw = re.split(r'(?:(?<=[\u0964.!?])\s+|\n+)', text.strip())
    clean = []
    for s in raw:
        s = re.sub(r'\s+', ' ', s).strip()
        if s and not is_noisy_sentence(s):
            clean.append(s)
    return clean


# =========================================================
# SECTION 3: SENTENCE-LEVEL RERANKER
# =========================================================

def rerank_sentences(claim: str, documents: list, model) -> list:
    """
    Scores every clean sentence across all documents against the claim
    using the mMARCO cross-encoder.

    Returns all scored candidates sorted best-first so the caller can
    pick greedily until the word-count target is met:
        [{"doc": int, "sent": int, "text": str, "score": float}, ...]
    """
    all_sentences = []
    for doc_id, doc in enumerate(documents):
        for sent_id, sentence in enumerate(split_into_sentences(doc)):
            all_sentences.append((doc_id, sent_id, sentence))

    if not all_sentences:
        return []

    pairs  = [[claim, s[2]] for s in all_sentences]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = [
        {"doc": doc_id, "sent": sent_id, "text": sentence, "score": float(score)}
        for (doc_id, sent_id, sentence), score in zip(all_sentences, scores)
    ]
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


# =========================================================
# SECTION 4: LOAD MODELS  (VRAM-guarded, FP16-optimised)
# =========================================================

cuda_available = torch.cuda.is_available()

if cuda_available:
    free_vram, _ = torch.cuda.mem_get_info()
    free_vram_gb  = free_vram / (1024 ** 3)
    print(f"GPU Free VRAM: {free_vram_gb:.2f} GB")
    if free_vram_gb < 4.0:
        print("Warning: < 4 GB VRAM free -- falling back to CPU to avoid OOM.")
        cuda_available = False
    torch.cuda.empty_cache()

use_gpu      = cuda_available
device_label = "GPU (CUDA)" if use_gpu else "CPU"

# ---- Stage 1: BGE multilingual document reranker ----
print(f"\nLoading BGE Document Reranker ({device_label})...")
bge_reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=use_gpu,
    devices="cuda" if use_gpu else None,
    batch_size=4 if use_gpu else 64       # conservative GPU batch to avoid OOM
)

# Monkey-patch for Hugging Face Transformers v5+ compatibility
if not hasattr(bge_reranker.tokenizer, "prepare_for_model"):
    def _prepare_for_model(self, ids, pair_ids=None, truncation=None,
                           max_length=None, padding=False, **kwargs):
        bos = self.bos_token_id or 0
        eos = self.eos_token_id or 2
        d   = pair_ids or []
        if truncation == "only_second" and max_length is not None:
            d = d[:max(0, max_length - len(ids) - 4)]
        input_ids = [bos] + ids + [eos, eos] + d + [eos]
        return BatchEncoding({"input_ids": input_ids,
                              "attention_mask": [1] * len(input_ids)})

    bge_reranker.tokenizer.prepare_for_model = types.MethodType(
        _prepare_for_model, bge_reranker.tokenizer
    )
    print("  -> Tokenizer monkey-patched for Transformers v5+ compatibility.")

# ---- Stage 2: mMARCO multilingual sentence reranker ----
print(f"Loading mMARCO Sentence Reranker ({device_label})...")
sentence_reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512,
    device="cuda" if use_gpu else "cpu"
)

print("Both models loaded successfully!\n")


# =========================================================
# SECTION 5: CONFIGURATION
# =========================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, "input",  "retrived_documents.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output", "ranked_documents.json")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

TOP_K_DOCS          = 3    # Primary document pool size (Stage 1 top-K)
CHECKPOINT_INTERVAL = 50   # Save progress every N claims

# Evidence word-count target window
TARGET_WORDS_MIN    = 80   # Minimum acceptable evidence length
TARGET_WORDS_TARGET = 150  # Ideal target -- stop filling once reached
TARGET_WORDS_MAX    = 180  # Hard cap -- never exceed


# =========================================================
# SECTION 6: LOAD DATASET
# =========================================================

with open(INPUT_FILE, "r", encoding="utf-8") as fh:
    data = json.load(fh)

claims_list = data if isinstance(data, list) else data.get("claims", [])
print(f"Loaded dataset -- {len(claims_list)} claims to process.\n")

final_output = []


# =========================================================
# SECTION 7: MAIN PROCESSING LOOP
# =========================================================

for index, item in enumerate(claims_list, start=1):

    claim_id   = item.get("id")   or item.get("ID")   or f"claim_{index}"
    claim_text = item.get("text") or item.get("Text") or item.get("claim")
    documents  = item.get("documents", [])

    if not claim_text or not documents:
        print(f"[{index}/{len(claims_list)}] Skipping {claim_id}: missing text or documents.")
        continue

    print(f"[{index}/{len(claims_list)}] Processing {claim_id}  ({len(documents)} docs)...")

    # ---- Extract raw document texts ----
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get("text") or doc.get("Text", ""))
        else:
            doc_texts.append(str(doc))
    if not doc_texts:
        continue

    # ----------------------------------------------------------
    # STAGE 1 -- Document-Level Reranking  (BGE-M3)
    # Score every [claim, document] pair; sort best-first.
    # ----------------------------------------------------------
    doc_pairs  = [[claim_text, txt] for txt in doc_texts]
    doc_scores = bge_reranker.compute_score(doc_pairs)

    # FlagEmbedding returns a bare float for single-doc inputs
    if isinstance(doc_scores, float):
        doc_scores = [doc_scores]

    scored_docs = sorted(
        [{"text": txt, "score": float(sc)} for txt, sc in zip(doc_texts, doc_scores)],
        key=lambda x: x["score"], reverse=True
    )

    # ----------------------------------------------------------
    # STAGE 2 -- Sentence-Level Reranking  (mMARCO)
    # First try top TOP_K_DOCS; expand to all docs if still
    # below the minimum word-count target.
    # ----------------------------------------------------------
    selected   = []
    word_count = 0

    for pool_size in [TOP_K_DOCS, len(scored_docs)]:
        pool_texts   = [d["text"] for d in scored_docs[:pool_size]]
        ranked_sents = rerank_sentences(claim_text, pool_texts, sentence_reranker)

        # ---- Dynamic greedy fill ----
        selected   = []
        word_count = 0

        for sent in ranked_sents:
            s_words = len(sent["text"].split())
            if word_count + s_words <= TARGET_WORDS_MAX:
                selected.append(sent)
                word_count += s_words
            if word_count >= TARGET_WORDS_TARGET:
                break

        # Stop expanding pool once the minimum target is reached
        if word_count >= TARGET_WORDS_MIN:
            break

    # ---- Restore chronological order (doc-position, sent-position) ----
    selected.sort(key=lambda x: (x["doc"], x["sent"]))

    # ---- Deduplicate while preserving order ----
    seen, unique_texts = set(), []
    for sent in selected:
        if sent["text"] not in seen:
            seen.add(sent["text"])
            unique_texts.append(sent["text"])

    evidence_paragraph = " ".join(unique_texts)
    print(f"   -> Evidence: {word_count} words | {len(unique_texts)} sentences")

    final_output.append({
        "ID":       claim_id,
        "Text":     claim_text,
        "Evidence": evidence_paragraph
    })

    # ---- Periodic checkpoint ----
    if index % CHECKPOINT_INTERVAL == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
            json.dump(final_output, fh, indent=4, ensure_ascii=False)
        print(f"   [Checkpoint] Saved progress for {index} claims.")


# =========================================================
# SECTION 8: SAVE FINAL OUTPUT
# =========================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
    json.dump(final_output, fh, indent=4, ensure_ascii=False)

print("\n" + "=" * 55)
print("Batch Ranking & Extraction Completed Successfully!")
print(f"Total claims processed : {len(final_output)}")
print(f"Output saved to        : {OUTPUT_FILE}")
print("=" * 55)