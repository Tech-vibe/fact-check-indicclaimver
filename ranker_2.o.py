import os
# Prevents the OpenMP duplicate library crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import types
import torch
from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
from transformers.tokenization_utils_base import BatchEncoding

# ---------------------------------------------------------
# HELPER FUNCTION: Split text into clean sentences
# ---------------------------------------------------------
def split_into_sentences(text):
    if not text:
        return []

    # Clean up excess whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Split after ., !, ?, or Devanagari/Bengali danda (।), keeping the punctuation attached
    sentences = re.split(r'(?<=[।.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]

# ---------------------------------------------------------
# HELPER FUNCTION: Sentence-Level Reranking
# ---------------------------------------------------------
def rerank_sentences(claim, documents, model, top_k=15):
    all_sentences = []

    # Extract all sentences from all candidate documents
    for doc_id, doc in enumerate(documents):
        sentences = split_into_sentences(doc)
        for sent_id, sentence in enumerate(sentences):
            all_sentences.append((doc_id, sent_id, sentence))

    if len(all_sentences) == 0:
        return []

    # Pair the claim with every single sentence
    pairs = [[claim, s[2]] for s in all_sentences]

    # Predict relevance scores
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = []
    for (doc_id, sent_id, sentence), score in zip(all_sentences, scores):
        ranked.append({
            "doc": doc_id,
            "sent": sent_id,
            "text": sentence,
            "score": float(score)  # Convert to float for JSON safety
        })

    # 1. Select top-k highest scoring sentences (Now set to 15)
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]

    # 2. Restore their original chronological order (by Document ID, then Sentence ID)
    ranked = sorted(ranked, key=lambda x: (x["doc"], x["sent"]))

    return [x["text"] for x in ranked]

# ---------------------------------------------------------
# STEP 1: Load Models (BGE & mMARCO)
# ---------------------------------------------------------
cuda_available = torch.cuda.is_available()
print(f"Loading BGE Reranker (CUDA Available: {cuda_available})...")

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=cuda_available,                     # Enable FP16 on GPU for memory/speed
    devices="cuda" if cuda_available else None,  # Use GPU if available
    batch_size=32 if cuda_available else 128
)

# Monkey-patch prepare_for_model for Hugging Face transformers v5+ compatibility
if not hasattr(reranker.tokenizer, "prepare_for_model"):
    def prepare_for_model(self, ids, pair_ids=None, truncation=None, max_length=None, padding=False, **kwargs):
        bos = self.bos_token_id if self.bos_token_id is not None else 0
        eos = self.eos_token_id if self.eos_token_id is not None else 2
        q_inp = ids
        d_inp = pair_ids if pair_ids is not None else []
        
        if truncation == 'only_second' and max_length is not None:
            allowed_d_len = max_length - len(q_inp) - 4
            d_inp = d_inp[:max(0, allowed_d_len)]
            
        input_ids = [bos] + q_inp + [eos, eos] + d_inp + [eos]
        attention_mask = [1] * len(input_ids)
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    reranker.tokenizer.prepare_for_model = types.MethodType(prepare_for_model, reranker.tokenizer)
    print("Tokenizer monkey-patched successfully for Transformers compatibility!")

print("Loading Sentence CrossEncoder (mMARCO)...")
sentence_model = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512,
    device="cuda" if cuda_available else "cpu"
)

print("Models loaded successfully!")

# ---------------------------------------------------------
# STEP 2: Read Batch JSON Input File (Enforcing UTF-8)
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "input", "output_bengali")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output", "output_ranked_documents.json")

# Crucial fix: encoding="utf-8" prevents Mojibake (gibberish characters)
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

claims_list = data if isinstance(data, list) else data.get("claims", [])
print(f"\nLoaded dataset. Total claims found to process: {len(claims_list)}")

final_output = []
TOP_K_DOCS = 3             # Maximum number of full documents to keep
TOP_K_SENTENCES = 15       # Maximum number of sentences to extract per claim (UPDATED TO 15)
CHECKPOINT_INTERVAL = 50   # Save progress every 50 records

# Make output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ---------------------------------------------------------
# STEP 3: Process Every Claim in the File
# ---------------------------------------------------------
for index, item in enumerate(claims_list, start=1):
    
    claim_id = item.get("id") or item.get("ID") or f"claim_{index}"
    claim_text = item.get("text") or item.get("Text") or item.get("claim")
    documents = item.get("documents", [])
    
    if not claim_text or not documents:
        print(f"Skipping index {index}: Missing text or candidate documents.")
        continue
        
    print(f"[{index}/{len(claims_list)}] Processing ID: {claim_id} ({len(documents)} docs)...")
    
    # Safely extract text whether 'doc' is a dict or string
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get("text") or doc.get("Text", ""))
        else:
            doc_texts.append(str(doc))
            
    if not doc_texts:
        continue
    
    # --- STAGE 1: DOCUMENT-LEVEL RANKING (BGE) ---
    pairs = [[claim_text, txt] for txt in doc_texts]
    scores = reranker.compute_score(pairs)
    
    # Handle single-document fallback
    if isinstance(scores, float):
        scores = [scores]
    
    scored_docs = [{"text": txt, "score": float(score)} for txt, score in zip(doc_texts, scores)]
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # Select the Top-K best documents
    top_doc_texts = [doc["text"] for doc in scored_docs[:TOP_K_DOCS]]

    # --- STAGE 2: SENTENCE-LEVEL RANKING (mMARCO) ---
    top_sentences = rerank_sentences(
        claim=claim_text,
        documents=top_doc_texts,
        model=sentence_model,
        top_k=TOP_K_SENTENCES  # Passes the new 15-sentence limit
    )

    # Recombine the chronologically sorted sentences into a single paragraph
    evidence_paragraph = " ".join(top_sentences)

    # Build final result
    result_dict = {
        "ID": claim_id,
        "Text": claim_text,
        "Evidence": evidence_paragraph
    }
    final_output.append(result_dict)

    # --- CHECKPOINT: Save Progress Periodically ---
    if index % CHECKPOINT_INTERVAL == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"   [Checkpoint] Progress saved for the first {index} claims.")

# ---------------------------------------------------------
# STEP 4: Save Final Combined Output
# ---------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print("\n" + "="*50)
print(f"Batch Ranking & Extraction Completed Successfully!")
print(f"Saved {len(final_output)} records to: {OUTPUT_FILE}")
print("="*50)