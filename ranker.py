import os
# Prevents the OpenMP duplicate library crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from FlagEmbedding import FlagReranker

import torch

# ---------------------------------------------------------
# HELPER FUNCTION: Extract First N Sentences
# ---------------------------------------------------------
def extract_first_n_sentences(text, n=10):
    """
    Splits text into sentences using English and Indic punctuation,
    and returns only the first 'n' sentences joined together.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Split on English (.!?) and Indic (।|) punctuation, keeping the punctuation attached
    sentences = re.split(r'(?<=[.!?|।])\s+', text.strip())
    
    # Remove empty strings caused by multiple spaces
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take only the first N sentences and join them back into a single string
    return " ".join(sentences[:n])

# -------------------------------
# STEP 1: Load BGE Reranker Model
# -------------------------------
cuda_available = torch.cuda.is_available()
print(f"Loading BGE Reranker (CUDA Available: {cuda_available})...")
reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3", 
    use_fp16=cuda_available,                     # Enable FP16 on GPU for memory/speed optimization
    devices="cuda" if cuda_available else None,  # Use GPU if available, fallback to CPU
    batch_size=32 if cuda_available else 128     # Use memory-safe batch size of 32 on GPU
)

# Monkey-patch prepare_for_model for Hugging Face transformers v5+ compatibility
if not hasattr(reranker.tokenizer, "prepare_for_model"):
    import types
    from transformers.tokenization_utils_base import BatchEncoding

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

print("Model loaded successfully!")

# --------------------------------
# STEP 2: Read Batch JSON Input File
# --------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "input", "retrived.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output", "ranked_documents.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle cases where the JSON is a direct list or wrapped inside a dictionary
claims_list = data if isinstance(data, list) else data.get("claims", [])
print(f"\nLoaded dataset. Total claims found to process: {len(claims_list)}")

final_output = []
TOP_K = 3  # Maximum number of evidence pieces to keep per claim
CHECKPOINT_INTERVAL = 50  # Save progress to the file after every 50 records

# --------------------------------
# STEP 3: Process Every Claim in the File
# --------------------------------
for index, item in enumerate(claims_list, start=1):
    
    # Safely extract ID and Text
    claim_id = item.get("id") or item.get("ID") or f"claim_{index}"
    claim_text = item.get("text") or item.get("Text") or item.get("claim")
    documents = item.get("documents", [])
    
    if not claim_text or not documents:
        print(f"Skipping index {index}: Missing text or candidate documents.")
        continue
        
    print(f"[{index}/{len(claims_list)}] Processing ID: {claim_id} ({len(documents)} docs)...")
    
    # Safely extract text whether 'doc' is a dictionary OR a plain string
    doc_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_texts.append(doc.get("text") or doc.get("Text", ""))
        else:
            doc_texts.append(str(doc))
    
    # Create pairs for the cross-encoder model
    pairs = [[claim_text, txt] for txt in doc_texts]
    
    # Compute relevance scores across all languages
    scores = reranker.compute_score(pairs)
    
    # Pair documents with scores temporarily to sort them
    scored_docs = []
    for txt, score in zip(doc_texts, scores):
        scored_docs.append({
            "text": txt,
            "score": float(score)
        })
    
    # Sort documents by score descending (highest relevance first)
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # --- THE 10-SENTENCE FIX ---
    # Extract ONLY the first 10 sentences from the Top-K items
    top_evidence_texts = []
    for doc in scored_docs[:TOP_K]:
        truncated_text = extract_first_n_sentences(doc["text"], n=10)
        top_evidence_texts.append(truncated_text)
    
    # Create the final dictionary using your exact requested capital format
    result_dict = {
        "ID": claim_id,
        "Text": claim_text
    }
    
    # Dynamically add Evidence1, Evidence2, Evidence3, etc.
    for i, text in enumerate(top_evidence_texts, start=1):
        result_dict[f"Evidence{i}"] = text
        
    # Append structured result to the master output array
    final_output.append(result_dict)

    # --------------------------------
    # CHECKPOINT: Save Progress Periodically
    # --------------------------------
    if index % CHECKPOINT_INTERVAL == 0:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"   [Checkpoint] Progress saved for the first {index} claims.")

# --------------------------------
# STEP 4: Save Final Combined Output
# --------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Write the final array (capturing any remaining records after the last checkpoint)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print("\n" + "="*40)
print("Batch Ranking Completed Successfully!")
print(f"All processed claims compiled into: {OUTPUT_FILE}")
print("="*40)