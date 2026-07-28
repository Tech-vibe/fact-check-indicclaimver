from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker

import os
import json
import re
import torch

def split_into_sentences(text):
    if not text:
        return []

    text = re.sub(r'\s+', ' ', text).strip()

    # Split after ., !, ?, or Devanagari danda
    sentences = re.split(r'(?<=[।.!?])\s*', text)

    return [s.strip() for s in sentences if s.strip()]

def rerank_sentences(claim, documents, model, top_k=10):

    all_sentences = []

    for doc_id, doc in enumerate(documents):

        sentences = split_into_sentences(doc)

        for sent_id, sentence in enumerate(sentences):
            all_sentences.append(
                (doc_id, sent_id, sentence)
            )

    if len(all_sentences) == 0:
        return []

    pairs = [
        (claim, s[2])
        for s in all_sentences
    ]

    scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    ranked = []

    for (doc_id, sent_id, sentence), score in zip(all_sentences, scores):
        ranked.append({
            "doc": doc_id,
            "sent": sent_id,
            "text": sentence,
            "score": score
        })

    # Select top-k by score
    ranked = sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    # Restore document/sentence order
    ranked = sorted(
        ranked,
        key=lambda x: (x["doc"], x["sent"])
    )

    return [x["text"] for x in ranked]


cuda_available = torch.cuda.is_available()

print("Loading BGE Reranker...")


reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=cuda_available
)

print("Loading Sentence CrossEncoder...")

sentence_model = CrossEncoder(
    #"cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512
)

print("Models loaded successfully.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "input", "input_from_retrieval.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output", "output_ranked_documents.json")

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

    # -----------------------------
    # Select Top-K Documents
    # -----------------------------
    top_docs = scored_docs[:TOP_K]

    top_doc_texts = [
        doc["text"]
        for doc in top_docs
    ]

    # -----------------------------
    # Sentence Reranking
    # -----------------------------
    top_sentences = rerank_sentences(
        claim_text,
        top_doc_texts,
        sentence_model,
        top_k=10
    )

    evidence_paragraph = " ".join(top_sentences)

    result_dict = {
        "ID": claim_id,
        "Text": claim_text,
        "Evidence": evidence_paragraph
    }

    final_output.append(result_dict)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print(f"\nSaved {len(final_output)} records to {OUTPUT_FILE}")
