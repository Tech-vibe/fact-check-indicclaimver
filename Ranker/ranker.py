import os
# Prevents the OpenMP duplicate library crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import types
import unicodedata
import torch
from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
from transformers.tokenization_utils_base import BatchEncoding
import difflib


def is_sentence_duplicate(candidate: str, accepted_sentences: list, similarity_threshold: float = 0.70) -> bool:
    """
    Checks if a candidate sentence is a fuzzy duplicate or sub-phrase of any
    already accepted sentence.
    """
    norm_cand = re.sub(r'[^\w\u0900-\u097F\u0980-\u09FF\u0600-\u06FF]', '', candidate.lower())
    if not norm_cand:
        return True

    for acc in accepted_sentences:
        norm_acc = re.sub(r'[^\w\u0900-\u097F\u0980-\u09FF\u0600-\u06FF]', '', acc.lower())
        if not norm_acc:
            continue

        # 1. Exact normalized match
        if norm_cand == norm_acc:
            return True

        # 2. Substring containment check
        if len(norm_cand) > 20 and len(norm_acc) > 20:
            if norm_cand in norm_acc or norm_acc in norm_cand:
                return True

        # 3. Fuzzy ratio match
        if difflib.SequenceMatcher(None, norm_cand, norm_acc).ratio() >= similarity_threshold:
            return True

    return False


class NoiseFilter:
    """
    5-Layer Enhanced Indic-Aware Noise Filter.
    Removes web page boilerplate, ads, navigation bars, dates, and non-text noise.
    """
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

    @classmethod
    def is_noisy(cls, sentence: str) -> bool:
        words = sentence.split()

        # Layer 1: Minimum length check
        if len(words) < 6:
            return True

        # Layer 2: Keyword blacklist
        if cls._NOISE_KEYWORDS.search(sentence):
            return True

        # Layer 3: Structural web noise
        if cls._NOISE_STRUCTURAL.search(sentence):
            return True

        # Layer 4: Unicode category check ('L' for Letter, 'M' for Indic matras)
        no_space = sentence.replace(' ', '')
        if no_space:
            lm_chars = sum(1 for c in sentence if unicodedata.category(c).startswith(('L', 'M')))
            if lm_chars / len(no_space) < 0.50:
                return True

        # Layer 5: ALL-CAPS text check
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        if words and upper_words / len(words) > 0.50:
            return True

        return False


class SentenceSplitter:
    """
    Multilingual Sentence Splitter for English and Indic languages (Devanagari/Bengali).
    Handles English (. ! ?), Indic danda (। / \u0964), and newline breaks.
    """
    @staticmethod
    def split(text: str) -> list:
        if not text:
            return []
        raw_sentences = re.split(r'(?:(?<=[\u0964.!?])\s+|\n+)', text.strip())
        clean_sentences = []
        for s in raw_sentences:
            s = re.sub(r'\s+', ' ', s).strip()
            if s and not NoiseFilter.is_noisy(s):
                clean_sentences.append(s)
        return clean_sentences


class RankerPipeline:
    """
    Unified 2-Stage Reranking and Evidence Extraction Pipeline for IndicClaimVer 2026.
    Stage 1: Document Reranking (BAAI/bge-reranker-v2-m3)
    Stage 2: Sentence Reranking (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
    """
    def __init__(self, top_k_docs: int = 3, target_min: int = 80, target_words: int = 150, target_max: int = 180):
        self.top_k_docs = top_k_docs
        self.target_min = target_min
        self.target_words = target_words
        self.target_max = target_max

        # CUDA & VRAM check
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            free_vram, _ = torch.cuda.mem_get_info()
            free_vram_gb = free_vram / (1024 ** 3)
            print(f"[RANKER] GPU Free VRAM: {free_vram_gb:.2f} GB")
            if free_vram_gb < 4.0:
                print("[RANKER] Warning: < 4 GB VRAM free -- falling back to CPU.")
                self.cuda_available = False
            else:
                torch.cuda.empty_cache()

        device_label = "GPU (CUDA)" if self.cuda_available else "CPU"

        # Load Stage 1 BGE Document Reranker
        print(f"[RANKER] Loading BGE Document Reranker ({device_label})...")
        self.bge_reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=self.cuda_available,
            devices="cuda" if self.cuda_available else None,
            batch_size=4 if self.cuda_available else 64
        )

        # Tokenizer patch for Transformers v5+ compatibility
        if not hasattr(self.bge_reranker.tokenizer, "prepare_for_model"):
            def _prepare_for_model(self_tok, ids, pair_ids=None, truncation=None,
                                   max_length=None, padding=False, **kwargs):
                bos = self_tok.bos_token_id or 0
                eos = self_tok.eos_token_id or 2
                d = pair_ids or []
                if truncation == "only_second" and max_length is not None:
                    d = d[:max(0, max_length - len(ids) - 4)]
                input_ids = [bos] + ids + [eos, eos] + d + [eos]
                return BatchEncoding({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

            self.bge_reranker.tokenizer.prepare_for_model = types.MethodType(
                _prepare_for_model, self.bge_reranker.tokenizer
            )

        # Load Stage 2 mMARCO Sentence Reranker
        print(f"[RANKER] Loading mMARCO Sentence Reranker ({device_label})...")
        self.sentence_reranker = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            max_length=512,
            device="cuda" if self.cuda_available else "cpu"
        )
        print("[RANKER] All ranking models initialized successfully!\n")

    def rerank_sentences(self, claim: str, documents: list) -> list:
        """
        Scores sentences across provided documents against the claim text.
        """
        all_sentences = []
        for doc_id, doc in enumerate(documents):
            for sent_id, sentence in enumerate(SentenceSplitter.split(doc)):
                all_sentences.append((doc_id, sent_id, sentence))

        if not all_sentences:
            return []

        pairs = [[claim, s[2]] for s in all_sentences]
        scores = self.sentence_reranker.predict(pairs, batch_size=32, show_progress_bar=False)

        ranked = [
            {"doc": doc_id, "sent": sent_id, "text": sentence, "score": float(score)}
            for (doc_id, sent_id, sentence), score in zip(all_sentences, scores)
        ]
        return sorted(ranked, key=lambda x: x["score"], reverse=True)

    def process_claim(self, claim_item: dict) -> dict:
        """
        Processes a single claim dictionary object and returns the ranked evidence item.
        """
        claim_id = claim_item.get("ID") or claim_item.get("id") or "claim_unknown"
        claim_text = claim_item.get("Text") or claim_item.get("text") or claim_item.get("claim", "")
        documents = claim_item.get("documents") or claim_item.get("retrieved_documents") or []

        doc_texts = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_texts.append(doc.get("text") or doc.get("Text", ""))
            else:
                doc_texts.append(str(doc))

        if not claim_text or not doc_texts:
            return {
                "ID": claim_id,
                "Text": claim_text,
                "Evidence": ""
            }

        # Stage 1: Document Reranking (BGE-M3)
        doc_pairs = [[claim_text, txt] for txt in doc_texts]
        doc_scores = self.bge_reranker.compute_score(doc_pairs)
        if isinstance(doc_scores, float):
            doc_scores = [doc_scores]

        scored_docs = sorted(
            [{"text": txt, "score": float(sc)} for txt, sc in zip(doc_texts, doc_scores)],
            key=lambda x: x["score"], reverse=True
        )

        # Stage 2: Sentence Reranking (mMARCO) with dynamic pool expansion and fuzzy deduplication
        selected = []
        word_count = 0

        for pool_size in [self.top_k_docs, len(scored_docs)]:
            pool_texts = [d["text"] for d in scored_docs[:pool_size]]
            ranked_sents = self.rerank_sentences(claim_text, pool_texts)

            selected = []
            accepted_texts = []
            word_count = 0
            for sent in ranked_sents:
                sent_text = sent["text"]
                if is_sentence_duplicate(sent_text, accepted_texts):
                    continue

                s_words = len(sent_text.split())
                if word_count + s_words <= self.target_max:
                    selected.append(sent)
                    accepted_texts.append(sent_text)
                    word_count += s_words
                if word_count >= self.target_words:
                    break

            if word_count >= self.target_min:
                break

        # Chronological sentence restoration
        selected.sort(key=lambda x: (x["doc"], x["sent"]))
        evidence_paragraph = " ".join([s["text"] for s in selected])

        return {
            "ID": claim_id,
            "Text": claim_text,
            "Evidence": evidence_paragraph
        }
