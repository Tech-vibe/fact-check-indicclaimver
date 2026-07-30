# Fact-Check IndicClaimVer: Ranker Module

A high-performance, multi-stage document and sentence reranking system tailored for Indic language claim verification and fact-checking workflows (supporting English, Bengali, Hindi, and code-mixed inputs).

---

## 📌 Overview

The **Ranker Module** takes candidate retrieved documents for a given claim and performs two-stage relevance scoring and evidence sentence extraction:
1. **Document-Level Reranking**: Uses `BAAI/bge-reranker-v2-m3` to score and rank retrieved documents based on claim relevance.
2. **Sentence-Level Extraction & Reranking**: Splits top documents into sentences using multilingual punctuation logic (handling English `.!?` and Indic `।`), ranks individual sentences using `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, and reconstructs a chronologically coherent evidence paragraph.

---

## ✨ Features

- **Multi-Stage Reranking Pipeline**: Combines cross-encoder document scoring with sentence-level fine-grained extraction.
- **Multilingual & Indic Support**: Sentence segmentation handles Indic sentence terminators (e.g., Devanagari and Bengali *danda* `।`).
- **Transformers v5+ Compatibility**: Includes automated tokenization patching for modern Hugging Face `transformers` releases.
- **Batch Processing & Checkpointing**: Automatic progress saving every 50 records to handle large datasets safely.
- **Hardware Acceleration**: Automatic GPU (`CUDA`) detection with FP16 precision for memory and speed optimization, with graceful fallback to CPU.

---

## 🛠️ Project Structure

```text
IndicClaimVer/
├── input/                          # Input JSON datasets from retrieval stage
│   └── retrived_documents.json
├── output/                         # Processed and ranked evidence JSON outputs
│   └── ranked_documents.json
├── ranker.py                       # Unified two-stage reranking & noise-filtered evidence extraction script
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 💻 Prerequisites & Requirements

- **Python**: `3.8` or higher
- **PyTorch**: Recommended with CUDA support if using GPU acceleration

### Python Packages (`requirements.txt`)
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `sentence-transformers >= 2.2.2`
- `FlagEmbedding >= 1.2.0`

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tech-vibe/fact-check-indicclaimver.git
   cd fact-check-indicclaimver
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### 1. Preparing Input Data
Place your JSON input file in `input/retrived_documents.json`. The expected format is:

```json
[
  {
    "id": "claim_001",
    "text": "The claim text to be verified",
    "documents": [
      "Candidate document 1 text...",
      "Candidate document 2 text..."
    ]
  }
]
```

### 2. Running the Reranker (`ranker.py`)
To execute the unified two-stage reranking (Document BGE-M3 + Sentence mMARCO with 5-layer noise filtering and dynamic word-count targeting):

```bash
python ranker.py
```

Output will be saved under `output/ranked_documents.json`:

```json
[
  {
    "ID": "claim_001",
    "Text": "The claim text to be verified",
    "Evidence": "Extracted top relevant evidence sentences..."
  }
]
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
