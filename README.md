# IndicClaimVerifier 2026 — End-to-End Multilingual Fact Verification Pipeline

An end-to-end, high-performance automated fact-verification system for Indic language claims (supporting English, Hindi, Bengali, Marathi, and code-mixed inputs).

---

## 🏗️ Architecture & Pipeline Layout

The system consists of three specialized submodules streaming claim data concurrently:

```text
Indic_Claim_Ver/
├── Input/
│   └── claims.json                     # Input claims to verify
├── Main/
│   └── main.py                         # Master Orchestrator (Concurrent streaming manager)
├── Retrieval/                          # Subtask 1: Multilingual Web & Wiki Evidence Search Engine
│   ├── main.py
│   ├── retrieval.py
│   └── requirement.txt
├── Ranker/                             # Subtask 2: 2-Stage Neural Reranker (BGE-M3 + mMARCO)
│   ├── main.py
│   ├── ranker.py
│   ├── utils.py
│   └── requirements.txt
├── LLM/                                # Subtask 3: LLM Reasoning & Fact Verification Engine (Qwen3-8B)
│   ├── llm_pipeline.py
│   ├── input_sanitizer.py
│   └── requirements.txt
├── Output/
│   ├── Retrieved_Output/
│   │   └── retrieved_documents.json    # Streamed output from Retrieval Engine
│   ├── Ranker_Output/
│   │   └── topk_Output.json            # Streamed output from Neural Ranker
│   └── LLM_Output/
│       └── Submission.json             # Final Competition Submission Output
└── requirements.txt                    # Master dependencies
```

---

## ⚡ Data Flow & Concurrent Execution

```text
       Input/claims.json
              │
              ▼
   [ Main/main.py Orchestrator ]
              │
    ┌─────────┼────────────────────────┐
    │         │                        │
    ▼         ▼                        ▼
[ Retrieval ] ──(retrieved_documents.json)──► [ Ranker ] ──(topk_Output.json)──► [ LLM Module ] ──► Submission.json
```

1. **Incremental Atomic Streaming**:
   - `Retrieval` writes each processed claim immediately to `retrieved_documents.json` via atomic `.tmp` swapping without waiting for the full dataset to finish.
   - `Ranker` monitors `retrieved_documents.json`, reranks incoming evidence using **BGE-M3** and **mMARCO**, and streams top-$k$ output to `topk_Output.json`.
   - `LLM Module` monitors `topk_Output.json`, invokes **Qwen3-8B** on GPU (`RTX 4050`), generates structured predictions (`SUPPORTS`/`REFUTES`) with natural justifications, and streams to `Submission.json`.
2. **GPU & Memory Resilience**:
   - Dynamic VRAM monitor automatically uses GPU CUDA when free VRAM $\ge 4.0\text{ GB}$ and falls back smoothly to CPU RAM if GPU VRAM is constrained.
3. **Automated Submission Validator**:
   - Validates all 10 competition compliance rules (UTF-8 encoding, JSON array schema, word count ranges, prediction label integrity) upon completion.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ and Git are installed. Clone the repository and set up your Python environment:

```cmd
git clone https://github.com/Tech-vibe/fact-check-indicclaimver.git
cd Indic_Claim_Ver
python -m venv LLM\.venv
```

### 2. Install PyTorch (Tailored to Hardware)

- **For NVIDIA GPU Users (CUDA 12.1 Hardware Acceleration)**:
  ```cmd
  LLM\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- **For AMD / Intel / CPU-Only Users**:
  ```cmd
  LLM\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
  ```

### 3. Install Master Dependencies
```cmd
LLM\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Download GGUF Model Weight File
Download your preferred GGUF quantized model file into `LLM/models/`:
- **Default (Recommended)**: `Qwen3-8B-IQ4_XS.gguf` (Path: `LLM/models/Qwen3-8B-IQ4_XS.gguf`)
- **For Low VRAM GPUs (4 GB VRAM)**: If your GPU has limited VRAM, you can use any smaller GGUF model (e.g., `Qwen2.5-3B-Instruct-Q4_K_M.gguf`, `Llama-3.2-3B-Instruct.gguf`).
- **Auto-Discovery**: The pipeline automatically detects and loads **any `.gguf` file** placed inside `LLM/models/` without requiring manual config edits!

---

## 💻 Running The Full Pipeline

Place your target claims in `Input/claims.json` and execute the master orchestrator from the project root:

```cmd
LLM\.venv\Scripts\python.exe Main\main.py
```

### Running Individual Submodules Directly:

```cmd
# Subtask 1: Retrieval
LLM\.venv\Scripts\python.exe Retrieval/main.py

# Subtask 2: Ranker
LLM\.venv\Scripts\python.exe Ranker/main.py

# Subtask 3: LLM Verifier
LLM\.venv\Scripts\python.exe LLM/llm_pipeline.py Output/Ranker_Output/topk_Output.json Output/LLM_Output/Submission.json
```

---

## 🔧 Troubleshooting & Hardware Resilience

- **Low VRAM / 4 GB GPU Optimization**:
  - For GPUs with 4 GB VRAM, drop a **3B or 4B GGUF model** (e.g. `Qwen2.5-3B-Instruct`) into `LLM/models/`. The LLM engine will automatically select it and fit fully inside VRAM!
  - The `Ranker` module dynamically monitors VRAM. If free VRAM is below $4.0\text{ GB}$, it automatically switches to CPU RAM to prevent CUDA Out-Of-Memory (OOM) crashes while keeping the LLM on GPU.
- **Interrupts & Resuming Pipeline Runs**:
  - All submodules write output incrementally per claim. If a run is interrupted, re-running `Main/main.py` will automatically resume where it left off, skipping already verified claims.
- **Port 8080 Conflicts**:
  - The local `llama-server.exe` runs on port 8080. If another process is using port 8080, close it or edit the port setting in `LLM/llm_pipeline.py`.

---

## 📄 Output Schema (`Submission.json`)

```json
[
  {
    "ID": "S2/1001",
    "Evidence": "High quality evidence summary extracted and cleaned from top ranked sources...",
    "Prediction": "SUPPORTS",
    "Justification": "Clear, objective justification verifying all specific points in the claim..."
  }
]
```

---

## 📜 License

Distributed under the MIT License.
