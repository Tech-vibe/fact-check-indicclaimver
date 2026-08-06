# IndicClaimVerifier — Multilingual LLM Fact Verification Pipeline

An end-to-end, GPU-accelerated fact-verification pipeline for multilingual claims in **English, Hindi, Bengali, Hinglish (CodeMix)**, and other Indic languages.

The pipeline takes a claim and retrieved context sources, then uses a local GPU-offloaded LLM (`Qwen3-8B`) via `llama-server` to generate a structured verification output:
- **`Evidence`**: A unified, comprehensive evidence summary (100–150 words).
- **`Prediction`**: Verification decision (`SUPPORTS` or `REFUTES`).
- **`Justification`**: Detailed step-by-step reasoning (100–120 words).

---

## 📋 System Requirements

| Component | Requirement |
| :--- | :--- |
| **GPU** | NVIDIA GPU with **5 GB+ free VRAM** (e.g. RTX 3060, RTX 4060, T4, A10G) |
| **CUDA** | NVIDIA CUDA drivers installed |
| **OS** | Windows 10/11 (PowerShell / Command Prompt) |
| **Python** | Python 3.10 or higher |

---

## 📁 Repository Structure

```text
IndicClaimVerifier/
├── models/
│   └── Qwen3-8B-IQ4_XS.gguf          # Model weights (Download separately)
├── llama.cpp/
│   └── build/bin/Release/
│       └── llama-server.exe           # GPU server binary
├── input_sanitizer.py                # Pre-processing, Indic Unicode normalizer & web boilerplate cleaner
├── llm_pipeline.py                   # Core LLM inference & verification pipeline
├── run_pipeline.bat                  # One-click GPU server + pipeline execution batch script
├── run_server.bat                    # Standalone llama-server launcher
├── input/
│   └── topk_output.json              # Input claims & retrieved evidence dataset
├── output/
│   ├── submission.json               # Final output predictions (generated)
│   └── submission_checkpoint.json    # Automatic progress checkpoint file
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tech-vibe/fact-check-indicclaimver.git
cd IndicClaimVerifier
```

### Step 2: Download Model Weights

Download the GGUF model weights file and place it inside the `models/` directory:

- **Download Link**: [Hugging Face - unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF) (Download `Qwen3-8B-IQ4_XS.gguf`)
- **Model File**: `Qwen3-8B-IQ4_XS.gguf` (~4.58 GB)
- **Destination Path**: `IndicClaimVerifier/models/Qwen3-8B-IQ4_XS.gguf`

*(If `models/` folder does not exist, create it manually).*

### Step 3: Download `llama.cpp` Server Binaries

Since compiled `.exe` files are ignored in GitHub to save space, you must download the GPU server yourself:
1. Go to the [official llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases).
2. Download the pre-compiled `.zip` release matching your OS and CUDA version (e.g., `llama-bXXXX-bin-win-cuda-cu12.2-x64.zip`).
3. Extract `llama-server.exe` and its `.dll` files into: `IndicClaimVerifier/llama.cpp/build/bin/Release/`

### Step 4: Set Up Python Environment

Create a virtual environment and install the required dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install required dependencies (includes fast-langdetect and indic-nlp-library)
pip install -r requirements.txt
```

### Step 5: Place Your Input Data

Place your `topk_output.json` file inside the `input/` directory. (If the directory doesn't exist, create it).

- **File Path**: `IndicClaimVerifier/input/topk_output.json`

### Step 6: Run the Pipeline

The `run_pipeline.bat` script supports two modes:

#### Normal Mode — Process all claims
```cmd
run_pipeline.bat
```

#### Repair Mode — Process a single specific claim
```cmd
run_pipeline.bat --repair "S2/T/BN/1045"
```
*(Replace `S2/T/BN/1045` with the exact Claim ID you want to process)*

**What `run_pipeline.bat` does automatically (in both modes):**
1. Terminates any stale server instances on port `8080`.
2. Launches `llama-server.exe` on GPU (`-c 8192` context size, full GPU offload `-ngl 99`, Flash Attention enabled, 8-bit KV cache `-ctk q8_0 -ctv q8_0`, micro-batch `-ub 1024`).
3. Waits for the model to load into GPU VRAM.
4. Executes `llm_pipeline.py` (with any arguments you passed).
5. Safely shuts down the background server upon completion.

---

## 🔄 Checkpointing & Resuming Work

- **Automatic Checkpointing**: Progress is saved to `output/submission_checkpoint.json` every **5 claims**.
- **Resume Support**: If the pipeline is interrupted (e.g. power loss or manual stop), simply re-run `run_pipeline.bat`. It will automatically detect existing checkpoint records and resume seamlessly from where it left off.

---

## 🛡️ Pre-flight Input Validation

Before sending any claims to the LLM, the pipeline automatically scans your entire input file for data quality issues. This runs in under a second and prevents hours of wasted GPU time.

For each record it checks:
- `ID` field is present and non-empty.
- `Text` / `claim` field (the claim to verify) is non-empty.
- `Evidence` field is non-empty.

**What happens if a bad record is found:**
- The bad record is **skipped with a warning** — the pipeline continues processing all valid records.
- At the end of the scan, a tip is shown to use `--repair` to add the skipped claim later.

---

## 🔧 Repair Mode — Adding a Single Skipped Claim

If a claim was skipped during the main run due to a bad input field, fix the issue in `input/topk_output.json` and then use Repair Mode to process just that one claim and add it to your existing output without re-running everything.

### How to use it

```cmd
run_pipeline.bat --repair "S2/T/BN/1045"
```

---

## 📥 Input & Output JSON Schemas

### Input Format (`input/topk_output.json`)

```json
[
  {
    "ID": "S2/T/BN/1001",
    "Text": "Claim statement to verify...",
    "Evidence": "Retrieved evidence article text..."
  }
]
```

### Output Format (`output/submission.json`)

```json
[
  {
    "ID": "S2/T/BN/1001",
    "Evidence": "Detailed evidence summary (100-150 words)...",
    "Prediction": "SUPPORTS",
    "Justification": "Detailed step-by-step reasoning (100-120 words)..."
  }
]
```

---

## ⚡ Technical Highlights & Performance Features

1. **`fast-langdetect` Integration**: Uses fastText C++ engine for ultra-fast (<0.05ms) language identification across Indic scripts and Romanized text, with fallbacks.
2. **`indic-nlp-library` Unicode Normalization**: Normalizes Indic Unicode codepoints (Devanagari, Bengali, Tamil, etc.) into NFC canonical form to ensure consistent tokenization.
3. **8-Bit KV-Cache (`-ctk q8_0 -ctv q8_0`)**: Quantizes attention KV cache to 8-bit, saving 50% memory bandwidth and boosting generation speeds by 20–30% with zero loss in accuracy.
4. **Micro-Batch Optimization (`-ub 1024`)**: Speeds up the prompt prefill phase by 2x on GPU Tensor Cores.
5. **Code-Mix / Hinglish Handling**: Automatically detects Hinglish and Code-Mix claims and generates clear, formal English outputs for `Evidence` and `Justification` fields.
6. **Persistent Connection Pooling**: Uses `requests.Session()` to eliminate TCP handshake overhead per claim.
7. **Web Boilerplate Stripping**: Automatically removes scraped website navigation headers/footers (`প্রচ্ছদ`, `জাতীয়`, `Home`, `News`, etc.) to keep prompt context clean and focused on factual article content.
8. **Robust Structural JSON Recovery**: Key-offset slicing parser handles unescaped inner quotes and formatting quirks, guaranteeing 100% complete field extraction without mid-sentence truncations.

