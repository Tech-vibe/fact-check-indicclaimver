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
| **GPU** | NVIDIA GPU with **6 GB+ free VRAM** (e.g. RTX 3060, RTX 4060, T4, A10G) |
| **CUDA** | NVIDIA CUDA drivers installed |
| **OS** | Windows 10/11 (PowerShell / Command Prompt) |
| **Python** | Python 3.10 or higher |

---

## 📁 Repository Structure

```text
IndicClaimVerifier/
├── models/
│   └── Qwen3-8B-Q4_K_M.gguf          # Model weights (Download separately)
├── llama.cpp/
│   └── build/bin/Release/
│       └── llama-server.exe           # GPU server binary
├── input_sanitizer.py                # Pre-processing & web boilerplate cleaner
├── llm_pipeline.py                   # Core LLM inference & verification pipeline
├── run_pipeline.bat                  # One-click GPU server + pipeline execution batch script
├── run_server.bat                    # Standalone llama-server launcher
├── input/
│   └── topk_output.json              # Input claims & retrieved evidence dataset
├── output/
│   ├── submission.json               # Final output predictions (generated)
│   └── submission_checkpoint.json    # Automatic progress checkpoint file
└── README.md                         # Documentation
```

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/IndicClaimVerifier.git
cd IndicClaimVerifier
```

### Step 2: Download Model Weights

Download the GGUF model weights file and place it inside the `models/` directory:

- **Download Link**: [Hugging Face - Qwen3-8B GGUF](https://huggingface.co/models?search=Qwen3-8B-GGUF) (Download `Qwen3-8B-Q4_K_M.gguf`)
- **Model File**: `Qwen3-8B-Q4_K_M.gguf`
- **Destination Path**: `IndicClaimVerifier/models/Qwen3-8B-Q4_K_M.gguf`

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

# Install required dependencies
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
2. Launches `llama-server.exe` on GPU (`-c 8192` context size, full GPU offload `-ngl 99`, Flash Attention enabled).
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

Example warning output:
```
[WARN] Record [499] (ID='S2/1500'): Missing or empty 'Evidence' field.
[WARN] 1 record(s) skipped. Fix them and use --repair <CLAIM_ID> to add them later.
```

---

## 🔧 Repair Mode — Adding a Single Skipped Claim

If a claim was skipped during the main run due to a bad input field, fix the issue in `input/topk_output.json` and then use Repair Mode to process just that one claim and add it to your existing output without re-running everything.

### How to use it

```cmd
run_pipeline.bat --repair "S2/T/BN/1045"
```

### What Repair Mode does

1. Starts the GPU server.
2. Finds the claim with the given ID in `input/topk_output.json`.
3. Validates that the claim's fields are now properly filled in.
4. Sends it through the full LLM pipeline.
5. Loads the existing `output/submission.json`.
6. **Replaces** the entry in-place if the ID already exists, or **appends** it if it's new.
7. Saves the updated output file.
8. Re-runs the full submission validation on the complete output.

### Important notes

- ✅ All other records in the output file are **untouched**.
- ✅ The checkpoint file is **never modified** by Repair Mode.
- ✅ If the output file doesn't exist yet, it will be created.
- ❌ If the Claim ID is not found in the input file, it will exit with a clear error.
- ❌ If the field is still empty after your fix, it will refuse to run and tell you to fix it first.

## 📥 Input & Output JSON Schemas

### Input Format (`input/topk_output.json`)

```json
[
  {
    "ID": "S2/1001",
    "Text": "Claim statement to verify...",
    "Evidence1": "Retrieved evidence article 1...",
    "Evidence2": "Retrieved evidence article 2...",
    "Evidence3": "Retrieved evidence article 3..."
  }
]
```

### Output Format (`output/submission.json`)

```json
[
  {
    "ID": "S2/1001",
    "Evidence": "Detailed evidence summary (100-150 words)...",
    "Prediction": "SUPPORTS",
    "Justification": "Detailed step-by-step reasoning (100-120 words)..."
  }
]
```

---

## ⚡ Technical Highlights

1. **Multilingual Script Support**: Auto-detects native Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, and Romanized Hinglish scripts.
2. **Web Boilerplate Stripping**: Automatically removes scraped website navigation headers/footers (`প্রচ্ছদ`, `জাতীয়`, `Home`, `News`, etc.) to keep prompt context clean and focused on factual article content.
3. **Robust Structural JSON Recovery**: Key-offset slicing parser handles unescaped inner quotes and formatting quirks, guaranteeing 100% complete field extraction without mid-sentence truncations.
4. **Indic Script Token Optimization**: Configured with `REPEAT_PENALTY = 1.0` and `MAX_TOKENS = 3072` to allow full 8–12 sentence Indic outputs without premature cutoffs.

---

## 🛠️ Troubleshooting & FAQs

- **`llama-server.exe` fails to start / VRAM error**:
  Ensure no other process is using port `8080` and your GPU has at least 6 GB free VRAM.
- **Console text encoding on Windows**:
  The pipeline automatically configures UTF-8 encoding for stdout/stderr to render Devanagari and Bengali characters cleanly in the Windows terminal.
