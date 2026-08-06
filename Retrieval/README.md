# IndicClaimVer 2026 — Retrieval Module (Subtask 1)

Multilingual web & Wikipedia evidence document retrieval engine tailored for Indic claim verification (supporting English, Hindi, Bengali, Marathi, and code-mixed claims).

## Features

- Fetches evidence documents from DuckDuckGo web search and Wikipedia APIs.
- Per-claim atomic incremental writing (`retrieved_documents.json`) to allow streaming downstream to Ranker without waiting for full batch completion.
- UTF-8 console and JSON output handling.

## Setup

```bash
pip install -r requirement.txt
```

## Usage

```bash
# Executed by Main orchestrator or directly:
python main.py [input_json] [output_json] [language]
```
If no arguments are passed, defaults to:
- Input: `\Indic_Claim_Ver\Input\claims.json`
- Output: `\Indic_Claim_Ver\Output\Retrieved_Output\retrieved_documents.json`
