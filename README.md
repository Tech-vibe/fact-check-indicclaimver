# IndicClaimVer 2026 — Retrieval Module

Retrieval module for the IndicClaimVer 2026 fact-checking pipeline.

## What This Module Does

Fetches evidence documents from Wikipedia and web search
for multilingual claims (English, Hindi, Bengali, Marathi).
Outputs a JSON file with claims and retrieved documents.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```
