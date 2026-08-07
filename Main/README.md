# IndicClaimVer — Master Pipeline Orchestrator Module

This module orchestrates the end-to-end multi-stage pipeline for Indic claim verification.

## Functionality
- Reads claim JSON input from `Input/claims.json`.
- Concurrently triggers `Retrieval/main.py`, `Ranker/main.py`, and `LLM/llm_pipeline.py`.
- Monitors progress in real-time across all stages.
- Features self-healing auto-recovery: automatically restarts any module sub-process if interrupted.

## Usage
From the project root directory:
```bash
python Main/main.py
```
