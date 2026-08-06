import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_input_file(input_dir: Path) -> Path:
    """
    Finds the input claims JSON file in the Input directory.
    Searches for claims.json, input.json, or the first available .json file.
    """
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)

    candidates = ["claims.json", "input.json", "topk_output.json"]
    for c in candidates:
        p = input_dir / c
        if p.exists():
            return p

    # Fallback: find any .json file inside Input/
    json_files = list(input_dir.glob("*.json"))
    if json_files:
        return json_files[0]

    return input_dir / "claims.json"


def count_claims(input_file: Path) -> int:
    """
    Counts the total number of claims in the input JSON file.
    """
    if not input_file.exists():
        print(f"[MAIN ERROR] Input file '{input_file}' does not exist.")
        print(f"Please place your input claims JSON file inside '{input_file.parent}'.")
        sys.exit(1)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict) and "claims" in data:
            return len(data["claims"])
        else:
            print(f"[MAIN ERROR] Unexpected JSON structure in '{input_file}'. Expected list of claims.")
            sys.exit(1)
    except Exception as e:
        print(f"[MAIN ERROR] Failed to read input file '{input_file}': {e}")
        sys.exit(1)


def read_json_count(file_path: Path) -> int:
    """
    Safely reads record count from a JSON output file.
    """
    if not file_path.exists():
        return 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return 0
            data = json.loads(content)
            if isinstance(data, list):
                return len(data)
    except Exception:
        pass
    return 0


def main():
    # Pipeline root directory (Indic_Claim_Ver)
    main_script_dir = Path(__file__).resolve().parent
    root_dir = main_script_dir.parent

    input_dir   = root_dir / "Input"
    output_dir  = root_dir / "Output"
    
    retrieval_out_dir = output_dir / "Retrieved_Output"
    ranker_out_dir    = output_dir / "Ranker_Output"
    llm_out_dir       = output_dir / "LLM_Output"

    # Ensure output directories exist
    for d in [retrieval_out_dir, ranker_out_dir, llm_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_file = find_input_file(input_dir)
    retrieved_out_file = retrieval_out_dir / "retrieved_documents.json"
    ranker_out_file    = ranker_out_dir / "topk_Output.json"
    submission_out_file = llm_out_dir / "Submission.json"

    print("=" * 65)
    print(" IndicClaimVer 2026 — Master Pipeline Orchestrator")
    print("=" * 65)
    print(f" Root Directory   : {root_dir}")
    print(f" Input File       : {input_file}")
    print(f" Retrieved Output : {retrieved_out_file}")
    print(f" Ranker Output    : {ranker_out_file}")
    print(f" LLM Submission   : {submission_out_file}")

    # 1. Count total claims
    total_count = count_claims(input_file)
    print(f" Total Claims     : {total_count}")
    print("=" * 65 + "\n")

    python_exe = sys.executable

    # 2. Trigger Retrieval Module in background
    print(f"[MAIN] Triggering Retrieval Module (1/3)...")
    retrieval_cmd = [
        python_exe,
        str(root_dir / "Retrieval" / "main.py"),
        str(input_file),
        str(retrieved_out_file)
    ]
    proc_retrieval = subprocess.Popen(retrieval_cmd, cwd=str(root_dir / "Retrieval"))

    # 3. Wait for Retrieval to write initial claim output to trigger Ranker
    print(f"[MAIN] Waiting for Retrieval to initialize output...")
    while not retrieved_out_file.exists() or read_json_count(retrieved_out_file) == 0:
        if proc_retrieval.poll() is not None:
            print("[MAIN ERROR] Retrieval module exited unexpectedly before producing output.")
            sys.exit(1)
        time.sleep(1.0)

    print(f"[MAIN] Retrieved output detected ({read_json_count(retrieved_out_file)} claim(s)).")

    # 4. Trigger Ranker Module in background
    print(f"[MAIN] Triggering Ranker Module (2/3)...")
    ranker_cmd = [
        python_exe,
        str(root_dir / "Ranker" / "main.py"),
        str(total_count)
    ]
    proc_ranker = subprocess.Popen(ranker_cmd, cwd=str(root_dir / "Ranker"))

    # 5. Wait for Ranker to write initial claim output to trigger LLM
    print(f"[MAIN] Waiting for Ranker to initialize output...")
    while not ranker_out_file.exists() or read_json_count(ranker_out_file) == 0:
        if proc_ranker.poll() is not None:
            print("[MAIN ERROR] Ranker module exited unexpectedly before producing output.")
            sys.exit(1)
        time.sleep(1.0)

    print(f"[MAIN] Ranker output detected ({read_json_count(ranker_out_file)} claim(s)).")

    # 6. Trigger LLM Module in background
    print(f"[MAIN] Triggering LLM Module (3/3)...")
    llm_cmd = [
        python_exe,
        str(root_dir / "LLM" / "llm_pipeline.py"),
        str(ranker_out_file),
        str(submission_out_file),
        str(total_count)
    ]
    proc_llm = subprocess.Popen(llm_cmd, cwd=str(root_dir / "LLM"))

    print("\n" + "=" * 65)
    print(" ALL PIPELINE MODULES ACTIVE & STREAMING CONCURRENTLY!")
    print("=" * 65 + "\n")

    # 7. Monitor Concurrent Execution
    last_r_count = -1
    last_rk_count = -1
    last_llm_count = -1

    while True:
        r_count   = read_json_count(retrieved_out_file)
        rk_count  = read_json_count(ranker_out_file)
        llm_count = read_json_count(submission_out_file)

        if (r_count != last_r_count) or (rk_count != last_rk_count) or (llm_count != last_llm_count):
            print(f"[PIPELINE PROGRESS] Retrieval: {r_count}/{total_count} | Ranker: {rk_count}/{total_count} | LLM: {llm_count}/{total_count}")
            last_r_count, last_rk_count, last_llm_count = r_count, rk_count, llm_count

        # Check if all processes finished or if LLM completed target count
        if llm_count >= total_count:
            print(f"\n[MAIN] All {total_count} claims successfully verified!")
            break

        # Check if any module crashed
        if proc_llm.poll() is not None and proc_llm.returncode != 0:
            print("[MAIN ERROR] LLM module failed.")
            break

        time.sleep(2.0)

    # Clean up background processes
    for p in [proc_retrieval, proc_ranker, proc_llm]:
        if p.poll() is None:
            p.wait()

    print("\n" + "=" * 65)
    print(" PIPELINE EXECUTION COMPLETE!")
    print(f" Final Submission Output : {submission_out_file}")
    print(f" Verified Claim Count   : {read_json_count(submission_out_file)}/{total_count}")
    print("=" * 65)


if __name__ == "__main__":
    main()
