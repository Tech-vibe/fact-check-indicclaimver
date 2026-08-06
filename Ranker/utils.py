import os
import json
import time

def resolve_file_path(path: str, default_filename: str) -> str:
    """
    If path is a directory or path without file extension, resolves to path/default_filename.
    """
    if not path:
        return path
    if os.path.isdir(path) or not os.path.splitext(path)[1]:
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, default_filename)
    return path


def safe_read_json(filepath: str, retries: int = 3, delay: float = 0.2) -> list:
    """
    Safely reads a JSON file that may be actively written to by another process (e.g. Retrieval).
    Returns a Python list of objects, or empty list [] if file does not exist or is invalid.
    """
    filepath = resolve_file_path(filepath, "retrieved_documents.json")
    if not os.path.exists(filepath):
        return []

    for attempt in range(retries):
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("claims", [])
                return []
        except (json.JSONDecodeError, IOError, PermissionError):
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return []
    return []


def atomic_write_json(filepath: str, data: list) -> None:
    """
    Writes JSON data atomically to prevent corruption or partial file reads
    by downstream modules (e.g. LLM).
    """
    filepath = resolve_file_path(filepath, "topk_Output.json")
    dir_name = os.path.dirname(os.path.abspath(filepath))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temp_filepath = f"{filepath}.tmp"
    with open(temp_filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(temp_filepath, filepath)
