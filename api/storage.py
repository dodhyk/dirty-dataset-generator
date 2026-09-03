import json
import re
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from core.models import GenerationResult

# Base storage directory
BASE_STORAGE_DIR = Path(__file__).resolve().parent.parent / "generated"

# Safe UUID pattern
UUID_REGEX = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")


def ensure_storage_dir() -> Path:
    BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_STORAGE_DIR


def validate_generation_id(generation_id: str) -> None:
    if not isinstance(generation_id, str) or not UUID_REGEX.match(generation_id.strip().lower()):
        raise ValueError(f"Invalid generation_id format: '{generation_id}'")


def save_generation_result(result: GenerationResult) -> Tuple[str, Dict[str, Path]]:
    """Save clean csv, dirty csv, and corruption report json in an isolated directory."""
    ensure_storage_dir()
    gen_id = str(uuid.uuid4())
    gen_dir = BASE_STORAGE_DIR / gen_id
    gen_dir.mkdir(parents=True, exist_ok=False)

    clean_filename = f"{result.preset}_clean.csv"
    dirty_filename = f"{result.preset}_dirty.csv"
    report_filename = "corruption_report.json"

    clean_path = gen_dir / clean_filename
    dirty_path = gen_dir / dirty_filename
    report_path = gen_dir / report_filename

    # Save clean dataset
    result.clean_df.to_csv(clean_path, index=False)

    # Save dirty dataset
    result.dirty_df.to_csv(dirty_path, index=False)

    # Save corruption report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.report.to_dict(), f, indent=2)

    return gen_id, {
        "clean": clean_path,
        "dirty": dirty_path,
        "report": report_path,
    }


def get_generation_file(generation_id: str, file_type: str) -> Path:
    """Retrieve the file path for a given generation_id and file_type ('clean', 'dirty', 'report').
    
    Guarantees isolation and strictly prevents path traversal.
    """
    validate_generation_id(generation_id)
    ensure_storage_dir()

    gen_dir = (BASE_STORAGE_DIR / generation_id).resolve()

    # Path traversal protection: Ensure gen_dir is strictly inside BASE_STORAGE_DIR
    if not gen_dir.is_relative_to(BASE_STORAGE_DIR.resolve()):
        raise PermissionError("Access denied: path traversal attempt detected")

    if not gen_dir.exists() or not gen_dir.is_dir():
        raise FileNotFoundError(f"Generation '{generation_id}' not found")

    if file_type == "clean":
        matches = list(gen_dir.glob("*_clean.csv"))
    elif file_type == "dirty":
        matches = list(gen_dir.glob("*_dirty.csv"))
    elif file_type == "report":
        matches = list(gen_dir.glob("corruption_report.json"))
    else:
        raise ValueError(f"Unknown file_type '{file_type}'. Must be 'clean', 'dirty', or 'report'.")

    if not matches or not matches[0].exists():
        raise FileNotFoundError(f"Requested {file_type} file for generation '{generation_id}' not found")

    target_file = matches[0].resolve()
    if not target_file.is_relative_to(gen_dir):
        raise PermissionError("Access denied: path traversal attempt detected")

    return target_file
