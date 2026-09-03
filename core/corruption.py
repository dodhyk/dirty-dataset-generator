from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from core.models import (
    ColumnRole,
    ColumnSpec,
    CorruptionRecord,
    CorruptionReport,
    DataType,
)
from core.presets.base import BasePreset


def _to_json_safe(val: Any) -> Any:
    """Convert numpy / pandas values to JSON-serializable types."""
    if pd.isna(val):
        return None
    # Boolean must precede integer because bool is a subclass of int in Python
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return str(val)


def apply_corruption(
    clean_df: pd.DataFrame,
    preset: BasePreset,
    dirtiness: float,
    seed: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, CorruptionReport]:
    """Select dirtiness% rows and apply corruptions to eligible feature columns.
    
    ID and Target columns are strictly protected and never modified.
    Rows not selected remain completely untouched.
    """
    total_rows = len(clean_df)
    dirty_df = clean_df.copy()

    schema = preset.get_schema()
    spec_by_col: Dict[str, ColumnSpec] = {col.name: col for col in schema}
    eligible_columns = preset.get_eligible_feature_columns()

    # Calculate number of dirty rows based on percentage of rows
    if dirtiness <= 0.0 or total_rows == 0:
        dirty_row_count = 0
    else:
        dirty_row_count = int(round(total_rows * dirtiness / 100.0))
        # Ensure if dirtiness > 0, at least 1 row is selected
        dirty_row_count = max(1, min(dirty_row_count, total_rows))

    if dirty_row_count == 0 or len(eligible_columns) == 0:
        report = CorruptionReport(
            preset=preset.name,
            total_rows=total_rows,
            dirty_rows=0,
            dirtiness=float(dirtiness),
            seed=seed,
            total_corruptions=0,
            corruptions=[],
        )
        return dirty_df, report

    dirty_indices = rng.choice(total_rows, size=dirty_row_count, replace=False)
    # Sort indices for deterministic order
    dirty_indices = np.sort(dirty_indices)

    corruptions: List[CorruptionRecord] = []

    for row_idx in dirty_indices:
        # Determine number of corruptions on this row (1 to 3)
        num_corruptions = int(rng.integers(1, min(4, len(eligible_columns) + 1)))
        chosen_cols = rng.choice(eligible_columns, size=num_corruptions, replace=False)

        for col_name in chosen_cols:
            col_spec = spec_by_col[col_name]
            orig_val = dirty_df.at[row_idx, col_name]

            corrupted_val, corr_type = _corrupt_cell(orig_val, col_spec, rng)

            # Ensure column can safely hold NaN/None and mixed types without LossySetitemError
            if dirty_df[col_name].dtype != object:
                dirty_df[col_name] = dirty_df[col_name].astype(object)

            dirty_df.at[row_idx, col_name] = corrupted_val

            corruptions.append(
                CorruptionRecord(
                    row=int(row_idx),
                    column=col_name,
                    type=corr_type,
                    original_value=_to_json_safe(orig_val),
                    corrupted_value=_to_json_safe(corrupted_val),
                )
            )

    report = CorruptionReport(
        preset=preset.name,
        total_rows=total_rows,
        dirty_rows=dirty_row_count,
        dirtiness=float(dirtiness),
        seed=seed,
        total_corruptions=len(corruptions),
        corruptions=corruptions,
    )

    return dirty_df, report


def _corrupt_cell(
    val: Any, spec: ColumnSpec, rng: np.random.Generator
) -> Tuple[Any, str]:
    """Corrupt a single cell based on its data type using allowed MVP corruption strategies."""
    d_type = spec.data_type

    if d_type in (DataType.INTEGER, DataType.DECIMAL):
        # Choose between outlier or missing_value
        choices = ["missing_value", "outlier"]
        corr_type = rng.choice(choices, p=[0.45, 0.55])

        if corr_type == "missing_value":
            return np.nan, "missing_value"
        else:
            # Numeric outlier
            if d_type == DataType.INTEGER:
                orig_int = int(val)
                if orig_int == 0:
                    outlier = int(rng.integers(50, 200)) if rng.random() < 0.7 else -99
                elif rng.random() < 0.7:
                    outlier = orig_int * int(rng.integers(8, 20))
                else:
                    outlier = -orig_int
                return outlier, "outlier"
            else:
                orig_float = float(val)
                if abs(orig_float) < 1e-6:
                    outlier = round(float(rng.uniform(50.0, 200.0)), 2) if rng.random() < 0.7 else -150.0
                elif rng.random() < 0.7:
                    outlier = round(orig_float * float(rng.uniform(6.0, 15.0)), 2)
                else:
                    outlier = round(-orig_float, 2)
                return outlier, "outlier"

    elif d_type in (DataType.CATEGORICAL, DataType.STRING):
        choices = ["missing_value", "whitespace", "case_inconsistency", "typo"]
        corr_type = rng.choice(choices, p=[0.30, 0.25, 0.25, 0.20])

        s_val = str(val) if val is not None else ""

        if corr_type == "missing_value":
            return np.nan, "missing_value"
        elif corr_type == "whitespace":
            # Add whitespace padding
            spaces = " " * int(rng.integers(1, 4))
            if rng.random() < 0.5:
                return f"{spaces}{s_val}", "whitespace"
            else:
                return f"{s_val}{spaces}", "whitespace"
        elif corr_type == "case_inconsistency":
            if s_val.isupper():
                return s_val.lower(), "case_inconsistency"
            elif s_val.islower():
                return s_val.upper(), "case_inconsistency"
            else:
                candidate = s_val.upper() if rng.random() < 0.5 else s_val.lower()
                return (candidate if candidate != s_val else s_val.swapcase()), "case_inconsistency"
        else:  # typo
            chars = list(s_val)
            # Find adjacent pairs that are different
            diff_pairs = [i for i in range(len(chars) - 1) if chars[i] != chars[i + 1]]
            if diff_pairs:
                swap_idx = diff_pairs[int(rng.integers(0, len(diff_pairs)))]
                chars[swap_idx], chars[swap_idx + 1] = chars[swap_idx + 1], chars[swap_idx]
                return "".join(chars), "typo"
            elif len(chars) > 0:
                # All characters are identical (e.g. 'AAA'); mutate one character
                sub_char = "x" if chars[0].lower() != "x" else "z"
                chars[0] = sub_char
                return "".join(chars), "typo"
            else:
                return "x", "typo"

    elif d_type == DataType.BOOLEAN:
        return np.nan, "missing_value"

    elif d_type == DataType.TIMESTAMP:
        choices = ["missing_value", "whitespace"]
        corr_type = rng.choice(choices, p=[0.6, 0.4])
        if corr_type == "missing_value":
            return np.nan, "missing_value"
        else:
            return f" {val} ", "whitespace"

    # Default fallback
    return np.nan, "missing_value"
