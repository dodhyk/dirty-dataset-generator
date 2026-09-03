import pytest
import numpy as np
import pandas as pd
from core.generator import generate


def _cell_equal(val_clean, val_dirty) -> bool:
    """Helper to check cell equality handling NaN and string/type representations."""
    if pd.isna(val_clean) and pd.isna(val_dirty):
        return True
    if pd.isna(val_clean) != pd.isna(val_dirty):
        return False
    # If both numeric
    try:
        if isinstance(val_clean, (int, float, np.number)) and isinstance(val_dirty, (int, float, np.number)):
            return float(val_clean) == float(val_dirty)
    except (ValueError, TypeError):
        pass
    return str(val_clean) == str(val_dirty)


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_unselected_rows_exact_cell_equality(preset):
    rows = 500
    dirtiness = 20.0  # 100 dirty rows, 400 untouched rows
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=999)

    clean_df = res.clean_df
    dirty_df = res.dirty_df
    reported_dirty_rows = set(c.row for c in res.report.corruptions)

    assert len(reported_dirty_rows) == 100

    # Test all unselected rows
    unselected_rows = [r for r in range(rows) if r not in reported_dirty_rows]
    assert len(unselected_rows) == 400

    for r in unselected_rows:
        for col in clean_df.columns:
            clean_val = clean_df.at[r, col]
            dirty_val = dirty_df.at[r, col]
            assert _cell_equal(clean_val, dirty_val), (
                f"Row {r} was unselected but column {col} changed from {clean_val} to {dirty_val}!"
            )


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_actual_dirty_rows_match_report(preset):
    rows = 300
    dirtiness = 15.0  # 45 dirty rows
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=42)

    clean_df = res.clean_df
    dirty_df = res.dirty_df
    reported_dirty_rows = set(c.row for c in res.report.corruptions)

    # Detect actual dirty rows by comparing all cells
    actual_differing_rows = set()
    for r in range(rows):
        row_differs = False
        for col in clean_df.columns:
            if not _cell_equal(clean_df.at[r, col], dirty_df.at[r, col]):
                row_differs = True
                break
        if row_differs:
            actual_differing_rows.add(r)

    # Actual differing rows must equal reported dirty rows exactly
    assert actual_differing_rows == reported_dirty_rows, (
        f"Mismatch between actual dirty rows ({len(actual_differing_rows)}) and reported ({len(reported_dirty_rows)})"
    )
