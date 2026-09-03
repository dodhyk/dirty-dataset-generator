import pytest
import pandas as pd
import numpy as np
from core.generator import generate
from core.presets import get_preset


def _val_matches_reported(actual, reported) -> bool:
    if pd.isna(actual) and (reported is None or (isinstance(reported, float) and np.isnan(reported))):
        return True
    if reported is None:
        return pd.isna(actual)
    try:
        if isinstance(actual, (int, float, np.number)) and isinstance(reported, (int, float, np.number)):
            return float(actual) == float(reported)
    except (ValueError, TypeError):
        pass
    return str(actual) == str(reported)


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_corruption_report_ground_truth_consistency(preset):
    rows = 400
    dirtiness = 20.0
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=54321)

    clean_df = res.clean_df
    dirty_df = res.dirty_df
    report = res.report

    # 1. Total corruptions matches list length
    assert report.total_corruptions == len(report.corruptions)
    assert report.dirty_rows == len(set(c.row for c in report.corruptions))

    # Track reported (row, col) pairs
    reported_changes = set()

    for record in report.corruptions:
        r = record.row
        col = record.column
        reported_changes.add((r, col))

        # Row index in bounds
        assert 0 <= r < rows

        # Original value in record matches clean_df
        clean_actual = clean_df.at[r, col]
        assert _val_matches_reported(clean_actual, record.original_value), (
            f"Row {r}, Col {col}: clean_df {clean_actual} != report original {record.original_value}"
        )

        # Corrupted value in record matches dirty_df
        dirty_actual = dirty_df.at[r, col]
        assert _val_matches_reported(dirty_actual, record.corrupted_value), (
            f"Row {r}, Col {col}: dirty_df {dirty_actual} != report corrupted {record.corrupted_value}"
        )

        # The cell must have genuinely changed
        assert not _val_matches_reported(clean_actual, dirty_actual), (
            f"Row {r}, Col {col}: report recorded corruption of type {record.type}, but value was unchanged ({clean_actual})!"
        )

    # 2. Check no unrecorded changes exist in the dataset
    for r in range(rows):
        for col in clean_df.columns:
            clean_v = clean_df.at[r, col]
            dirty_v = dirty_df.at[r, col]
            if not _val_matches_reported(clean_v, dirty_v):
                assert (r, col) in reported_changes, (
                    f"Cell difference at Row {r}, Column {col} ({clean_v} -> {dirty_v}) was NOT recorded in report!"
                )


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_multiple_corruptions_per_row(preset):
    """Section 19: Test that when a single row receives multiple corruptions, each change is recorded."""
    rows = 600
    dirtiness = 30.0
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=888)

    # Count corruptions per row
    corruptions_by_row = {}
    for c in res.report.corruptions:
        corruptions_by_row.setdefault(c.row, []).append(c.column)

    # Verify that multi-corruption rows exist
    multi_corrupt_rows = {r: cols for r, cols in corruptions_by_row.items() if len(cols) > 1}
    assert len(multi_corrupt_rows) > 0, "No multiple corruptions detected in dirty rows"

    # Verify columns corrupted within same row are distinct
    for r, cols in multi_corrupt_rows.items():
        assert len(cols) == len(set(cols)), f"Row {r} has duplicate column corruptions: {cols}"


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_corruption_scope_only_eligible_features(preset):
    """Section 20: Test that corruptions occur exclusively on eligible feature columns."""
    rows = 300
    dirtiness = 40.0
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=333)

    preset_obj = get_preset(preset)
    protected_cols = set(preset_obj.get_protected_columns())
    eligible_cols = set(preset_obj.get_eligible_feature_columns())

    for record in res.report.corruptions:
        assert record.column in eligible_cols, f"Column {record.column} is not an eligible feature!"
        assert record.column not in protected_cols, f"Protected column {record.column} was corrupted!"
