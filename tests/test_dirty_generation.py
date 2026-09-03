import pytest
import numpy as np
import pandas as pd
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_dirty_row_percentage_and_protection(preset):
    rows = 1000
    dirtiness = 20.0
    result = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=999)

    clean_df = result.clean_df
    dirty_df = result.dirty_df
    report = result.report

    # 1. Total rows remain identical
    assert len(dirty_df) == rows
    assert report.total_rows == rows

    # 2. Dirty rows count matches dirtiness percentage (20% of 1000 = 200)
    expected_dirty = int(round(rows * dirtiness / 100.0))
    assert report.dirty_rows == expected_dirty
    assert report.dirtiness == dirtiness

    # 3. Protected columns are NEVER modified
    id_col = "customer_id" if "customer_id" in clean_df.columns else "house_id"
    target_col = "price" if preset == "house_price" else ("churn" if preset == "customer_churn" else "purchased_again")

    # Verify ID column bitwise equality
    pd.testing.assert_series_equal(clean_df[id_col], dirty_df[id_col], check_names=True)
    assert dirty_df[id_col].nunique() == rows

    # Verify Target column bitwise equality
    pd.testing.assert_series_equal(clean_df[target_col], dirty_df[target_col], check_names=True)

    # 4. Untouched rows are completely identical to clean rows
    changed_rows = set(c.row for c in report.corruptions)
    assert len(changed_rows) == expected_dirty

    untouched_rows = [i for i in range(rows) if i not in changed_rows]
    clean_untouched = clean_df.iloc[untouched_rows].reset_index(drop=True)
    dirty_untouched = dirty_df.iloc[untouched_rows].reset_index(drop=True)
    pd.testing.assert_frame_equal(clean_untouched, dirty_untouched, check_dtype=False)

    # 5. Corruption report accuracy
    assert report.total_corruptions == len(report.corruptions)
    assert report.total_corruptions >= expected_dirty

    for rec in report.corruptions:
        assert rec.column != id_col
        assert rec.column != target_col
        # Verify row index in bounds
        assert 0 <= rec.row < rows


def test_zero_dirtiness():
    result = generate(preset="customer_churn", rows=200, dirtiness=0.0, seed=123)
    assert result.report.dirty_rows == 0
    assert result.report.total_corruptions == 0
    pd.testing.assert_frame_equal(result.clean_df, result.dirty_df)


def test_full_dirtiness():
    result = generate(preset="house_price", rows=100, dirtiness=100.0, seed=123)
    assert result.report.dirty_rows == 100
    assert result.report.total_corruptions >= 100
    # ID and target must still be protected
    pd.testing.assert_series_equal(result.clean_df["house_id"], result.dirty_df["house_id"])
    pd.testing.assert_series_equal(result.clean_df["price"], result.dirty_df["price"])
