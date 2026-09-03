import pytest
import pandas as pd
from core.generator import generate


@pytest.mark.parametrize(
    "preset,id_col,target_col,is_classification",
    [
        ("customer_churn", "customer_id", "churn", True),
        ("ecommerce", "customer_id", "purchased_again", True),
        ("house_price", "house_id", "price", False),
    ],
)
def test_id_uniqueness_and_non_null(preset, id_col, target_col, is_classification):
    rows = 500
    res = generate(preset=preset, rows=rows, dirtiness=25.0, seed=123)
    clean_df = res.clean_df
    dirty_df = res.dirty_df

    # ID validation on Clean
    assert clean_df[id_col].notna().all()
    assert clean_df[id_col].nunique() == rows
    assert len(clean_df[id_col]) == rows

    # ID validation on Dirty
    assert dirty_df[id_col].notna().all()
    assert dirty_df[id_col].nunique() == rows
    assert len(dirty_df[id_col]) == rows


@pytest.mark.parametrize(
    "preset,id_col,target_col,is_classification",
    [
        ("customer_churn", "customer_id", "churn", True),
        ("ecommerce", "customer_id", "purchased_again", True),
        ("house_price", "house_id", "price", False),
    ],
)
def test_target_validity(preset, id_col, target_col, is_classification):
    rows = 500
    res = generate(preset=preset, rows=rows, dirtiness=20.0, seed=123)
    clean_df = res.clean_df

    # Target cannot be null
    assert clean_df[target_col].notna().all()

    if is_classification:
        # Binary target in {0, 1}
        vals = set(clean_df[target_col].unique())
        assert vals.issubset({0, 1})
        # Non-pathological: neither all 0s nor all 1s
        p_mean = clean_df[target_col].mean()
        assert 0.05 < p_mean < 0.95, f"Pathological target distribution for {preset}: mean={p_mean}"
    else:
        # Regression: continuous positive float
        assert pd.api.types.is_numeric_dtype(clean_df[target_col])
        assert (clean_df[target_col] > 0).all()


@pytest.mark.parametrize(
    "preset,id_col,target_col",
    [
        ("customer_churn", "customer_id", "churn"),
        ("ecommerce", "customer_id", "purchased_again"),
        ("house_price", "house_id", "price"),
    ],
)
@pytest.mark.parametrize("dirtiness", [10.0, 50.0, 100.0])
def test_protected_fields_never_corrupted(preset, id_col, target_col, dirtiness):
    """Mandatory Test: Protected fields (ID and Target) MUST remain bitwise identical across all rows."""
    rows = 300
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=777)
    clean_df = res.clean_df
    dirty_df = res.dirty_df

    # 1. Clean ID == Dirty ID
    pd.testing.assert_series_equal(
        clean_df[id_col],
        dirty_df[id_col],
        check_names=True,
        check_dtype=True,
        obj=f"{preset} ID column",
    )

    # 2. Clean Target == Dirty Target
    pd.testing.assert_series_equal(
        clean_df[target_col],
        dirty_df[target_col],
        check_names=True,
        check_dtype=True,
        obj=f"{preset} Target column",
    )

    # 3. Neither ID nor Target is ever listed in the corruption report
    corrupted_columns = {c.column for c in res.report.corruptions}
    assert id_col not in corrupted_columns, f"ID column {id_col} found in corruption report!"
    assert target_col not in corrupted_columns, f"Target column {target_col} found in corruption report!"
