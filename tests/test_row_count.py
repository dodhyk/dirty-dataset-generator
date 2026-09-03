import pytest
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
@pytest.mark.parametrize("rows", [100, 1000, 10000])
def test_row_counts_clean_and_dirty(preset, rows):
    dirtiness = 15.0
    result = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=42)

    # 1. Clean dataset row count matches requested
    assert len(result.clean_df) == rows, f"Clean dataset row count mismatch for {preset} at {rows} rows"
    assert result.clean_df.shape[0] == rows

    # 2. Dirty dataset row count matches requested
    assert len(result.dirty_df) == rows, f"Dirty dataset row count mismatch for {preset} at {rows} rows"
    assert result.dirty_df.shape[0] == rows

    # 3. Report total rows matches requested
    assert result.report.total_rows == rows
