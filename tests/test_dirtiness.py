import pytest
import pandas as pd
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
@pytest.mark.parametrize(
    "dirtiness,expected_dirty_rows",
    [
        (0.0, 0),
        (5.0, 50),
        (10.0, 100),
        (20.0, 200),
        (50.0, 500),
        (100.0, 1000),
    ],
)
def test_dirtiness_row_percentage(preset, dirtiness, expected_dirty_rows):
    rows = 1000
    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=12345)

    assert res.report.dirtiness == dirtiness
    assert res.report.total_rows == rows
    assert res.report.dirty_rows == expected_dirty_rows

    if dirtiness == 0.0:
        # Zero dirtiness: dirty dataset == clean dataset exactly
        pd.testing.assert_frame_equal(res.clean_df, res.dirty_df)
        assert res.report.total_corruptions == 0
        assert len(res.report.corruptions) == 0
    elif dirtiness == 100.0:
        # Full dirtiness: exactly 1000 dirty rows
        assert res.report.dirty_rows == rows
        assert res.report.total_corruptions >= rows
        # ID and Target remain 100% protected
        id_col = "customer_id" if "customer_id" in res.clean_df.columns else "house_id"
        target_col = "price" if preset == "house_price" else ("churn" if preset == "customer_churn" else "purchased_again")
        pd.testing.assert_series_equal(res.clean_df[id_col], res.dirty_df[id_col])
        pd.testing.assert_series_equal(res.clean_df[target_col], res.dirty_df[target_col])
