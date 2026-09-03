import pytest
import pandas as pd
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_reproducibility_same_seed(preset):
    seed = 42
    rows = 200
    dirtiness = 25.0

    gen_a = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=seed)
    gen_b = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=seed)

    # 1. Clean datasets must be identical
    pd.testing.assert_frame_equal(gen_a.clean_df, gen_b.clean_df)

    # 2. Dirty datasets must be identical
    pd.testing.assert_frame_equal(gen_a.dirty_df, gen_b.dirty_df)

    # 3. Reports must be identical
    assert gen_a.report.to_dict() == gen_b.report.to_dict()


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_divergence_different_seeds(preset):
    rows = 200
    dirtiness = 25.0

    gen_a = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=111)
    gen_b = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=222)

    # Clean datasets should differ
    assert not gen_a.clean_df.equals(gen_b.clean_df)

    # Dirty datasets should differ
    assert not gen_a.dirty_df.equals(gen_b.dirty_df)


def test_unspecified_seed_records_seed_in_report():
    result = generate(preset="customer_churn", rows=100, dirtiness=10.0)
    assert result.report.seed is not None
    assert isinstance(result.report.seed, int)
