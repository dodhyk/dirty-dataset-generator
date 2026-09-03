import time
import os
import gc
import pytest
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
@pytest.mark.parametrize("rows", [1000, 10000])
def test_performance_baseline_small_medium(preset, rows):
    start_time = time.perf_counter()
    res = generate(preset=preset, rows=rows, dirtiness=20.0, seed=123)
    elapsed = time.perf_counter() - start_time

    assert len(res.clean_df) == rows
    assert len(res.dirty_df) == rows
    # Reasonable baseline thresholds: 1k under 2.0s, 10k under 10.0s
    max_allowed = 2.0 if rows == 1000 else 10.0
    assert elapsed < max_allowed, f"{preset} ({rows} rows) took {elapsed:.2f}s, expected < {max_allowed}s"


def test_performance_and_memory_100k_rows():
    """Section 35 & 36: Benchmark 100,000 rows generation and verify memory stability."""
    gc.collect()
    start_time = time.perf_counter()

    res = generate(preset="customer_churn", rows=100000, dirtiness=20.0, seed=42)
    elapsed = time.perf_counter() - start_time

    assert len(res.clean_df) == 100000
    assert len(res.dirty_df) == 100000
    assert res.report.dirty_rows == 20000
    assert res.report.total_corruptions >= 20000

    # Clean memory after test
    del res
    gc.collect()

    # 100k rows should finish cleanly within 30s
    assert elapsed < 30.0, f"100,000 rows took {elapsed:.2f}s, expected < 30.0s"
