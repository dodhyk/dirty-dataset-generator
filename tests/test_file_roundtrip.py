import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_csv_and_json_file_roundtrip(preset):
    rows = 200
    dirtiness = 20.0
    seed = 42

    res = generate(preset=preset, rows=rows, dirtiness=dirtiness, seed=seed)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        clean_file = tmp_path / f"{preset}_clean.csv"
        dirty_file = tmp_path / f"{preset}_dirty.csv"
        report_file = tmp_path / "corruption_report.json"

        # 1. Write files
        res.clean_df.to_csv(clean_file, index=False)
        res.dirty_df.to_csv(dirty_file, index=False)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(res.report.to_dict(), f, indent=2)

        # 2. Verify files exist and have non-zero size
        assert clean_file.exists() and clean_file.stat().st_size > 0
        assert dirty_file.exists() and dirty_file.stat().st_size > 0
        assert report_file.exists() and report_file.stat().st_size > 0

        # 3. Read back clean CSV
        loaded_clean = pd.read_csv(clean_file)
        assert len(loaded_clean) == rows
        assert list(loaded_clean.columns) == list(res.clean_df.columns)
        assert loaded_clean.isna().sum().sum() == 0

        # 4. Read back dirty CSV
        loaded_dirty = pd.read_csv(dirty_file)
        assert len(loaded_dirty) == rows
        assert list(loaded_dirty.columns) == list(res.dirty_df.columns)

        # 5. Read back JSON corruption report
        with open(report_file, "r", encoding="utf-8") as f:
            loaded_report = json.load(f)

        assert loaded_report["preset"] == preset
        assert loaded_report["total_rows"] == rows
        assert loaded_report["dirty_rows"] == 40
        assert loaded_report["seed"] == seed
        assert loaded_report["total_corruptions"] == len(res.report.corruptions)
        assert len(loaded_report["corruptions"]) == len(res.report.corruptions)
