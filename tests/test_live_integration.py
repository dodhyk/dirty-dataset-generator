import json
import pytest
from ui.client import ApiClient


def test_live_api_integration():
    client = ApiClient(base_url="http://127.0.0.1:8000")
    assert client.check_health() is True

    presets = client.get_presets()
    assert len(presets) == 3
    assert set(presets) == {"customer_churn", "ecommerce", "house_price"}

    for preset in presets:
        rows = 150
        dirtiness = 20.0
        seed = 42

        res = client.generate_dataset(preset=preset, rows=rows, dirtiness=dirtiness, seed=seed)
        assert res["preset"] == preset
        assert res["rows"] == rows
        assert res["dirty_rows"] == 30
        assert res["seed"] == seed

        clean_bytes = client.download_file(res["clean_download_url"])
        dirty_bytes = client.download_file(res["dirty_download_url"])
        report_bytes = client.download_file(res["report_download_url"])

        assert len(clean_bytes) > 0
        assert len(dirty_bytes) > 0
        assert len(report_bytes) > 0

        # Parse CSV lines
        clean_lines = clean_bytes.decode("utf-8").strip().splitlines()
        dirty_lines = dirty_bytes.decode("utf-8").strip().splitlines()
        assert len(clean_lines) == rows + 1  # header + rows
        assert len(dirty_lines) == rows + 1

        # Parse JSON report
        report = json.loads(report_bytes.decode("utf-8"))
        assert report["preset"] == preset
        assert report["total_rows"] == rows
        assert report["dirty_rows"] == 30
        assert report["dirtiness"] == dirtiness
        assert report["seed"] == seed
        assert report["total_corruptions"] == len(report["corruptions"])
