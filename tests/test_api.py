import json
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_presets_endpoint():
    response = client.get("/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert "customer_churn" in data["presets"]
    assert "ecommerce" in data["presets"]
    assert "house_price" in data["presets"]


def test_generate_success():
    payload = {
        "preset": "customer_churn",
        "rows": 100,
        "dirtiness": 15.0,
        "seed": 42,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generation_id" in data
    assert data["preset"] == "customer_churn"
    assert data["rows"] == 100
    assert data["dirtiness"] == 15.0
    assert data["dirty_rows"] == 15
    assert data["seed"] == 42
    assert "clean_download_url" in data
    assert "dirty_download_url" in data
    assert "report_download_url" in data

    gen_id = data["generation_id"]

    # Test clean download
    clean_resp = client.get(f"/download/{gen_id}/clean")
    assert clean_resp.status_code == 200
    assert clean_resp.headers["content-type"].startswith("text/csv")
    assert b"customer_id" in clean_resp.content

    # Test dirty download
    dirty_resp = client.get(f"/download/{gen_id}/dirty")
    assert dirty_resp.status_code == 200
    assert dirty_resp.headers["content-type"].startswith("text/csv")
    assert b"customer_id" in dirty_resp.content

    # Test report download
    report_resp = client.get(f"/download/{gen_id}/report")
    assert report_resp.status_code == 200
    report_json = report_resp.json()
    assert report_json["preset"] == "customer_churn"
    assert report_json["total_rows"] == 100
    assert report_json["dirty_rows"] == 15


def test_generate_invalid_preset():
    payload = {
        "preset": "non_existent_preset",
        "rows": 100,
        "dirtiness": 10.0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "Unknown preset" in response.json()["detail"]


def test_generate_invalid_rows():
    payload = {
        "preset": "customer_churn",
        "rows": 0,
        "dirtiness": 10.0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


def test_generate_negative_rows():
    payload = {
        "preset": "customer_churn",
        "rows": -50,
        "dirtiness": 10.0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


def test_generate_invalid_dirtiness():
    payload = {
        "preset": "customer_churn",
        "rows": 100,
        "dirtiness": 150.0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 422


def test_download_invalid_generation_id_format():
    response = client.get("/download/invalid-uuid-123/clean")
    assert response.status_code == 400


def test_download_nonexistent_generation_id():
    response = client.get("/download/11111111-2222-3333-4444-555555555555/clean")
    assert response.status_code == 404
