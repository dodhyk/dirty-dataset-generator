import ast
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_path_traversal_attempts_blocked():
    """Section 34: Test that path traversal strings in download endpoints are rejected."""
    traversal_payloads = [
        "../etc/passwd",
        "..\\..\\windows\\system32",
        "....//....//generated",
        "%2e%2e%2f",
        "c:/windows/system32/cmd.exe",
        "foo/bar",
    ]

    for payload in traversal_payloads:
        for file_type in ["clean", "dirty", "report"]:
            resp = client.get(f"/download/{payload}/{file_type}")
            # Must reject with 400 (bad format) or 404/403, never 200 or 500
            assert resp.status_code in (400, 403, 404), (
                f"Payload '{payload}' yielded unexpected status {resp.status_code}"
            )


def test_arbitrary_file_access_blocked():
    """Ensure accessing non-existent generation IDs returns 404 without exposing internal paths."""
    resp = client.get("/download/00000000-0000-0000-0000-000000000000/clean")
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    # Ensure raw server filesystem path is not exposed
    assert "C:\\" not in data["detail"]
    assert "Users\\" not in data["detail"]


def test_malformed_request_body_handling():
    """Ensure malformed json or unexpected fields do not crash server."""
    resp = client.post("/generate", json={"preset": "customer_churn", "rows": "not_an_int"})
    assert resp.status_code == 422

    resp = client.post("/generate", content="invalid raw json string")
    assert resp.status_code == 422


def test_streamlit_architectural_separation():
    """Section 33: Ensure Streamlit UI layer strictly NEVER imports 'core' or 'generator'."""
    project_root = Path(__file__).resolve().parent.parent
    ui_files = [
        project_root / "ui" / "app.py",
        project_root / "ui" / "client.py",
    ]

    for ui_file in ui_files:
        assert ui_file.exists(), f"UI file {ui_file} not found"
        with open(ui_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(ui_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("core"), (
                        f"Architectural Violation: {ui_file.name} directly imports '{alias.name}'"
                    )
                    assert alias.name != "generator", (
                        f"Architectural Violation: {ui_file.name} directly imports 'generator'"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("core"), (
                    f"Architectural Violation: {ui_file.name} imports from '{module}'"
                )
                assert module != "generator", (
                    f"Architectural Violation: {ui_file.name} imports from 'generator'"
                )
