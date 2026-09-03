from typing import Any, Dict, List, Optional
import requests


class ApiError(Exception):
    """Custom exception representing API communication or validation errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DatasetApiClient:
    """Dedicated HTTP client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> bool:
        """Check if FastAPI backend is healthy and responding."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "ok"
            return False
        except requests.RequestException:
            return False

    def get_presets(self) -> List[str]:
        """Fetch available dataset presets from backend."""
        try:
            resp = requests.get(f"{self.base_url}/presets", timeout=self.timeout)
            if resp.status_code != 200:
                self._handle_error_response(resp)
            data = resp.json()
            return data.get("presets", [])
        except requests.RequestException as e:
            raise ApiError(f"Failed to connect to backend at {self.base_url}: {str(e)}")

    def generate_dataset(
        self,
        preset: str,
        rows: int,
        dirtiness: float,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trigger dataset generation on the backend."""
        payload: Dict[str, Any] = {
            "preset": preset,
            "rows": rows,
            "dirtiness": dirtiness,
        }
        if seed is not None:
            payload["seed"] = seed

        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self._handle_error_response(resp)
            return resp.json()
        except requests.RequestException as e:
            raise ApiError(f"Backend communication error during generation: {str(e)}")

    def download_file(self, relative_download_url: str) -> bytes:
        """Download a generated file (clean, dirty, or report) by URL path."""
        url = f"{self.base_url}{relative_download_url}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                self._handle_error_response(resp)
            return resp.content
        except requests.RequestException as e:
            raise ApiError(f"Error downloading file from backend: {str(e)}")

    def _handle_error_response(self, resp: requests.Response) -> None:
        try:
            err_data = resp.json()
            detail = err_data.get("detail")
            if isinstance(detail, list):
                # Pydantic validation error list
                messages = []
                for item in detail:
                    loc = " -> ".join(str(l) for l in item.get("loc", []))
                    msg = item.get("msg", "Invalid input")
                    messages.append(f"{loc}: {msg}")
                detail_str = "; ".join(messages)
            else:
                detail_str = str(detail or resp.text)
        except Exception:
            detail_str = resp.text or f"HTTP {resp.status_code}"

        raise ApiError(detail_str, status_code=resp.status_code)


# Alias for backward/convenience compatibility
ApiClient = DatasetApiClient

