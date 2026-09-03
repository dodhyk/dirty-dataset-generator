from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    preset: str = Field(..., description="Preset dataset identifier")
    rows: int = Field(..., gt=0, le=500000, description="Number of rows to generate (1 to 500,000)")
    dirtiness: float = Field(..., ge=0.0, le=100.0, description="Percentage of dirty rows (0.0 to 100.0)")
    seed: Optional[int] = Field(None, ge=0, description="Optional random seed for reproducibility")

    @field_validator("preset")
    @classmethod
    def validate_preset_name(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Preset cannot be empty")
        return cleaned


class GenerateResponse(BaseModel):
    generation_id: str
    preset: str
    rows: int
    dirtiness: float
    dirty_rows: int
    seed: int
    total_corruptions: int
    clean_download_url: str
    dirty_download_url: str
    report_download_url: str


class PresetsResponse(BaseModel):
    presets: List[str]


class HealthResponse(BaseModel):
    status: str
