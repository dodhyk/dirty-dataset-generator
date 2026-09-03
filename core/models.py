from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional
import pandas as pd


class ColumnRole(str, Enum):
    ID = "id"
    FEATURE = "feature"
    TARGET = "target"
    TRANSACTION_ID = "transaction_id"
    PRODUCT_ID = "product_id"


class DataType(str, Enum):
    INTEGER = "integer"
    DECIMAL = "decimal"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    STRING = "string"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: DataType
    role: ColumnRole
    is_protected: bool = False


@dataclass
class GenerationConfig:
    preset: str
    rows: int
    dirtiness: float  # Percentage (0 - 100)
    seed: Optional[int] = None

    def __post_init__(self):
        if self.rows <= 0:
            raise ValueError(f"rows must be positive, got {self.rows}")
        if not (0.0 <= self.dirtiness <= 100.0):
            raise ValueError(f"dirtiness must be between 0.0 and 100.0, got {self.dirtiness}")


@dataclass
class CorruptionRecord:
    row: int
    column: str
    type: str
    original_value: Any
    corrupted_value: Any

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "type": self.type,
            "original_value": self.original_value,
            "corrupted_value": self.corrupted_value,
        }


@dataclass
class CorruptionReport:
    preset: str
    total_rows: int
    dirty_rows: int
    dirtiness: float
    seed: int
    total_corruptions: int
    corruptions: List[CorruptionRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "total_rows": self.total_rows,
            "dirty_rows": self.dirty_rows,
            "dirtiness": self.dirtiness,
            "seed": self.seed,
            "total_corruptions": self.total_corruptions,
            "corruptions": [c.to_dict() for c in self.corruptions],
        }


@dataclass
class GenerationResult:
    preset: str
    clean_df: pd.DataFrame
    dirty_df: pd.DataFrame
    report: CorruptionReport
