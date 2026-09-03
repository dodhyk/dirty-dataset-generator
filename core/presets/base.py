from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd

from core.models import ColumnSpec


class BasePreset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the preset."""
        pass

    @abstractmethod
    def get_schema(self) -> List[ColumnSpec]:
        """Return the column specifications for this preset."""
        pass

    def get_protected_columns(self) -> List[str]:
        """Return list of column names that are protected from corruption."""
        return [col.name for col in self.get_schema() if col.is_protected]

    def get_eligible_feature_columns(self) -> List[str]:
        """Return list of column names eligible for corruption."""
        return [col.name for col in self.get_schema() if not col.is_protected]

    @abstractmethod
    def generate_clean(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        """Generate clean synthetic dataset with realistic distributions and relationships."""
        pass
