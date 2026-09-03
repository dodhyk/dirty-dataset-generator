from typing import List
import numpy as np
import pandas as pd

from core.models import ColumnRole, ColumnSpec, DataType
from core.presets.base import BasePreset
from core.distributions import (
    generate_unique_ids,
    sample_truncated_normal_int,
    sample_truncated_normal_float,
    sample_weighted_categorical,
    sample_bernoulli,
)


class HousePricePreset(BasePreset):
    @property
    def name(self) -> str:
        return "house_price"

    def get_schema(self) -> List[ColumnSpec]:
        return [
            ColumnSpec("house_id", DataType.STRING, ColumnRole.ID, is_protected=True),
            ColumnSpec("area_sqft", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("bedrooms", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("bathrooms", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("floors", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("year_built", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("location", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("property_type", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("condition", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("garage", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("has_garden", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("has_pool", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("distance_to_city_km", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("property_tax", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("price", DataType.DECIMAL, ColumnRole.TARGET, is_protected=True),
        ]

    def generate_clean(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        house_ids = generate_unique_ids("HOUSE", rows, digits=6)

        property_types = sample_weighted_categorical(
            rng, ["Single Family", "Townhouse", "Apartment", "Villa"], [0.50, 0.25, 0.15, 0.10], rows
        )
        locations = sample_weighted_categorical(
            rng, ["Suburban", "Downtown", "Uptown", "Rural", "Waterfront"], [0.40, 0.25, 0.18, 0.10, 0.07], rows
        )
        conditions = sample_weighted_categorical(
            rng, ["Poor", "Fair", "Good", "Excellent", "Luxury"], [0.08, 0.22, 0.45, 0.18, 0.07], rows
        )

        area_sqft = sample_truncated_normal_int(rng, mean=1850, std=680, low=500, high=6000, size=rows)

        # Bedrooms roughly correlated with area
        bedrooms = np.clip(np.round(area_sqft / 550.0 + rng.normal(0, 0.6, size=rows)), 1, 7).astype(int)

        # Bathrooms
        raw_baths = np.clip(np.round((bedrooms * 0.75 + rng.normal(0, 0.4, size=rows)) * 2) / 2, 1.0, 5.0)
        bathrooms = np.round(raw_baths, 1)

        # Floors
        floors = sample_weighted_categorical(rng, [1.0, 1.5, 2.0, 2.5, 3.0], [0.35, 0.15, 0.35, 0.08, 0.07], rows)
        floors = np.array(floors, dtype=float)

        year_built = sample_truncated_normal_int(rng, mean=2003, std=17, low=1960, high=2025, size=rows)
        garage = sample_weighted_categorical(rng, [0, 1, 2, 3, 4], [0.15, 0.35, 0.38, 0.09, 0.03], rows)
        garage = np.array(garage, dtype=int)

        has_garden = sample_bernoulli(rng, 0.62, rows)
        has_pool = sample_bernoulli(rng, 0.20, rows)
        distance_to_city_km = sample_truncated_normal_float(rng, mean=14.0, std=8.0, low=0.5, high=45.0, decimals=1, size=rows)

        # Continuous Regression Price Calculation
        loc_val = {
            "Downtown": 95000.0,
            "Waterfront": 160000.0,
            "Uptown": 45000.0,
            "Suburban": 0.0,
            "Rural": -38000.0,
        }
        cond_mult = {
            "Poor": 0.76,
            "Fair": 0.90,
            "Good": 1.00,
            "Excellent": 1.16,
            "Luxury": 1.36,
        }

        price = np.zeros(rows, dtype=float)
        for i in range(rows):
            base = 90000.0
            base += area_sqft[i] * 145.0
            base += bedrooms[i] * 14000.0
            base += bathrooms[i] * 16000.0
            base += loc_val[locations[i]]
            base *= cond_mult[conditions[i]]
            base += garage[i] * 9000.0
            if has_garden[i]:
                base += 8500.0
            if has_pool[i]:
                base += 28000.0
            base -= distance_to_city_km[i] * 2100.0
            base += (year_built[i] - 1980) * 850.0
            # Realistic additive market noise
            noise = rng.normal(0, 12000.0)
            final_p = max(base + noise, 50000.0)
            price[i] = round(final_p, 2)

        # Property tax continuous decimal: roughly 1.1% to 1.3% of house price
        tax_rate = rng.uniform(0.011, 0.013, size=rows)
        property_tax = np.round(price * tax_rate, 2)

        return pd.DataFrame(
            {
                "house_id": house_ids,
                "area_sqft": area_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "floors": floors,
                "year_built": year_built,
                "location": locations,
                "property_type": property_types,
                "condition": conditions,
                "garage": garage,
                "has_garden": has_garden,
                "has_pool": has_pool,
                "distance_to_city_km": distance_to_city_km,
                "property_tax": property_tax,
                "price": price,
            }
        )
