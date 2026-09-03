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


class CustomerChurnPreset(BasePreset):
    @property
    def name(self) -> str:
        return "customer_churn"

    def get_schema(self) -> List[ColumnSpec]:
        return [
            ColumnSpec("customer_id", DataType.STRING, ColumnRole.ID, is_protected=True),
            ColumnSpec("gender", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("age", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("tenure_months", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("monthly_charges", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("total_charges", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("contract_type", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("internet_service", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("payment_method", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("paperless_billing", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("senior_citizen", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("partner", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("dependents", DataType.BOOLEAN, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("churn", DataType.INTEGER, ColumnRole.TARGET, is_protected=True),
        ]

    def generate_clean(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        customer_ids = generate_unique_ids("CUST", rows, digits=6)
        genders = sample_weighted_categorical(rng, ["Male", "Female"], [0.5, 0.5], rows)
        senior_citizen = sample_bernoulli(rng, 0.16, rows)

        ages = np.zeros(rows, dtype=int)
        for i in range(rows):
            if senior_citizen[i]:
                ages[i] = sample_truncated_normal_int(rng, mean=72, std=5, low=65, high=88, size=1)[0]
            else:
                ages[i] = sample_truncated_normal_int(rng, mean=38, std=11, low=18, high=64, size=1)[0]

        partners = sample_bernoulli(rng, 0.48, rows)
        dependents = np.zeros(rows, dtype=bool)
        for i in range(rows):
            p_dep = 0.45 if partners[i] else 0.10
            dependents[i] = sample_bernoulli(rng, p_dep, 1)[0]

        tenure_months = sample_truncated_normal_int(rng, mean=32, std=24, low=1, high=72, size=rows)
        contract_types = sample_weighted_categorical(
            rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.20], rows
        )
        internet_services = sample_weighted_categorical(
            rng, ["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], rows
        )
        payment_methods = sample_weighted_categorical(
            rng,
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            [0.34, 0.23, 0.22, 0.21],
            rows,
        )
        paperless_billing = sample_bernoulli(rng, 0.59, rows)

        monthly_charges = np.zeros(rows, dtype=float)
        for i in range(rows):
            service = internet_services[i]
            if service == "No":
                mc = sample_truncated_normal_float(rng, mean=21.0, std=3.0, low=18.0, high=26.0, decimals=2, size=1)[0]
            elif service == "DSL":
                mc = sample_truncated_normal_float(rng, mean=55.0, std=10.0, low=30.0, high=85.0, decimals=2, size=1)[0]
            else:  # Fiber optic
                mc = sample_truncated_normal_float(rng, mean=88.0, std=12.0, low=65.0, high=120.0, decimals=2, size=1)[0]
            monthly_charges[i] = mc

        # total_charges has meaningful relationship: tenure * monthly_charges + small discount/charge noise
        noise_ratio = rng.normal(loc=1.0, scale=0.03, size=rows)
        noise_ratio = np.clip(noise_ratio, 0.90, 1.10)
        total_charges = np.round(tenure_months * monthly_charges * noise_ratio, 2)
        total_charges = np.maximum(total_charges, monthly_charges)

        # Realistic Churn Target (Classification)
        log_odds = -1.6
        log_odds += np.where(np.array(contract_types) == "Month-to-month", 1.25, 0.0)
        log_odds += np.where(np.array(contract_types) == "Two year", -1.1, 0.0)
        log_odds += np.where(np.array(internet_services) == "Fiber optic", 0.85, 0.0)
        log_odds += np.where(np.array(payment_methods) == "Electronic check", 0.55, 0.0)
        log_odds -= 0.038 * tenure_months
        log_odds += 0.015 * (monthly_charges - 65.0)
        log_odds += 0.35 * senior_citizen.astype(int)
        log_odds -= 0.25 * partners.astype(int)
        log_odds += rng.normal(loc=0.0, scale=0.25, size=rows)

        prob_churn = 1.0 / (1.0 + np.exp(-log_odds))
        churn = (rng.random(size=rows) < prob_churn).astype(int)

        df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "gender": genders,
                "age": ages,
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract_type": contract_types,
                "internet_service": internet_services,
                "payment_method": payment_methods,
                "paperless_billing": paperless_billing,
                "senior_citizen": senior_citizen,
                "partner": partners,
                "dependents": dependents,
                "churn": churn,
            }
        )
        return df
