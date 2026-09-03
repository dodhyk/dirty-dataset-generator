import pytest
import numpy as np
from core.generator import generate


@pytest.mark.parametrize("preset", ["customer_churn", "ecommerce", "house_price"])
def test_clean_generation_row_counts_and_uniqueness(preset):
    rows = 250
    result = generate(preset=preset, rows=rows, dirtiness=0.0, seed=42)
    clean_df = result.clean_df

    # 1. Correct row count
    assert len(clean_df) == rows

    # 2. Unique ID
    id_col = "customer_id" if "customer_id" in clean_df.columns else "house_id"
    assert clean_df[id_col].nunique() == rows

    # 3. No NaN in clean data
    assert clean_df.isna().sum().sum() == 0


def test_customer_churn_clean_semantics():
    result = generate(preset="customer_churn", rows=500, dirtiness=0.0, seed=123)
    df = result.clean_df

    # Check target values are binary (0 or 1)
    assert set(df["churn"].unique()).issubset({0, 1})

    # Age domain
    assert df["age"].min() >= 18
    assert df["age"].max() <= 100

    # Tenure domain
    assert df["tenure_months"].min() >= 1
    assert df["tenure_months"].max() <= 72

    # Relationship: total_charges >= monthly_charges
    assert (df["total_charges"] >= df["monthly_charges"] - 1.0).all()

    # Categorical domain
    assert set(df["contract_type"].unique()).issubset({"Month-to-month", "One year", "Two year"})
    assert set(df["internet_service"].unique()).issubset({"DSL", "Fiber optic", "No"})


def test_ecommerce_clean_semantics():
    result = generate(preset="ecommerce", rows=500, dirtiness=0.0, seed=123)
    df = result.clean_df

    # Target is binary
    assert set(df["purchased_again"].unique()).issubset({0, 1})

    # Numeric domains
    assert (df["total_orders"] >= 1).all()
    assert (df["total_spent"] > 0).all()
    assert (df["avg_order_value"] > 0).all()
    assert (df["discount_usage"] <= df["total_orders"]).all()
    assert (df["days_since_last_order"] >= 0).all()


def test_house_price_clean_semantics():
    result = generate(preset="house_price", rows=500, dirtiness=0.0, seed=123)
    df = result.clean_df

    # Target continuous decimal > 0
    assert (df["price"] > 30000).all()

    # Features domains
    assert (df["area_sqft"] >= 400).all()
    assert (df["bedrooms"] >= 1).all()
    assert (df["bathrooms"] >= 1.0).all()
    assert (df["property_tax"] > 0).all()

    # Correlation check: larger area should positively correlate with price
    corr = df["area_sqft"].corr(df["price"])
    assert corr > 0.5
