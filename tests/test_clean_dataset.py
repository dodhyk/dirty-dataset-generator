import pytest
import numpy as np
import pandas as pd


def test_customer_churn_clean_integrity(churn_gen_1k):
    df = churn_gen_1k.clean_df

    # Zero nulls
    assert df.isna().sum().sum() == 0

    # ID uniqueness
    assert df["customer_id"].nunique() == 1000

    # Reasonable domains
    assert (df["age"] >= 18).all() and (df["age"] <= 100).all()
    assert (df["tenure_months"] >= 1).all() and (df["tenure_months"] <= 72).all()
    assert (df["monthly_charges"] >= 18.0).all() and (df["monthly_charges"] <= 125.0).all()
    assert (df["total_charges"] >= df["monthly_charges"] - 1.0).all()

    # Allowed categoricals
    assert set(df["gender"].unique()).issubset({"Male", "Female"})
    assert set(df["contract_type"].unique()).issubset({"Month-to-month", "One year", "Two year"})
    assert set(df["internet_service"].unique()).issubset({"DSL", "Fiber optic", "No"})
    assert set(df["payment_method"].unique()).issubset(
        {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        }
    )

    # Boolean integrity
    for b_col in ["paperless_billing", "senior_citizen", "partner", "dependents"]:
        assert df[b_col].dtype == bool
        assert set(df[b_col].unique()).issubset({True, False})

    # Target
    assert set(df["churn"].unique()).issubset({0, 1})


def test_ecommerce_clean_integrity(ecom_gen_1k):
    df = ecom_gen_1k.clean_df

    # Zero nulls
    assert df.isna().sum().sum() == 0

    # ID uniqueness
    assert df["customer_id"].nunique() == 1000
    assert df["order_id"].nunique() == 1000

    # Numeric domains
    assert (df["age"] >= 18).all() and (df["age"] <= 80).all()
    assert (df["total_orders"] >= 1).all()
    assert (df["total_spent"] > 0).all()
    assert (df["avg_order_value"] > 0).all()
    assert (df["items_per_order"] >= 1.0).all()
    assert (df["days_since_last_order"] >= 0).all()
    assert (df["discount_usage"] >= 0).all()
    assert (df["discount_usage"] <= df["total_orders"]).all()

    # Allowed categoricals
    assert set(df["gender"].unique()).issubset({"Female", "Male", "Other"})
    assert set(df["device_type"].unique()).issubset({"Mobile App", "Desktop Web", "Mobile Web"})
    assert set(df["membership_type"].unique()).issubset({"Regular", "Silver", "Gold", "Platinum"})
    assert set(df["payment_method"].unique()).issubset(
        {"E-Wallet", "Credit Card", "Bank Transfer", "Cash on Delivery", "Buy Now Pay Later"}
    )

    # Target
    assert set(df["purchased_again"].unique()).issubset({0, 1})


def test_house_price_clean_integrity(house_gen_1k):
    df = house_gen_1k.clean_df

    # Zero nulls
    assert df.isna().sum().sum() == 0

    # ID uniqueness
    assert df["house_id"].nunique() == 1000

    # Numeric domains
    assert (df["area_sqft"] >= 450).all()
    assert (df["bedrooms"] >= 1).all() and (df["bedrooms"] <= 10).all()
    assert (df["bathrooms"] >= 1.0).all()
    assert (df["floors"] >= 1.0).all() and (df["floors"] <= 4.0).all()
    assert (df["year_built"] >= 1950).all() and (df["year_built"] <= 2026).all()
    assert (df["garage"] >= 0).all() and (df["garage"] <= 5).all()
    assert (df["distance_to_city_km"] >= 0.0).all()
    assert (df["property_tax"] > 0).all()
    assert (df["price"] > 35000.0).all()

    # Categoricals
    assert set(df["property_type"].unique()).issubset(
        {"Single Family", "Townhouse", "Apartment", "Villa"}
    )
    assert set(df["location"].unique()).issubset(
        {"Suburban", "Downtown", "Uptown", "Rural", "Waterfront"}
    )
    assert set(df["condition"].unique()).issubset(
        {"Poor", "Fair", "Good", "Excellent", "Luxury"}
    )

    # Booleans
    assert df["has_garden"].dtype == bool
    assert df["has_pool"].dtype == bool
