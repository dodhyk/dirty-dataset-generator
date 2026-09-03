import pytest
import numpy as np


def test_customer_churn_distributions(churn_gen_1k):
    df = churn_gen_1k.clean_df

    # Gender ratio: expected ~50%, tolerance 40% - 60%
    male_ratio = (df["gender"] == "Male").mean()
    assert 0.40 <= male_ratio <= 0.60, f"Male ratio out of bounds: {male_ratio:.3f}"

    # Senior citizen: expected ~16%, tolerance 8% - 25%
    senior_ratio = df["senior_citizen"].mean()
    assert 0.08 <= senior_ratio <= 0.25, f"Senior ratio out of bounds: {senior_ratio:.3f}"

    # Partner: expected ~48%, tolerance 38% - 58%
    partner_ratio = df["partner"].mean()
    assert 0.38 <= partner_ratio <= 0.58, f"Partner ratio out of bounds: {partner_ratio:.3f}"

    # Contract types: month-to-month should be largest share (>35%)
    m2m_ratio = (df["contract_type"] == "Month-to-month").mean()
    assert 0.35 <= m2m_ratio <= 0.70, f"Month-to-month contract ratio out of bounds: {m2m_ratio:.3f}"

    # Numeric distributions: mean age ~38-48
    assert 30 <= df["age"].mean() <= 55
    # Mean tenure ~25-40
    assert 20 <= df["tenure_months"].mean() <= 45
    # Monthly charges mean ~50-80
    assert 45 <= df["monthly_charges"].mean() <= 85


def test_ecommerce_distributions(ecom_gen_1k):
    df = ecom_gen_1k.clean_df

    # Membership: Regular should be largest segment (expected ~50%, tolerance 35% - 65%)
    reg_ratio = (df["membership_type"] == "Regular").mean()
    assert 0.35 <= reg_ratio <= 0.65, f"Regular membership ratio out of bounds: {reg_ratio:.3f}"

    # Device type: Mobile App should be the leading device (>45%)
    app_ratio = (df["device_type"] == "Mobile App").mean()
    assert 0.45 <= app_ratio <= 0.80, f"Mobile App device ratio out of bounds: {app_ratio:.3f}"

    # Age: mean ~30-40
    assert 26 <= df["age"].mean() <= 42

    # Orders: mean orders ~5-12
    assert 4 <= df["total_orders"].mean() <= 15


def test_house_price_distributions(house_gen_1k):
    df = house_gen_1k.clean_df

    # Property type: Single family should be dominant (expected ~50%, tolerance 35% - 65%)
    sf_ratio = (df["property_type"] == "Single Family").mean()
    assert 0.35 <= sf_ratio <= 0.65, f"Single Family ratio out of bounds: {sf_ratio:.3f}"

    # Has pool: expected ~20%, tolerance 10% - 30%
    pool_ratio = df["has_pool"].mean()
    assert 0.10 <= pool_ratio <= 0.32, f"Pool ratio out of bounds: {pool_ratio:.3f}"

    # Has garden: expected ~62%, tolerance 50% - 75%
    garden_ratio = df["has_garden"].mean()
    assert 0.48 <= garden_ratio <= 0.75, f"Garden ratio out of bounds: {garden_ratio:.3f}"

    # Area sqft: mean ~1500 - 2200
    assert 1400 <= df["area_sqft"].mean() <= 2300

    # Price mean: expected ~300k - 700k
    assert 250000 <= df["price"].mean() <= 750000
