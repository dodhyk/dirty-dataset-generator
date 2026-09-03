import pytest
import numpy as np
import pandas as pd


def test_customer_churn_feature_target_relationships(churn_gen_1k):
    df = churn_gen_1k.clean_df

    # 1. Total charges relationship: total_charges ~ tenure_months * monthly_charges
    calculated_expected = df["tenure_months"] * df["monthly_charges"]
    ratio = df["total_charges"] / calculated_expected
    # Ratio should be tightly clustered around 1.0 (mean within [0.95, 1.05])
    assert 0.95 <= ratio.mean() <= 1.05

    # 2. Contract type relationship with Churn
    # Month-to-month contracts should have significantly higher churn than Two year contracts
    churn_by_contract = df.groupby("contract_type")["churn"].mean()
    m2m_churn = churn_by_contract.get("Month-to-month", 0.0)
    two_yr_churn = churn_by_contract.get("Two year", 0.0)
    assert m2m_churn > two_yr_churn + 0.15, (
        f"Expected Month-to-month churn ({m2m_churn:.2f}) > Two year churn ({two_yr_churn:.2f}) + 0.15"
    )

    # 3. Tenure relationship with Churn: longer tenure customers churn less
    low_tenure_churn = df[df["tenure_months"] <= 12]["churn"].mean()
    high_tenure_churn = df[df["tenure_months"] >= 48]["churn"].mean()
    assert low_tenure_churn > high_tenure_churn + 0.10, (
        f"Expected low tenure churn ({low_tenure_churn:.2f}) > high tenure churn ({high_tenure_churn:.2f})"
    )


def test_ecommerce_feature_target_relationships(ecom_gen_1k):
    df = ecom_gen_1k.clean_df

    # 1. total_spent is derived from total_orders * avg_order_value
    expected_spent = df["total_orders"] * df["avg_order_value"]
    diff = np.abs(df["total_spent"] - expected_spent)
    assert (diff < 1.0).all()

    # 2. Membership relationship with purchased_again
    purchase_by_mem = df.groupby("membership_type")["purchased_again"].mean()
    plat_purchase = purchase_by_mem.get("Platinum", 0.0)
    reg_purchase = purchase_by_mem.get("Regular", 0.0)
    assert plat_purchase > reg_purchase, (
        f"Platinum repurchase ({plat_purchase:.2f}) should exceed Regular ({reg_purchase:.2f})"
    )

    # 3. days_since_last_order relationship with purchased_again
    recent_buyers_prob = df[df["days_since_last_order"] <= 30]["purchased_again"].mean()
    distant_buyers_prob = df[df["days_since_last_order"] >= 90]["purchased_again"].mean()
    assert recent_buyers_prob > distant_buyers_prob, (
        f"Recent buyers repurchase ({recent_buyers_prob:.2f}) should exceed distant ({distant_buyers_prob:.2f})"
    )


def test_house_price_feature_target_relationships(house_gen_1k):
    df = house_gen_1k.clean_df

    # 1. Area correlation with Price: large positive correlation
    corr_area_price = df["area_sqft"].corr(df["price"])
    assert corr_area_price > 0.60, f"Area-price correlation ({corr_area_price:.2f}) should be > 0.60"

    # 2. Bedrooms correlation with Area
    corr_bed_area = df["bedrooms"].corr(df["area_sqft"])
    assert corr_bed_area > 0.40, f"Bedrooms-area correlation ({corr_bed_area:.2f}) should be > 0.40"

    # 3. Location premium: Waterfront / Downtown median price > Rural
    loc_medians = df.groupby("location")["price"].median()
    waterfront_med = loc_medians.get("Waterfront", 0.0)
    rural_med = loc_medians.get("Rural", 0.0)
    assert waterfront_med > rural_med * 1.25, (
        f"Waterfront median ({waterfront_med}) should exceed Rural ({rural_med}) by > 25%"
    )

    # 4. Property tax tightly coupled with price
    corr_tax_price = df["property_tax"].corr(df["price"])
    assert corr_tax_price > 0.95, f"Property tax-price correlation ({corr_tax_price:.2f}) should be > 0.95"
