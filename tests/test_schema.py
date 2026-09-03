import pytest
from core.models import ColumnRole, DataType
from core.presets import get_preset, get_available_presets


def test_available_presets():
    presets = get_available_presets()
    assert "customer_churn" in presets
    assert "ecommerce" in presets
    assert "house_price" in presets
    assert len(presets) == 3


def test_customer_churn_schema():
    preset = get_preset("customer_churn")
    schema = preset.get_schema()
    col_names = [c.name for c in schema]
    expected_cols = [
        "customer_id",
        "gender",
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "payment_method",
        "paperless_billing",
        "senior_citizen",
        "partner",
        "dependents",
        "churn",
    ]
    assert col_names == expected_cols

    # Protected columns verification
    protected = preset.get_protected_columns()
    assert "customer_id" in protected
    assert "churn" in protected
    assert len(protected) == 2

    # Column roles
    role_map = {c.name: c.role for c in schema}
    assert role_map["customer_id"] == ColumnRole.ID
    assert role_map["churn"] == ColumnRole.TARGET
    assert role_map["gender"] == ColumnRole.FEATURE


def test_ecommerce_schema():
    preset = get_preset("ecommerce")
    schema = preset.get_schema()
    col_names = [c.name for c in schema]
    expected_cols = [
        "customer_id",
        "order_id",
        "product_id",
        "product_name",
        "age",
        "gender",
        "location",
        "device_type",
        "membership_type",
        "total_orders",
        "total_spent",
        "avg_order_value",
        "items_per_order",
        "days_since_last_order",
        "discount_usage",
        "preferred_category",
        "payment_method",
        "timestamp",
        "shipping_address",
        "purchased_again",
    ]
    assert col_names == expected_cols

    protected = preset.get_protected_columns()
    assert "customer_id" in protected
    assert "purchased_again" in protected
    assert "order_id" not in protected
    assert len(protected) == 2


def test_house_price_schema():
    preset = get_preset("house_price")
    schema = preset.get_schema()
    col_names = [c.name for c in schema]
    expected_cols = [
        "house_id",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "floors",
        "year_built",
        "location",
        "property_type",
        "condition",
        "garage",
        "has_garden",
        "has_pool",
        "distance_to_city_km",
        "property_tax",
        "price",
    ]
    assert col_names == expected_cols

    protected = preset.get_protected_columns()
    assert "house_id" in protected
    assert "price" in protected
    assert len(protected) == 2
