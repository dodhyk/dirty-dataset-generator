import pytest
from core.presets import get_available_presets, get_preset
from core.presets.base import BasePreset
from core.models import ColumnRole, DataType
from core.generator import generate


def test_preset_discovery():
    presets = get_available_presets()
    assert isinstance(presets, list)
    assert len(presets) == 3
    assert set(presets) == {"customer_churn", "ecommerce", "house_price"}


@pytest.mark.parametrize("name", ["customer_churn", "ecommerce", "house_price"])
def test_preset_retrieval_and_type(name):
    preset = get_preset(name)
    assert isinstance(preset, BasePreset)
    assert preset.name == name


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("invalid_preset_xyz")


@pytest.mark.parametrize("name", ["customer_churn", "ecommerce", "house_price"])
def test_preset_schema_completeness(name):
    preset = get_preset(name)
    schema = preset.get_schema()
    assert len(schema) > 0

    col_names = [col.name for col in schema]
    # No duplicate column names
    assert len(col_names) == len(set(col_names))

    # Must have exactly 1 ID and 1 Target
    roles = [col.role for col in schema]
    assert roles.count(ColumnRole.ID) == 1
    assert roles.count(ColumnRole.TARGET) == 1

    # Protected columns must include ID and Target
    protected = preset.get_protected_columns()
    assert len(protected) >= 2

    id_cols = [c.name for c in schema if c.role == ColumnRole.ID]
    target_cols = [c.name for c in schema if c.role == ColumnRole.TARGET]
    assert id_cols[0] in protected
    assert target_cols[0] in protected


@pytest.mark.parametrize("name", ["customer_churn", "ecommerce", "house_price"])
def test_preset_exception_free_generation(name):
    # Generating with standard parameters must never throw exceptions
    result = generate(preset=name, rows=50, dirtiness=10.0, seed=123)
    assert result.clean_df is not None
    assert result.dirty_df is not None
    assert result.report is not None
    assert len(result.clean_df) == 50
    assert len(result.dirty_df) == 50
