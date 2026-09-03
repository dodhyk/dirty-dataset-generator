from datetime import datetime
import pytest
import numpy as np
import pandas as pd

from core.models import DataType
from core.presets import get_preset


@pytest.mark.parametrize(
    "preset_fixture_name,preset_name",
    [
        ("churn_gen_1k", "customer_churn"),
        ("ecom_gen_1k", "ecommerce"),
        ("house_gen_1k", "house_price"),
    ],
)
def test_clean_data_types(request, preset_fixture_name, preset_name):
    gen_result = request.getfixturevalue(preset_fixture_name)
    df = gen_result.clean_df
    preset = get_preset(preset_name)
    schema = preset.get_schema()

    for col in schema:
        series = df[col.name]

        if col.data_type == DataType.INTEGER:
            # Must be integer compatible
            assert np.issubdtype(series.dtype, np.integer), f"Column {col.name} should be integer dtype, got {series.dtype}"
            # Check values are exact integers
            for val in series:
                assert int(val) == val

        elif col.data_type == DataType.DECIMAL:
            # Must be floating-point numeric
            assert np.issubdtype(series.dtype, np.floating), f"Column {col.name} should be float dtype, got {series.dtype}"
            assert not np.isinf(series).any(), f"Column {col.name} contains infinite values"
            assert not series.isna().any(), f"Column {col.name} contains NaN"

        elif col.data_type == DataType.BOOLEAN:
            # Must be boolean type with strictly True/False values
            assert series.dtype == bool or np.issubdtype(series.dtype, np.bool_), (
                f"Column {col.name} should be bool dtype, got {series.dtype}"
            )
            unique_vals = set(series.unique())
            assert unique_vals.issubset({True, False})

        elif col.data_type == DataType.CATEGORICAL:
            # Must be string / categorical with valid strings
            assert (
                series.dtype == object
                or isinstance(series.dtype, (pd.CategoricalDtype, pd.StringDtype))
                or pd.api.types.is_string_dtype(series)
            )
            assert not series.isna().any()
            for val in series:
                assert isinstance(val, str)
                assert len(val.strip()) > 0

        elif col.data_type == DataType.STRING:
            assert series.dtype == object or isinstance(series.dtype, pd.StringDtype) or pd.api.types.is_string_dtype(series)
            for val in series:
                assert isinstance(val, str)
                assert len(val.strip()) > 0

        elif col.data_type == DataType.TIMESTAMP:
            # Valid parseable ISO 8601 timestamp string
            assert series.dtype == object or isinstance(series.dtype, pd.StringDtype) or pd.api.types.is_string_dtype(series)
            for val in series:
                assert isinstance(val, str)
                parsed = datetime.fromisoformat(val)
                assert parsed is not None
                assert 2020 <= parsed.year <= 2030
