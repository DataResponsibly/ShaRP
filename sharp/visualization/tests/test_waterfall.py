import pytest
import pandas as pd
import numpy as np
from sharp.visualization._waterfall import format_value, _waterfall


@pytest.mark.parametrize(
    "value, format_str, expected",
    [
        (123.45000, "%.2f", "123.45"),
        (123.00000, "%.2f", "123"),
        (-123.45000, "%.2f", "\u2212123.45"),
        (-123.00000, "%.2f", "\u2212123"),
        (0.00000, "%.2f", "0"),
        ("123.45", "%.2f", "123.45"),
        ("-123.45", "%.2f", "\u2212123.45"),
    ],
)
def test_format_value(value, format_str, expected):
    assert format_value(value, format_str) == expected


@pytest.fixture
def shap_values():
    return {
        "base_values": 0.5,
        "features": np.array([1.0, 2.0, 3.0]),
        "feature_names": ["feature1", "feature2", "feature3"],
        "values": pd.Series([0.1, -0.2, 0.3]),
    }


def test_waterfall_plot(shap_values):
    fig = _waterfall(shap_values, max_display=3, show=False)
    assert fig is not None
    assert len(fig.axes) > 0


@pytest.mark.parametrize("max_display", [1, 2, 3])
def test_waterfall_max_display(shap_values, max_display):
    fig = _waterfall(shap_values, max_display=max_display, show=False)
    assert fig is not None
    assert (
        len(fig.axes[0].patches) == max_display * 2
    )  # Each feature has two patches (positive and negative)


def test_waterfall_no_features():
    shap_values = {
        "base_values": 0.5,
        "features": None,
        "feature_names": ["feature1", "feature2", "feature3"],
        "values": pd.Series([0.1, -0.2, 0.3]),
    }
    fig = _waterfall(shap_values, max_display=3, show=False)
    assert fig is not None
    assert len(fig.axes) > 0


def test_waterfall_no_values():
    shap_values = {
        "base_values": 0.5,
        "features": np.array([1.0, 2.0, 3.0]),
        "feature_names": ["feature1", "feature2", "feature3"],
        "values": pd.Series([]),
    }
    fig = _waterfall(shap_values, max_display=3, show=False)
    assert fig is not None
    assert len(fig.axes) > 0
