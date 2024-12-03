import pytest
import numpy as np
import pandas as pd
from sharp.visualization._aggregate import group_boxplot


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "group": np.random.choice(["A", "B", "C", "D", "E"], 100),
        }
    )
    y = np.random.rand(100)
    contributions = np.random.rand(100, 2)
    feature_names = ["feature1", "feature2"]
    return X, y, contributions, feature_names


def test_group_boxplot_group_by_bins(sample_data):
    X, y, contributions, feature_names = sample_data
    ax = group_boxplot(
        X, y, contributions, feature_names=feature_names, group=5, show=False
    )
    assert ax is not None
    assert len(ax.get_xticklabels()) == 5


def test_group_boxplot_group_by_variable(sample_data):
    X, y, contributions, feature_names = sample_data
    ax = group_boxplot(
        X, y, contributions, feature_names=feature_names, group="group", show=False
    )
    assert ax is not None
    assert len(ax.get_xticklabels()) == len(X["group"].unique())


def test_group_boxplot_show(sample_data):
    X, y, contributions, feature_names = sample_data
    ax = group_boxplot(
        X, y, contributions, feature_names=feature_names, group=5, show=True
    )
    assert ax is None
