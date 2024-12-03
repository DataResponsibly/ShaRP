import pytest
import pandas as pd
import numpy as np
from sharp.utils._utils import _optional_import
from sharp.visualization._visualization import ShaRPViz


@pytest.fixture
def mock_sharp():
    """
    Fixture to create a mock ShaRP object with dummy feature names.
    """

    class MockSharp:
        def __init__(self):
            self.feature_names_ = np.array(["Feature1", "Feature2", "Feature3"])
            self.qoi="rank"
            self.measure="shapley"

    return MockSharp()


def test_bar(mock_sharp):
    """
    Test the bar method of ShaRPViz.
    """
    sharpviz = ShaRPViz(mock_sharp)
    scores = [0.1, 0.5, 0.4]

    plt = _optional_import("matplotlib.pyplot")
    fig, ax = plt.subplots(1, 1)

    result_ax = sharpviz.bar(scores, ax=ax)
    assert result_ax is not None
    assert result_ax.get_ylabel() == "Contribution to QoI"
    assert result_ax.get_xlabel() == "Features"
    assert len(result_ax.patches) == len(scores)  # Number of bars matches scores


def test_waterfall(mock_sharp):
    """
    Test the waterfall method of ShaRPViz.
    """
    sharpviz = ShaRPViz(mock_sharp)
    contributions = [0.2, -0.1, 0.3]
    feature_values = ["A", "B", "C"]
    mean_target_value = 1.0

    result = sharpviz.waterfall(
        contributions=contributions,
        feature_values=feature_values,
        mean_target_value=mean_target_value,
    )
    assert result is not None


def test_box(mock_sharp):
    """
    Test the box method of ShaRPViz.
    """
    sharpviz = ShaRPViz(mock_sharp)
    X = pd.DataFrame(
        np.random.randn(100, 3), columns=["Feature1", "Feature2", "Feature3"]
    )
    y = np.random.choice([0, 1], size=100)
    contributions = pd.DataFrame(
        np.random.randn(100, 3), columns=["Feature1", "Feature2", "Feature3"]
    )

    plt = _optional_import("matplotlib.pyplot")
    fig, ax = plt.subplots(1, 1)

    result = sharpviz.box(X, y, contributions, ax=ax)
    assert result is not None


def test_bar_without_ax(mock_sharp):
    """
    Test the bar method without providing an ax.
    """
    sharpviz = ShaRPViz(mock_sharp)
    scores = [0.3, 0.5, 0.2]

    result_ax = sharpviz.bar(scores)
    assert result_ax is not None
    assert result_ax.get_ylabel() == "Contribution to QoI"
    assert result_ax.get_xlabel() == "Features"
