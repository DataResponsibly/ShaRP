import pandas as pd
import pytest
import numpy as np
from sharp.metrics import kendall_agreement, jaccard_agreement
from sklearn.utils import check_random_state

RNG_SEED = 123
rng = check_random_state(RNG_SEED)

right = [
    pd.DataFrame([[1, 2] for _ in range(2)]),
    pd.DataFrame([[1, 2, 3] for _ in range(6)]),
]
left = [
    pd.DataFrame([[1, 2], [2, 1]]),
    pd.DataFrame([[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 1, 2], [2, 3, 1], [3, 2, 1]]),
]
kendall_results = [[1, 0], [1, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0]]


@pytest.mark.parametrize("a, b, result", zip(right, left, kendall_results))
def test_kendall_tau(a, b, result):
    np.testing.assert_array_almost_equal(kendall_agreement(a, b), result, decimal=10)


@pytest.mark.parametrize("a, b", zip(right, left))
def test_jaccard(a, b):
    np.testing.assert_allclose(jaccard_agreement(a, b, n_features=None), 1)
