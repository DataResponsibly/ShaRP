import pytest
import numpy as np
from sklearn.utils import check_random_state
from sharp.qoi import TopKQoI, FlipQoI
from sharp.utils import scores_to_ordering

# Set up some envrionment variables
RNG_SEED = 123
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)
rank = scores_to_ordering(y)


@pytest.mark.parametrize("top_k", range(0, X.shape[0], 5))
def test_top_k(top_k):
    qoi = TopKQoI(target_function=score_function, top_k=top_k, X=X)
    np.testing.assert_allclose(qoi.estimate(X), (rank <= top_k).astype(int))


@pytest.mark.parametrize("item_idx", range(0, X.shape[0], 5))
def test_top_k_vs_flip(item_idx):
    top_k_qoi = TopKQoI(target_function=score_function, top_k=rank[item_idx], X=X)
    flip_qoi = FlipQoI(
        target_function=lambda row: score_function(row) >= y[item_idx], X=X
    )
    np.testing.assert_array_equal(top_k_qoi.estimate(X), flip_qoi.estimate(X))
