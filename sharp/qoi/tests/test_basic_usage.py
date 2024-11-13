import pytest
from itertools import product
import numpy as np
from sklearn.utils import check_random_state
from sharp.qoi import get_qoi
from sharp.utils import scores_to_ordering

# Set up some envrionment variables
RNG_SEED = 123
N_SAMPLES = 20
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)
rank = scores_to_ordering(y)

qois_str = ["rank", "rank_score", "top_k", "diff", "flip", "likelihood"]
qois_obj = [get_qoi(qoi) for qoi in qois_str]

pairs1 = rng.randint(0, X.shape[0], size=3)
pairs2 = rng.randint(0, X.shape[0], size=3)

# @pytest.mark.parametrize("qoi_str, item_idx", product(qois_str, range(X.shape[0])))
# def test_qoi_estimation(qoi_str, item_idx):
#     item = X[item_idx:item_idx + 1]
#     qoi = get_qoi(qoi_str)(target_function=score_function, X=X)
#     if qoi_str == "rank":
#         assert qoi.estimate(rows=item).item() == rank[item_idx]
#     elif qoi_str == "top_k":
#         assert qoi.estimate(rows=item).item() == (rank[item_idx] <= 10).astype(int)
#     else:
#         np.testing.assert_allclose(qoi.estimate(rows=item), y[item_idx], atol=1e-7)


@pytest.mark.parametrize("qoi_str", qois_str)
def test_qoi_estimation(qoi_str):
    qoi = get_qoi(qoi_str)(target_function=score_function, X=X)
    if qoi_str == "rank":
        assert (qoi.estimate(rows=X) == rank).all()
    elif qoi_str == "top_k":
        assert (qoi.estimate(rows=X) == (rank <= 10).astype(int)).all()
    elif qoi_str == "likelihood":
        assert qoi.estimate(rows=X).item() == y.mean()
    else:
        np.testing.assert_allclose(qoi.estimate(rows=X), y, atol=1e-7)


@pytest.mark.parametrize(
    "qoi_str, item_idx1, item_idx2",
    product(qois_str, pairs1, pairs2),
)
def test_qoi_calculation(qoi_str, item_idx1, item_idx2):
    item1 = X[item_idx1 : item_idx1 + 1]
    item2 = X[item_idx2 : item_idx2 + 1]
    qoi = get_qoi(qoi_str)(target_function=score_function, X=X)
    if qoi_str == "rank":
        assert (
            qoi.calculate(rows1=item1, rows2=item2).item()
            == rank[item_idx2] - rank[item_idx1]
        )
    elif qoi_str == "top_k":
        assert qoi.calculate(rows1=item1, rows2=item2).item() == (
            (rank[item_idx1] <= 10).astype(int) - (rank[item_idx2] <= 10).astype(int)
        )
    elif qoi_str == "flip":
        assert qoi.calculate(rows1=item1, rows2=item2).item() == (
            y[item_idx1] != y[item_idx2]
        ).astype(int)
    else:
        np.testing.assert_allclose(
            qoi.calculate(rows1=item1, rows2=item2),
            y[item_idx1] - y[item_idx2],
            atol=1e-7,
        )
