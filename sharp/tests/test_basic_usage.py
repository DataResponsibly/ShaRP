import pytest
from itertools import product
import numpy as np
from sklearn.utils import check_random_state
from sharp import ShaRP
from sharp.qoi import get_qoi
from sharp._measures import MEASURES

# Set up some envrionment variables
RNG_SEED = 123
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)

rank_qois_str = ["rank", "rank_score", "top_k"]
rank_qois_obj = [get_qoi(qoi) for qoi in rank_qois_str]

clf_qois_str = ["diff", "flip", "likelihood"]
clf_qois_obj = [get_qoi(qoi) for qoi in clf_qois_str]

measures = list(MEASURES.keys())


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)


@pytest.mark.parametrize("qoi, measure", product(rank_qois_str, measures))
def test_explain_all_rank_str_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for ranking only).
    """
    xai = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure=measure,
        random_state=RNG_SEED,
    )
    xai.fit(X)
    contributions = xai.all(X)

    if measure == "shapley":
        np.testing.assert_allclose(contributions.mean(), 0, atol=1e-07)
    else:
        # Calculation of other measures other than shapley does not guarantee mean
        # contributions to ranking are going to be zero
        np.testing.assert_allclose(contributions.mean(), 0, atol=0.5)


@pytest.mark.parametrize("qoi, measure", product(rank_qois_obj, measures))
def test_explain_all_rank_obj_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for ranking only).
    """
    qoi = qoi(target_function=score_function, X=X)
    xai = ShaRP(
        qoi=qoi,
        measure=measure,
        random_state=RNG_SEED,
    )
    xai.fit(X)
    contributions = xai.all(X)

    if measure == "shapley":
        np.testing.assert_allclose(contributions.mean(), 0, atol=1e-07)
    else:
        # Calculation of other measures other than shapley does not guarantee mean
        # contributions to ranking are going to be zero
        np.testing.assert_allclose(contributions.mean(), 0, atol=0.5)


@pytest.mark.parametrize("qoi, measure", product(clf_qois_str, measures))
def test_explain_all_clf_str_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for classification only).
    """
    clf_function = lambda X: (score_function(X) > 0.5).astype(int)  # noqa

    xai = ShaRP(
        qoi=qoi,
        target_function=clf_function,
        measure=measure,
        random_state=RNG_SEED,
    )
    xai.fit(X)
    contributions = xai.all(X)

    if qoi == "flip":
        assert (contributions >= 0).all()

    if measure == "shapley":
        val = (contributions.sum(axis=1) >= (1 - clf_function(X).mean())).astype(int)
        assert (val == clf_function(X)).all()
    elif measure in ["unary", "set"] and qoi != "flip":
        assert (contributions.sum(axis=1) >= 0).mean() == clf_function(X).mean()


@pytest.mark.parametrize("qoi, measure", product(clf_qois_obj, measures))
def test_explain_all_clf_obj_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for classification only).
    """
    clf_function = lambda X: (score_function(X) > 0.5).astype(int)  # noqa

    qoi = qoi(target_function=clf_function, X=X)
    xai = ShaRP(
        qoi=qoi,
        measure=measure,
        random_state=RNG_SEED,
    )

    xai.fit(X)
    contributions = xai.all(X)

    if qoi.__class__.__name__ == "FlipQoI":
        assert (contributions >= 0).all()

    if measure == "shapley":
        val = (contributions.sum(axis=1) >= (1 - clf_function(X).mean())).astype(int)
        assert (val == clf_function(X)).all()
    elif measure in ["unary", "set"] and qoi.__class__.__name__ != "FlipQoI":
        assert (contributions.sum(axis=1) >= 0).mean() == clf_function(X).mean()
