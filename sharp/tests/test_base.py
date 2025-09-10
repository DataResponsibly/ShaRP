"""
Tests code in `base.py`.
"""

import numpy as np
import pytest
from sharp import ShaRP
from sharp.qoi import TopKQoI

RNG_SEED = 123
N_SAMPLES = 20


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


def test_default_qoi():
    """
    Reproduces issue #44: Defining ShaRP without an explicit QoI raises an AttributeError
    """
    _X = np.random.random((100, 3))
    sharp = ShaRP(target_function=lambda x: x.sum(axis=1))
    sharp.fit(_X)
    sharp.all(_X[:5])


@pytest.mark.parametrize("top_k", [1, 5, 10])
def test_sharp_topk_qoi_with_different_k(top_k):
    """
    Test ShaRP with the TopK QoI with different values of k.

    Source: Issue #68
    """
    rng = np.random.RandomState(RNG_SEED)
    X = np.concatenate(
        [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))],
        axis=1,
    )

    xai = ShaRP(
        qoi="top_k", target_function=score_function, random_state=RNG_SEED, top_k=top_k
    )
    xai.fit(X)
    contributions = xai.all(X)

    # Check qoi type and top_k parameter
    assert isinstance(xai.qoi_, TopKQoI)
    assert xai.qoi_.top_k == top_k

    # Output shape
    assert contributions.shape == X.shape

    # For shapley, mean should be close to zero
    np.testing.assert_allclose(contributions.mean(), 0, atol=0.001)

    # Check that top_k is respected in the QoI output
    topk_est = xai.qoi_.estimate(X)
    assert ((topk_est == 0) | (topk_est == 1)).all()
    assert topk_est.sum() <= X.shape[0]
