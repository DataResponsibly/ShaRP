"""
==================
ShaRP: Basic usage
==================

This example shows a simple application of ShaRP over a toy dataset.
"""

import numpy as np
from sklearn.utils import check_random_state
from sharp import ShaRP

RNG_SEED = 42
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)

xai = ShaRP(
    qoi="ranking",
    target_function=score_function,
    measure="shapley",
    random_state=RNG_SEED,
)

# Feature importance (single feature)
xai.feature(0, X)

# Feature importance (single observation)
xai.individual(9, X)

# Feature importance (all cells)
xai.all(X)

# Pairwise comparison (one vs one)
xai.pairwise(X[2], X[3])

# Pairwise comparison (one vs group)
xai.pairwise(X[2], X[5:10])
