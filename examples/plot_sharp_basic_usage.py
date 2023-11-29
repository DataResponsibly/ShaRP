"""
==================
ShaRP: Basic usage
==================

This example shows a simple application of ShaRP over a toy dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
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
    sample_size=50,
)
xai.fit(X, y)

print("Global contribution of a single feature:", xai.feature(0, X))

print("Global feature contributions:", xai.all(X).mean(axis=0))

individual_scores = xai.individual(9, X)
print("Feature contributions to a single observation: ", individual_scores)

pair_scores = xai.pairwise(X[2], X[3])
print("Pairwise comparison (one vs one):", pair_scores)

print("Pairwise comparison (one vs group):", xai.pairwise(X[2], X[5:10]))

# Visualization of feature contributions
print("Sample 2 feature values:", X[2])
print("Sample 3 feature values:", X[3])
fig, axes = plt.subplots(1, 2)

# Bar plot comparing two points
xai.plot.bar(pair_scores, ax=axes[0])
axes[0].set_title("Pairwise comparison between Sample 2 and 3")

# Waterfall explaining rank for sample 2
axes[1] = xai.plot.waterfall(individual_scores)
axes[1].xlabel("Ranking explanation for Sample 9")

plt.show()
