"""
Basic usage
===========

This example shows a simple application of ShaRP over a toy dataset.

We will start by setting up the imports, envrironment variables and a basic score
function that will be used to determine rankings.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sharp import ShaRP
from sharp.utils import scores_to_ordering

from itertools import combinations
import time
import pandas as pd
import seaborn as sns


# Set up some envrionment variables
RNG_SEED = 42
N_SAMPLES = 5000
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.1 * X[:, 0] + 0.1 * X[:, 1] + 0.1 * X[:, 2] + 0.1 * X[:, 3] + 0.1 * X[:, 4] + \
        0.1 * X[:, 5] + 0.1 * X[:, 6] + 0.1 * X[:, 7] + 0.1 * X[:, 8] + 0.1 * X[:, 9]


######################################################################################
# We can now generate a simple mock dataset with 2 features, one sampled from a normal
# distribution, another from a bernoulli.

X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.normal(1, 0.5, size=(N_SAMPLES, 1)),
     rng.normal(size=(N_SAMPLES, 1)), rng.normal(1, 0.5, size=(N_SAMPLES, 1)),
     rng.normal(size=(N_SAMPLES, 1)), rng.normal(1, 0.5, size=(N_SAMPLES, 1)),
     rng.normal(size=(N_SAMPLES, 1)), rng.normal(1, 0.5, size=(N_SAMPLES, 1)),
     rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)
rank = scores_to_ordering(y)



################# TEST

rest_cols_idx = np.arange(X.shape[1])
rest_cols_idx = rest_cols_idx[rest_cols_idx != 1]

# Obtain all coalitions
coalitions = []
for set_size in range(0, len(rest_cols_idx) + 1):
    for set_cols_idx in combinations(rest_cols_idx, set_size):
        coalitions.append(set_cols_idx)
# coalitions = np.array(coalitions)

print(len(coalitions))
print(2 ** (X.shape[1]-1))

# print(coalitions[rng.choice(np.arange(len(coalitions)), size=10, replace=False)])

######################################################################################
# Run ShaRP for all features using different sample and coalition sizes

x_explain = X[rng.choice(np.arange(X.shape[0]), size=1000)]

# Calculate actual exact contributions
xai = ShaRP(
        qoi="rank",
        target_function=score_function,
        measure="shapley",
        sample_size=None,
        coalition_size=512,
        replace=False,
        random_state=RNG_SEED,
    )
xai.fit(x_explain)

start = time.time()
ftr_contrs_exact = xai.all(x_explain).mean(axis=0)
end = time.time()
print(end - start)

# Now caluclate the approximations
sizes = range(10, 512, 50)
ftr_contrs = pd.DataFrame(ftr_contrs_exact, columns=["exact"])
ftr_contrs_error = []
times = []

for coal_size in sizes:
    print(coal_size)
    xai = ShaRP(
        qoi="rank",
        target_function=score_function,
        measure="shapley",
        sample_size=None,
        coalition_size=coal_size,
        replace=False,
        random_state=RNG_SEED,
    )
    xai.fit(x_explain)

    start = time.time()
    ftr_contrs_curr = xai.all(x_explain).mean(axis=0)
    ftr_contrs = pd.concat([ftr_contrs, pd.DataFrame(ftr_contrs_curr, columns=[coal_size])], axis=1)
    end = time.time()
    ftr_contrs_error.append((ftr_contrs_curr - ftr_contrs_exact).mean())
    times.append(end - start)
    print(times)


# individual_scores = xai.individual(9, x_explain)
# print("Feature contributions to a single observation: ", individual_scores)

print(ftr_contrs)
print(ftr_contrs_error)
print(times)

######################################################################################
# We can also turn these into visualizations:

plt.plot(sizes, times)
plt.show()

plt.plot(sizes, ftr_contrs_error)
plt.show()

ftr_contrs.to_csv('/Users/vp/Desktop/times-test.csv', index=False)

# plt.style.use("seaborn-v0_8-whitegrid")
#
# # Visualization of feature contributions
# print("Sample 2 feature values:", X[2])
# print("Sample 3 feature values:", X[3])
# fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")
#
# # Bar plot comparing two points
# xai.plot.bar(pair_scores, ax=axes[0], color="#ff0051")
# axes[0].set_title(
#     f"Pairwise comparison - Sample 2 (rank {rank[2]}) vs 3 (rank {rank[3]})",
#     fontsize=12,
#     y=-0.2,
# )
# axes[0].set_xlabel("")
# axes[0].set_ylabel("Contribution to rank", fontsize=12)
# axes[0].tick_params(axis="both", which="major", labelsize=12)
#
# # Waterfall explaining rank for sample 2
# axes[1] = xai.plot.waterfall(
#     individual_scores, feature_values=X[9], mean_target_value=rank.mean()
# )
# ax = axes[1].gca()
# ax.set_title("Rank explanation for Sample 9", fontsize=12, y=-0.2)
#
# plt.show()
