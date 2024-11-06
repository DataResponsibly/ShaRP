"""
Extended Usage of ShaRP with Banzhaf Values
============================================

This example extends the basic functionality of the ShaRP library by introducing
the Banzhaf values for feature influence assessment. The Banzhaf values provide
an alternative to the Shapley values, emphasizing a feature's power in forming
winning coalitions rather than ensuring fair contribution distribution.

In contrast to the Shapley values, which assign contributions to features based
on a cooperative game theory approach, the Banzhaf values examine the strength
of each feature by calculating its influence in potential combinations. This
can be particularly useful for analyzing features in scenarios where influence
or power dynamics are more relevant than equal contribution.

In this example, we walk through:
- Setting up a toy dataset and a basic ranking function.
- Applying both Shapley and Banzhaf approaches to calculate feature contributions for rankings.
- Comparing and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sharp import ShaRP
from sharp.utils import scores_to_ordering

# Set up some envrionment variables
RNG_SEED = 42
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


#############################################################################################
# We can now generate a simple mock dataset with 2 features, one sampled from a normal
# distribution, another from a bernoulli.

X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)
rank = scores_to_ordering(y)


#############################################################################################
# Next, we will configure ``ShaRP`` with `measure` parameter set to `shapley` and `banzhaff`:

xai_shapley = ShaRP(
    qoi="rank",
    target_function=score_function,
    measure="shapley",
    sample_size=None,
    replace=False,
    random_state=RNG_SEED,
)
xai_shapley.fit(X)

xai_banzhaf = ShaRP(
    qoi="rank",
    target_function=score_function,
    measure="banzhaff",
    sample_size=None,
    replace=False,
    random_state=RNG_SEED,
)
xai_banzhaf.fit(X)


#############################################################################################
# Let's take a look at some banzhaf values used for ranking explanations:

print("Aggregate contribution of a single feature:", xai_banzhaf.feature(0, X))
print("Aggregate feature contributions:", xai_banzhaf.all(X).mean(axis=0))

# Compute feature contributions for a single observation using both approaches
individual_scores_banzhaf = xai_banzhaf.individual(6, X)
individual_scores_shapley = xai_shapley.individual(6, X)
print("Banzhaf - Feature contributions for sample 6:", [float(value) for value in individual_scores_banzhaf])
print("Shapley - Feature contributions for sample 6:", [float(value) for value in individual_scores_shapley])

# Compute pairwise feature contributions using Banzhaf approach
pair_scores_banzhaf = xai_banzhaf.pairwise(X[2], X[3])
pair_scores_group_banzhaf = xai_banzhaf.pairwise(X[2], X[5:10])
print("Banzhaf - Pairwise comparison (one vs one):", [float(value) for value in pair_scores_banzhaf])
print("Banzhaf - Pairwise comparison (one vs group):", [float(value) for value in pair_scores_group_banzhaf])

# Compute pairwise feature contributions using Shapley approach
pair_scores_shapley = xai_shapley.pairwise(X[2], X[3])
pair_scores_group_shapley = xai_shapley.pairwise(X[2], X[5:10])
print("Shapley - Pairwise comparison (one vs one):", [float(value) for value in pair_scores_shapley])
print("Shapley - Pairwise comparison (one vs group):", [float(value) for value in pair_scores_group_shapley])


#############################################################################################
# We can also turn these into visualizations:

plt.style.use("seaborn-v0_8-whitegrid")

# Visualization of feature contributions with Shapley values
print("Sample 2 feature values:", X[2])
print("Sample 3 feature values:", X[3])
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")

# Bar plot comparing two points
xai_shapley.plot.bar(pair_scores_shapley, ax=axes[0], color="#ff0051")
axes[0].set_title(
    f"Shapley - Pairwise comparison - Sample 2 (rank {rank[2]}) vs 3 (rank {rank[3]})",
    fontsize=12,
    y=-0.2,
)
axes[0].set_xlabel("")
axes[0].set_ylabel("Contribution to rank", fontsize=12)
axes[0].tick_params(axis="both", which="major", labelsize=12)

# Waterfall explaining rank for sample 9
axes[1] = xai_shapley.plot.waterfall(
    individual_scores_shapley, feature_values=X[9], mean_target_value=rank.mean()
)
ax = axes[1].gca()
ax.set_title("Shapley - Rank explanation for Sample 9", fontsize=12, y=-0.2)

plt.show()


# Visualization of feature contributions with Banzhaf values
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")

# Bar plot comparing two points
xai_banzhaf.plot.bar(pair_scores_banzhaf, ax=axes[0], color="#ff0051")
axes[0].set_title(
    f"Banzhaf - Pairwise comparison - Sample 2 (rank {rank[2]}) vs 3 (rank {rank[3]})",
    fontsize=12,
    y=-0.2,
)
axes[0].set_xlabel("")
axes[0].set_ylabel("Contribution to rank", fontsize=12)
axes[0].tick_params(axis="both", which="major", labelsize=12)

# Waterfall explaining rank for sample 9
axes[1] = xai_banzhaf.plot.waterfall(
    individual_scores_banzhaf, feature_values=X[9], mean_target_value=rank.mean()
)
ax = axes[1].gca()
ax.set_title("Banzhaf - Rank explanation for Sample 9", fontsize=12, y=-0.2)

plt.show()