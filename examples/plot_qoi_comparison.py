"""
Comparing different Quantities of Interest (QOIs) in ShaRP
==========================================================

This example demonstrates how to use ShaRP with different QOIs to analyze 
and explain ranking outcomes.

In this example, we will:
- Set up and define two QOIs, rank and score.
- Compare feature contributions under each QOI for several examples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
import certifi
import ssl
import urllib.request

from sharp import ShaRP
from sharp.utils import scores_to_ordering


# Set up some envrionment variables
RNG_SEED = 42
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)


#############################################################################################
# We will use CS Rankings dataset, which contains data for 189 Computer Science departments 
# in the US, e.g., publication count of the faculty across 4 research areas: AI, Systems, 
# Theory, and Inter- disciplinary.

CS_RANKS_URL = "https://zenodo.org/records/11234896/files/csrankings_raw.csv"
context = ssl.create_default_context(cafile=certifi.where())

csrank_data = (
    pd.read_csv(urllib.request.urlopen(CS_RANKS_URL, context=context))
    .drop(columns="Unnamed: 0")
    .rename(columns={"Count": "Rank"})
    .set_index("Institution")
)

# Let's also preprocess this data
def preprocess_csrank_data(df):
    X = df.drop(columns=["Rank", "Score"])
    X = X.iloc[:, X.columns.str.contains("Count")]
    X = X / X.max()
    ranks = df.Rank
    scores = df.Score
    return X, ranks, scores

X, _, _ = preprocess_csrank_data(csrank_data)
X.head()

# Here we will define the scoring function
def csrank_score(X):
    weights = np.array([5, 12, 3, 7])

    # multiplier contains the maximum values in the original dataset
    multiplier = np.array([71.4, 12.6, 21.1, 13.8])

    if np.array(X).ndim == 1:
        X = np.array(X).reshape(1, -1)

    return np.clip(
        (np.array(X) * multiplier) ** weights + 1, a_min=1, a_max=np.inf
    ).prod(axis=1) ** (1 / weights.sum())

score = csrank_score(X)
rank = scores_to_ordering(score)


#############################################################################################
# Next, we will configure ``ShaRP`` with `qoi` parameter set to `rank` and `rank_score`:

xai_rank = ShaRP(
    qoi="rank",
    target_function=csrank_score,
    measure="shapley",
    sample_size=None,
    replace=False,
    random_state=RNG_SEED,
    n_jobs=-1,
)
xai_rank.fit(X)

xai_score = ShaRP(
    qoi="rank_score",
    target_function=csrank_score,
    measure="shapley",
    sample_size=None,
    replace=False,
    random_state=RNG_SEED,
    n_jobs=-1,
)
xai_score.fit(X)


#############################################################################################
# Let's take a look at some contributions for both QOIs

contributions_rank = xai_rank.all(X)
contributions_rank[:5]

contributions_score = xai_score.all(X)
contributions_score[:5]

# Now let's plot the waterfall plots for different universities and check 
# if the results for `score` and `rank` differ

# Plot for Stanford University, ranked #6 in QS World University Rankings 2025
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
xai_rank.plot.waterfall(
    contributions=contributions_rank[5],
    feature_values=X.iloc[5].to_numpy(),
    mean_target_value=rank.mean()
)
plt.title("Stanford University - Waterfall Plot for Rank")

plt.subplot(1, 2, 2)
xai_score.plot.waterfall(
    contributions=contributions_score[5],
    feature_values=X.iloc[5].to_numpy(),
    mean_target_value=score.mean()
)
plt.title("Stanford University - Waterfall Plot for Score")

plt.tight_layout()
plt.show()


# Plot for University of Texas at Austin, ranked #66 in QS World University Rankings 2025
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
xai_rank.plot.waterfall(
    contributions=contributions_rank[15],
    feature_values=X.iloc[15].to_numpy(),
    mean_target_value=rank.mean()
)
plt.title("University of Texas at Austin - Waterfall Plot for Rank")

plt.subplot(1, 2, 2)
xai_score.plot.waterfall(
    contributions=contributions_score[15],
    feature_values=X.iloc[15].to_numpy(),
    mean_target_value=score.mean()
)
plt.title("University of Texas at Austin - Waterfall Plot for Score")

plt.tight_layout()
plt.show()


# Plot for Indiana University, ranked #355 in QS World University Rankings 2025
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
xai_rank.plot.waterfall(
    contributions=contributions_rank[53],
    feature_values=X.iloc[53].to_numpy(),
    mean_target_value=rank.mean()
)
plt.title("Indiana University - Waterfall Plot for Rank")

plt.subplot(1, 2, 2)
xai_score.plot.waterfall(
    contributions=contributions_score[53],
    feature_values=X.iloc[53].to_numpy(),
    mean_target_value=score.mean()
)
plt.title("Indiana University - Waterfall Plot for Score")

plt.tight_layout()
plt.show()
