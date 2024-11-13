"""
Comparison of feature contributions for advantaged and disadvantaged groups
===========================================================================

This example demonstrates how feature contributions differ for advantaged and
disadvantaged groups after training ``ShaRP`` model on the whole data.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from sharp import ShaRP
from sharp.utils import check_inputs

plt.style.use("seaborn-v0_8-whitegrid")

#########################################################################################
# Let's start with data preparation

X, y = fetch_openml(
    data_id=43141, parser="auto", return_X_y=True, read_csv_kwargs={"nrows": 150}
)

# Get the indices of samples that belong to each group
adv_idx = X[X["SEX"] == 1].index
dis_idx = X[X["SEX"] == 2].index

# Reduce the number of features in the dataset
numerical_cols = ["AGEP", "WKHP"]
ordinal_cols = ["SCHL"]

X = X.filter(items=numerical_cols + ordinal_cols)

X.head()


# Here we will define the scoring function
def score_function(X):
    X, _ = check_inputs(X)
    return 0.2 * X[:, 0] + 0.3 * X[:, 1] + 0.5 * X[:, 2]


# Standardize X and calculate scores
scaler = MinMaxScaler()
X.iloc[:, :] = scaler.fit_transform(X)
scores = score_function(X)


#########################################################################################
# Next, we will configure ``ShaRP`` and fit it on the whole dataset:

xai = ShaRP(
    qoi="rank",
    target_function=score_function,
    measure="shapley",
    sample_size=None,
    replace=False,
    random_state=42,
    n_jobs=-1,
)

xai.fit(X)


#########################################################################################
# Let's take a look at contributions for both QOIs

contributions = xai.all(X)

# Now let's create box plots and compare feature contributions for privileged and
# protected groups

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

for ax, idx, title in zip(
    axes.flatten(),
    [X.index, adv_idx, dis_idx],
    ["All", "Advantaged group", "Disadvantaged group"],
):
    xai.plot.box(X=X.loc[idx], y=scores[idx], contributions=contributions[idx], ax=ax)
    ax.set_xlabel(title)

plt.show()

#########################################################################################
# We can also compare contributions across groups overall:

X["Sex"] = ""
X.loc[adv_idx, "Sex"] = "Male"
X.loc[dis_idx, "Sex"] = "Female"
xai.plot.box(X, scores, contributions, group="Sex")
plt.show()
