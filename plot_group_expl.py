"""

==========================================================


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from sharp import ShaRP
from sharp.utils import check_inputs


#############################################################################################
# Let's start with data preparation

X, y = fetch_openml(
    data_id=43141, parser="auto", return_X_y=True, read_csv_kwargs={"nrows": 150}
)

# Get the indices of samples that belong to each group
privil_group_indexes, protec_group_indexes = X[X["SEX"] == 1].index, X[X["SEX"] == 2].index

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


#############################################################################################
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


#############################################################################################
# Let's take a look at some contributions for both QOIs

contributions = xai.all(X)[:10]

# Now let's create boxplots and compare feature contributions to different groups
xai.plot.strata_boxplot(X, scores, contributions)
xai.plot.strata_boxplot(X.loc[privil_group_indexes], scores[privil_group_indexes], contributions[privil_group_indexes])
xai.plot.strata_boxplot(X.loc[protec_group_indexes], scores[protec_group_indexes], contributions[protec_group_indexes])