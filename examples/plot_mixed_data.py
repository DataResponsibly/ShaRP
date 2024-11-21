"""
ShaRP for classification on large datasets with mixed data types
================================================================

This example showcases a more complex setting, where we will develop and interpret a
classification model using a larger dataset with both categorical and continuous
features.

``sharp`` is designed to operate over the unprocessed input space, to ensure every
"Frankenstein" point generated to compute feature contributions are plausible. This means
that the function producing the scores (or class predictions) should take as input the
raw dataset, and every preprocessing step leading to the black box predictions/scores
should be included within it.

We will start by downloading the German Credit dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sharp import ShaRP

#####################################################################################
# Let's get the data first. We will use the dataset that classifies people described 
# by a set of attributes as good or bad credit risks.

df = fetch_openml(data_id=31, parser="auto")["frame"]
df.head(5)

######################################################################
# Split X and y (input and target) from `df` and split train and test:

X = df.drop(columns="class")
y = df["class"]

categorical_features = X.dtypes.apply(
    lambda dtype: isinstance(dtype, pd.CategoricalDtype)
).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

#########################################################################################
# Now we will set up model. Here, we will use a pipeline to combine all the preprocessing
# steps. However, to use ``sharp``, it is also sufficient to pass any function
# (containing all the preprocessing steps) that takes a numpy array as input and outputs
# the model's predictions.

transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=False), categorical_features),
        ("minmax", MinMaxScaler(), ~categorical_features),
    ],
    remainder="passthrough",
    n_jobs=-1,
)
classifier = LogisticRegression(random_state=42)
model = make_pipeline(transformer, classifier)
model.fit(X_train.values, y_train.values)

#########################################################################################
# We can now use ``sharp`` to explain our model's predictions! If we consider the dataset
# to be too large, we have a few options to reduce computational complexity, such as
# configuring the ``n_jobs`` parameter, setting a value on ``sample_size``, or setting
# ``measure=unary``.

xai = ShaRP(
    qoi="flip",
    target_function=model.predict,
    measure="unary",
    sample_size=None,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
xai.fit(X_test)

unary_values = pd.DataFrame(xai.all(X_test), columns=X.columns)
unary_values

##############################################################
# Finally, we can plot the mean contributions of each feature:

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots()
xai.plot.bar(unary_values.mean(), ax=ax)
ax.set_ylim(bottom=0)
ax.tick_params(labelrotation=90)
fig.tight_layout()
plt.show()
