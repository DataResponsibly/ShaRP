import numpy as np
from sklearn.utils.validation import check_array, _get_feature_names


def check_feature_names(X):
    """
    Retrieve feature names from X.
    """
    feature_names = _get_feature_names(X)

    if feature_names is None:
        feature_names = np.indices([X.shape[1]]).squeeze()

    return feature_names


def check_inputs(X, y=None):
    """
    Converts X and y inputs to numpy arrays.
    """
    if y is not None:
        y = np.array(y)

    return check_array(X, dtype="object"), y


def check_measure():
    """
    If None, return a default function. If str, grab function from a dict. if function,
    return itself.
    """
    pass
