import numpy as np
from sklearn.utils.validation import check_array, _get_feature_names

from sharp.qoi import QOI_OBJECTS
from sharp._measures import MEASURES


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


def check_measure(measure):
    """
    If None, return a default function. If str, grab function from a dict. if function,
    check if it's valid and return itself.
    """
    if measure is None:
        return MEASURES["shapley"]
    elif isinstance(measure, str):
        return MEASURES[measure]
    else:
        return measure


def check_qoi(qoi, target_function=None, X=None):
    """
    If None, return a default function. If str, grab function from a dict. if function,
    check if it's valid and return itself.
    """
    if isinstance(qoi, str):
        # Target function is always required regardless of QoI
        params = {"target_function": target_function}

        if target_function is None:
            msg = "If `qoi` is of type `str`, `target_function` cannot be None."
            raise TypeError(msg)

        if QOI_OBJECTS[qoi]._qoi_type == "rank":
            # Add dataset to list of parameters if QoI is rank-based
            params["X"] = X

            if X is None:
                msg = "If `qoi` is `str` and rank-based, `X` cannot be None."
                raise TypeError(msg)

    elif qoi is None:
        params = {"target_function": target_function}
        qoi = "qoi"

        if target_function is None:
            msg = "If `qoi` is `None`, `target_function` cannot be None."
            raise TypeError(msg)

    else:
        return qoi

    qoi = QOI_OBJECTS[qoi](**params)
    return qoi
