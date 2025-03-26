import numpy as np
from sklearn.utils.validation import check_array, _get_feature_names

from sharp.qoi import get_qoi
from sharp._measures import MEASURES


def check_feature_names(X):
    """
    Retrieve feature names from X.
    """
    feature_names = _get_feature_names(X)

    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(X.shape[1])])

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


def check_qoi(qoi, target_function=None, X=None, cache=True):
    """
    If None, return a default function. If str, grab function from a dict. if function,
    check if it's valid and return itself.
    """
    # Target function is always required regardless of QoI
    params = {"target_function": target_function, "cache": cache}
    define_qoi = True
    if isinstance(qoi, str):

        params["X"] = X

        if target_function is None:
            msg = "If `qoi` is of type `str`, `target_function` cannot be None."
            raise TypeError(msg)

        if get_qoi(qoi)._qoi_type == "rank" and X is None:
            msg = "If `qoi` is `str` and rank-based, `X` cannot be None."
            raise TypeError(msg)

    elif qoi is None:
        qoi = "qoi"

        if target_function is None:
            msg = "If `qoi` is `None`, `target_function` cannot be None."
            raise TypeError(msg)

    else:
        define_qoi = False

    if define_qoi:
        qoi = get_qoi(qoi)(**params)

    if qoi._qoi_type == "rank":
        qoi.sort_base()

    return qoi
