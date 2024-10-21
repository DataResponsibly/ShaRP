import numpy as np
from ._base import _MEASURES


def cross_method_explanation_consistency(
    results1, results2, measure="kendall", **kwargs
):
    res_ = _MEASURES[measure](results1, results2, **kwargs)
    mean = res_.mean()
    sem = np.std(res_) / np.sqrt(res_.size)
    return mean, sem
