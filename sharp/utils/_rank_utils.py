import numpy as np


def scores_to_ordering(y, direction=-1):
    """
    Converts an array with scores to a ranking.

    If higher rank values are better, set direction to 1 instead.
    """
    temp = np.argsort(y * direction)
    ranks = np.zeros(*y.shape, dtype=int)
    ranks[temp] = np.arange(*y.shape) + 1
    return ranks
