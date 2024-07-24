"""
Quantitative Input Influence measures.
"""

import numpy as np
from math import comb
from itertools import combinations


def _set(row, col_idx, X, qoi, sample_size, replace, rng, **kwargs):
    """
    Calculates the QII for a single or set of attributes in a single row.

    Parameters
    ----------
    row_idx: pd.Series
        The index of the row to be explained.

    col_idx: [str, list]
        The index of the attribute to be explained.

    dataset: [pd.DataFrame, np.ndarray]
        The dataframe/array used to test the classifier.

    qoi: qoi object
        The quantity of interest used to measure feature importance.

    sample_size: int, default=30
        How many times we calculate the qoi.

    Returns
    -------
    scores: int
        The QII score of the attribute, i.e., how this attribute contributes to the QOI.
    """

    # Draw new samples uniformly at random
    X_sampled = X[rng.choice(np.arange(X.shape[0]), size=sample_size, replace=replace)]

    # Unary/Set approach
    X_modded = np.repeat(row.reshape(1, -1), repeats=sample_size, axis=0)
    X_modded[:, col_idx] = X_sampled[:, col_idx]

    # Return score
    return qoi.calculate(row.reshape(1, -1), X_modded)


def _marginal(row, col_idx, set_cols_idx, X, qoi, sample_size, replace, rng, **kwargs):
    """
    Calculates the marginal QII for a single or set of attributes in a single row.

    Parameters
    ----------
      row [pandas.series]:
          The dataframe row we are explaining.
      column [str]:
          The column we are explaining.
      set_columns [str, list]:
          The attribute (list) that we are going to use for the marginal.
      dataset [pandas.dataframe]:
          The dataset to use in order to test the classifier.
      classifier [function]:
          The machine learning model we used to predict the data.
      sample_size [int], default=30:
          how many times we calculate the QII.
      return [int]:
          the QII score of the attribute,
          -- how this attribute contribute to the machine.
    """
    # Draw new samples uniformly at random
    X_sampled = X[rng.choice(np.arange(X.shape[0]), size=sample_size, replace=replace)]

    # Marginal approach
    X_modded1 = np.repeat(row.reshape(1, -1), repeats=sample_size, axis=0)

    # For X_modded1, keep:
    # - ``X_sampled`` values for the columns in ``set_cols_idx``
    X_modded1[:, set_cols_idx] = X_sampled[:, set_cols_idx]

    # For X_modded2, keep:
    # - ``X_sampled`` values for the columns in ``set_cols_idx`` and ``col_idx``
    X_modded2 = X_modded1.copy()
    X_modded2[:, col_idx] = X_sampled[:, col_idx]

    # Return score
    return qoi.calculate(X_modded1, X_modded2)


def _shapley(row, col_idx, X, qoi, sample_size, coalition_size, replace, rng, **kwargs):
    """
    Calculates the Shapley for a single attribute of a single row.

    Parameters
    ----------
      row [pandas.series]:
          The dataframe row we are explaining.
      dataset [pandas.dataframe]:
          The dataset to use in order to test the classifier.
      target [str]:
          The feature we are explaining
      model [function]:
          The machine learning model we used to predict the data.
      random_state [int]:
          Random state seed.
      iterate_time [int], default=30:
          how many times we calculate the marginal per coalition.

      return [int]:
          the Shapley score of the attribute for the feature,
          -- how this attribute contributes to the feature's prediction.
    """

    # Get indices for all columns except the one being explained
    rest_cols_idx = np.arange(X.shape[1])
    rest_cols_idx = rest_cols_idx[rest_cols_idx != col_idx]

    # Set up variable to track the total score for the specific attribute
    total_score = 0

    # Calculate the marginal score of every combination for ``col_idx`` vs rest
    iterable = [
        set_cols_idx
        for set_size in range(0, coalition_size + 1)
        for set_cols_idx in combinations(rest_cols_idx, set_size)
    ]

    for set_cols_idx in iterable:
        score = _marginal(
            row=row,
            col_idx=col_idx,
            set_cols_idx=set_cols_idx,
            X=X,
            qoi=qoi,
            sample_size=sample_size,
            replace=replace,
            rng=rng,
        )
        total_score += score / (comb(X.shape[1] - 1, len(set_cols_idx)) * X.shape[1])

    return total_score


def _banzhaff(
    row, col_idx, X, qoi, sample_size, coalition_size, replace, rng, **kwargs
):
    """
    Calculates the Shapley for a single attribute of a single row.

    Parameters
    ----------
      row [pandas.series]:
          The dataframe row we are explaining.
      dataset [pandas.dataframe]:
          The dataset to use in order to test the classifier.
      target [str]:
          The feature we are explaining
      model [function]:
          The machine learning model we used to predict the data.
      random_state [int]:
          Random state seed.
      iterate_time [int], default=30:
          how many times we calculate the marginal per coalition.

      return [int]:
          the Shapley score of the attribute for the feature,
          -- how this attribute contributes to the feature's prediction.
    """
    # Get indices for all columns except the one being explained
    rest_cols_idx = np.arange(X.shape[1])
    rest_cols_idx = rest_cols_idx[rest_cols_idx != col_idx]

    # Set up variable to track the total score for the specific attribute
    total_score = 0

    # Calculate the marginal score of every combination for ``col_idx`` vs rest
    iterable = [
        set_cols_idx
        for set_size in range(0, coalition_size + 1)
        for set_cols_idx in combinations(rest_cols_idx, set_size)
    ]

    for set_cols_idx in iterable:
        score = _marginal(
            row=row,
            col_idx=col_idx,
            set_cols_idx=set_cols_idx,
            X=X,
            qoi=qoi,
            sample_size=sample_size,
            replace=replace,
            rng=rng,
        )
        total_score += score / 2 ** (X.shape[1] - 1)

    return total_score


MEASURES = {
    "unary": _set,
    "set": _set,
    "marginal": _marginal,
    "shapley": _shapley,
    "banzhaff": _banzhaff,
}
