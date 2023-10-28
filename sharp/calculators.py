"""
Implementation notes:
- SHAP samples the data to setup up the explainer object.
    - See documentation from shap.maskers.Independent
"""

from sklearn.utils import check_random_state
from .methods import _set_qii_row, _marginal_qii_row
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd


def group_set_qii(qoi, columns, dataset, sample_size=None, random_state=None):
    """
    This is the function that to calculate the QII for a single attribute
    (unary QII) or a set of attributes (set QII).

    Parameters
    ----------
      dataset [pandas.dataframe]:
        The dataset the we input to the model.
      model [function]:
        The machine learning model we used to predict the data.
      columns [str, list]:
        List of attributes that we are going to calculate.
      iterate_time [int]:
        How many times we iterate to calculate the QII. Given a default value.

      return [int], the QII score of the attribute, -- how this set of
        attributes contribute to the machine.
    """
    random_state = check_random_state(random_state)
    # TODO - check dataframe function
    # TODO - check sample size function
    # TODO     - Sampling based on the info provided in paper
    if sample_size is None:
        sample_size = 30

    qii = dataset.apply(
        lambda row: _set_qii_row(
            qoi=qoi,
            row=row,
            columns=columns,
            dataset=dataset,
            sample_size=sample_size,
            rng=random_state,
        ),
        axis=1,
    ).values

    return qii


def group_marginal_qii(
    qoi, column, set_columns, dataset, sample_size=None, random_state=None
):
    """
    This is the function that to calculate the marginal QII for a attribute.

    Parameters
    ----------
      column [str]:
          Name of the target column the function going to calculate.
      set_columns [list]:
          List of columns that going to change in order to calculate the
          marginal value.
      dataset [Pandas.Dataframe/Pandas.Series]:
          The dataset we calculate.
      model [function]:
          The machine learning model we used to predict the data.
      iterate_time [int]:
          How many times we iterate to calculate the QII. Given a default
          value.

      return [int], the QII score of the attribute,
          -- how this set of attributes contribute to the machine.
    """
    random_state = check_random_state(random_state)
    # TODO - check dataframe function
    # TODO - check sample size function
    # TODO     - Sampling based on the info provided in paper
    if sample_size is None:
        sample_size = 30

    qii = dataset.apply(
        lambda row: _marginal_qii_row(
            qoi=qoi,
            row=row,
            column=column,
            set_columns=set_columns,
            dataset=dataset,
            sample_size=sample_size,
            rng=random_state,
        ),
        axis=1,
    ).values

    return qii


# How can we compair the specific difference?
# Or we can suppose all the output of the machine are numbers for now?
# Should I make QII an matrix like shapley value did?


def shapley_score(qoi, row, dataset, target, random_state, iterate_time=30):
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

    # Get all column names and remove feature being explained
    ftr_names = dataset.columns.values
    coal_ftr_names = np.delete(ftr_names, np.where(ftr_names == target))

    total_score = 0
    for set_size in range(len(coal_ftr_names) + 1):
        for set_columns in combinations(coal_ftr_names, set_size):
            # To calculate the marginal score of each column in
            score = _marginal_qii_row(
                qoi=qoi,
                row=row,
                column=target,
                set_columns=set_columns,
                dataset=dataset,
                rng=random_state,
                sample_size=iterate_time,
            )
            total_score = total_score + score / (
                comb(len(coal_ftr_names), set_size) * len(ftr_names)
            )
    return total_score


def banzhaf_score(qoi, row, dataset, target, random_state, iterate_time=30):
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
    # Get all column names and remove feature being explained
    ftr_names = dataset.columns.values
    coal_ftr_names = np.delete(ftr_names, np.where(ftr_names == target))

    total_score = 0
    for set_size in range(len(coal_ftr_names) + 1):
        for set_columns in combinations(coal_ftr_names, set_size):
            # To calculate the marginal score of each column in
            score = _marginal_qii_row(
                qoi=qoi,
                row=row,
                column=target,
                set_columns=set_columns,
                dataset=dataset,
                rng=random_state,
                sample_size=iterate_time,
            )
            total_score = total_score + score / 2 ** (len(ftr_names) - 1)
    return total_score


def group_set_qii(qoi, columns, dataset, sample_size=None, random_state=None):
    """
    This is the function that to calculate the QII for a single attribute
    (unary QII) or a set of attributes (set QII).

    Parameters
    ----------
      dataset [pandas.dataframe]:
        The dataset the we input to the model.
      model [function]:
        The machine learning model we used to predict the data.
      columns [str, list]:
        List of attributes that we are going to calculate.
      iterate_time [int]:
        How many times we iterate to calculate the QII. Given a default value.

      return [int], the QII score of the attribute, -- how this set of
        attributes contribute to the machine.
    """
    random_state = check_random_state(random_state)
    # TODO - check dataframe function
    # TODO - check sample size function
    # TODO     - Sampling based on the info provided in paper
    if sample_size is None:
        sample_size = 30

    qii = dataset.apply(
        lambda row: _set_qii_row(
            qoi=qoi,
            row=row,
            columns=columns,
            dataset=dataset,
            sample_size=sample_size,
            rng=random_state,
        ),
        axis=1,
    ).values

    return qii
