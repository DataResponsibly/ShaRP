from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator


class BaseQoI(BaseEstimator, metaclass=ABCMeta):
    """
    Base class to implement Quantities of Interest (QoI) for classification or regression
    tasks. It should not be used directly. Any QoI must define at least 2 functions:

      :func:`qoi.estimate`:
          A function that predicts the model's result for one or multiple items.
      :func:`qoi.calculate`:
          A function that applies a difference metric to the predictions of two items, or
          two sets of items. If sets, then the mean difference is returned.

    Parameters
    ----------
    target_function : function
        Method used to predict a label or score. The output of this function
        should be a 1-dimensional array with the expected target (i.e., label or score)
        for each of the passed observations.
    """

    _qoi_type = "score"

    def __init__(self, target_function=None, X=None):
        # NOTE: This parameter name is not descriptive enough
        self.target_function = target_function
        self.X = X

    def estimate(self, rows):
        """
        Prepares and runs ``self.target_function`` for a set of samples.

        Parameters
        ----------
        rows : array-like of shape (n_samples, n_features)
            Samples over which ``target_function`` will be applied.

        Returns
        -------
        target_output : np.ndarray of shape (n_samples,)
            Label predictions or score estimations.
        """
        return self._estimate(rows)

    def calculate(self, rows1, rows2):
        """
        Calculates the influence score based on the ``target_function`` outputs for
        ``rows1`` and ``rows2``.

        Parameters
        ----------
        rows1 : array-like of shape (n_samples, n_features)
            First set of samples to compare.

        rows2 : array-like of shape (n_samples, n_features)
            Second set of samples to compare.

        Returns
        -------
        influence_score : float
            Influence score for ``rows1``, compared to ``rows2``.
        """
        return self._calculate(rows1, rows2)

    @abstractmethod
    def _estimate(self, rows):
        pass

    @abstractmethod
    def _calculate(self, rows1, rows2):
        pass


class BaseRankingQoI(BaseQoI, metaclass=ABCMeta):
    """
    Base class to implement Quantities of Interest (QoI) for ranking tasks. It should not
    be used directly. Any QoI must define at least 2 functions:

      :func:`qoi.estimate`:
          A function that predicts the model's result for one or multiple items.
      :func:`qoi.calculate`:
          A function that applies a difference metric to the predictions of two items, or
          two sets of items. If sets, then the mean difference is returned.

    Parameters
    ----------
    target_function : function
        Method used to calculate a score. The output of this function
        should be a 1-dimensional array with the expected target (i.e., label, score, or
        ranking) for each of the passed observations.

    X : array-like of shape (n_samples, n_features)
        The input samples.
    """

    _qoi_type = "rank"

    def rank(self, X, X_base=None):
        """
        Computes the ranking of samples ``X`` within a dataset ``X_base``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to calculate the rankings for.

        Returns
        -------
        rankings : np.ndarray of shape (n_samples,)
            Rankings for all samples in ``X``.
        """

        # Retrieve reference dataset and calculate scores to get rankings for
        if X_base is None and self.X is None:
            msg = (
                "Either ``self.X`` (defined in the object initialization) or ``X_base`` "
                "(a paremeter from the ``rank`` method) must be defined."
            )
            raise ValueError(msg)
        elif X_base is None:
            X_base = self.X

            if not hasattr(self, "_scores_base"):
                self._scores_base = self.target_function(X_base)

            scores_base = deepcopy(self._scores_base)
        else:
            scores_base = self.target_function(X_base)

        ranks_all = []
        for row in X:
            score = self.target_function(row.reshape(1, -1))[0]
            rank = (scores_base > score).sum() + 1
            ranks_all.append(rank)

        return np.array(ranks_all)
