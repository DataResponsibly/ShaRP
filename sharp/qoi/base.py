from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
from multiprocessing import Manager


class TargetFunctionWrapper:

    def __init__(self, target_function):
        self.target_function = target_function
        self._hashmap = Manager().dict()

    def _check_hash(self, row):
        row_str = str(row)
        try:
            return self._hashmap[row_str]
        except KeyError:
            pred = self.target_function(row)
            self._hashmap[row_str] = pred
            return pred

    def __call__(self, rows):
        results = [self._check_hash(row.reshape(1, -1)) for row in rows]
        return np.array(results).reshape(-1)


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

    def __init__(self, target_function=None, X=None, cache=True):
        self.target_function = (
            TargetFunctionWrapper(target_function) if cache else target_function
        )
        self.cache = cache
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


class BaseRankQoI(BaseQoI, metaclass=ABCMeta):
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
    _X_sorted = False

    def sort_base(self):
        idx_sorted = np.argsort(self.target_function(self.X))
        self.X = self.X[idx_sorted]
        self._X_sorted = True
        return self

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
            if not self._X_sorted:
                self.sort_base()

            if not hasattr(self, "_scores_base"):
                self._scores_base = self.target_function(self.X)

            X_base = self.X
            scores_base = self._scores_base

            scores = self.target_function(X)
            ranks_all = scores_base.shape[0] - np.searchsorted(
                scores_base, scores, side="left"
            )
            return ranks_all
        else:
            scores_base = self.target_function(X_base)

            ranks_all = []
            for row in X:
                score = self.target_function(row.reshape(1, -1))[0]
                rank = (scores_base > score).sum() + 1
                ranks_all.append(rank)

            return np.array(ranks_all)
