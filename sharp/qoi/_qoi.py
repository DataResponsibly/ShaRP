import copy
from .base import BaseQoI, BaseRankQoI


class DiffQoI(BaseQoI):
    """
    A general QoI, suitable for models/methods that output label predictions or scores.
    ``target_function`` can output either scores or binary labels.

    Parameters
    ----------
    target_function : function
        Method used to predict a label or score. The output of this function
        should be a 1-dimensional array with the expected target (i.e., label or score)
        for each of the passed observations.

    Notes
    -----
    This QoI was formerly defined as just ``QoI``.
    """

    def _estimate(self, rows):
        return self.target_function(rows)

    def _calculate(self, rows1, rows2):
        return (self.estimate(rows1) - self.estimate(rows2)).mean()


class FlipQoI(BaseQoI):
    """
    Implements equation 4 from [1]_. This QoI is designed for classification, using label
    predictions. Although it was originally intended for binary classification,
    multiclass problems may be quantified directly using this QoI. This QoI's influence
    score quantifies how "pivotal" a given feature is. ``target_function`` should output
    class predictions.

    References
    ----------
    .. [1] Datta, A., Sen, S., & Zick, Y. (2016). Algorithmic transparency via
        quantitative input influence: Theory and experiments with learning systems. In
        2016 IEEE symposium on security and privacy (SP) (pp. 598-617). IEEE.

    Notes
    -----
    This QoI was formerly defined as ``BCFlipped``.
    """

    def _estimate(self, rows):
        return self.target_function(rows)

    def _calculate(self, rows1, rows2):
        y_pred1 = self.estimate(rows1)
        y_pred2 = self.estimate(rows2)
        return 1 - (y_pred2 == y_pred1).mean()


class LikelihoodQoI(BaseQoI):
    """
    Implements equation 3 from [1]_. This QoI is designed for binary classification
    problems only.  It calculates the difference between the likelihoods for ``rows1``
    and ``rows2`` to obtain the positive label. ``target_function`` should output either
    scores or class label predictions.

    References
    ----------
    .. [1] Datta, A., Sen, S., & Zick, Y. (2016). Algorithmic transparency via
        quantitative input influence: Theory and experiments with learning systems. In
        2016 IEEE symposium on security and privacy (SP) (pp. 598-617). IEEE.

    Notes
    -----
    This QoI was formerly defined as ``BCLikelihood``.
    """

    def _estimate(self, rows):
        y_pred = self.target_function(rows)  # .squeeze()
        y_pred_mean = (y_pred if y_pred.ndim == 1 else y_pred[:, -1]).mean()
        return y_pred_mean

    def _calculate(self, rows1, rows2):
        return self.estimate(rows1) - self.estimate(rows2)  # .mean()


class RankQoI(BaseRankQoI):
    """
    Rank specific QoI. Uses rank as the quantity being measured. The influence score
    is based on the comparison between the rank of a sample and synthetic data (based on
    the original sample). ``target_function`` should output scores.

    Notes
    -----
    This QoI was formerly defined as ``RankingRank``.
    """

    def _estimate(self, rows):
        return self.rank(rows)

    def _calculate(self, rows1, rows2):
        return (self.estimate(rows2) - self.estimate(rows1)).mean()


class RankScoreQoI(BaseRankQoI):
    """
    A general, ranking-oriented QoI, similar to ``DiffQoI``. ``target_function`` must
    output scores.

    Notes
    -----
    This QoI was formerly defined as ``RankingScore``.
    """

    def _estimate(self, rows):
        return self.target_function(rows)

    def _calculate(self, rows1, rows2):
        return (self.estimate(rows1) - self.estimate(rows2)).mean()


class TopKQoI(BaseRankQoI):
    """
    Rank-specific QoI. Estimates the likelihood of reaching the top-K as the
    quantity of interest.

    Parameters
    ----------
    top_k : int, default=10
        The number of items to consider as part of the top-ranked group.
    """

    def __init__(self, target_function=None, top_k=10, X=None, cache=True):
        super().__init__(target_function=target_function, X=X, cache=cache)
        self.top_k = top_k

    def _estimate(self, rows):
        ranks = self.rank(rows)
        return (ranks <= self.top_k).astype(int)

    def _calculate(self, rows1, rows2):
        return (self.estimate(rows1) - self.estimate(rows2)).mean()


_QOI_OBJECTS = {
    "diff": DiffQoI,
    "flip": FlipQoI,
    "likelihood": LikelihoodQoI,
    "rank": RankQoI,
    "rank_score": RankScoreQoI,
    "top_k": TopKQoI,
}


def get_qoi_names():
    """Get the names of all available quantities of interest.

    These names can be passed to :func:`~sharp.qoi.get_qoi` to
    retrieve the QoI object.

    Returns
    -------
    list of str
        Names of all available quantities of interest.

    Examples
    --------
    >>> from sharp.qoi import get_qoi_names
    >>> all_qois = get_qoi_names()
    >>> type(all_qois)
    <class 'list'>
    >>> all_qois[:3]
    ['diff', 'flip', 'likelihood']
    >>> "ranking" in all_qois
    True
    """
    return sorted(_QOI_OBJECTS.keys())


def get_qoi(qoi):
    """Get a quantity of interest from string.

    :func:`~sharp.qoi.get_qoi_names` can be used to retrieve the names
    of all available quantities of interest.

    Parameters
    ----------
    qoi : str, callable or None
        Quantity of interest as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    quantity : callable
        The quantity of interest.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_qoi` twice for the same scorer results in two
    separate QoI objects.
    """
    if isinstance(qoi, str):
        try:
            quantity = copy.deepcopy(_QOI_OBJECTS[qoi])
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sklearn.metrics.get_scorer_names() "
                "to get valid options." % qoi
            )
    else:
        quantity = qoi
    return quantity
