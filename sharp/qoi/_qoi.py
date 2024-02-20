from .base import BaseQoI, BaseRankingQoI


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


class RankingQoI(BaseRankingQoI):
    """
    Ranking specific QoI. Uses rank as the quantity being measured. The influence score
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


class RankingScoreQoI(BaseRankingQoI):
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


class TopKQoI(BaseRankingQoI):
    """
    Ranking-specific QoI. Estimates the likelihood of reaching the top-K as the
    quantity of interest.

    Parameters
    ----------
    top_k : int, default=10
        The number of items to consider as part of the top-ranked group.
    """

    def __init__(self, target_function=None, top_k=10, X=None):
        super().__init__(target_function=target_function, X=X)
        self.top_k = top_k

    def _estimate(self, rows):
        ranks = self.rank(rows)
        return (ranks <= self.top_k).astype(int)

    def _calculate(self, rows1, rows2):
        return (self.estimate(rows1) - self.estimate(rows2)).mean()


QOI_OBJECTS = {
    "diff": DiffQoI,
    "flip": FlipQoI,
    "likelihood": LikelihoodQoI,
    "ranking": RankingQoI,
    "ranking_score": RankingScoreQoI,
    "top_k": TopKQoI,
}
