from .ranking import Ranking
from itertools import combinations
import pandas as pd


class QoI:
    """
    General Quantity of Interest (QoI) class.
    It is suitable for models that have a predict function.

    This class consists of 4 functions. Details for each will be given below. They are:
      :func:`qoi.__init__`:
          the constructor.
      :func:`qoi.estimate`:
          a function that predicts the model's result for one multiple items.
      :func:`qoi.calculate`:
          a function that applies a difference metric to the predictions of two or two sets of items. If sets, then the
          mean difference is returned.
      :func:`qoi.pairwise`:
          a function that applies :func:`qoi.calculate` on two different pairs of items and returns the difference.
    """
    def __init__(self, model):
        """
        Constructor of general QoI.

        Parameters
        ----------
          model [scikit-learn model]:
              scikit-learn model or any model that has a predict function.
        """
        self.model = model

    def estimate(self, rows):
        """
        Main function of the class, here we acquire the model's prediction for one or multiple rows of items.
        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [list]:
              Predictions for all rows.
        """
        return self.model.predict(rows)

    def calculate(self, q1, q2):
        """
        Calculates the mean numerical difference between the predictions for q1 and the predictions for q2.

        Parameters
        ----------
          q1 [pandas.series]:
              Dataframe rows that are first in the subtraction.
          q2 [pandas.series]:
              Dataframe rows that are second in the subtraction.
        Returns
        -------
          return [float]:
              Mean difference between the predictions for the rows of q1 and the rows of q2.
        """
        return (self.estimate(q1) - self.estimate(q2)).mean()


class BCFlipped(QoI):
    """
    This class is implementing eq 4 from the paper cited below. This subclass is extending QoI class for
    classification specifically. It models a new QoI for BC, the probability that the labels changed.
    In other words it's 1 minus the probability that the labels remained the same. In the paper, this is
    interpreted as how "pivotal" the feature we are currently examining is.

    Datta, A.; Sen, S.; and Zick, Y. 2016. Algorithmic transparency via quantitative input influence: Theory and
    experiments with learning systems. In 2016 IEEE symposium on security and privacy (SP), 598–617. IEEE.
    """
    def calculate(self, q1, q2):
        """
        Calculates the expectation of the probability that the q2 set of points changed labels compared
        to q1. This indicates how "pivotal" the current feature was for the current coalition.

        Parameters
        ----------
          q1 [pandas.series]:
              Dataframe rows that are first in the subtraction.
          q2 [pandas.series]:
              Dataframe rows that are second in the subtraction.
        Returns
        -------
          return [float]:
              Probability that the label of q2 changed compared to q1
        """
        y_pred_mod = self.estimate(q2)
        y_pred = self.estimate(q1)
        return 1 - (y_pred_mod == y_pred).astype(int).mean()


class BCLikelihood(QoI):
    """
    This class is implementing eq 3 from the paper cited below. This subclass is extending QoI class for
    classification specifically. It models a new QoI, the probability that q1 received a label minus the
    probability that the corresponding q2 points did.

    Datta, A.; Sen, S.; and Zick, Y. 2016. Algorithmic transparency via quantitative input influence: Theory and
    experiments with learning systems. In 2016 IEEE symposium on security and privacy (SP), 598–617. IEEE.
    """
    def __init__(self, model, label):
        """
        Constructor. For this QoI in addition to the model, we also need the label that we are checking the
        probabilities for.

        Parameters
        ----------
          model [scikit-learn model]:
              scikit-learn model or any model that has a predict function.
          #TODO str or int?
          label []:
              label of the class for which we are calculating the difference in probabilities
        """
        self.model = model
        self.label = label

    def estimate(self, rows):
        """
        Estimation method for this class. Here we don't want the class label but instead the probability of
        being in that class.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [float]:
              Average probability that the rows given are in the class of the label.
        """
        return self.model.predict_proba(rows)[:, self.label].mean()

# TODO: MAYBE make another base class that is a subclass of your new one and make all the below its subclasses.
class RankingRank(QoI):
    """
    Ranking specific QoI. This QoI is using rank as the quantity being measured. The rank of the original
    datapoint and the Frankestein datapoint are compared.

    We need to change all functions of the base QoI for multiple reasons. First, we want to calculate the
    ranking once (in the constructor) and save it in a raning.Ranking variable. Secondly, we are defining our
    own ranking class and therefore the estimation changes. Finally, the calculation requires us to reverse
    the order of the original and the Frankestein points because unlike the general case, smaller numbers
    are better in the case of ranking.
    """
    def __init__(self, dataset, ranking_function):
        """
        Constructor. Stores the ranking in a variable of type [ranking.Ranking], so we won't have to apply
        the function everytime we rank some new item.

        Parameters
        ----------
          dataset [pandas.dataframe]:
              Dataframe that we want to sort
          #TODO type below
          ranking_function []:
              Function that we will pass to our ranking class. It is applied to one datapoint of the dataframe.
        """
        self.ranking = Ranking(ranking_function, dataset)

    def estimate(self, rows):
        """
        Uses the ranking from the class to estimate the ranks of the series in the input.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [int, list]:
              Predicted rank per row.
        """
        return self.ranking.predict_rank(rows)

    def calculate(self, q1, q2):
        """
        Calculates the average rank difference between q2 and q1 Frankenstein points. The significant
        difference compared to the rest of the classes is that here q2 and q1 are reversed. The reason
        is that for the QoI rank, smaller values are better.

        Parameters
        ----------
         q1 [pandas.series]:
              Dataframe rows that are first in the subtraction.
          q2 [pandas.series]:
              Dataframe rows that are second in the subtraction.

        Returns
        -------
          return [float]:
              Average rank difference between the predicted ranks of q2 and the predicted ranks for q1.
        """
        return (self.estimate(q2) - self.estimate(q1)).mean()

    def get_ranking(self):
        """
        Returns the ranking of all points in the original dataset.

        Returns
        -------
          return [pandas.series]:
              Ranks for all the points in the RankingRank.dataset
        """
        return self.ranking.get_all()


class RankingScore(QoI):
    """
    Ranking specific QoI. This QoI is using score as the quantity being measured. While this QoI is similar to
    the standard Shapley values, we still need to use the ranking.Ranking class to use the ranking function (or
    at least pass the function). Here we use the ranking class.

    We need to change some functions of the base QoI for two reasons. First, we want to calculate the ranking
    once (in the constructor) and save it in a raning.Ranking variable. Finally, we are defining our  own
    ranking class and therefore the estimation changes.
    """
    def __init__(self, dataset, ranking_function):
        """
        Uses the ranking from the class to estimate the ranks of the series in the input.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [int, list]:
              Predicted rank per row.
        """
        self.ranking = Ranking(ranking_function, dataset)

    def estimate(self, rows):
        """
        Uses the ranking from the class to estimate the scores of the series in the input.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [float, list]:
              Predicted score per row.
        """
        return self.ranking.predict_score(rows)

    def get_ranking(self):
        """
        Returns the ranking of all points in the original dataset.

        Returns
        -------
          return [pandas.series]:
              Ranks for all the points in the RankingRank.dataset
        """
        return self.ranking.get_all()


class RankingTopK(QoI):
    """
    Ranking specific QoI. This QoI is using the probability of being in the topK as the quantity being measured.
    We are requiring another input variable, the number k of items being selected. We use the ranking.Ranking
    class to calculate first the rank and then whether the items being estimated pass the threshold k.

    We need to change some functions of the base QoI for two reasons. First, we want to calculate the ranking
    once (in the constructor) and save it in a raning.Ranking variable. Also, save k. Finally, we are defining
    our own ranking class and therefore the estimation changes.
    """
    def __init__(self, dataset, ranking_function, k):
        """
        Uses the ranking from the class to estimate the ranks of the series in the input.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [int, list]:
              Predicted rank per row.
        """
        self.ranking = Ranking(ranking_function, dataset)
        self.k = k

    def estimate(self, rows):
        """
        Uses the ranking from the class to estimate whether of each item in the series is in the topK.

        Parameters
        ----------
          rows [pandas.series]:
              Dataframe rows on which we want to use the model.

        Returns
        -------
          return [int, list]:
              Whether the item/items in the input atre in the topK
        """
        ranks = self.ranking.predict_rank(rows)
        return (ranks <= self.k).astype(int)

    def get_ranking(self):
        """
        Returns the ranking of all points in the original dataset.

        Returns
        -------
          return [pandas.series]:
              Ranks for all the points in the RankingRank.dataset
        """
        return self.ranking.get_all()

