"""
Base object used to set up explainability objects.

Topics that must be covered:
- Single-observation explanation
- Global-input explanation
  * Set qii
  * Unary qii
  * Marginal qii
  * Shapley
  * Banzhaff

Scores or Bool?

TODO: Check params functions (data/object types and such)
TODO: Parallelization
"""
from abc import ABC, abstractmethod
from utils.validation import check_feature_names, check_inputs, check_measure
from ._visualization import BaseVisualization


class BaseQII(ABC):
    """
    This is a base class, serves as a template for QII objects. It should not be called
    directly.

    Parameters
    ----------
    estimator : ML classifier
    qoi : Quantity of interest
    X : reference input
    y : target
    measure : measure used to estimate feature contributions (unary, set, banzhaf, etc.)
    sample_size : amount of perturbations applied per data point
    predict_method : estimator's function that provides inference
    random_state : random seed
    """

    def __init__(
        self,
        estimator,
        qoi,
        X=None,  # NOTE: might be removed
        y=None,
        measure="unary",
        sample_size=32,  # TODO: Set to None and predefine according to QII paper!
        predict_method="predict",
        random_state=None,
    ):
        self.estimator = estimator
        self.qoi = qoi
        self._X = X
        self._y = y
        self.measure = measure
        self.sample_size = sample_size
        self.predict_method = predict_method
        self.random_state = random_state
        self.plot = BaseVisualization()

    def _predict(self, X):
        if hasattr(self.estimator, self.predict_method):
            return getattr(self.estimator, self.predict_method)(X)
        else:
            raise AttributeError("Attribute bla bla not found")  # TODO

    def get_params(self):
        pass  # TODO

    def set_params(self):
        pass  # TODO

    @abstractmethod
    def individual(self, X=None, y=None):
        pass

    @abstractmethod
    def all(self, X=None, y=None):
        pass


class TabularQII(BaseQII):
    """
    QII object for classification tasks over tabular data.
    """

    def _check_params(self, X, y):
        if self._X is not None:
            reference_X_ = self._X
            f_ = check_feature_names(reference_X_)

        feature_names = check_feature_names(X)
        if f_ != feature_names:
            raise KeyError("Features in reference X and target X do not match.")

        tested_X_ = check_inputs(X)
        measure_ = check_measure(self.measure)

        return feature_names, reference_X_, tested_X_, measure_

    def individual(self, X=None, y=None):
        feature_names, reference_X_, X_, measure_ = self._check_params(X, y)
        X

    def feature(self, feature, X=None, y=None):
        feature_names, reference_X_, X_, measure_ = self._check_params(X, y)

    def all(self, X=None, y=None):
        X = check_inputs(X)
