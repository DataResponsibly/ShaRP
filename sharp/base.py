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

TODO: Check params functions (data/object types and such)
TODO: Parallelization/Vectorization
TODO: Ensure inputs are converted to numpy arrays
"""
import numpy as np
from sklearn.utils import check_random_state
from .utils import check_feature_names, check_inputs, check_measure, check_qoi
from .visualization._visualization import ShaRPViz


class ShaRP:
    """
    Explains the contributions of features to different aspects of a ranked outcome,
    based on Shapley values.

    This algorithm is an implementation of Shapley for Rankings and Preferences (ShaRP),
    as presented in [1]_.

    If QoI is None, ``target_function`` and parameters ``X`` and ``y`` need to be passed.
    if QoI is not None, ``target_function`` is ignored.

    Parameters
    ----------
    estimator : ML classifier

    qoi : Quantity of interest

    measure : measure used to estimate feature contributions (unary, set, banzhaf, etc.)

    sample_size : amount of perturbations applied per data point

    predict_method : estimator's function that provides inference

    random_state : random seed

    X : reference input

    y : target

    Attributes
    ----------
    TODO

    Notes
    -----
    See the original paper: [1]_ for more details.

    References
    ----------
    .. [1] V. Pliatsika, J. Fonseca, T. Wang, J. Stoyanovich, "ShaRP: Explaining
       Rankings with Shapley Values", Under submission.
    """

    def __init__(
        self,
        qoi=None,
        target_function=None,
        measure="shapley",
        sample_size=32,
        random_state=None,
        **kwargs
    ):
        self.qoi = qoi
        self.target_function = target_function
        self.measure = measure
        self.sample_size = sample_size
        self.random_state = random_state
        self.plot = ShaRPViz(self)
        self._X = kwargs["X"] if "X" in kwargs.keys() else None
        self._y = kwargs["y"] if "y" in kwargs.keys() else None

    def _check_params(self, X, y):  # TODO: refactor _check_params
        if X is None:
            X = self._X

        if y is None:
            y = self._y

        feature_names = check_feature_names(X)

        X_, y_ = check_inputs(X, y)
        measure_ = check_measure(self.measure)
        qoi_ = check_qoi(
            self.qoi,
            target_function=self.target_function,
            X=X_,
        )

        if not hasattr(self, "_rng"):
            self._rng = check_random_state(self.random_state)

        return feature_names, X_, y_, measure_, qoi_

    def get_params(self):
        pass  # TODO

    def set_params(self):
        pass  # TODO

    def individual(self, sample, X=None, y=None, **kwargs):
        """
        set_cols_idx should be passed in kwargs if measure is marginal
        """
        feature_names, X_, y_, measure_, qoi_ = self._check_params(X, y)

        if "set_cols_idx" in kwargs.keys():
            set_cols_idx = kwargs["set_cols_idx"]
        else:
            set_cols_idx = None

        if isinstance(sample, int):
            sample = X_[sample]

        influences = []
        for col_idx in range(len(feature_names)):
            cell_influence = measure_(
                row=sample,
                col_idx=col_idx,
                set_cols_idx=set_cols_idx,
                X=X_,
                qoi=qoi_,
                sample_size=self.sample_size,
                rng=self._rng,
            )
            influences.append(cell_influence)
        return influences

    def feature(self, feature, X=None, y=None, **kwargs):
        """
        set_cols_idx should be passed in kwargs if measure is marginal
        """
        feature_names, X_, y_, measure_, qoi_ = self._check_params(X, y)
        col_idx = feature_names.index(feature) if type(feature) is str else feature

        if "set_cols_idx" in kwargs.keys():
            set_cols_idx = kwargs["set_cols_idx"]
        else:
            set_cols_idx = None

        influences = []
        for sample_idx in range(X_.shape[0]):
            sample = X_[sample_idx]
            cell_influence = measure_(
                row=sample,
                col_idx=col_idx,
                set_cols_idx=set_cols_idx,
                X=X_,
                qoi=qoi_,
                sample_size=self.sample_size,
                rng=self._rng,
            )
            influences.append(cell_influence)

        return np.mean(influences)

    def all(self, X=None, y=None, **kwargs):
        """
        set_cols_idx should be passed in kwargs if measure is marginal
        """
        X_, y_ = check_inputs(X, y)

        influences = []
        for sample_idx in range(X_.shape[0]):
            individual_influence = self.individual(sample_idx, X_, **kwargs)
            influences.append(individual_influence)

        return np.array(influences)

    def pairwise(self, sample1, sample2, **kwargs):
        """
        Compare two samples, or one sample against a set of samples.

        set_cols_idx should be passed in kwargs if measure is marginal
        """
        if "X" in kwargs.keys():
            X = kwargs["X"]

            if type(sample1) in [int, list]:
                sample1 = X[sample1]

            if type(sample2) in [int, list]:
                sample2 = X[sample2]

        sample2 = sample2.reshape(1, -1) if sample2.ndim == 1 else sample2

        return self.individual(sample1, X=sample2, **kwargs)
