# Adapted from:
# https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/tests/test_docstring_parameters.py

import inspect
import warnings
import importlib
from pkgutil import walk_packages
from inspect import signature

import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils import IS_PYPY
from sklearn.utils._testing import check_docstring_parameters
from sklearn.utils._testing import _get_func_name
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y
from sklearn.utils.estimator_checks import _enforce_estimator_tags_X
from sklearn.utils._test_common.instance_generator import _construct_instances
from sklearn.utils.deprecation import _is_deprecated

import sharp
from sharp.qoi.base import BaseQoI
from sharp.utils._testing import all_elements


def is_qoi(estimator):
    """Return True if the given estimator is a QoI, False otherwise.

    Parameters
    ----------
    estimator : object
        Estimator to test.

    Returns
    -------
    is_sampler : bool
        True if estimator is a QoI, otherwise False.
    """
    if issubclass(estimator, BaseQoI):
        return True
    return False


# walk_packages() ignores DeprecationWarnings, now we need to ignore
# FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    sharp_path = sharp.__path__
    PUBLIC_MODULES = set(
        [
            pckg[1]
            for pckg in walk_packages(prefix="sharp.", path=sharp_path)
            if not ("._" in pckg[1] or ".tests." in pckg[1])
        ]
    )

# functions to ignore args / docstring of
_DOCSTRING_IGNORES = []  # type: ignore

# Methods where y param should be ignored if y=None by default
_METHODS_IGNORE_NONE_Y = [
    "fit",
    "score",
    "fit_predict",
    "fit_transform",
    "partial_fit",
    "predict",
]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipif(IS_PYPY, reason="test segfaults on PyPy")
def test_docstring_parameters():  # noqa: C901
    # Test module docstring formatting

    # Skip test if numpydoc is not found
    pytest.importorskip(
        "numpydoc", reason="numpydoc is required to test the docstrings"
    )

    from numpydoc import docscrape

    incorrect = []
    for name in PUBLIC_MODULES:
        if name.endswith(".conftest"):
            # pytest tooling, not part of the scikit-learn API
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        # Exclude non-scikit-learn classes
        classes = [cls for cls in classes if cls[1].__module__.startswith("sharp")]
        for cname, cls in classes:
            this_incorrect = []
            if cname in _DOCSTRING_IGNORES or cname.startswith("_"):
                continue
            if inspect.isabstract(cls):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError(
                    "Error for __init__ of %s in %s:\n%s" % (cls, name, w[0])
                )

            cls_init = getattr(cls, "__init__", None)

            if _is_deprecated(cls_init):
                continue
            elif cls_init is not None:
                this_incorrect += check_docstring_parameters(cls.__init__, cdoc)

            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                if _is_deprecated(method):
                    continue
                param_ignore = None
                # Now skip docstring test for y when y is None
                # by default for API reason
                if method_name in _METHODS_IGNORE_NONE_Y:
                    sig = signature(method)
                    if "y" in sig.parameters and sig.parameters["y"].default is None:
                        param_ignore = ["y"]  # ignore y for fit and score
                result = check_docstring_parameters(method, ignore=param_ignore)
                this_incorrect += result

            incorrect += this_incorrect

        functions = inspect.getmembers(module, inspect.isfunction)
        # Exclude imported functions
        functions = [fn for fn in functions if fn[1].__module__ == name]
        for fname, func in functions:
            # Don't test private methods / functions
            if fname.startswith("_"):
                continue
            if fname == "configuration" and name.endswith("setup"):
                continue
            name_ = _get_func_name(func)
            if not any(d in name_ for d in _DOCSTRING_IGNORES) and not _is_deprecated(
                func
            ):
                incorrect += check_docstring_parameters(func)

    msg = "\n".join(incorrect)
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error:\n" + msg)


@ignore_warnings(category=FutureWarning)
def test_tabs():
    # Test that there are no tabs in our source files
    for importer, modname, ispkg in walk_packages(sharp.__path__, prefix="sharp."):
        if IS_PYPY:
            continue

        # because we don't import
        mod = importlib.import_module(modname)

        try:
            source = inspect.getsource(mod)
        except IOError:  # user probably should have run "make clean"
            continue
        assert "\t" not in source, (
            '"%s" has tabs, please remove them ',
            "or add it to the ignore list" % modname,
        )


def _construct_compose_pipeline_instance(Estimator):
    # Minimal / degenerate instances: only useful to test the docstrings.
    if Estimator.__name__ == "Pipeline":
        return Estimator(steps=[("clf", LogisticRegression())])


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name, Obj", all_elements())
def test_fit_docstring_attributes(name, Obj):
    pytest.importorskip("numpydoc")
    from numpydoc import docscrape

    if Obj.__name__ in _DOCSTRING_IGNORES:
        return

    doc = docscrape.ClassDoc(Obj)
    attributes = doc["Attributes"]

    est = next(_construct_instances(Obj))

    X, y = make_classification(
        n_samples=20,
        n_features=3,
        n_redundant=0,
        n_classes=2,
        random_state=2,
    )

    y = _enforce_estimator_tags_y(est, y)
    X = _enforce_estimator_tags_X(est, X)

    if "oob_score" in est.get_params():
        est.set_params(oob_score=True)

    if is_qoi(Obj) or Obj.__name__ == "ShaRP":
        est.X = X
        est.target_function = (
            lambda X: sum([X[:, i] for i in range(X.shape[1])]) / X.shape[1]
        )
    else:
        raise TypeError(f"Object type not recognized: {type(est)}")

    skipped_attributes = set([])

    for attr in attributes:
        if attr.name in skipped_attributes:
            continue
        desc = " ".join(attr.desc).lower()
        # As certain attributes are present "only" if a certain parameter is
        # provided, this checks if the word "only" is present in the attribute
        # description, and if not the attribute is required to be present.
        if "only " in desc:
            continue
        # ignore deprecation warnings
        with ignore_warnings(category=FutureWarning):
            assert hasattr(est, attr.name)

    fit_attr = _get_all_fitted_attributes(est)
    fit_attr_names = [attr.name for attr in attributes]
    undocumented_attrs = set(fit_attr).difference(fit_attr_names)
    undocumented_attrs = set(undocumented_attrs).difference(skipped_attributes)
    if undocumented_attrs:
        raise AssertionError(
            f"Undocumented attributes for {Obj.__name__}: {undocumented_attrs}"
        )


def _get_all_fitted_attributes(estimator):
    "Get all the fitted attributes of an estimator including properties"
    # attributes
    fit_attr = list(estimator.__dict__.keys())

    # properties
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)

        for name in dir(estimator.__class__):
            obj = getattr(estimator.__class__, name)
            if not isinstance(obj, property):
                continue

            # ignore properties that raises an AttributeError and deprecated
            # properties
            try:
                getattr(estimator, name)
            except (AttributeError, FutureWarning):
                continue
            fit_attr.append(name)

    return [k for k in fit_attr if k.endswith("_") and not k.startswith("_")]
