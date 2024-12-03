import pytest
from sharp.utils._utils import _optional_import, parallel_loop


def test_optional_import_existing_module():
    module = _optional_import("math")
    assert module is not None


def test_optional_import_non_existing_module():
    with pytest.raises(ImportError):
        _optional_import("non_existing_module")


def test_optional_import_partial_module():
    module = _optional_import("os.path")
    assert module is not None


def test_parallel_loop():
    def square(x):
        return x * x

    iterable = range(10)

    # Test parallel loop without progress bar
    results = parallel_loop(square, iterable, n_jobs=2, progress_bar=False)
    assert results == [x * x for x in iterable]

    # Test parallel loop with progress bar
    results = parallel_loop(
        square, iterable, n_jobs=2, progress_bar=True, description="Test"
    )
    assert results == [x * x for x in iterable]
