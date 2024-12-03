import pytest
from sharp.utils._utils import _optional_import


def test_optional_import_existing_module():
    module = _optional_import("math")
    assert module is not None


def test_optional_import_non_existing_module():
    with pytest.raises(ImportError):
        _optional_import("non_existing_module")


def test_optional_import_partial_module():
    module = _optional_import("os.path")
    assert module is not None
