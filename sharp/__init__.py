"""
Implementation of the ShaRP algorithm (Shapley for Rankings and Preferences).

``sharp`` is a library containing the implementation of the ShaRP
algorithm (Shapley for Rankings and Preferences), a framework that can be used
to explain the contributions of features to different aspects of a ranked
outcome, based on Shapley values.

Subpackages
-----------
qoi
utils
visualization
"""

import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sharp when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SHARP_SETUP__'
    __SHARP_SETUP__  # type: ignore
except NameError:
    __SHARP_SETUP__ = False

if __SHARP_SETUP__:
    sys.stderr.write("Partial import of imblearn during the build process.\n")
    # We are not importing the rest of sharp during the build
    # process, as it may not be compiled yet
else:
    # from . import utils

    from ._version import __version__
    from .base import ShaRP

    __all__ = ["__version__", "ShaRP"]
