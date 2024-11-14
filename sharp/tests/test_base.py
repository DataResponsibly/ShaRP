"""
Tests code in `base.py`.
"""

import numpy as np
from sharp import ShaRP


def test_default_qoi():
    """
    Reproduces issue #44: Defining ShaRP without an explicit QoI raises an AttributeError
    """
    _X = np.random.random((100, 3))
    sharp = ShaRP(target_function=lambda x: x.sum(axis=1))
    sharp.fit(_X)
    sharp.all(_X[:5])
