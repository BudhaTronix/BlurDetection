from __future__ import annotations

import numpy as np

from Code.Utils.utils import returnClass


def test_return_class_binning() -> None:
    values = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 1.0], dtype=float)
    result = returnClass(4, values.copy())
    assert np.array_equal(result, np.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0]))


def test_return_class_handles_out_of_range_values() -> None:
    values = np.array([-0.3, 1.4], dtype=float)
    result = returnClass(4, values.copy())
    assert np.array_equal(result, np.array([0.0, 3.0]))
