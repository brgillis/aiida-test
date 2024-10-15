"""Unit test of linear regression with errors function
"""

import numpy as np
from samples.wlr import linregress_with_errors

def test_linregress_with_errors_simple():
    """Unit test of linregress_with_errors.
    """

    # Some test input
    x = np.array([1, 2, 4, 8, 7])
    y = np.array([10, 11, 9, 7, 12])
    y_err = np.array([0.1, 0.2, 0.1, 0.2, 0.4])

    # Calculate the regression
    weighted_results = linregress_with_errors(x, y, y_err)

    # Test they match expectations
    assert np.isclose(weighted_results.slope, -0.34995112414467133)
    assert np.isclose(weighted_results.intercept, 10.54740957966764)
    assert np.isclose(weighted_results.slope_err, 0.028311906106274067)
    assert np.isclose(weighted_results.intercept_err, 0.1076724332578932)
    assert np.isclose(
        weighted_results.slope_intercept_covar, -0.0024828934506353857)
