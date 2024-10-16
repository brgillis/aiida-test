"""Unit test of linear regression with errors function
"""

import aiida
import numpy as np
from calc_functions.wlr_functions import wf_linregress_with_errors
from samples.wlr import linregress_with_errors

# Some test input
x = np.array([1., 2, 4, 8, 7])
y = np.array([10., 11, 9, 7, 12])
y_err = np.array([0.1, 0.2, 0.1, 0.2, 0.4])

def test_linregress_with_errors_simple():
    """Unit test of linregress_with_errors.
    """

    # Calculate the regression
    results = linregress_with_errors(x, y, y_err)

    # Test they match expectations
    assert np.isclose(results.slope, -0.34995112414467133)
    assert np.isclose(results.intercept, 10.54740957966764)
    assert np.isclose(results.slope_err, 0.028311906106274067)
    assert np.isclose(results.intercept_err, 0.1076724332578932)
    assert np.isclose(results.slope_intercept_covar, -0.0024828934506353857)

def test_wf_linregress_with_errors():
    """Unit test of wf_linregress_with_errors to ensure it returns the same
    result as the normal function.
    """

    # Load the appropriate AiiDA profile
    aiida.load_profile("presto")

    # Calculate with the two different methods
    results = linregress_with_errors(x, y, y_err)
    wf_results = wf_linregress_with_errors(x, y, y_err)

    # Test it's all the same between the two methods
    assert np.isclose(results.slope, wf_results["slope"])
    assert np.isclose(results.intercept, wf_results["intercept"])
    assert np.isclose(results.slope_err, wf_results["slope_err"])
    assert np.isclose(results.intercept_err, wf_results["intercept_err"])
    assert np.isclose(results.slope_intercept_covar, wf_results["slope_intercept_covar"])
