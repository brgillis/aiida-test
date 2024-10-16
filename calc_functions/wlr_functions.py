"""@file calc_functions/wlr_functions.py

Created 2024-10-16 by Bryan Gillis.

AiiDA calculation functions for the weighted linear regression workflow.
"""

from numbers import Number
from collections.abc import Iterable
import numpy as np

from samples.wlr import LinregressResults

def wf_linregress_with_errors(l_x: Iterable[Number],
                           l_y: Iterable[Number],
                           l_y_err: Iterable[Number]) -> LinregressResults:
    """Algorithmic implementation of linear regression with errors on dependent variable,
    which I'll work piece-by-piece to convert into purely function blocks that can be
    made AiiDA calculation functions.

    Parameters
    ----------
    l_x : Iterable[Number]
        Array of independent variable
    l_y : Iterable[Number]
        Array of dependent variable
    l_y_err : Iterable[Number]
        Array of errors on dependent variable

    Returns
    -------
    LinregressResults
    """

    # Calculate the weights
    l_w = get_weights_from_errors(l_y_err)
    w = sum_array(l_w)

    # Catch possible bad data, indicated here by zero (or negative) weight
    if w <= 0:
        w = 0
        xm: float = 0
        x2m: float = 0
        ym: float = 0
        xym: float = 0
    else:
        xm: float = divide(sum_array(multiply(l_x, l_w)), w)
        x2m: float = divide(sum_array(multiply(square(l_x), l_w)), w)
        ym: float = divide(sum_array(multiply(l_y, l_w)), w)
        xym: float = divide(sum_array(multiply(l_x, l_y, l_w)), w)

    dx2m = subtract(x2m, square(xm))
    dxym = subtract(xym, multiply(xm, ym))

    if not dx2m > 0:
        slope = np.inf
        intercept = np.nan
    else:
        slope = divide(dxym, dx2m)
        intercept = subtract(ym, multiply(xm, slope))

    if dx2m <= 0 or w == 0:
        slope_err: float = np.inf
        intercept_err: float = np.nan
        slope_intercept_covar: float = np.nan
    else:
        slope_err: float = sqrt(divide(1., multiply(w, dx2m)))
        intercept_err: float = sqrt(divide(add(1.0, divide(square(xm), dx2m)), w))
        slope_intercept_covar: float = divide(-xm, multiply(w, dx2m))

    return LinregressResults(slope=slope,
                             intercept=intercept,
                             slope_err=slope_err,
                             intercept_err=intercept_err,
                             slope_intercept_covar=slope_intercept_covar)

def add(x, y, *args):
    """Adds two or more values
    """
    s = x+y
    for arg in args:
        s += arg
    return s

def subtract(x, y):
    """Subtracts one value from another
    """
    return x - y

def multiply(x, y, *args):
    """Multiplies two or more values
    """
    p = x*y
    for arg in args:
        p *= arg
    return p

def divide(x, y):
    """Divides one value by another
    """
    return x / y

def square(x):
    """Squares a value
    """
    return x ** 2

def sqrt(x):
    """Gets the square root of a value
    """
    return np.sqrt(x)

def sum_array(l_x: Iterable[Number]) -> float:
    """Calculates the sum of an array, excluding any NaN values
    """
    return np.nansum(l_x)

def get_weights_from_errors(l_y_err: Iterable[Number]) -> np.ndarray:
    """Converts an array of weights to an array of errors
    """
    return np.asarray(l_y_err) ** -2
