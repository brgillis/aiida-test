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

    dx2m = x2m - xm ** 2
    dxym = xym - xm * ym

    if not dx2m > 0:
        slope = np.inf
        intercept = np.nan
    else:
        slope = dxym / dx2m
        intercept = ym - xm * slope

    if dx2m <= 0 or w == 0:
        slope_err: float = np.inf
        intercept_err: float = np.nan
        slope_intercept_covar: float = np.nan
    else:
        slope_err: float = np.sqrt(1. / (w * dx2m))
        intercept_err: float = np.sqrt((1.0 + xm ** 2 / dx2m) / w)
        slope_intercept_covar: float = -xm / (w * dx2m)

    return LinregressResults(slope=slope,
                             intercept=intercept,
                             slope_err=slope_err,
                             intercept_err=intercept_err,
                             slope_intercept_covar=slope_intercept_covar)

def divide(x: Number, y: Number) -> Number:
    """Divides one value by another
    """
    return x / y

def square(x: Number) -> Number:
    """Squares a value
    """
    return x ** 2

def multiply(x: Number, y: Number, *args: Number) -> Number:
    """Multiplies two or more values
    """
    p: Number = x*y
    for arg in args:
        p *= arg
    return p

def sum_array(l_x: Iterable[Number]) -> float:
    """Calculates the sum of an array, excluding any NaN values
    """
    return np.nansum(l_x)

def get_weights_from_errors(l_y_err: Iterable[Number]) -> np.ndarray:
    """Converts an array of weights to an array of errors
    """
    return np.asarray(l_y_err) ** -2
