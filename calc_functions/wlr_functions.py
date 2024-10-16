"""@file calc_functions/wlr_functions.py

Created 2024-10-16 by Bryan Gillis.

AiiDA calculation functions for the weighted linear regression workflow.
"""

from numbers import Number
from collections.abc import Iterable
import numpy as np
from aiida.engine import calcfunction, workfunction
from aiida.orm import ArrayData, Dict, Float

@workfunction
def wf_linregress_with_errors(l_x: Iterable[Number],
                              l_y: Iterable[Number],
                              l_y_err: Iterable[Number]) -> Dict:
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
    Dict
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
        xm: float = divide(sum_array(a_multiply(l_x, l_w)), w)
        x2m: float = divide(sum_array(a_multiply(a_square(l_x), l_w)), w)
        ym: float = divide(sum_array(a_multiply(l_y, l_w)), w)
        xym: float = divide(sum_array(a_multiply(l_x, l_y, l_w)), w)

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

    d_results = Dict({"slope": slope.value,
                 "intercept": intercept.value,
                 "slope_err": slope_err.value,
                 "intercept_err": intercept_err.value,
                 "slope_intercept_covar": slope_intercept_covar.value})
    d_results.store()

    return d_results

@calcfunction
def add(x, y, *args):
    """Adds two or more values
    """
    s = x+y
    for arg in args:
        s += arg
    return s

@calcfunction
def a_add(l_x, l_y, *args):
    """Adds two or more arrays elementwise
    """
    s = l_x.get_array() + l_y.get_array()
    for arg in args:
        s += arg.get_array()
    return ArrayData(s)

@calcfunction
def subtract(x, y):
    """Subtracts one value from another
    """
    return x - y

@calcfunction
def a_subtract(l_x, l_y):
    """Subtracts one array from another elementwise
    """
    return ArrayData(l_x.get_array() - l_y.get_array())

@calcfunction
def multiply(x, y, *args):
    """Multiplies two or more values
    """
    p = x*y
    for arg in args:
        p *= arg
    return p

@calcfunction
def a_multiply(l_x, l_y, *args):
    """Multiplies two or more arrays elementwise
    """
    p = l_x.get_array() * l_y.get_array()
    for arg in args:
        p *= arg.get_array()
    return ArrayData(p)

@calcfunction
def divide(x, y):
    """Divides one value by another
    """
    return x / y

@calcfunction
def a_divide(l_x, l_y):
    """Divides one array by another elementwise
    """
    return ArrayData(l_x.get_array() / l_y.get_array())

@calcfunction
def square(x):
    """Squares a value
    """
    return x ** 2

@calcfunction
def a_square(l_x):
    """Squares an array elementwise
    """
    return ArrayData(l_x.get_array() ** 2)

@calcfunction
def a_sqrt(l_x):
    """Gets the square root of an array elementwise
    """
    return ArrayData(np.sqrt(l_x.get_array()))

@calcfunction
def sqrt(x):
    """Gets the square root of a scalar value
    """
    return Float(np.sqrt(x.value))

@calcfunction
def sum_array(l_x: Iterable[Number]) -> float:
    """Calculates the sum of an array, excluding any NaN values
    """
    return Float(np.nansum(l_x.get_array()))

@calcfunction
def get_weights_from_errors(l_y_err: Iterable[Number]) -> np.ndarray:
    """Converts an array of weights to an array of errors
    """
    l_y_err = np.asarray(l_y_err.get_array())
    return ArrayData(np.power(l_y_err, -2))
