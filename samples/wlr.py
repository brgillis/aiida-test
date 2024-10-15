"""This file provides an algorithmic implementation of a weighted least-squares regression, which can be used for
comparisons and reference.
"""

from dataclasses import dataclass
from numbers import Number
from collections.abc import Iterable

import numpy as np

@dataclass
class LinregressResults:
    """Dataclass to hold the results of a linear regression calculation
    """
    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    slope_intercept_covar: float

def linregress_with_errors(l_x: Iterable[Number],
                           l_y: Iterable[Number],
                           l_y_err: Iterable[Number]) -> LinregressResults:
    """Algorithmic implementation of linear regression with errors on dependent variable

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
    
    # Silently coerce input to numpy arrays
    l_x = np.asarray(l_x)
    l_y = np.asarray(l_y)
    l_y_err = np.asarray(l_y_err)

    # Calculate the weights
    l_w = l_y_err ** -2
    w: float = np.nansum(l_w)

    # Catch possible bad data, indicated here by zero (or negative) weight
    if w <= 0:
        w = 0
        xm: float = 0
        x2m: float = 0
        ym: float = 0
        xym: float = 0
    else:
        xm: float = np.nansum(l_x * l_w) / w
        x2m: float = np.nansum(l_x ** 2 * l_w) / w
        ym: float = np.nansum(l_y * l_w) / w
        xym: float = np.nansum(l_x * l_y * l_w) / w

    dx2m = x2m - xm ** 2
    dxym = xym - xm * ym

    if not (dx2m > 0):
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