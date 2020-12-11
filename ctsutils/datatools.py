import numpy as np
from scipy.optimize import newton

def get_nth_smallest_curve(x, y, n):
    """
    Arguments:
    x -- 1d array
    y -- columns of different y values (all sharing the same x values)
    n=0: lowest
    n=1: 2nd lowest, ...
    """

    y_requested = np.array([])

    for k in x:
        # for each x value, get the nth lowest y value
        y_requested = np.append(y_requested, np.sort(np.ravel(y[x == k]))[n])

    return x, y_requested  # as x, y data

def extract_x_from_interpolation(y_specific, x, f):
    # find x where f(x) = y_specific
    # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

    root = None
    # try to find a root
    try:
        root = newton(lambda x: f(x) - y_specific, (max(x) + min(x))/2.)
        # import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
        # print("hey, root: ", root, " for ", x, y_specific)
    except ValueError as err:
        print("skipping because of ValueError: {0}".format(err))

    return root
