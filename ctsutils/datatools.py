import numpy as np

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
