"""Sample distance covariance calculation.

This module implements distance covariance or distance correlation
calculations. It supports the use of arbitrary (pseudo)-distances.
"""

import numpy as np


def pairwise_distances(X, distance_fn):
    """Pairwise distance calculation with an arbitrary distance function."""
    n = len(X)
    D = np.empty((n, n), dtype=float)

    for i, x in enumerate(X):
        for j, y in enumerate(X):
            D[i, j] = distance_fn(x, y)

    return D


def _means(X):
    """Calculate matrix-based means.

    Parameters
    ----------
    X : np.array of shape (n, n)
        Input matrix

    Returns
    -------
    Tuple of np.array
        Vector of row means, vector of column means, and overall matrix
        mean.
    """
    row_mean = np.mean(X, axis=1, keepdims=True)
    col_mean = np.mean(X, axis=0, keepdims=True)
    mean = np.mean(X)

    return row_mean, col_mean, mean


def dcov(X, Y, distance_fn):
    """Evaluate distance correlation between two samples.

    This function performs the main distance correlation calculation,
    given two sets of samples ``X`` and ``Y``.

    Parameters
    ----------
    X
        First set of samples (of arbitrary type)

    Y
        Second set of samples (of arbitrary type)

    distance_fn: callable
        Binary distance function. Must be able to handle ``X`` and ``Y`
        inputs and return a scalar value.

    Returns
    -------
    float
        Sample distance covariance
    """
    n = len(X)
    m = len(Y)

    assert n == m, RuntimeError('Inputs must have the same cardinality')

    A = pairwise_distances(X, distance_fn)
    B = pairwise_distances(Y, distance_fn)

    A_row_mean, A_col_mean, A_mean = _means(A)
    B_row_mean, B_col_mean, B_mean = _means(B)

    A = A - A_row_mean - A_col_mean + A_mean
    B = B - B_row_mean - B_col_mean + B_mean

    d = 1 / (n**2) * np.sum(np.multiply(A, B))
    return d

########################################################################
# Demo
########################################################################

def l1(x, y):
    return np.sum(np.abs(x - y))


X = [1, 2, 3]
Y = [4, 5, 6]

print(dcov(X, Y, l1))
