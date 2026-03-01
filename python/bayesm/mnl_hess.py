"""Compute MNL negative expected Hessian."""

import numpy as np


def mnl_hess(beta, y, X):
    """
    Compute MNL -E[Hessian].
    
    Parameters
    ----------
    beta : array (k,) - coefficients
    y : array (n,) - choices (0-indexed: 0, ..., j-1)
    X : array (n*j, k) - design matrix
    
    Returns
    -------
    array (k, k) - negative expected Hessian
    """
    n = len(y)
    j = X.shape[0] // n
    k = X.shape[1]
    
    Xbeta = X @ beta
    Xbeta = Xbeta.reshape((n, j), order='C')
    Xbeta = np.exp(Xbeta)
    denom = Xbeta.sum(axis=1, keepdims=True)
    Prob = Xbeta / denom
    
    Hess = np.zeros((k, k))
    for i in range(n):
        p = Prob[i, :]
        A = np.diag(p) - np.outer(p, p)
        Xt = X[j * i : j * (i + 1), :]
        Hess += Xt.T @ A @ Xt
    
    return Hess
