"""MNP log-likelihood using GHK simulator."""

import os
import numpy as np
from numpy.linalg import cholesky

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp


def llmnp(beta, Sigma, X, y, r):
    """
    Evaluate MNP log-likelihood using GHK simulator.
    
    Parameters
    ----------
    beta : array (k,)
        Coefficient vector
    Sigma : array (p-1, p-1)
        Covariance matrix of differenced errors
    X : array (n*(p-1), k)
        Design matrix (differenced system, stacked across observations)
    y : array (n,)
        Vector of choice indicators (1-indexed: 1, ..., p)
    r : int
        Number of GHK draws
    
    Returns
    -------
    logl : float
        Log-likelihood value
    """
    pm1 = Sigma.shape[0]
    k = len(beta)
    n = len(y)
    
    mu = (X @ beta).reshape(pm1, n, order='F')
    logl = 0.0
    above = np.zeros(pm1)
    
    for j in range(pm1):
        mask = (y == (j + 1))
        if not np.any(mask):
            continue
        muj = mu[:, mask]
        Aj = -np.eye(pm1)
        Aj[:, j] = 1.0
        trunpt = (-Aj @ muj).flatten(order='F')
        Lj = cholesky(Aj @ Sigma @ Aj.T)
        probs = cpp.ghkvec(Lj, trunpt, above, r, False, np.zeros(1))
        logl += np.sum(np.log(probs + 1e-50))
    
    mask = (y == (pm1 + 1))
    if np.any(mask):
        trunpt = (-mu[:, mask]).flatten(order='F')
        Lj = cholesky(Sigma)
        above_p = np.ones(pm1)
        probs = cpp.ghkvec(Lj, trunpt, above_p, r, False, np.zeros(1))
        logl += np.sum(np.log(probs + 1e-50))
    
    return logl
