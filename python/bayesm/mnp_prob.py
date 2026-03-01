"""MNP probability computation using GHK simulator."""

import os
import numpy as np
from numpy.linalg import cholesky

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp


def mnp_prob(beta, Sigma, X, r=100):
    """
    Compute MNP choice probabilities using GHK simulator.
    
    Parameters
    ----------
    beta : array (k,)
        Coefficient vector
    Sigma : array (p-1, p-1)
        Covariance matrix of differenced errors
    X : array (p-1, k)
        Design matrix for one observation (differenced system)
    r : int
        Number of GHK draws
    
    Returns
    -------
    prob : array (p,)
        Choice probabilities for each alternative
    """
    pm1 = Sigma.shape[0]
    k = len(beta)
    mu = X @ beta
    above = np.zeros(pm1)
    prob = np.zeros(pm1 + 1)
    
    for j in range(pm1):
        Aj = -np.eye(pm1)
        Aj[:, j] = 1.0
        trunpt = -Aj @ mu
        Lj = cholesky(Aj @ Sigma @ Aj.T)
        prob[j] = cpp.ghkvec(Lj, trunpt, above, r, False, np.zeros(1))[0]
    
    prob[pm1] = 1.0 - np.sum(prob[:pm1])
    return prob
