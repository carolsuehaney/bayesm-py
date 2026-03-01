"""
Conditional moments for multivariate normal

Converted from condMom.R
"""

import numpy as np


def cond_mom(x, mu, sigi, i):
    """
    Compute moments of conditional distribution of ith element of normal given all others
    
    Parameters
    ----------
    x : array_like
        Vector of values to condition on
    mu : array_like
        Mean vector of length(x)-dim MVN
    sigi : array_like
        Inverse of covariance matrix (Sigma^-1)
    i : int
        Element to condition on (0-indexed in Python, 1-indexed in R)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'cmean': conditional mean (scalar)
        - 'cvar': conditional variance (scalar)
    
    Notes
    -----
    Model: x ~ MVN(mu, Sigma)
    Computes moments of x_i given x_{-i}
    
    Converted from condMom.R (adjusted for 0-indexing)
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigi = np.asarray(sigi)
    
    sig = 1.0 / sigi[i, i]
    
    mask = np.arange(len(x)) != i
    m = mu[i] - np.dot(x[mask] - mu[mask], sigi[mask, i]) * sig
    
    return {'cmean': float(m), 'cvar': sig}
