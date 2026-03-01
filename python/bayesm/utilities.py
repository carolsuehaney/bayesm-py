"""
Core utility functions

Converted from BayesmFunctions.R and nmat.R
"""

import numpy as np


def pandterm(message):
    """
    Raise an error with a message
    
    Converted from BayesmFunctions.R
    """
    raise ValueError(message)


def nmat(vec):
    """
    Convert variance-covariance matrix (in vector form) to correlation matrix
    
    Parameters
    ----------
    vec : array_like
        Vector form of variance-covariance matrix
    
    Returns
    -------
    array_like
        Correlation matrix in vector form
    
    Converted from nmat.R
    """
    vec = np.asarray(vec)
    p = int(np.sqrt(len(vec)))
    sigma = vec.reshape((p, p))
    nsig = 1.0 / np.sqrt(np.diag(sigma))
    corr = nsig[:, np.newaxis] * sigma * nsig[np.newaxis, :]
    return corr.ravel()
