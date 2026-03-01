"""
Log marginal density using Newton-Raftery estimator

Converted from logMargDenNR.R
"""

import numpy as np


def log_marg_den_nr(ll):
    """
    Compute log marginal density using Newton-Raftery importance sampling
    
    Estimator: 1 / (1/g * sum_g exp(-log_like))
    where log_like is the likelihood evaluated at posterior draws
    
    Parameters
    ----------
    ll : array_like
        Vector of log-likelihood values evaluated at posterior draws
    
    Returns
    -------
    float
        Estimated log-marginal density
    """
    ll = np.asarray(ll)
    med = np.median(ll)
    return med - np.log(np.mean(np.exp(-ll + med)))
