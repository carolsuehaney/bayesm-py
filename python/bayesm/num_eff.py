"""
Numerical efficiency calculation for MCMC draws

Converted from numEff.R
"""

import numpy as np
from statsmodels.tsa.stattools import acf


def num_eff(x, m=None):
    """
    Compute Newey-West standard error and relative numerical efficiency
    
    Parameters
    ----------
    x : array_like
        Vector of MCMC draws
    m : int, optional
        Number of lags to truncate ACF. Default is computed as:
        min(length(x), (100/sqrt(5000)) * sqrt(length(x)))
        so m=100 if length(x)=5000 and grows with sqrt(length)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'stderr': numerical standard error
        - 'f': variance multiple (inefficiency factor)
        - 'm': number of lags used
    
    Notes
    -----
    Converted from numEff.R
    """
    x = np.asarray(x)
    n = len(x)
    
    if m is None:
        m = int(min(n, (100.0 / np.sqrt(5000)) * np.sqrt(n)))
    
    # Compute ACF (nlags includes lag 0)
    acf_vals = acf(x, nlags=m, fft=True)
    
    # Weights: m, m-1, ..., 1 divided by (m+1)
    wgt = np.arange(m, 0, -1) / (m + 1)
    
    # Inefficiency factor: 1 + 2 * weighted sum of autocorrelations (excluding lag 0)
    f = 1 + 2 * np.dot(wgt, acf_vals[1:])
    
    stderr = np.sqrt(np.var(x, ddof=1) * f / n)
    
    return {'stderr': stderr, 'f': f, 'm': m}
