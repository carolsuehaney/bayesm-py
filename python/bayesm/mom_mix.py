"""
Mixture moments computation

Converted from momMix.R
"""

import numpy as np
from .utilities import nmat


def mom_mix(probdraw, compdraw):
    """
    Compute moments of normal mixture averaged over MCMC draws
    
    Parameters
    ----------
    probdraw : array_like
        Matrix where ith row is ith draw of probabilities of mixture components
    compdraw : list
        List of lists of draws of mixture component moments
        Each element is a list of components from mixture Gibbs sampler
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'mu': mean vector
        - 'sigma': covariance matrix
        - 'sd': standard deviations
        - 'corr': correlation matrix
    """
    probdraw = np.asarray(probdraw)
    
    dim = len(compdraw[0][0][0])
    nc = len(compdraw[0])
    
    probdraw = probdraw.reshape((len(compdraw), nc))
    
    mu = np.zeros(dim)
    sigma = np.zeros((dim, dim))
    sd = np.zeros(dim)
    corr = np.zeros((dim, dim))
    
    for i in range(len(compdraw)):
        out = _mom(probdraw[i, :], compdraw[i])
        sd += np.sqrt(np.diag(out['sigma']))
        corr += nmat(out['sigma'].ravel()).reshape((dim, dim))
        mu += out['mu']
        sigma += out['sigma']
    
    n_draws = len(compdraw)
    mu = mu / n_draws
    sigma = sigma / n_draws
    sd = sd / n_draws
    corr = corr / n_draws
    
    return {'mu': mu, 'sigma': sigma, 'sd': sd, 'corr': corr}


def _mom(prob, comps):
    """
    Obtain mean and covariance from list of normal components
    
    Parameters
    ----------
    prob : array_like
        Vector of mixture probabilities
    comps : list
        List of components, each is [mu, rooti] where
        - mu: mean vector
        - rooti: inverse of upper triangular Cholesky root
    
    Returns
    -------
    dict
        Dictionary with 'mu' (mean vector) and 'sigma' (covariance matrix)
    """
    prob = np.asarray(prob)
    nc = len(comps)
    dim = len(comps[0][0])
    
    mu = np.zeros(dim)
    for i in range(nc):
        mu += prob[i] * comps[i][0]
    
    var = np.zeros((dim, dim))
    for i in range(nc):
        mui = comps[i][0]
        root = np.linalg.solve(comps[i][1], np.eye(dim))
        sigma_comp = root.T @ root
        var += prob[i] * sigma_comp + prob[i] * np.outer(mui - mu, mui - mu)
    
    return {'mu': mu, 'sigma': var}
