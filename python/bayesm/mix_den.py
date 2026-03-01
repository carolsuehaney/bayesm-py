"""
Mixture density functions

Converted from mixDen.R and mixDenBi.R
"""

import numpy as np
from scipy.stats import norm


def mix_den(x, pvec, comps):
    """
    Compute marginal densities for multivariate mixture of normals
    
    Parameters
    ----------
    x : array_like
        Array where ith column gives evaluations for density of ith variable
    pvec : array_like
        Prior probabilities of normal components
    comps : list
        List of mixture components, each is a list [mu, rooti] where:
        - mu: mean vector
        - rooti: inverse of upper triangular Cholesky root (Sigma^-1 = rooti @ rooti.T)
    
    Returns
    -------
    array_like
        Matrix with same shape as x, ith column gives marginal density of ith variable
    """
    x = np.asarray(x)
    pvec = np.asarray(pvec)
    
    # Get marginal means and standard deviations
    nc = len(comps)
    dim = len(comps[0][0])
    mu = np.zeros((nc, dim))
    sigma = np.zeros((nc, dim))
    
    for i in range(nc):
        mu[i, :] = comps[i][0]
        # root = inv(rooti)
        root = np.linalg.solve(comps[i][1], np.eye(dim))
        sigma[i, :] = np.sqrt(np.diag(root.T @ root))
    
    # Compute densities
    den = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(nc):
            den[:, i] += norm.pdf(x[:, i], loc=mu[j, i], scale=sigma[j, i]) * pvec[j]
    
    return den


def mix_den_bi(i, j, xi, xj, pvec, comps):
    """
    Compute marginal bivariate density from mixture of multivariate normals
    
    Parameters
    ----------
    i, j : int
        Indices of two variables (0-indexed in Python, 1-indexed in R)
    xi : array_like
        Grid of points for variable i
    xj : array_like
        Grid of points for variable j
    pvec : array_like
        Prior probabilities of normal components
    comps : list
        List of mixture components, each is a list [mu, rooti]
    
    Returns
    -------
    array_like
        Matrix with density values on grid (ngridxi x ngridxj)
    """
    xi = np.asarray(xi)
    xj = np.asarray(xj)
    pvec = np.asarray(pvec)
    
    nc = len(comps)
    dim = len(comps[0][0])
    
    # Extract bivariate components
    marmoms = []
    for comp_idx in range(nc):
        mu_full = comps[comp_idx][0]
        rooti_full = comps[comp_idx][1]
        
        # Get bivariate marginal
        mu = mu_full[[i, j]]
        root = np.linalg.solve(rooti_full, np.eye(dim))
        Sigma = root.T @ root
        sigma_biv = Sigma[np.ix_([i, j], [i, j])]
        rooti_biv = np.linalg.inv(np.linalg.cholesky(sigma_biv).T)
        
        marmoms.append({'mu': mu, 'rooti': rooti_biv})
    
    # Create grid
    ngridxi = len(xi)
    ngridxj = len(xj)
    z = np.column_stack([
        np.repeat(xi, ngridxj),
        np.tile(xj, ngridxi)
    ])
    
    # Compute density
    den = np.zeros((ngridxi, ngridxj))
    for comp_idx in range(nc):
        mu_comp = marmoms[comp_idx]['mu']
        rooti_comp = marmoms[comp_idx]['rooti']
        
        z_centered = (z - mu_comp).T
        quads = np.sum((rooti_comp @ z_centered) ** 2, axis=0)
        
        log_det_term = np.sum(np.log(np.diag(rooti_comp)))
        dencomp = np.exp(-np.log(2 * np.pi) + log_det_term - 0.5 * quads)
        dencomp = dencomp.reshape((ngridxi, ngridxj))
        den += dencomp * pvec[comp_idx]
    
    return den
