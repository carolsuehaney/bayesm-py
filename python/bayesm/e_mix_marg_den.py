"""
Expected mixture marginal density

Converted from eMixMargDen.R
"""

import numpy as np
from .mix_den import mix_den


def e_mix_marg_den(grid, probdraw, compdraw):
    """
    Compute marginal density of normal mixture averaged over MCMC draws
    
    Parameters
    ----------
    grid : array_like
        Array of grid points, grid[:, i] are ordinates for ith component
    probdraw : array_like
        Matrix where ith row is ith draw of probabilities of mixture components
    compdraw : list
        List of lists of draws of mixture component moments
    
    Returns
    -------
    array_like
        Array of same dimension as grid with density values
    """
    grid = np.asarray(grid)
    probdraw = np.asarray(probdraw)
    
    den = np.zeros_like(grid)
    for i in range(len(compdraw)):
        den += mix_den(grid, probdraw[i, :], compdraw[i])
    
    return den / len(compdraw)
