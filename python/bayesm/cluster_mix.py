"""Cluster observations based on mixture indicator draws."""

import numpy as np
from .constants import BayesmConstants
from .utilities import pandterm


def _z_to_sim(z):
    """Convert indicator vector to similarity matrix."""
    n = len(z)
    Sim = (z[:, None] == z[None, :]).astype(float)
    return Sim


def _sim_to_z(Sim):
    """Convert similarity matrix to indicator vector."""
    n = Sim.shape[0]
    z = np.zeros(n, dtype=int)
    groupn = 1
    
    for i in range(n):
        count = 0
        for j in range(n):
            if z[j] == 0 and Sim[j, i] == 1:
                z[j] = groupn
                count += 1
        if count > 0:
            groupn += 1
    
    return z


def cluster_mix(zdraw, cutoff=0.9, silent=False, nprint=None):
    """
    Cluster observations based on draws of mixture component indicators.
    
    Parameters
    ----------
    zdraw : array (R, nobs)
        Matrix of indicator draws (typically from rnmixGibbs).
        Row r contains the r-th draw of indicators for each observation.
        Elements take values 1, ..., p for p groups.
    cutoff : float
        Cutoff for clusterb (must be between 0.5 and 1.0)
    silent : bool
        If True, suppress progress output
    nprint : int
        Print progress every nprint draws (default: 100)
    
    Returns
    -------
    dict with:
        clustera : array (nobs,)
            Clustering from zdraw closest to posterior mean similarity
        clusterb : array (nobs,)
            Clustering from thresholding posterior mean similarity at cutoff
    """
    if nprint is None:
        nprint = BayesmConstants.nprint
    
    if nprint < 0:
        pandterm("nprint must be >= 0")
    
    R, nobs = zdraw.shape
    
    if not np.all(np.isin(zdraw, np.arange(1, nobs + 1))):
        pandterm("Bad zdraw argument -- all elements must be integers in 1:nobs")
    
    if not silent:
        unique, counts = np.unique(zdraw, return_counts=True)
        print("Table of zdraw values pooled over all rows:")
        for u, c in zip(unique, counts):
            print(f"  {int(u)}: {c}")
    
    if cutoff > 1 or cutoff < 0.5:
        pandterm(f"cutoff invalid, = {cutoff}")
    
    if not silent:
        print("Computing Posterior Expectation of Similarity Matrix")
        print("processing draws ...")
    
    Pmean = np.zeros((nobs, nobs))
    for rep in range(R):
        Pmean += _z_to_sim(zdraw[rep, :])
        if not silent and nprint > 0 and (rep + 1) % nprint == 0:
            print(f"  {rep + 1}")
    
    Pmean /= R
    
    if not silent:
        print()
        print("Look for zdraw which minimizes loss")
        print("processing draws ...")
    
    loss = np.zeros(R)
    for rep in range(R):
        loss[rep] = np.sum(np.abs(Pmean - _z_to_sim(zdraw[rep, :])))
        if not silent and nprint > 0 and (rep + 1) % nprint == 0:
            print(f"  {rep + 1}")
    
    index = np.argmin(loss)
    clustera = zdraw[index, :].astype(int)
    
    Sim = (Pmean >= cutoff).astype(float)
    clusterb = _sim_to_z(Sim)
    
    return {
        'clustera': clustera,
        'clusterb': clusterb
    }
