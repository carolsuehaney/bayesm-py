"""Ordered Probit Gibbs sampler."""

import os
import numpy as np
from scipy.optimize import minimize

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp

from .constants import BayesmConstants
from .utilities import pandterm


def _dstartoc(dstar):
    """Transform dstar to cutoffs."""
    return np.concatenate([[-100], [0], np.cumsum(np.exp(dstar)), [100]])


def _lldstar(dstar, y, mu):
    """Log-likelihood for dstar given data."""
    from scipy.stats import norm
    gamma = _dstartoc(dstar)
    y_int = y.astype(int)
    arg = norm.cdf(gamma[y_int] - mu) - norm.cdf(gamma[y_int - 1] - mu)
    arg = np.maximum(arg, 1e-50)
    return np.sum(np.log(arg))


def rordprobitGibbs(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for Ordered Probit Model.
    
    Model: z = X*beta + e, e ~ N(0,1)
           y = k if c[k-1] < z <= c[k]
    Prior: beta ~ N(betabar, A^-1)
           dstar ~ N(dstarbar, Ad^-1)
    
    Parameters
    ----------
    Data : dict
        y : array (nobs,) - ordinal outcomes (1, 2, ..., k)
        X : array (nobs, nvar) - design matrix
        k : int - number of ordinal categories
    Prior : dict, optional
        betabar : array (nvar,) - prior mean for beta (default: zeros)
        A : array (nvar, nvar) - prior precision for beta (default: 0.01*I)
        dstarbar : array (k-2,) - prior mean for dstar (default: zeros)
        Ad : array (k-2, k-2) - prior precision for dstar (default: I)
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
        s : float - RW Metropolis scale (default: 2.93/sqrt(k-2))
    
    Returns
    -------
    dict with:
        betadraw : array (R/keep, nvar)
        cutdraw : array (R/keep, k-1) - interior cutoffs (excluding -100, 0, 100)
        dstardraw : array (R/keep, k-2)
        accept : float - acceptance rate for cutoff draws
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with y, X, k")
    if 'X' not in Data:
        pandterm("Requires Data element X")
    if 'y' not in Data:
        pandterm("Requires Data element y")
    if 'k' not in Data:
        pandterm("Requires Data element k")
    
    X = np.asarray(Data['X'], dtype=np.float64)
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    k = Data['k']
    
    nvar = X.shape[1]
    nobs = len(y)
    ndstar = k - 2
    ncuts = k + 1
    
    if len(y) != X.shape[0]:
        pandterm("y and X not of same row dim")
    
    unique_y = np.unique(y.astype(int))
    if not np.all(np.isin(unique_y, np.arange(1, k + 1))):
        pandterm("some value of y is not valid")
    
    if Prior is None:
        betabar = np.zeros(nvar)
        A = BayesmConstants.A * np.eye(nvar)
        Ad = np.eye(ndstar)
        dstarbar = np.zeros(ndstar)
    else:
        betabar = Prior.get('betabar', np.zeros(nvar))
        A = Prior.get('A', BayesmConstants.A * np.eye(nvar))
        Ad = Prior.get('Ad', np.eye(ndstar))
        dstarbar = Prior.get('dstarbar', np.zeros(ndstar))
    
    if A.shape != (nvar, nvar):
        pandterm(f"bad dimensions for A: {A.shape}")
    if len(betabar) != nvar:
        pandterm(f"betabar wrong length: {len(betabar)}")
    if Ad.shape != (ndstar, ndstar):
        pandterm(f"bad dimensions for Ad: {Ad.shape}")
    if len(dstarbar) != ndstar:
        pandterm(f"dstarbar wrong length: {len(dstarbar)}")
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    if 'R' not in Mcmc:
        pandterm("requires Mcmc element R")
    
    R = Mcmc['R']
    keep = Mcmc.get('keep', BayesmConstants.keep)
    nprint = Mcmc.get('nprint', BayesmConstants.nprint)
    s = Mcmc.get('s', BayesmConstants.RRScaling / np.sqrt(ndstar) if ndstar > 0 else 1.0)
    
    if nprint < 0:
        pandterm("nprint must be >= 0")
    
    if nprint > 0:
        print()
        print("Starting Gibbs Sampler for Ordered Probit Model")
        print(f"   with {nobs} observations")
        print()
        print("Table of y values")
        unique, counts = np.unique(y.astype(int), return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        print()
        print("Prior Parms:")
        print(f"  betabar: {betabar}")
        print(f"  A diagonal: {np.diag(A)}")
        print()
        print(f"MCMC parms: R = {R}, keep = {keep}, s = {s:.3f}")
        print()
    
    XtX = X.T @ X
    betahat = np.linalg.solve(XtX, X.T @ y)
    
    dstarini = np.cumsum(np.full(ndstar, 0.1))
    
    def neg_lldstar(dstar):
        return -_lldstar(dstar, y, X @ betahat)
    
    result = minimize(neg_lldstar, dstarini, method='BFGS')
    dstaropt = result.x
    
    hess_approx = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(ndstar)
    if isinstance(hess_approx, np.ndarray):
        try:
            inc_root = np.linalg.cholesky(np.linalg.inv(-result.hess_inv + Ad)).T
        except:
            inc_root = np.linalg.cholesky(Ad).T
    else:
        inc_root = np.linalg.cholesky(Ad).T
    
    betadraw, cutdraw, dstardraw, accept = cpp.rordprobitGibbs_rcpp_loop(
        y, X, k, A, betabar, Ad, s, inc_root, dstarbar, betahat, R, keep
    )
    
    cutdraw_interior = cutdraw[:, 1:k]
    
    if nprint > 0:
        print(f"Acceptance rate: {accept:.3f}")
    
    return {
        'betadraw': betadraw,
        'cutdraw': cutdraw_interior,
        'dstardraw': dstardraw,
        'accept': accept
    }
