"""Univariate regression with conjugate prior - IID sampler."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp

from .constants import BayesmConstants
from .utilities import pandterm


def runireg(Data, Prior=None, Mcmc=None):
    """
    IID sampler for univariate regression with conjugate prior.
    
    Model: y = X @ beta + e, e ~ N(0, sigmasq)
    Prior: beta ~ N(betabar, sigmasq * A^-1), sigmasq ~ (nu*ssq)/chisq_nu
    
    Parameters
    ----------
    Data : dict
        y : array (n,) - response
        X : array (n, k) - design matrix
    Prior : dict, optional
        betabar : array (k,) - prior mean (default: zeros)
        A : array (k, k) - prior precision (default: 0.01*I)
        nu : float - prior df (default: 3)
        ssq : float - prior scale (default: var(y))
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
    
    Returns
    -------
    dict with:
        betadraw : array (R/keep, k)
        sigmasqdraw : array (R/keep,)
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with y and X")
    if 'X' not in Data:
        pandterm("Requires Data element X")
    if 'y' not in Data:
        pandterm("Requires Data element y")
    
    X = np.asarray(Data['X'], dtype=np.float64)
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    
    nvar = X.shape[1]
    nobs = len(y)
    
    if nobs != X.shape[0]:
        pandterm("length(y) != nrow(X)")
    
    if Prior is None:
        betabar = np.zeros(nvar)
        A = BayesmConstants.A * np.eye(nvar)
        nu = BayesmConstants.nu
        ssq = np.var(y)
    else:
        betabar = Prior.get('betabar', np.zeros(nvar))
        A = Prior.get('A', BayesmConstants.A * np.eye(nvar))
        nu = Prior.get('nu', BayesmConstants.nu)
        ssq = Prior.get('ssq', np.var(y))
    
    if A.shape != (nvar, nvar):
        pandterm(f"bad dimensions for A: {A.shape}")
    if len(betabar) != nvar:
        pandterm(f"betabar wrong length: {len(betabar)}")
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    if 'R' not in Mcmc:
        pandterm("requires Mcmc element R")
    
    R = Mcmc['R']
    keep = Mcmc.get('keep', BayesmConstants.keep)
    nprint = Mcmc.get('nprint', BayesmConstants.nprint)
    
    if nprint < 0:
        pandterm("nprint must be >= 0")
    
    if nprint > 0:
        print()
        print("Starting IID Sampler for Univariate Regression Model")
        print(f"  with {nobs} observations")
        print()
        print("Prior Parms:")
        print(f"  betabar: {betabar}")
        print(f"  A diagonal: {np.diag(A)}")
        print(f"  nu = {nu}, ssq = {ssq:.4f}")
        print()
        print(f"MCMC parms: R = {R}, keep = {keep}")
        print()
    
    betadraw, sigmasqdraw = cpp.runireg_rcpp_loop(
        y, X, betabar, A, nu, ssq, R, keep
    )
    
    return {
        'betadraw': betadraw,
        'sigmasqdraw': sigmasqdraw
    }
