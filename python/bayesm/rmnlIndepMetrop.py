"""MNL Independence Metropolis sampler."""

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
from .mnl_hess import mnl_hess


def rmnlIndepMetrop(Data, Prior=None, Mcmc=None):
    """
    Independence Metropolis sampler for Multinomial Logit.
    
    Model: Pr(y=j) = exp(x_j'beta) / sum_k exp(x_k'beta)
    Prior: beta ~ N(betabar, A^-1)
    
    Parameters
    ----------
    Data : dict
        p : int - number of alternatives
        y : array (nobs,) - choices (1-indexed: 1, ..., p)
        X : array (nobs*p, nvar) - design matrix
    Prior : dict, optional
        betabar : array (nvar,) - prior mean (default: zeros)
        A : array (nvar, nvar) - prior precision (default: 0.01*I)
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
        nu : float - df for Student-t candidate (default: 6)
    
    Returns
    -------
    dict with:
        betadraw : array (R/keep, nvar)
        loglike : array (R/keep,)
        acceptr : float - acceptance rate
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with p, y, X")
    if 'X' not in Data:
        pandterm("Requires Data element X")
    if 'y' not in Data:
        pandterm("Requires Data element y")
    if 'p' not in Data:
        pandterm("Requires Data element p")
    
    X = np.asarray(Data['X'], dtype=np.float64)
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    p = Data['p']
    
    nvar = X.shape[1]
    nobs = len(y)
    
    if len(y) != X.shape[0] // p:
        pandterm("length(y) != nrow(X)/p")
    
    if not np.all(np.isin(y.astype(int), np.arange(1, p + 1))):
        pandterm("invalid values in y vector -- must be integers in 1:p")
    
    if Prior is None:
        betabar = np.zeros(nvar)
        A = BayesmConstants.A * np.eye(nvar)
    else:
        betabar = Prior.get('betabar', np.zeros(nvar))
        A = Prior.get('A', BayesmConstants.A * np.eye(nvar))
    
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
    nu = Mcmc.get('nu', 6.0)
    
    if nprint < 0:
        pandterm("nprint must be >= 0")
    
    if nprint > 0:
        print()
        print("Starting Independence Metropolis Sampler for Multinomial Logit Model")
        print(f"  {nobs} obs with {p} alternatives")
        print()
        print("Table of y Values")
        unique, counts = np.unique(y.astype(int), return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        print()
        print("Prior Parms:")
        print(f"  betabar: {betabar}")
        print(f"  A diagonal: {np.diag(A)}")
        print()
        print(f"MCMC parms: R = {R}, keep = {keep}, nu = {nu}")
        print()
    
    y_0idx = (y - 1).astype(np.float64)
    
    def neg_llmnl(beta):
        return -cpp.llmnl(beta, y_0idx, X)
    
    result = minimize(neg_llmnl, np.zeros(nvar), method='BFGS')
    betastar = result.x
    
    mhess = mnl_hess(betastar, y_0idx.astype(int), X)
    candcov = np.linalg.inv(mhess)
    root = np.linalg.cholesky(candcov).T
    rooti = np.linalg.solve(root, np.eye(nvar))
    
    priorcov = np.linalg.inv(A)
    rootp = np.linalg.cholesky(priorcov).T
    rootpi = np.linalg.solve(rootp, np.eye(nvar))
    
    beta = betastar.copy()
    oldloglike = cpp.llmnl(beta, y_0idx, X)
    oldlpost = oldloglike + cpp.lndMvn(beta, betabar, rootpi)
    oldlimp = cpp.lndMvst(beta, nu, betastar, rooti, False)
    
    betadraw, loglike, naccept = cpp.rmnlIndepMetrop_rcpp_loop(
        R, keep, nu, betastar, root, y_0idx, X, betabar, rootpi, rooti, oldlimp, oldlpost
    )
    
    if nprint > 0:
        print(f"Acceptance rate: {naccept/R:.3f}")
    
    return {
        'betadraw': betadraw,
        'loglike': loglike,
        'acceptr': naccept / R
    }
