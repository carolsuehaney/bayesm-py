"""Binary Probit Gibbs sampler."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp

from .constants import BayesmConstants
from .utilities import pandterm


def rbprobitGibbs(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for Binary Probit Model.
    
    Model: y = 1 if w = X*beta + e > 0, e ~ N(0,1)
    Prior: beta ~ N(betabar, A^-1)
    
    Parameters
    ----------
    Data : dict
        y : array (nobs,) - binary outcomes (0 or 1)
        X : array (nobs, nvar) - design matrix
    Prior : dict, optional
        betabar : array (nvar,) - prior mean (default: zeros)
        A : array (nvar, nvar) - prior precision (default: 0.01*I)
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
    
    Returns
    -------
    dict with:
        betadraw : array (R/keep, nvar)
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with y, X")
    if 'X' not in Data:
        pandterm("Requires Data element X")
    if 'y' not in Data:
        pandterm("Requires Data element y")
    
    X = np.asarray(Data['X'], dtype=np.float64)
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    
    nvar = X.shape[1]
    nobs = len(y)
    
    if len(y) != X.shape[0]:
        pandterm("y and X not of same row dim")
    
    unique_y = np.unique(y)
    if not np.all(np.isin(unique_y, [0, 1])):
        pandterm("Invalid y, must be 0,1")
    
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
    
    if nprint < 0:
        pandterm("nprint must be >= 0")
    
    if nprint > 0:
        print()
        print("Starting Gibbs Sampler for Binary Probit Model")
        print(f"   with {nobs} observations")
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
        print(f"MCMC parms: R = {R}, keep = {keep}")
        print()
    
    beta = np.zeros(nvar)
    sigma = np.ones(nobs)
    
    XtX = X.T @ X
    root = np.linalg.cholesky(np.linalg.inv(XtX + A)).T
    Abetabar = A @ betabar
    
    above = np.where(y == 0, 1.0, 0.0)
    trunpt = np.zeros(nobs)
    
    betadraw = cpp.rbprobitGibbs_rcpp_loop(
        y, X, Abetabar, root, beta, sigma, trunpt, above, R, keep
    )
    
    return {'betadraw': betadraw}
