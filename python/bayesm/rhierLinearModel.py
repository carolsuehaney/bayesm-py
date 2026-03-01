"""Hierarchical linear model Gibbs sampler."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp

from .constants import BayesmConstants
from .utilities import pandterm


def rhierLinearModel(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for hierarchical linear model.
    
    Model: y_i = X_i @ beta_i + e_i, e_i ~ N(0, tau_i)
    Prior: 
        tau_i ~ nu_e * ssq_i / chisq(nu_e)
        beta_i ~ N(Z[i,:] @ Delta, Vbeta)
        vec(Delta) | Vbeta ~ N(vec(Deltabar), Vbeta (x) A^-1)
        Vbeta ~ IW(nu, V)
    
    Parameters
    ----------
    Data : dict
        regdata : list of dicts, each with 'y' (n_i,) and 'X' (n_i, nvar)
        Z : array (nreg, nz), optional - covariates for Delta (default: ones)
    Prior : dict, optional
        Deltabar : array (nz, nvar) - prior mean for Delta
        A : array (nz, nz) - prior precision for Delta
        nu_e : float - df for tau_i (default: 3)
        ssq : array (nreg,) - scale for tau_i (default: var(y_i))
        nu : float - df for Vbeta (default: nvar + 3)
        V : array (nvar, nvar) - scale for Vbeta
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
    
    Returns
    -------
    dict with:
        betadraw : array (nreg, nvar, R/keep)
        taudraw : array (R/keep, nreg)
        Deltadraw : array (R/keep, nz*nvar)
        Vbetadraw : array (R/keep, nvar*nvar)
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with regdata and Z")
    if 'regdata' not in Data:
        pandterm("Requires Data element regdata")
    
    regdata = Data['regdata']
    nreg = len(regdata)
    
    if 'Z' not in Data or Data['Z'] is None:
        Z = np.ones((nreg, 1))
    else:
        Z = np.asarray(Data['Z'], dtype=np.float64)
        if Z.shape[0] != nreg:
            pandterm(f"nrow(Z) {Z.shape[0]} != number of regressions {nreg}")
    
    nz = Z.shape[1]
    
    nvar = None
    for i, rd in enumerate(regdata):
        if not isinstance(rd.get('X'), np.ndarray):
            pandterm(f"regdata[{i}]['X'] must be a numpy array")
        if nvar is None:
            nvar = rd['X'].shape[1]
        elif rd['X'].shape[1] != nvar:
            pandterm(f"regdata[{i}]['X'] has wrong number of columns")
    
    def getvar(rd):
        v = np.var(rd['y'])
        return v if v > 0 and np.isfinite(v) else 1.0
    
    if Prior is None:
        Deltabar = np.zeros((nz, nvar))
        A = BayesmConstants.A * np.eye(nz)
        nu_e = BayesmConstants.nu_e
        ssq = np.array([getvar(rd) for rd in regdata])
        nu = nvar + BayesmConstants.nuInc
        V = nu * np.eye(nvar)
    else:
        Deltabar = Prior.get('Deltabar', np.zeros((nz, nvar)))
        A = Prior.get('A', BayesmConstants.A * np.eye(nz))
        nu_e = Prior.get('nu_e', BayesmConstants.nu_e)
        ssq = Prior.get('ssq', np.array([getvar(rd) for rd in regdata]))
        nu = Prior.get('nu', nvar + BayesmConstants.nuInc)
        V = Prior.get('V', nu * np.eye(nvar))
    
    if A.shape != (nz, nz):
        pandterm(f"bad dimensions for A: {A.shape}")
    if Deltabar.shape != (nz, nvar):
        pandterm(f"bad dimensions for Deltabar: {Deltabar.shape}")
    if len(ssq) != nreg:
        pandterm(f"bad length for ssq: {len(ssq)}")
    if V.shape != (nvar, nvar):
        pandterm(f"bad dimensions for V: {V.shape}")
    
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
        print("Starting Gibbs Sampler for Linear Hierarchical Model")
        print(f"  {nreg} Regressions")
        print(f"  {nz} Variables in Z (if 1, then only intercept)")
        print()
        print(f"MCMC parms: R = {R}, keep = {keep}")
        print()
    
    regdata_cpp = []
    tau = np.array([getvar(rd) for rd in regdata])
    for rd in regdata:
        y = np.asarray(rd['y'], dtype=np.float64).flatten()
        X = np.asarray(rd['X'], dtype=np.float64)
        regdata_cpp.append({
            'y': y,
            'X': X,
            'XpX': X.T @ X,
            'Xpy': X.T @ y
        })
    
    Delta = np.zeros((nz, nvar))
    Vbeta = np.eye(nvar)
    
    betadraw, taudraw, Deltadraw, Vbetadraw = cpp.rhierLinearModel_rcpp_loop(
        regdata_cpp, Z, Deltabar, A, nu, V, nu_e, ssq, tau, Delta, Vbeta, R, keep
    )
    
    return {
        'betadraw': betadraw,
        'taudraw': taudraw,
        'Deltadraw': Deltadraw,
        'Vbetadraw': Vbetadraw
    }
