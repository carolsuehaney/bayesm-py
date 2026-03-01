"""Linear Instrumental Variables Gibbs sampler."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp

from .constants import BayesmConstants
from .utilities import pandterm


def rivGibbs(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for Linear IV Model.
    
    Model: x = z'delta + e1
           y = beta*x + w'gamma + e2
           (e1, e2) ~ N(0, Sigma)
    
    Prior: delta ~ N(md, Ad^-1)
           (beta, gamma) ~ N(mbg, Abg^-1)
           Sigma ~ IW(nu, V)
    
    Parameters
    ----------
    Data : dict
        y : array (n,) - dependent variable
        x : array (n,) - endogenous variable
        z : array (n, dimd) - instruments
        w : array (n, dimg) - exogenous variables
    Prior : dict, optional
        md : array (dimd,) - prior mean for delta
        Ad : array (dimd, dimd) - prior precision for delta
        mbg : array (1+dimg,) - prior mean for (beta, gamma)
        Abg : array (1+dimg, 1+dimg) - prior precision for (beta, gamma)
        nu : float - df for IW prior on Sigma (default: 3)
        V : array (2, 2) - scale for IW prior on Sigma
    Mcmc : dict
        R : int - number of draws
        keep : int - thinning (default: 1)
        nprint : int - print interval (default: 100, 0 for silent)
    
    Returns
    -------
    dict with:
        deltadraw : array (R/keep, dimd)
        betadraw : array (R/keep,)
        gammadraw : array (R/keep, dimg)
        Sigmadraw : array (R/keep, 4) - vectorized 2x2 Sigma
    """
    if Data is None:
        pandterm("Requires Data argument -- dict with z, w, x, y")
    if 'z' not in Data:
        pandterm("Requires Data element z")
    if 'w' not in Data:
        pandterm("Requires Data element w")
    if 'x' not in Data:
        pandterm("Requires Data element x")
    if 'y' not in Data:
        pandterm("Requires Data element y")
    
    z = np.asarray(Data['z'], dtype=np.float64)
    w = np.asarray(Data['w'], dtype=np.float64)
    x = np.asarray(Data['x'], dtype=np.float64).flatten()
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    
    n = len(y)
    dimd = z.shape[1]
    dimg = w.shape[1]
    
    if len(x) != n:
        pandterm("length(y) != length(x)")
    if z.shape[0] != n:
        pandterm("length(y) != nrow(z)")
    if w.shape[0] != n:
        pandterm("length(y) != nrow(w)")
    
    if Prior is None:
        md = np.zeros(dimd)
        Ad = BayesmConstants.A * np.eye(dimd)
        mbg = np.zeros(1 + dimg)
        Abg = BayesmConstants.A * np.eye(1 + dimg)
        nu = 3.0
        V = np.eye(2)
    else:
        md = Prior.get('md', np.zeros(dimd))
        Ad = Prior.get('Ad', BayesmConstants.A * np.eye(dimd))
        mbg = Prior.get('mbg', np.zeros(1 + dimg))
        Abg = Prior.get('Abg', BayesmConstants.A * np.eye(1 + dimg))
        nu = Prior.get('nu', 3.0)
        V = Prior.get('V', nu * np.eye(2))
    
    if Ad.shape != (dimd, dimd):
        pandterm(f"bad dimensions for Ad: {Ad.shape}")
    if len(md) != dimd:
        pandterm(f"md wrong length: {len(md)}")
    if Abg.shape != (1 + dimg, 1 + dimg):
        pandterm(f"bad dimensions for Abg: {Abg.shape}")
    if len(mbg) != (1 + dimg):
        pandterm(f"mbg wrong length: {len(mbg)}")
    
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
        print("Starting Gibbs Sampler for Linear IV Model")
        print()
        print(f" nobs = {n}; {dimd} instruments; {dimg} included exog vars")
        print("     Note: the numbers above include intercepts if in z or w")
        print()
        print("Prior Parms:")
        print(f"  md: {md}")
        print(f"  mbg: {mbg}")
        print(f"  nu: {nu}")
        print()
        print(f"MCMC parms: R = {R}, keep = {keep}")
        print()
    
    deltadraw, betadraw, gammadraw, Sigmadraw = cpp.rivGibbs_rcpp_loop(
        y, x, z, w, mbg, Abg, md, Ad, V, nu, R, keep
    )
    
    return {
        'deltadraw': deltadraw,
        'betadraw': betadraw,
        'gammadraw': gammadraw,
        'Sigmadraw': Sigmadraw
    }
