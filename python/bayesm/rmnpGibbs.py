"""Multinomial probit Gibbs sampler."""
import numpy as np
from .utilities import pandterm
from . import _cpp as cpp

def rmnpGibbs(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for multinomial probit model.
    
    Model: w_i = X_i * beta + e, e ~ N(0, Sigma), dim = p-1
           y_i = j if w_ij > w_i-j for j=1,...,p-1
           y_i = p if all w_i < 0
    
    Prior: beta ~ N(betabar, A^-1), Sigma ~ IW(nu, V)
    """
    if Data is None:
        pandterm("Requires Data argument")
    p = Data.get('p')
    if p is None:
        pandterm("Requires Data element p")
    y = Data.get('y')
    X = Data.get('X')
    if y is None or X is None:
        pandterm("Requires Data elements y and X")
    
    y = np.asarray(y).flatten()
    n = len(y)
    k = X.shape[1]
    pm1 = p - 1
    
    if X.shape[0] / n != pm1:
        pandterm(f"X has {X.shape[0]} rows; must be = (p-1)*n")
    
    if Prior is None:
        betabar = np.zeros(k)
        A = 0.01 * np.eye(k)
        nu = pm1 + 3
        V = nu * np.eye(pm1)
    else:
        betabar = Prior.get('betabar', np.zeros(k))
        A = Prior.get('A', 0.01 * np.eye(k))
        nu = Prior.get('nu', pm1 + 3)
        V = Prior.get('V', nu * np.eye(pm1))
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    R = Mcmc.get('R')
    if R is None:
        pandterm("requires Mcmc element R")
    keep = Mcmc.get('keep', 1)
    beta0 = Mcmc.get('beta0', np.zeros(k))
    sigma0 = Mcmc.get('sigma0', np.eye(pm1))
    
    y_int = y.astype(np.int32)
    
    betadraw, sigmadraw = cpp.rmnpGibbs_rcpp_loop(
        int(R), int(keep), int(pm1),
        np.ascontiguousarray(y_int), np.asfortranarray(X),
        np.asfortranarray(beta0), np.asfortranarray(sigma0),
        np.asfortranarray(V), float(nu),
        np.asfortranarray(betabar), np.asfortranarray(A))
    
    return {'betadraw': betadraw, 'sigmadraw': sigmadraw}
