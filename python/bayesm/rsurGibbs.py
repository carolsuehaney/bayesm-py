"""SUR Gibbs sampler - seemingly unrelated regression."""
import numpy as np
from .utilities import pandterm
from . import _cpp as cpp

def rsurGibbs(Data, Prior=None, Mcmc=None):
    """
    Gibbs sampler for Seemingly Unrelated Regression.
    
    Model: y_i = X_i * beta_i + e_i for i=1,...,nreg
           (e_1k,...,e_nreg,k) ~ N(0, Sigma) for k=1,...,nobs
    
    Prior: beta ~ N(betabar, A^-1), Sigma ~ IW(nu, V)
    """
    if Data is None:
        pandterm("Requires Data argument")
    regdata = Data.get('regdata')
    if regdata is None:
        pandterm("Requires Data element regdata")
    
    nreg = len(regdata)
    nobs = len(regdata[0]['y'])
    nvar = 0
    indreg = np.zeros(nreg + 1)
    
    for reg in range(nreg):
        y_reg = regdata[reg]['y']
        X_reg = regdata[reg]['X']
        if len(y_reg) != nobs or X_reg.shape[0] != nobs:
            pandterm(f"incorrect dimensions for regression {reg}")
        indreg[reg] = nvar + 1
        nvar += X_reg.shape[1]
    indreg[nreg] = nvar + 1
    
    if Prior is None:
        betabar = np.zeros(nvar)
        A = 0.01 * np.eye(nvar)
        nu = nreg + 3
        V = nu * np.eye(nreg)
    else:
        betabar = Prior.get('betabar', np.zeros(nvar))
        A = Prior.get('A', 0.01 * np.eye(nvar))
        nu = Prior.get('nu', nreg + 3)
        V = Prior.get('V', nu * np.eye(nreg))
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    R = Mcmc.get('R')
    if R is None:
        pandterm("requires Mcmc element R")
    keep = Mcmc.get('keep', 1)
    
    E = np.zeros((nobs, nreg))
    for reg in range(nreg):
        y_reg = regdata[reg]['y']
        X_reg = regdata[reg]['X']
        beta_ols = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
        E[:, reg] = y_reg - X_reg @ beta_ols
    
    Sigma = (E.T @ E + 0.01 * np.eye(nreg)) / nobs
    Sigmainv = np.linalg.inv(Sigma)
    
    nk = np.array([regdata[i]['X'].shape[1] for i in range(nreg)], dtype=float)
    cumnk = np.cumsum(nk)
    
    Xstar = np.hstack([regdata[i]['X'] for i in range(nreg)])
    Y = np.column_stack([regdata[i]['y'] for i in range(nreg)])
    XspXs = Xstar.T @ Xstar
    Abetabar = A @ betabar
    
    regdata_list = [{'y': np.asfortranarray(regdata[i]['y']), 
                     'X': np.asfortranarray(regdata[i]['X'])} for i in range(nreg)]
    
    betadraw, Sigmadraw = cpp.rsurGibbs_rcpp_loop(
        regdata_list, np.asfortranarray(indreg), np.asfortranarray(cumnk),
        np.asfortranarray(nk), np.asfortranarray(XspXs), np.asfortranarray(Sigmainv),
        np.asfortranarray(A), np.asfortranarray(Abetabar), float(nu),
        np.asfortranarray(V), int(nvar), np.asfortranarray(E), np.asfortranarray(Y),
        int(R), int(keep))
    
    return {'betadraw': betadraw, 'Sigmadraw': Sigmadraw, 'nreg': nreg}
