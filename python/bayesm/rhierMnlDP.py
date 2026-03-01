# rhierMnlDP - Hierarchical MNL with Dirichlet Process prior
import numpy as np
from scipy.special import digamma
from scipy.optimize import minimize
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp


def _llmnl(beta, y, X):
    # X is n*p x nvar, y is n (choices 1 to p)
    n = len(y)
    nvar = len(beta)
    p = X.shape[0] // n
    Xbeta = (X @ beta).reshape(n, p)
    Xbeta = Xbeta - Xbeta.max(axis=1, keepdims=True)
    Xbeta = np.exp(Xbeta)
    denom = Xbeta.sum(axis=1)
    ll = sum(np.log(Xbeta[i, int(y[i]) - 1]) for i in range(n)) - np.sum(np.log(denom))
    return ll


def _mnlHess(beta, y, X):
    # X is n*p x nvar
    n = len(y)
    k = len(beta)
    p = X.shape[0] // n
    Xbeta = (X @ beta).reshape(n, p)
    Xbeta = Xbeta - Xbeta.max(axis=1, keepdims=True)
    Xbeta = np.exp(Xbeta)
    prob = Xbeta / Xbeta.sum(axis=1, keepdims=True)
    
    H = np.zeros((k, k))
    for i in range(n):
        Xi = X[i * p:(i + 1) * p, :]
        pi = prob[i, :]
        H += Xi.T @ (np.diag(pi) - np.outer(pi, pi)) @ Xi
    return -H


def rhierMnlDP(Data, Prior=None, Mcmc=None):
    """
    Hierarchical Multinomial Logit with Dirichlet Process prior.
    
    Uses RW Metropolis and a DP mixture of normals for heterogeneity.
    """
    if Data is None:
        pandterm("Requires Data argument -- list of p,lgtdata, and (possibly) Z")
    
    if 'p' not in Data:
        pandterm("Requires Data element p (# choice alternatives)")
    p = Data['p']
    
    if 'lgtdata' not in Data:
        pandterm("Requires Data element lgtdata")
    lgtdata = Data['lgtdata']
    nlgt = len(lgtdata)
    
    drawdelta = 'Z' in Data and Data['Z'] is not None
    if drawdelta:
        Z = np.asarray(Data['Z'], dtype=np.float64)
        nz = Z.shape[1]
    else:
        Z = np.zeros((nlgt, 1))
        nz = 1
    
    ypooled = np.concatenate([np.asarray(lgtdata[i]['y']).flatten() for i in range(nlgt)])
    Xpooled = np.vstack([np.asarray(lgtdata[i]['X']) for i in range(nlgt)])
    nvar = Xpooled.shape[1]
    
    alimdef = BayesmConstants.DPalimdef
    nulimdef = BayesmConstants.DPnulimdef
    vlimdef = BayesmConstants.DPvlimdef
    
    if Prior is None:
        Prior = {}
    
    lambda_hyper = Prior.get('lambda_hyper', {})
    alim = lambda_hyper.get('alim', alimdef)
    nulim = lambda_hyper.get('nulim', nulimdef)
    vlim = lambda_hyper.get('vlim', vlimdef)
    
    Prioralpha = Prior.get('Prioralpha', {})
    Istarmin = Prioralpha.get('Istarmin', BayesmConstants.DPIstarmin)
    Istarmax = Prioralpha.get('Istarmax', min(50, int(0.1 * nlgt)))
    power = Prioralpha.get('power', BayesmConstants.DPpower)
    gamma_const = BayesmConstants.gamma
    alphamin = np.exp(digamma(Istarmin) - np.log(gamma_const + np.log(nlgt)))
    alphamax = np.exp(digamma(Istarmax) - np.log(gamma_const + np.log(nlgt)))
    
    if drawdelta:
        Ad = Prior.get('Ad', BayesmConstants.A * np.eye(nvar * nz))
        deltabar = Prior.get('deltabar', np.zeros(nz * nvar))
    else:
        Ad = np.eye(1)
        deltabar = np.zeros(1)
    
    if Mcmc is None:
        pandterm("Requires Mcmc list argument")
    
    R = Mcmc.get('R')
    if R is None:
        pandterm("Requires R argument in Mcmc list")
    keep = Mcmc.get('keep', BayesmConstants.keep)
    s = Mcmc.get('s', BayesmConstants.RRScaling / np.sqrt(nvar))
    maxuniq = Mcmc.get('maxuniq', BayesmConstants.DPmaxuniq)
    gridsize = Mcmc.get('gridsize', BayesmConstants.DPgridsize)
    
    # Initialize betas with optim
    oldbetas = np.zeros((nlgt, nvar))
    lgtdata_cpp = []
    for i in range(nlgt):
        y_i = np.asarray(lgtdata[i]['y'], dtype=np.float64).flatten()
        X_i = np.asarray(lgtdata[i]['X'], dtype=np.float64)
        
        def negll(b):
            return -_llmnl(b, y_i, X_i)
        
        res = minimize(negll, np.zeros(nvar), method='BFGS')
        oldbetas[i, :] = res.x
        hess = _mnlHess(res.x, y_i, X_i)
        # Ensure hess is negative definite (regularize if needed)
        eigvals = np.linalg.eigvalsh(hess)
        if np.max(eigvals) > -1e-6:
            hess = hess - (np.max(eigvals) + 0.01) * np.eye(nvar)
        
        lgtdata_cpp.append({
            'y': y_i,
            'X': X_i,
            'hess': hess
        })
    
    alim = np.asarray(alim, dtype=np.float64)
    nulim = np.asarray(nulim, dtype=np.float64)
    vlim = np.asarray(vlim, dtype=np.float64)
    deltabar = np.asarray(deltabar, dtype=np.float64)
    Ad = np.asarray(Ad, dtype=np.float64)
    
    Deltadraw, betadraw, probdraw, loglike, alphadraw, Istardraw, adraw, nudraw, vdraw, compdraw = cpp.rhierMnlDP_rcpp_loop(
        R, keep, lgtdata_cpp, Z, deltabar, Ad,
        power, alphamin, alphamax, nlgt, alim, nulim, vlim,
        drawdelta, nvar, oldbetas, s, maxuniq, gridsize,
        BayesmConstants.A, BayesmConstants.nuInc, BayesmConstants.DPalpha)
    
    nmix = {
        'probdraw': probdraw,
        'zdraw': None,
        'compdraw': compdraw
    }
    
    result = {
        'betadraw': betadraw,
        'nmix': nmix,
        'alphadraw': alphadraw,
        'Istardraw': Istardraw,
        'adraw': adraw,
        'nudraw': nudraw,
        'vdraw': vdraw,
        'loglike': loglike
    }
    
    if drawdelta:
        result['Deltadraw'] = Deltadraw
    
    return result
