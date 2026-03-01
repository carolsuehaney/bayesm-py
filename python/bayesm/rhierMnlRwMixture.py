# rhierMnlRwMixture - Hierarchical MNL with Mixture of Normals
import numpy as np
from scipy.optimize import minimize
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp


def _llmnl_con(betastar, y, X, SignRes):
    """Log-likelihood for MNL with sign constraints."""
    beta = betastar.copy()
    sign_idx = SignRes != 0
    beta[sign_idx] = SignRes[sign_idx] * np.exp(betastar[sign_idx])
    
    n = len(y)
    j = X.shape[0] // n
    Xbeta = X @ beta
    
    xby = np.zeros(n)
    denom = np.zeros(n)
    
    for i in range(n):
        for p in range(j):
            denom[i] += np.exp(Xbeta[i * j + p])
        xby[i] = Xbeta[i * j + int(y[i]) - 1]
    
    return np.sum(xby - np.log(denom))


def _mnlHess_con(betastar, y, X, SignRes):
    """Compute Hessian for MNL with sign constraints."""
    beta = betastar.copy()
    beta[SignRes != 0] = SignRes[SignRes != 0] * np.exp(betastar[SignRes != 0])
    
    n = len(y)
    j = X.shape[0] // n
    k = X.shape[1]
    
    Xbeta = X @ beta
    Xbeta = Xbeta.reshape(n, j)
    Xbeta = np.exp(Xbeta)
    denom = Xbeta.sum(axis=1)
    Prob = Xbeta / denom[:, np.newaxis]
    
    Hess = np.zeros((k, k))
    for i in range(n):
        p = Prob[i, :]
        A = np.diag(p) - np.outer(p, p)
        Xt = X[j * i:j * (i + 1), :]
        Hess += Xt.T @ A @ Xt
    
    lam = np.ones(len(SignRes))
    lam[SignRes == 1] = beta[SignRes == 1]
    lam[SignRes == -1] = -beta[SignRes == -1]
    Hess = Hess * np.outer(lam, lam)
    
    return Hess


def rhierMnlRwMixture(Data, Prior, Mcmc):
    """
    Hierarchical Multinomial Logit with Mixture of Normals heterogeneity.
    """
    if Data is None or 'p' not in Data:
        pandterm("Requires Data element p (# choice alternatives)")
    if 'lgtdata' not in Data:
        pandterm("Requires Data element lgtdata")
    
    p = Data['p']
    lgtdata = Data['lgtdata']
    nlgt = len(lgtdata)
    drawdelta = True
    
    if 'Z' not in Data or Data['Z'] is None:
        drawdelta = False
        Z = np.zeros((1, 1))
    else:
        Z = np.asarray(Data['Z'], dtype=np.float64)
        if Z.shape[0] != nlgt:
            pandterm(f"Nrow(Z) {Z.shape[0]} ne number logits {nlgt}")
    
    nz = Z.shape[1] if drawdelta else 1
    
    ypooled = []
    Xpooled = []
    for i, ld in enumerate(lgtdata):
        if 'y' not in ld or 'X' not in ld:
            pandterm(f"Requires elements y and X in lgtdata[{i}]")
        ypooled.extend(ld['y'])
        Xpooled.append(ld['X'])
    
    ypooled = np.array(ypooled)
    Xpooled = np.vstack(Xpooled)
    nvar = Xpooled.shape[1]
    
    ncomp = Prior.get('ncomp')
    if ncomp is None:
        pandterm("Requires Prior element ncomp")
    
    SignRes = Prior.get('SignRes', np.zeros(nvar))
    SignRes = np.asarray(SignRes, dtype=np.float64)
    
    has_constraints = np.sum(np.abs(SignRes)) > 0
    
    if has_constraints:
        mubar = Prior.get('mubar', np.zeros(nvar) + 2 * np.abs(SignRes))
        Amu = Prior.get('Amu', np.array([[BayesmConstants.A * 10]]))
        nu = Prior.get('nu', nvar + BayesmConstants.nuInc + 12)
        V = Prior.get('V', nu * np.diag(np.abs(SignRes) * 0.1 + (1 - np.abs(SignRes)) * 4))
    else:
        mubar = Prior.get('mubar', np.zeros(nvar))
        Amu = Prior.get('Amu', np.array([[BayesmConstants.A]]))
        nu = Prior.get('nu', nvar + BayesmConstants.nuInc)
        V = Prior.get('V', nu * np.eye(nvar))
    
    mubar = np.atleast_2d(mubar)
    if np.isscalar(Amu):
        Amu = np.array([[Amu]])
    
    if drawdelta:
        Ad = Prior.get('Ad', BayesmConstants.A * np.eye(nvar * nz))
        deltabar = Prior.get('deltabar', np.zeros(nz * nvar))
    else:
        Ad = np.zeros((1, 1))
        deltabar = np.zeros(1)
    
    a = Prior.get('a', np.repeat(BayesmConstants.a, ncomp))
    
    s = Mcmc.get('s', BayesmConstants.RRScaling / np.sqrt(nvar))
    w = Mcmc.get('w', BayesmConstants.w)
    keep = Mcmc.get('keep', BayesmConstants.keep)
    R = Mcmc.get('R')
    if R is None:
        pandterm("Requires R argument in Mcmc")
    
    betainit = np.zeros(nvar)
    noRes = np.zeros(nvar)
    
    def neg_llmnl(beta, y, X, SignRes):
        return -_llmnl_con(beta, y, X, SignRes)
    
    out = minimize(neg_llmnl, betainit, args=(ypooled, Xpooled, noRes), method='BFGS')
    betainit = out.x
    betainit[SignRes != 0] = 0
    
    out = minimize(neg_llmnl, betainit, args=(ypooled, Xpooled, SignRes), method='Nelder-Mead')
    betapooled = out.x
    
    H = _mnlHess_con(betapooled, ypooled, Xpooled, SignRes)
    rootH = np.linalg.cholesky(H).T
    
    oldbetas = np.zeros((nlgt, nvar))
    lgtdata_cpp = []
    
    def llmnlFract(beta, y, X, betapooled, rootH, w, wgt, SignRes):
        z = rootH @ (beta - betapooled)
        return (1 - w) * _llmnl_con(beta, y, X, SignRes) + w * wgt * (-0.5 * (z @ z))
    
    def neg_llmnlFract(beta, y, X, betapooled, rootH, w, wgt, SignRes):
        return -llmnlFract(beta, y, X, betapooled, rootH, w, wgt, SignRes)
    
    for i in range(nlgt):
        y_i = np.asarray(lgtdata[i]['y'], dtype=np.float64)
        X_i = np.asarray(lgtdata[i]['X'], dtype=np.float64)
        wgt = len(y_i) / len(ypooled)
        
        out = minimize(neg_llmnlFract, betapooled, 
                       args=(y_i, X_i, betapooled, rootH, w, wgt, SignRes),
                       method='BFGS', options={'maxiter': 100})
        
        if out.success:
            hess = _mnlHess_con(out.x, y_i, X_i, SignRes)
            betafmle = out.x
        else:
            hess = np.eye(nvar)
            betafmle = np.zeros(nvar)
        
        oldbetas[i, :] = betafmle
        lgtdata_cpp.append({'y': y_i, 'X': X_i, 'hess': hess})
    
    ninc = nlgt // ncomp
    ind = np.zeros(nlgt)
    for i in range(ncomp - 1):
        ind[i * ninc:(i + 1) * ninc] = i + 1
    ind[(ncomp - 1) * ninc:] = ncomp
    if ncomp == 1:
        ind[:] = 1
    
    olddelta = np.zeros(nz * nvar) if drawdelta else np.zeros(1)
    oldprob = np.repeat(1.0 / ncomp, ncomp)
    
    Deltadraw, betadraw, probdraw, loglike, compdraw = cpp.rhierMnlRwMixture_rcpp_loop(
        lgtdata_cpp, Z, deltabar, Ad, mubar, Amu, nu, V, s,
        R, keep, drawdelta, olddelta, a, oldprob, oldbetas, ind, SignRes)
    
    nmix = {'probdraw': probdraw, 'zdraw': None, 'compdraw': compdraw}
    
    result = {'betadraw': betadraw, 'nmix': nmix, 'loglike': loglike, 'SignRes': SignRes}
    if drawdelta:
        result['Deltadraw'] = Deltadraw
    
    return result
