"""Negative binomial regression with random walk Metropolis."""
import numpy as np
from scipy.optimize import minimize
from .utilities import pandterm
from . import _cpp as cpp

def _llnegbin_py(par, X, y, nvar):
    """Log-likelihood for negative binomial (for optimization)."""
    beta = par[:nvar]
    alpha = np.exp(par[nvar]) + 1e-50
    mean = np.exp(X @ beta)
    prob = alpha / (alpha + mean)
    prob = np.clip(prob, 1e-100, 1 - 1e-100)
    from scipy.special import gammaln
    ll = gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1) + alpha * np.log(prob) + y * np.log(1 - prob)
    return np.sum(ll)

def rnegbinRw(Data, Prior=None, Mcmc=None):
    """
    Random walk Metropolis sampler for negative binomial regression.
    
    Model: (y|lambda,alpha) ~ NegBin(mean=lambda, overdispersion=alpha)
           ln(lambda) = X * beta
    
    Prior: beta ~ N(betabar, A^-1), alpha ~ Gamma(a, b)
    """
    if Data is None:
        pandterm("Requires Data argument")
    X = Data.get('X')
    y = Data.get('y')
    if X is None or y is None:
        pandterm("Requires Data elements X and y")
    
    nvar = X.shape[1]
    nobs = len(y)
    if len(y) != X.shape[0]:
        pandterm("Mismatch in number of observations")
    
    if Prior is None:
        betabar = np.zeros(nvar)
        A = 0.01 * np.eye(nvar)
        a = 0.5
        b = 0.1
    else:
        betabar = Prior.get('betabar', np.zeros(nvar))
        A = Prior.get('A', 0.01 * np.eye(nvar))
        a = Prior.get('a', 0.5)
        b = Prior.get('b', 0.1)
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    R = Mcmc.get('R')
    if R is None:
        pandterm("requires Mcmc element R")
    keep = Mcmc.get('keep', 1)
    beta0 = Mcmc.get('beta0', np.zeros(nvar))
    s_alpha = Mcmc.get('s_alpha', 2.93)
    s_beta = Mcmc.get('s_beta', 2.93 / np.sqrt(nvar))
    fixalpha = 'alpha' in Mcmc
    alpha_fixed = Mcmc.get('alpha', 1.0)
    
    par0 = np.zeros(nvar + 1)
    result = minimize(lambda p: -_llnegbin_py(p, X, y, nvar), par0, method='L-BFGS-B',
                      bounds=[(None, None)] * nvar + [(None, np.log(1e8))])
    
    beta_mle = result.x[:nvar]
    alpha_mle = np.exp(result.x[nvar])
    varcovinv = np.eye(nvar + 1)
    try:
        from scipy.optimize import approx_fprime
        def neg_ll(p): return -_llnegbin_py(p, X, y, nvar)
        hess = np.zeros((nvar + 1, nvar + 1))
        eps = 1e-5
        for i in range(nvar + 1):
            e = np.zeros(nvar + 1)
            e[i] = eps
            hess[i, :] = (approx_fprime(result.x + e, neg_ll, eps) - approx_fprime(result.x - e, neg_ll, eps)) / (2 * eps)
        varcovinv = hess
    except:
        varcovinv = 0.1 * np.eye(nvar + 1)
    
    beta = beta0
    betacvar = s_beta * np.linalg.inv(varcovinv[:nvar, :nvar] + 0.01 * np.eye(nvar))
    betaroot = np.linalg.cholesky(betacvar)
    alpha = alpha_fixed if fixalpha else alpha_mle
    alphacvar = s_alpha / (varcovinv[nvar, nvar] + 0.01)
    alphacroot = np.sqrt(alphacvar)
    rootA = np.linalg.cholesky(A)
    
    betadraw, alphadraw, nacceptbeta, nacceptalpha = cpp.rnegbinRw_rcpp_loop(
        np.asfortranarray(y), np.asfortranarray(X), np.asfortranarray(betabar),
        np.asfortranarray(rootA), float(a), float(b), np.asfortranarray(beta),
        float(alpha), bool(fixalpha), np.asfortranarray(betaroot), float(alphacroot),
        int(R), int(keep))
    
    return {'betadraw': betadraw, 'alphadraw': alphadraw,
            'acceptrbeta': nacceptbeta / (R / keep), 'acceptralpha': nacceptalpha / (R / keep)}
