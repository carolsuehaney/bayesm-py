"""
rhierNegbinRw - Hierarchical Negative Binomial with Random Walk Metropolis
"""
import numpy as np
from scipy.optimize import minimize
from .constants import BayesmConstants
from ._cpp import _bayesm_cpp


def _llnegbinR(par, X, y, nvar):
    """Log-likelihood for negative binomial."""
    beta = par[:nvar]
    alpha = np.exp(par[nvar]) + 1e-50
    mean = np.exp(X @ beta)
    prob = alpha / (alpha + mean)
    prob = np.maximum(prob, 1e-100)
    from scipy.stats import nbinom
    ll = nbinom.logpmf(y, n=alpha, p=prob)
    return np.sum(ll)


def _llnegbinFract(par, X, y, Xpooled, ypooled, w, wgt, nvar, lnalpha):
    """Fractional log-likelihood at unit level."""
    theta = np.concatenate([par, [lnalpha]])
    return (1 - w) * _llnegbinR(theta, X, y, nvar) + w * wgt * _llnegbinR(theta, Xpooled, ypooled, nvar)


def rhierNegbinRw(Data, Prior=None, Mcmc=None):
    """
    Hierarchical Negative Binomial regression with Random Walk Metropolis.
    
    Parameters
    ----------
    Data : dict
        regdata : list of dicts with 'y' and 'X' for each unit
        Z : array (nreg, nz), optional - unit-level characteristics
    Prior : dict, optional
        Deltabar, Adelta, nu, V, a, b
    Mcmc : dict
        R : int - number of draws
        keep, s_alpha, s_beta, w, Vbeta0, Delta0, alpha (fixed), nprint
    
    Returns
    -------
    dict with Betadraw, alphadraw, Vbetadraw, Deltadraw, llike, acceptrbeta, acceptralpha
    """
    if Data is None:
        raise ValueError("Data is required")
    
    regdata = Data.get('regdata')
    if regdata is None:
        raise ValueError("Data must contain regdata")
    
    nreg = len(regdata)
    Z = Data.get('Z')
    if Z is None:
        Z = np.ones((nreg, 1))
    Z = np.asarray(Z, dtype=np.float64)
    nz = Z.shape[1]
    
    # Get dimensions
    nvar = regdata[0]['X'].shape[1]
    
    # Pool data
    ypooled = np.concatenate([reg['y'].ravel() for reg in regdata])
    Xpooled = np.vstack([reg['X'] for reg in regdata])
    nobs = len(ypooled)
    
    # Prior defaults
    if Prior is None:
        Prior = {}
    
    Deltabar = Prior.get('Deltabar', np.zeros((nz, nvar)))
    Adelta = Prior.get('Adelta', BayesmConstants.A * np.eye(nz))
    nu = Prior.get('nu', nvar + BayesmConstants.nuInc)
    V = Prior.get('V', nu * np.eye(nvar))
    a = Prior.get('a', BayesmConstants.agammaprior)
    b = Prior.get('b', BayesmConstants.bgammaprior)
    
    Deltabar = np.asarray(Deltabar, dtype=np.float64)
    Adelta = np.asarray(Adelta, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    
    # MCMC parameters
    if Mcmc is None:
        raise ValueError("Mcmc is required")
    
    R = Mcmc.get('R')
    if R is None:
        raise ValueError("Mcmc must contain R")
    
    Vbeta0 = Mcmc.get('Vbeta0', np.eye(nvar))
    Delta0 = Mcmc.get('Delta0', np.zeros((nz, nvar)))
    keep = Mcmc.get('keep', BayesmConstants.keep)
    s_alpha = Mcmc.get('s_alpha', BayesmConstants.RRScaling)
    s_beta = Mcmc.get('s_beta', BayesmConstants.RRScaling / np.sqrt(nvar))
    w = Mcmc.get('w', BayesmConstants.w)
    
    fixalpha = 'alpha' in Mcmc
    alpha_init = Mcmc.get('alpha', 1.0)
    
    Vbeta0 = np.asarray(Vbeta0, dtype=np.float64)
    Delta0 = np.asarray(Delta0, dtype=np.float64)
    
    print("Starting Random Walk Metropolis Sampler for Hierarchical Negative Binomial Regression")
    print(f"  {nobs} obs; {nvar} covariates; {nz} individual characteristics")
    
    # Initialize via MLE on pooled data
    par0 = np.zeros(nvar + 1)
    
    def neg_ll(par):
        return -_llnegbinR(par, Xpooled, ypooled, nvar)
    
    bounds = [(None, None)] * nvar + [(None, np.log(1e8))]
    result = minimize(neg_ll, par0, method='L-BFGS-B', bounds=bounds)
    
    beta_mle = result.x[:nvar]
    alpha_mle = np.exp(result.x[nvar])
    
    # Compute hessian numerically
    from scipy.optimize import approx_fprime
    eps = 1e-5
    n_params = nvar + 1
    hess = np.zeros((n_params, n_params))
    for i in range(n_params):
        def grad_i(par):
            return approx_fprime(par, neg_ll, eps)[i]
        hess[i, :] = approx_fprime(result.x, grad_i, eps)
    varcovinv = hess
    
    Delta = Delta0.copy()
    Beta = np.tile(beta_mle, (nreg, 1))
    Vbetainv = np.linalg.inv(Vbeta0)
    
    if not fixalpha:
        alpha = alpha_mle
    else:
        alpha = alpha_init
    
    alphacvar = s_alpha / max(varcovinv[nvar, nvar], 0.01)
    alphacroot = np.sqrt(alphacvar)
    
    print(f"beta_mle = {beta_mle}")
    print(f"alpha_mle = {alpha_mle}")
    
    # Compute unit-level hessians
    print(f"Initializing Metropolis candidate densities for {nreg} units...")
    
    if nobs > 1000:
        sind = np.random.choice(nobs, size=1000, replace=False)
        ypooleds = ypooled[sind]
        Xpooleds = Xpooled[sind, :]
    else:
        ypooleds = ypooled
        Xpooleds = Xpooled
    
    hess_list = []
    for i in range(nreg):
        wgt = len(regdata[i]['y']) / len(ypooleds)
        
        def neg_ll_frac(par):
            return -_llnegbinFract(par, regdata[i]['X'], regdata[i]['y'].ravel(),
                                   Xpooleds, ypooleds, w, wgt, nvar, result.x[nvar])
        
        res2 = minimize(neg_ll_frac, result.x[:nvar], method='BFGS',
                       options={'gtol': 1e-6, 'disp': False})
        
        if res2.success:
            hess_i = np.zeros((nvar, nvar))
            for j in range(nvar):
                def grad_j(par):
                    return approx_fprime(par, neg_ll_frac, eps)[j]
                hess_i[j, :] = approx_fprime(res2.x, grad_j, eps)
            hess_list.append({'hess': hess_i})
        else:
            hess_list.append({'hess': np.eye(nvar)})
        
        if (i + 1) % 50 == 0:
            print(f"  completed unit #{i + 1}")
    
    # Prepare data for C++
    regdata_cpp = []
    for i in range(nreg):
        regdata_cpp.append({
            'y': np.asarray(regdata[i]['y'], dtype=np.float64).ravel(),
            'X': np.asarray(regdata[i]['X'], dtype=np.float64),
            'hess': np.asarray(hess_list[i]['hess'], dtype=np.float64)
        })
    
    rootA = np.linalg.cholesky(Vbetainv).T
    
    # Call C++ loop
    result = _bayesm_cpp.rhierNegbinRw_rcpp_loop(
        regdata_cpp, Z, Beta, Delta,
        Deltabar, Adelta, nu, V, a, b,
        R, keep, s_beta, alphacroot, rootA, alpha, fixalpha
    )
    
    Betadraw, alphadraw, Vbetadraw, Deltadraw, llike, acceptrbeta, acceptralpha = result
    
    print(f"Acceptance rate beta: {acceptrbeta:.1f}%")
    if not fixalpha:
        print(f"Acceptance rate alpha: {acceptralpha:.1f}%")
    
    return {
        'Betadraw': Betadraw,
        'alphadraw': alphadraw,
        'Vbetadraw': Vbetadraw,
        'Deltadraw': Deltadraw,
        'llike': llike,
        'acceptrbeta': acceptrbeta,
        'acceptralpha': acceptralpha
    }
