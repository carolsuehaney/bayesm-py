"""Normal mixture Gibbs sampler."""
import numpy as np
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp

def rnmixGibbs(Data, Prior, Mcmc):
    """
    Gibbs sampler for mixture of multivariate normals.
    
    Model:
        y_i ~ N(mu_ind, Sigma_ind)
        ind ~ iid multinomial(p)
    
    Priors:
        mu_j ~ N(mubar, Sigma (x) A^-1)
        Sigma_j ~ IW(nu, V)
        p ~ Dirichlet(a)
    """
    if Data is None or 'y' not in Data:
        pandterm("Requires Data argument with element y")
    y = np.asarray(Data['y'], dtype=np.float64)
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    nobs, dimy = y.shape
    
    if Prior is None or 'ncomp' not in Prior:
        pandterm("Requires Prior$ncomp")
    ncomp = Prior['ncomp']
    
    Mubar = Prior.get('Mubar', np.zeros((1, dimy)))
    if np.ndim(Mubar) == 1:
        Mubar = Mubar.reshape(1, -1)
    Mubar = np.asarray(Mubar, dtype=np.float64)
    
    A = Prior.get('A', np.array([[BayesmConstants.A]]))
    if np.isscalar(A):
        A = np.array([[A]])
    A = np.asarray(A, dtype=np.float64)
    
    nu = Prior.get('nu', dimy + BayesmConstants.nuInc)
    V = Prior.get('V', nu * np.eye(dimy))
    V = np.asarray(V, dtype=np.float64)
    
    a = Prior.get('a', np.repeat(BayesmConstants.a, ncomp))
    a = np.asarray(a, dtype=np.float64)
    
    if nobs < 2 * ncomp:
        pandterm("too few obs, nobs should be >= 2*ncomp")
    
    if Mcmc is None or 'R' not in Mcmc:
        pandterm("Requires Mcmc$R")
    R = Mcmc['R']
    keep = Mcmc.get('keep', BayesmConstants.keep)
    
    # Initial values
    z = np.tile(np.arange(1, ncomp + 1), nobs // ncomp + 1)[:nobs].astype(np.float64)
    p = np.ones(ncomp) / ncomp
    
    # Call C++
    probdraw, zdraw, compdraw = cpp.rnmixGibbs_rcpp_loop(
        y, Mubar, A, nu, V, a, p, z, R, keep)
    
    return {
        'probdraw': probdraw,
        'zdraw': zdraw,
        'compdraw': compdraw
    }
