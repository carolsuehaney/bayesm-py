# rhierLinearMixture - Hierarchical Linear Model with Mixture of Normals
import numpy as np
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp


def rhierLinearMixture(Data, Prior, Mcmc):
    """
    Hierarchical Linear Model with Mixture of Normals heterogeneity.
    """
    if Data is None or 'regdata' not in Data:
        pandterm("Requires Data element regdata (list of data for each unit)")
    
    regdata = Data['regdata']
    nreg = len(regdata)
    drawdelta = True
    
    if 'Z' not in Data or Data['Z'] is None:
        drawdelta = False
        Z = np.zeros((1, 1))
    else:
        Z = np.asarray(Data['Z'], dtype=np.float64)
        if Z.shape[0] != nreg:
            pandterm(f"Nrow(Z) {Z.shape[0]} ne number regressions {nreg}")
    
    nz = Z.shape[1] if drawdelta else 1
    
    for i, rd in enumerate(regdata):
        if not isinstance(rd.get('X'), np.ndarray):
            pandterm(f"regdata[{i}]['X'] must be a matrix")
        if not isinstance(rd.get('y'), np.ndarray):
            pandterm(f"regdata[{i}]['y'] must be a vector")
    
    nvar = regdata[0]['X'].shape[1]
    
    nu_e = Prior.get('nu.e', BayesmConstants.nu_e)
    ssq = Prior.get('ssq', np.array([np.var(rd['y']) if np.var(rd['y']) > 0 else 1.0 for rd in regdata]))
    ncomp = Prior.get('ncomp')
    if ncomp is None:
        pandterm("Requires Prior element ncomp (num of mixture components)")
    
    mubar = Prior.get('mubar', np.zeros((1, nvar)))
    if isinstance(mubar, (list, np.ndarray)):
        mubar = np.atleast_2d(mubar)
    
    Amu = Prior.get('Amu', np.array([[BayesmConstants.A]]))
    if np.isscalar(Amu):
        Amu = np.array([[Amu]])
    
    nu = Prior.get('nu', nvar + BayesmConstants.nuInc)
    V = Prior.get('V', nu * np.eye(nvar))
    
    if drawdelta:
        Ad = Prior.get('Ad', BayesmConstants.A * np.eye(nvar * nz))
        deltabar = Prior.get('deltabar', np.zeros(nz * nvar))
    else:
        Ad = np.zeros((1, 1))
        deltabar = np.zeros(1)
    
    a = Prior.get('a', np.repeat(BayesmConstants.a, ncomp))
    
    keep = Mcmc.get('keep', BayesmConstants.keep)
    R = Mcmc.get('R')
    if R is None:
        pandterm("Requires R argument in Mcmc")
    
    regdata_cpp = []
    tau = np.zeros(nreg)
    for i, rd in enumerate(regdata):
        y = np.asarray(rd['y'], dtype=np.float64).flatten()
        X = np.asarray(rd['X'], dtype=np.float64)
        XpX = X.T @ X
        Xpy = X.T @ y
        regdata_cpp.append({'y': y, 'X': X, 'XpX': XpX, 'Xpy': Xpy})
        var_y = np.var(y)
        tau[i] = var_y if var_y > 0 else 1.0
    
    ninc = nreg // ncomp
    ind = np.zeros(nreg)
    for i in range(ncomp - 1):
        ind[i * ninc:(i + 1) * ninc] = i + 1
    ind[(ncomp - 1) * ninc:] = ncomp
    if ncomp == 1:
        ind[:] = 1
    
    olddelta = np.zeros(nz * nvar) if drawdelta else np.zeros(1)
    oldprob = np.repeat(1.0 / ncomp, ncomp)
    
    taudraw, Deltadraw, betadraw, probdraw, compdraw = cpp.rhierLinearMixture_rcpp_loop(
        regdata_cpp, Z, deltabar, Ad, mubar, Amu, nu, V, nu_e, ssq,
        R, keep, drawdelta, olddelta, a, oldprob, ind, tau)
    
    nmix = {'probdraw': probdraw, 'zdraw': None, 'compdraw': compdraw}
    
    result = {'taudraw': taudraw, 'betadraw': betadraw, 'nmix': nmix}
    if drawdelta:
        result['Deltadraw'] = Deltadraw
    
    return result
