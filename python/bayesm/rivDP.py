# rivDP - Linear IV Model with Dirichlet Process prior for errors
import numpy as np
from scipy.special import digamma
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp


def rivDP(Data, Prior=None, Mcmc=None):
    """
    Linear IV model with Dirichlet Process prior on error distribution.
    
    Model:
        x = z'delta + e1
        y = beta*x + w'gamma + e2
        (e1, e2) ~ DP mixture of normals
    """
    if Data is None:
        pandterm("Requires Data argument -- list of z,w,x,y")
    
    isgamma = 'w' in Data and Data['w'] is not None
    if isgamma:
        w = np.asarray(Data['w'], dtype=np.float64)
    else:
        w = np.zeros((1, 1))
    
    if 'z' not in Data:
        pandterm("Requires Data element z")
    z = np.asarray(Data['z'], dtype=np.float64)
    
    if 'x' not in Data:
        pandterm("Requires Data element x")
    x = np.asarray(Data['x'], dtype=np.float64).flatten()
    
    if 'y' not in Data:
        pandterm("Requires Data element y")
    y = np.asarray(Data['y'], dtype=np.float64).flatten()
    
    n = len(y)
    dimd = z.shape[1]
    dimg = w.shape[1] if isgamma else 1
    dimbg = 1 + dimg if isgamma else 1
    
    alimdef = BayesmConstants.DPalimdef
    nulimdef = BayesmConstants.DPnulimdef
    vlimdef = BayesmConstants.DPvlimdef
    
    if Prior is None:
        Prior = {}
    
    md = Prior.get('md', np.zeros(dimd))
    Ad = Prior.get('Ad', BayesmConstants.A * np.eye(dimd))
    mbg = Prior.get('mbg', np.zeros(dimbg))
    Abg = Prior.get('Abg', BayesmConstants.A * np.eye(dimbg))
    
    Prioralpha = Prior.get('Prioralpha', {})
    gamma_const = BayesmConstants.gamma
    Istarmin = Prioralpha.get('Istarmin', BayesmConstants.DPIstarmin)
    Istarmax = Prioralpha.get('Istarmax', int(0.1 * n))
    power = Prioralpha.get('power', BayesmConstants.DPpower)
    alphamin = np.exp(digamma(Istarmin) - np.log(gamma_const + np.log(n)))
    alphamax = np.exp(digamma(Istarmax) - np.log(gamma_const + np.log(n)))
    
    lambda_hyper = Prior.get('lambda_hyper', {})
    alim = lambda_hyper.get('alim', alimdef)
    nulim = lambda_hyper.get('nulim', nulimdef)
    vlim = lambda_hyper.get('vlim', vlimdef)
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    
    R = Mcmc.get('R')
    if R is None:
        pandterm("requires Mcmc argument, R")
    keep = Mcmc.get('keep', BayesmConstants.keep)
    maxuniq = Mcmc.get('maxuniq', BayesmConstants.DPmaxuniq)
    gridsize = Mcmc.get('gridsize', BayesmConstants.DPgridsize)
    SCALE = Mcmc.get('SCALE', BayesmConstants.DPSCALE)
    
    delta = Mcmc.get('delta')
    if delta is None:
        from numpy.linalg import lstsq
        delta, _, _, _ = lstsq(z, x, rcond=None)
    delta = np.asarray(delta, dtype=np.float64)
    
    scalex = 1.0
    scaley = 1.0
    if SCALE:
        scaley = np.std(y)
        scalex = np.std(x)
        meany = np.mean(y)
        meanx = np.mean(x)
        meanz = np.mean(z, axis=0)
        y = (y - meany) / scaley
        x = (x - meanx) / scalex
        z = (z - meanz) / scalex
        if isgamma:
            w = w / scaley
        delta = delta * scalex / scalex  # stays same after centering
    
    alim = np.asarray(alim, dtype=np.float64)
    nulim = np.asarray(nulim, dtype=np.float64)
    vlim = np.asarray(vlim, dtype=np.float64)
    
    deltadraw, betadraw, alphadraw, Istardraw, gammadraw, adraw, nudraw, vdraw, thetaNp1draw = cpp.rivDP_rcpp_loop(
        R, keep, dimd, mbg, Abg, md, Ad, y, isgamma, z, x, w, delta,
        power, alphamin, alphamax, n, gridsize, SCALE, maxuniq, scalex, scaley,
        alim, nulim, vlim, BayesmConstants.A, 2 + BayesmConstants.nuInc)
    
    nmix = {
        'probdraw': np.ones((len(alphadraw), 1)),
        'zdraw': None,
        'compdraw': thetaNp1draw
    }
    
    result = {
        'deltadraw': deltadraw,
        'betadraw': betadraw,
        'alphadraw': alphadraw,
        'Istardraw': Istardraw,
        'adraw': adraw,
        'nudraw': nudraw,
        'vdraw': vdraw,
        'nmix': nmix
    }
    
    if isgamma:
        result['gammadraw'] = gammadraw
    
    return result
