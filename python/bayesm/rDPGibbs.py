# rDPGibbs - Dirichlet Process Gibbs Sampler for Density Estimation
import numpy as np
from scipy.special import digamma
from .utilities import pandterm
from .constants import BayesmConstants
from . import _cpp as cpp


def rDPGibbs(Data, Prior, Mcmc):
    """
    Dirichlet Process Gibbs Sampler for density estimation.
    
    Model: y_i ~ f(y|theta_i), theta_i|G ~ G, G|lambda,alpha ~ DP(G0(lambda), alpha)
    """
    if Data is None or 'y' not in Data:
        pandterm("Requires Data element y")
    
    y = np.asarray(Data['y'], dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    nobs = y.shape[0]
    dimy = y.shape[1]
    
    alimdef = BayesmConstants.DPalimdef
    nulimdef = BayesmConstants.DPnulimdef
    vlimdef = BayesmConstants.DPvlimdef
    
    if Prior is None:
        pandterm("requires Prior argument")
    
    lambda_hyper = Prior.get('lambda_hyper', {})
    alim = lambda_hyper.get('alim', alimdef)
    nulim = lambda_hyper.get('nulim', nulimdef)
    vlim = lambda_hyper.get('vlim', vlimdef)
    
    Prioralpha = Prior.get('Prioralpha', {})
    Istarmin = Prioralpha.get('Istarmin', BayesmConstants.DPIstarmin)
    Istarmax = Prioralpha.get('Istarmax', min(50, int(0.1 * nobs)))
    power = Prioralpha.get('power', BayesmConstants.DPpower)
    
    gamma_const = BayesmConstants.gamma
    alphamin = np.exp(digamma(Istarmin) - np.log(gamma_const + np.log(nobs)))
    alphamax = np.exp(digamma(Istarmax) - np.log(gamma_const + np.log(nobs)))
    
    if Mcmc is None:
        pandterm("requires Mcmc argument")
    
    R = Mcmc.get('R')
    if R is None:
        pandterm("requires Mcmc element R")
    keep = Mcmc.get('keep', BayesmConstants.keep)
    maxuniq = Mcmc.get('maxuniq', BayesmConstants.DPmaxuniq)
    SCALE = Mcmc.get('SCALE', BayesmConstants.DPSCALE)
    gridsize = Mcmc.get('gridsize', BayesmConstants.DPgridsize)
    
    alim = np.asarray(alim, dtype=np.float64)
    nulim = np.asarray(nulim, dtype=np.float64)
    vlim = np.asarray(vlim, dtype=np.float64)
    
    alphadraw, Istardraw, adraw, nudraw, vdraw, inddraw, thetaNp1draw = cpp.rDPGibbs_rcpp_loop(
        R, keep, y, alim, nulim, vlim, SCALE, maxuniq,
        power, alphamin, alphamax, nobs, gridsize,
        BayesmConstants.A, BayesmConstants.nuInc, BayesmConstants.DPalpha)
    
    nmix = {
        'probdraw': np.ones((len(alphadraw), 1)),
        'zdraw': inddraw,
        'compdraw': thetaNp1draw
    }
    
    return {
        'alphadraw': alphadraw,
        'Istardraw': Istardraw,
        'adraw': adraw,
        'nudraw': nudraw,
        'vdraw': vdraw,
        'nmix': nmix
    }
