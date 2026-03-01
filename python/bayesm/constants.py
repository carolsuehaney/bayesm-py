"""
MCMC and Prior Constants

Converted from BayesmConstants.R
"""


class BayesmConstants:
    """Constants for Bayesian MCMC and prior specifications"""
    
    # MCMC
    keep = 1
    nprint = 100
    RRScaling = 2.38
    w = 0.1
    
    # Priors
    A = 0.01
    nuInc = 3
    a = 5
    nu_e = 3.0
    nu = 3.0
    agammaprior = 0.5
    bgammaprior = 0.1
    
    # Dirichlet Process
    DPalimdef = [0.01, 10]
    DPnulimdef = [0.01, 3]
    DPvlimdef = [0.1, 4]
    DPIstarmin = 1
    DPpower = 0.8
    DPalpha = 1.0
    DPmaxuniq = 200
    DPSCALE = True
    DPgridsize = 20
    
    # Mathematical Constants
    gamma = 0.5772156649015328606
    
    # BayesBLP
    BLPVOmega = [[1, 0.5], [0.5, 1]]
    BLPtol = 1e-6
