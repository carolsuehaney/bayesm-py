"""Scale Usage Model - Gibbs sampler for ordinal scale rating data."""
import numpy as np
from . import _bayesm_cpp


def rscaleUsage(Data, Prior=None, Mcmc=None):
    """
    Scale Usage Model MCMC sampler.
    
    Parameters
    ----------
    Data : dict
        k : int, scale of responses (1,2,...,k)
        x : ndarray (n x p), data matrix of ordinal responses
    Prior : dict, optional
        nu, V : Sigma ~ IW(nu, V)
        mubar, Am : mu ~ N(mubar, Am^{-1})
        gsigma : grid for sigma
        gl11, gl22, gl12 : grids for Lambda elements
        Lambdanu, LambdaV : Lambda ~ IW(Lambdanu, LambdaV)
        ge : grid for e
        gs : grid size (default 100)
    Mcmc : dict, optional
        R : number of MCMC draws (default 1000)
        keep : thinning (default 1)
        ndghk : number of draws for GHK (default 100)
        y, mu, Sigma, tau, sigma, Lambda, e : initial values
        domu, doSigma, dosigma, dotau, doLambda, doe : draw flags
    
    Returns
    -------
    dict with keys: Sigmadraw, mudraw, taudraw, sigmadraw, Lambdadraw, edraw
    """
    if Prior is None:
        Prior = {}
    if Mcmc is None:
        Mcmc = {}
    
    # Process Data
    if 'k' not in Data:
        raise ValueError("k not specified")
    k = int(Data['k'])
    if not (0 < k < 50):
        raise ValueError("Data$k must be integer between 1 and 50")
    
    if 'x' not in Data:
        raise ValueError("x (the data), not specified")
    x = np.asarray(Data['x'], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Data$x must be a matrix")
    n, p = x.shape
    if p < 2 or n < 1:
        raise ValueError(f"invalid dimensions for x: nrow,ncol: {n},{p}")
    x_int = x.astype(int)
    if not np.all((x_int >= 1) & (x_int <= k)):
        raise ValueError("each element of Data$x must be in 1,2...k")
    
    # Process Mcmc
    R = Mcmc.get('R', 1000)
    keep = Mcmc.get('keep', 1)
    ndghk = Mcmc.get('ndghk', 100)
    
    # Initial state
    y = Mcmc.get('y', x.astype(np.float64).copy())
    mu = Mcmc.get('mu', np.mean(y, axis=0))
    Sigma = Mcmc.get('Sigma', np.cov(y, rowvar=False))
    tau = Mcmc.get('tau', np.zeros(n))
    sigma = Mcmc.get('sigma', np.ones(n))
    Lambda = Mcmc.get('Lambda', np.array([[4.0, 0.0], [0.0, 0.5]]))
    e = Mcmc.get('e', 0.0)
    
    # Draw flags
    domu = Mcmc.get('domu', True)
    doSigma = Mcmc.get('doSigma', True)
    dosigma = Mcmc.get('dosigma', True)
    dotau = Mcmc.get('dotau', True)
    doLambda = Mcmc.get('doLambda', True)
    doe = Mcmc.get('doe', True)
    
    # Process Prior
    nu = Prior.get('nu', p + 3)
    V = Prior.get('V', nu * np.eye(p))
    mubar = Prior.get('mubar', np.full((p, 1), k / 2.0))
    Am = Prior.get('Am', 0.01 * np.eye(p))
    
    gs = Prior.get('gs', 100)
    gsigma = Prior.get('gsigma', 6.0 * np.arange(1, gs + 1) / gs)
    gl11 = Prior.get('gl11', 0.1 + 5.9 * np.arange(1, gs + 1) / gs)
    gl22 = Prior.get('gl22', 0.1 + 2.0 * np.arange(1, gs + 1) / gs)
    gl12 = Prior.get('gl12', -2.0 + 4.0 * np.arange(1, gs + 1) / gs)
    
    nuL = Prior.get('Lambdanu', 20)
    VL = Prior.get('LambdaV', (nuL - 3) * Lambda)
    ge = Prior.get('ge', -0.1 + 0.2 * np.arange(0, gs + 1) / gs)
    
    # Ensure correct types/shapes
    y = np.ascontiguousarray(y, dtype=np.float64)
    mu = np.ascontiguousarray(mu.flatten(), dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    tau = np.ascontiguousarray(tau.flatten(), dtype=np.float64)
    sigma = np.ascontiguousarray(sigma.flatten(), dtype=np.float64)
    Lambda = np.ascontiguousarray(Lambda, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    mubar = np.ascontiguousarray(mubar, dtype=np.float64)
    Am = np.ascontiguousarray(Am, dtype=np.float64)
    gsigma = np.ascontiguousarray(gsigma, dtype=np.float64)
    gl11 = np.ascontiguousarray(gl11, dtype=np.float64)
    gl22 = np.ascontiguousarray(gl22, dtype=np.float64)
    gl12 = np.ascontiguousarray(gl12, dtype=np.float64)
    VL = np.ascontiguousarray(VL, dtype=np.float64)
    ge = np.ascontiguousarray(ge, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    
    drSigma, drmu, drtau, drsigma, drLambda, dre = _bayesm_cpp.rscaleUsage_rcpp_loop(
        k, x, p, n,
        R, keep, ndghk,
        y, mu, Sigma, tau, sigma, Lambda, e,
        domu, doSigma, dosigma, dotau, doLambda, doe,
        nu, V, mubar, Am,
        gsigma, gl11, gl22, gl12,
        nuL, VL, ge
    )
    
    return {
        'Sigmadraw': drSigma,
        'mudraw': drmu,
        'taudraw': drtau,
        'sigmadraw': drsigma,
        'Lambdadraw': drLambda,
        'edraw': dre
    }
