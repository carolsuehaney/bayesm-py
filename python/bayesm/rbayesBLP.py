"""
rbayesBLP - BLP Demand Estimation with Random Coefficients
Jiang, Manchanda, and Rossi (2009) Bayesian BLP
"""
import numpy as np
from .constants import BayesmConstants
from ._cpp import _bayesm_cpp


def rbayesBLP(Data, Prior=None, Mcmc=None):
    """
    BLP demand estimation via Random Walk Metropolis-Hastings.
    
    Parameters
    ----------
    Data : dict with keys
        X : array (J*T, K) - product characteristics
        share : array (J*T,) - observed market shares
        J : int - number of alternatives (excluding outside option)
        Z : array (J*T, I), optional - instrumental variables
    Prior : dict, optional
        sigmasqR : prior variance for r parameters
        theta_hat : prior mean for theta_bar
        A : prior precision for theta_bar
        deltabar, Ad : prior for IV regression delta
        nu0, s0_sq : prior for tau_sq (non-IV case)
        VOmega : prior for Omega (IV case)
    Mcmc : dict
        R : int - number of MCMC draws
        H : int - number of simulation draws for MC integration
        s : float, optional - scaling for MH increment
        cand_cov : array, optional - covariance for MH increment
        keep : int - thinning
        initial_theta_bar, initial_r, initial_tau_sq, initial_Omega, initial_delta
        tol : float - contraction mapping tolerance
    
    Returns
    -------
    dict with keys: thetabardraw, rdraw, Sigmadraw, lldraw, acceptrate,
                    tausqdraw (non-IV) or Omegadraw, deltadraw (IV)
    """
    if Data is None:
        raise ValueError("Data is required")
    
    X = Data.get('X')
    share = Data.get('share')
    J = Data.get('J')
    Z = Data.get('Z')
    
    if X is None or share is None or J is None:
        raise ValueError("Data must contain X, share, and J")
    
    X = np.asarray(X, dtype=np.float64)
    share = np.asarray(share, dtype=np.float64).ravel()
    
    IV = Z is not None
    if IV:
        Z = np.asarray(Z, dtype=np.float64)
        I = Z.shape[1]
    else:
        Z = np.zeros((1, 1), dtype=np.float64)
        I = 1
    
    K = X.shape[1]
    T = len(share) // J
    
    # Prior defaults
    if Prior is None:
        Prior = {}
    
    if 'sigmasqR' not in Prior:
        c = 50
        sigmasqRoff = 1.0
        sigmasqRdiag = np.log((1 + np.sqrt(1 - 4 * (2 * (np.arange(K)) * sigmasqRoff - c))) / 2) / 4
        sigmasqR = np.concatenate([sigmasqRdiag, np.ones(K * (K - 1) // 2)])
    else:
        sigmasqR = np.asarray(Prior['sigmasqR'], dtype=np.float64)
    
    A = Prior.get('A', BayesmConstants.A * np.eye(K))
    theta_hat = Prior.get('theta_hat', np.zeros(K))
    nu0 = Prior.get('nu0', K + 1)
    s0_sq = Prior.get('s0_sq', 1.0)
    deltabar = Prior.get('deltabar', np.zeros(I))
    Ad = Prior.get('Ad', BayesmConstants.A * np.eye(I))
    VOmega = Prior.get('VOmega', np.array(BayesmConstants.BLPVOmega, dtype=np.float64))
    
    A = np.asarray(A, dtype=np.float64)
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    deltabar = np.asarray(deltabar, dtype=np.float64)
    Ad = np.asarray(Ad, dtype=np.float64)
    VOmega = np.asarray(VOmega, dtype=np.float64)
    
    # MCMC parameters
    if Mcmc is None:
        raise ValueError("Mcmc is required (at least R and H)")
    
    R = Mcmc.get('R')
    H = Mcmc.get('H')
    if R is None or H is None:
        raise ValueError("Mcmc must contain R and H")
    
    keep = Mcmc.get('keep', 1)
    tol = Mcmc.get('tol', BayesmConstants.BLPtol)
    
    initial_theta_bar = np.asarray(Mcmc.get('initial_theta_bar', np.zeros(K)), dtype=np.float64)
    initial_r = np.asarray(Mcmc.get('initial_r', np.zeros(K * (K + 1) // 2)), dtype=np.float64)
    initial_tau_sq = Mcmc.get('initial_tau_sq', 0.1)
    initial_Omega = np.asarray(Mcmc.get('initial_Omega', np.eye(2)), dtype=np.float64)
    initial_delta = np.asarray(Mcmc.get('initial_delta', np.zeros(I)), dtype=np.float64)
    
    # Tuning parameters
    s = Mcmc.get('s')
    cand_cov = Mcmc.get('cand_cov')
    tuning_auto = s is None or cand_cov is None
    
    if tuning_auto:
        s = BayesmConstants.RRScaling / np.sqrt(K * (K + 1) / 2)
        cand_cov = np.diag(np.concatenate([0.1 * np.ones(K), np.ones(K * (K - 1) // 2)]))
    
    cand_cov = np.asarray(cand_cov, dtype=np.float64)
    
    # MC integration draws
    v = np.random.randn(K, H)
    
    # Auto-tuning if needed
    minaccep = 0.3
    maxaccep = 0.5
    
    if tuning_auto:
        print("Tuning RW Metropolis-Hastings Increment...")
        complete1 = False
        
        initial_theta_bar2 = initial_theta_bar.copy()
        initial_r2 = initial_r.copy()
        initial_tau_sq2 = initial_tau_sq
        initial_Omega2 = initial_Omega.copy()
        initial_delta2 = initial_delta.copy()
        rdraws_list = []
        
        while not complete1:
            print(f"  try s={s:.6f}")
            result = _bayesm_cpp.rbayesBLP_rcpp_loop(
                IV, X, Z, share, J, T, v, 500,
                sigmasqR, A, theta_hat, deltabar, Ad,
                nu0, s0_sq, VOmega, s**2, cand_cov,
                initial_theta_bar2, initial_r2, initial_tau_sq2, initial_Omega2, initial_delta2,
                tol, 1
            )
            tausqdraw, Omegadraw, deltadraw, thetabardraw, rdraw, Sigmadraw, lldraw, acceptrate = result
            
            initial_theta_bar2 = thetabardraw[:, -1]
            initial_r2 = rdraw[:, -1]
            initial_tau_sq2 = tausqdraw[-1] if not IV else initial_tau_sq2
            if IV:
                initial_Omega2 = Omegadraw[:, -1].reshape(2, 2)
                initial_delta2 = deltadraw[:, -1]
            
            print(f"    acceptance rate is {acceptrate:.4f}")
            
            if 0.20 < acceptrate < 0.80:
                rdraws_list.append(rdraw)
                print("    (r draws stored)")
            
            if acceptrate < minaccep:
                s = s / 5
            elif acceptrate > maxaccep:
                s = s * 3
            else:
                complete1 = True
                print("\n    (tuning completed.)")
        
        if rdraws_list:
            rdraws = np.hstack(rdraws_list)
            scale_opt = s * np.sqrt(np.diag(cand_cov))
            Omega_cov = np.cov(rdraws)
            scale_Omega = np.sqrt(np.diag(Omega_cov))
            corr_opt = Omega_cov / np.outer(scale_Omega, scale_Omega)
            s = 1.0
            cand_cov = corr_opt * np.outer(scale_opt, scale_opt)
        
        print(f"\nTuning Completed: s={s}")
    
    # Main run
    print("Starting Random Walk Metropolis-Hastings Sampler for BLP")
    result = _bayesm_cpp.rbayesBLP_rcpp_loop(
        IV, X, Z, share, J, T, v, R,
        sigmasqR, A, theta_hat, deltabar, Ad,
        nu0, s0_sq, VOmega, s**2, cand_cov,
        initial_theta_bar, initial_r, initial_tau_sq, initial_Omega, initial_delta,
        tol, keep
    )
    tausqdraw, Omegadraw, deltadraw, thetabardraw, rdraw, Sigmadraw, lldraw, acceptrate = result
    
    out = {
        'thetabardraw': thetabardraw.T,
        'rdraw': rdraw.T,
        'Sigmadraw': Sigmadraw.T,
        'lldraw': lldraw,
        'acceptrate': acceptrate,
        's': s,
        'cand_cov': cand_cov
    }
    
    if IV:
        out['Omegadraw'] = Omegadraw.T
        out['deltadraw'] = deltadraw.T
    else:
        out['tausqdraw'] = tausqdraw
    
    return out
