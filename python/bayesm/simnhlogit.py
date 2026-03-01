"""Simulate non-homothetic logit model."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp


def simnhlogit(theta, lnprices, Xexpend):
    """
    Simulate data from non-homothetic logit model.
    
    Parameters
    ----------
    theta : array
        Parameter vector: [alpha (m), k (m), gamma (d), tau]
    lnprices : array (n, m)
        Log-prices faced by each consumer
    Xexpend : array (n, d)
        Covariates for expenditure function
    
    Returns
    -------
    dict with keys:
        y : array (n,)
            Simulated choices (1-indexed: 1, ..., m)
        Xexpend : array (n, d)
            Input Xexpend
        lnprices : array (n, m)
            Input lnprices
        theta : array
            Input theta
        prob : array (n, m)
            Choice probabilities
    
    Notes
    -----
    Non-homothetic model: ln(psi_i(u)) = alpha_i - exp(k_i)*u
    """
    m = lnprices.shape[1]
    n = lnprices.shape[0]
    d = Xexpend.shape[1]
    
    alpha = theta[:m]
    k = theta[m:2*m]
    gamma = theta[2*m:2*m+d]
    tau = theta[-1]
    
    iotam = np.ones(m)
    c1 = np.kron(Xexpend @ gamma, iotam) - lnprices.T.flatten(order='F') + np.tile(alpha, n)
    c2 = np.tile(np.exp(k), n)
    
    u = cpp.callroot(c1, c2, 1e-7, 20)
    v = np.tile(alpha, n) - u * np.tile(np.exp(k), n) - lnprices.T.flatten(order='F')
    vmat = v.reshape(m, n, order='F').T
    vmat = tau * vmat
    
    Prob = np.exp(vmat)
    denom = np.sum(Prob, axis=1, keepdims=True)
    Prob = Prob / denom
    
    y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = np.random.choice(np.arange(1, m + 1), p=Prob[i, :])
    
    return {
        'y': y,
        'Xexpend': Xexpend,
        'lnprices': lnprices,
        'theta': theta,
        'prob': Prob
    }
