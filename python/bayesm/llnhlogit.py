"""Non-homothetic logit log-likelihood."""

import os
import numpy as np

_cpp_path = os.path.join(os.path.dirname(__file__), '_cpp')
import sys
if _cpp_path not in sys.path:
    sys.path.insert(0, _cpp_path)
import _bayesm_cpp as cpp


def llnhlogit(theta, choice, lnprices, Xexpend):
    """
    Evaluate non-homothetic logit log-likelihood.
    
    Parameters
    ----------
    theta : array
        Parameter vector: [alpha (m), k (m), gamma (d), tau]
    choice : array (n,)
        Choice indicators (1-indexed: 1, ..., m)
    lnprices : array (n, m)
        Log-prices faced by each consumer
    Xexpend : array (n, d)
        Covariates for expenditure function
    
    Returns
    -------
    ll : float
        Log-likelihood value
    
    Notes
    -----
    Non-homothetic model: ln(psi_i(u)) = alpha_i - exp(k_i)*u
    """
    m = lnprices.shape[1]
    n = len(choice)
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
    
    ind = np.arange(n)
    vchosen = vmat[ind, choice - 1]
    lnprob = vchosen - np.log(np.sum(np.exp(vmat), axis=1))
    
    return np.sum(lnprob)
