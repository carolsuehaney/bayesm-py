import numpy as np
from bayesm import rhierMnlDP


def test_rhierMnlDP_basic():
    """Test basic hierarchical MNL with DP."""
    np.random.seed(42)
    nlgt = 20
    p = 3
    nvar = 2
    nobs_per_lgt = 10
    
    lgtdata = []
    for i in range(nlgt):
        beta_i = np.random.randn(nvar)
        y = []
        X_list = []
        for t in range(nobs_per_lgt):
            X_t = np.random.randn(p, nvar)
            utils = X_t @ beta_i
            probs = np.exp(utils - utils.max())
            probs = probs / probs.sum()
            choice = np.random.choice(p, p=probs) + 1
            y.append(choice)
            X_list.append(X_t)
        lgtdata.append({'y': np.array(y), 'X': np.vstack(X_list)})
    
    Data = {'p': p, 'lgtdata': lgtdata}
    Prior = {}
    Mcmc = {'R': 50, 'keep': 1}
    
    result = rhierMnlDP(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert 'alphadraw' in result
    assert 'Istardraw' in result
    assert result['betadraw'].shape == (nlgt, nvar, 50)


def test_rhierMnlDP_larger():
    """Test hierarchical MNL with DP with more units."""
    np.random.seed(123)
    nlgt = 30
    p = 2
    nvar = 2
    nobs_per_lgt = 12
    
    lgtdata = []
    for i in range(nlgt):
        beta_i = np.random.randn(nvar) * 0.5
        y = []
        X_list = []
        for t in range(nobs_per_lgt):
            X_t = np.random.randn(p, nvar)
            utils = X_t @ beta_i
            probs = np.exp(utils - utils.max())
            probs = probs / probs.sum()
            choice = np.random.choice(p, p=probs) + 1
            y.append(choice)
            X_list.append(X_t)
        lgtdata.append({'y': np.array(y), 'X': np.vstack(X_list)})
    
    Data = {'p': p, 'lgtdata': lgtdata}
    Prior = {}
    Mcmc = {'R': 40, 'keep': 2}
    
    result = rhierMnlDP(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert result['betadraw'].shape == (nlgt, nvar, 20)
