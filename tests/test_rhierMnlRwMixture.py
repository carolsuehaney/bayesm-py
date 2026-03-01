import numpy as np
from bayesm import rhierMnlRwMixture


def test_rhierMnlRwMixture_basic():
    """Test basic functionality without Z matrix."""
    np.random.seed(42)
    nlgt = 30
    nobs = 10
    p = 3
    nvar = 4
    ncomp = 2
    
    lgtdata = []
    for _ in range(nlgt):
        X = np.random.randn(nobs * p, nvar)
        y = np.random.randint(1, p + 1, nobs).astype(float)
        lgtdata.append({'y': y, 'X': X})
    
    Data = {'p': p, 'lgtdata': lgtdata}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 100, 'keep': 2}
    
    result = rhierMnlRwMixture(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert 'nmix' in result
    assert 'loglike' in result
    assert result['betadraw'].shape == (nlgt, nvar, 50)
    assert result['nmix']['probdraw'].shape == (50, ncomp)
    assert len(result['loglike']) == 50


def test_rhierMnlRwMixture_with_Z():
    """Test with Z matrix."""
    np.random.seed(123)
    nlgt = 20
    nobs = 8
    p = 3
    nvar = 3
    ncomp = 2
    nz = 2
    
    Z = np.random.randn(nlgt, nz)
    Z = Z - Z.mean(axis=0)
    
    lgtdata = []
    for _ in range(nlgt):
        X = np.random.randn(nobs * p, nvar)
        y = np.random.randint(1, p + 1, nobs).astype(float)
        lgtdata.append({'y': y, 'X': X})
    
    Data = {'p': p, 'lgtdata': lgtdata, 'Z': Z}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 50, 'keep': 1}
    
    result = rhierMnlRwMixture(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert 'Deltadraw' in result
    assert result['Deltadraw'].shape == (50, nz * nvar)
