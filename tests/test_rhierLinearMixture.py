import numpy as np
from bayesm import rhierLinearMixture


def test_rhierLinearMixture_basic():
    """Test basic functionality without Z matrix."""
    np.random.seed(42)
    nreg = 50
    nobs = 20
    nvar = 3
    ncomp = 2
    
    regdata = []
    for _ in range(nreg):
        X = np.column_stack([np.ones(nobs), np.random.randn(nobs, nvar - 1)])
        beta = np.random.randn(nvar)
        y = X @ beta + np.random.randn(nobs) * 0.5
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 200, 'keep': 2}
    
    result = rhierLinearMixture(Data, Prior, Mcmc)
    
    assert 'taudraw' in result
    assert 'betadraw' in result
    assert 'nmix' in result
    assert result['taudraw'].shape == (100, nreg)
    assert result['betadraw'].shape == (nreg, nvar, 100)
    assert result['nmix']['probdraw'].shape == (100, ncomp)


def test_rhierLinearMixture_with_Z():
    """Test with Z matrix (drawdelta=True)."""
    np.random.seed(123)
    nreg = 30
    nobs = 15
    nvar = 2
    nz = 2
    ncomp = 2
    
    Z = np.random.randn(nreg, nz)
    Z = Z - Z.mean(axis=0)  # de-mean Z
    
    regdata = []
    for i in range(nreg):
        X = np.column_stack([np.ones(nobs), np.random.randn(nobs, nvar - 1)])
        beta = np.random.randn(nvar)
        y = X @ beta + np.random.randn(nobs) * 0.5
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata, 'Z': Z}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 100, 'keep': 1}
    
    result = rhierLinearMixture(Data, Prior, Mcmc)
    
    assert 'taudraw' in result
    assert 'betadraw' in result
    assert 'Deltadraw' in result
    assert 'nmix' in result
    assert result['Deltadraw'].shape == (100, nz * nvar)
