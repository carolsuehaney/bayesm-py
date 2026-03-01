"""Tests for rhierNegbinRw - Hierarchical Negative Binomial."""
import numpy as np
import sys
sys.path.insert(0, '/Users/carolh/code/python_code/py_peter/python')

from bayesm import rhierNegbinRw


def test_rhierNegbinRw_basic():
    """Test basic hierarchical negative binomial."""
    np.random.seed(42)
    
    nreg = 20
    nobs_per = 30
    nvar = 2
    
    # Generate data
    beta_true = np.array([1.0, -0.5])
    alpha_true = 2.0
    
    regdata = []
    for i in range(nreg):
        X = np.column_stack([np.ones(nobs_per), np.random.randn(nobs_per)])
        lambda_i = np.exp(X @ beta_true + np.random.randn() * 0.3)
        prob = alpha_true / (alpha_true + lambda_i)
        y = np.random.negative_binomial(alpha_true, prob)
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata}
    Mcmc = {'R': 100, 'keep': 1}
    
    result = rhierNegbinRw(Data, Mcmc=Mcmc)
    
    assert 'Betadraw' in result
    assert 'alphadraw' in result
    assert 'Vbetadraw' in result
    assert 'Deltadraw' in result
    
    assert result['Betadraw'].shape == (nreg, nvar, 100)
    assert result['alphadraw'].shape == (100,)


def test_rhierNegbinRw_with_Z():
    """Test with unit-level characteristics Z."""
    np.random.seed(123)
    
    nreg = 15
    nobs_per = 25
    nvar = 2
    nz = 2
    
    Z = np.column_stack([np.ones(nreg), np.random.randn(nreg)])
    
    regdata = []
    for i in range(nreg):
        X = np.column_stack([np.ones(nobs_per), np.random.randn(nobs_per)])
        lambda_i = np.exp(X @ np.array([0.5, -0.3]))
        prob = 2.0 / (2.0 + lambda_i)
        y = np.random.negative_binomial(2, prob)
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata, 'Z': Z}
    Mcmc = {'R': 100, 'keep': 1}
    
    result = rhierNegbinRw(Data, Mcmc=Mcmc)
    
    assert result['Deltadraw'].shape == (100, nvar * nz)


if __name__ == '__main__':
    test_rhierNegbinRw_basic()
    print("test_rhierNegbinRw_basic passed")
    test_rhierNegbinRw_with_Z()
    print("test_rhierNegbinRw_with_Z passed")
