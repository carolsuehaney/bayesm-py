"""Tests for rscaleUsage - Scale Usage Model."""
import numpy as np
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from bayesm import rscaleUsage


def test_rscaleUsage_basic():
    """Basic test with synthetic ordinal data."""
    np.random.seed(42)
    
    n = 50
    p = 4
    k = 7
    
    x = np.random.randint(1, k + 1, size=(n, p)).astype(float)
    
    Data = {'k': k, 'x': x}
    Mcmc = {'R': 100, 'keep': 1, 'ndghk': 50}
    
    result = rscaleUsage(Data, Mcmc=Mcmc)
    
    assert 'Sigmadraw' in result
    assert 'mudraw' in result
    assert 'taudraw' in result
    assert 'sigmadraw' in result
    assert 'Lambdadraw' in result
    assert 'edraw' in result
    
    nkeep = 100
    assert result['Sigmadraw'].shape == (nkeep, p * p)
    assert result['mudraw'].shape == (nkeep, p)
    assert result['taudraw'].shape == (nkeep, n)
    assert result['sigmadraw'].shape == (nkeep, n)
    assert result['Lambdadraw'].shape == (nkeep, 4)
    assert result['edraw'].shape == (nkeep,)


def test_rscaleUsage_with_prior():
    """Test with custom prior settings."""
    np.random.seed(123)
    
    n = 30
    p = 3
    k = 5
    
    x = np.random.randint(1, k + 1, size=(n, p)).astype(float)
    
    Data = {'k': k, 'x': x}
    Prior = {
        'nu': p + 5,
        'V': (p + 5) * np.eye(p),
        'mubar': np.full((p, 1), k / 2),
        'Am': 0.01 * np.eye(p),
        'gs': 50,
    }
    Mcmc = {'R': 50, 'keep': 1, 'ndghk': 25}
    
    result = rscaleUsage(Data, Prior=Prior, Mcmc=Mcmc)
    
    assert result['mudraw'].shape[0] == 50
    assert result['mudraw'].shape[1] == p


def test_rscaleUsage_partial_draws():
    """Test with some draws disabled."""
    np.random.seed(456)
    
    n = 20
    p = 3
    k = 5
    
    x = np.random.randint(1, k + 1, size=(n, p)).astype(float)
    
    Data = {'k': k, 'x': x}
    Mcmc = {
        'R': 30,
        'keep': 1,
        'doe': False,
        'doLambda': False,
    }
    
    result = rscaleUsage(Data, Mcmc=Mcmc)
    
    assert result['mudraw'].shape[0] == 30
    e_vals = result['edraw']
    assert np.allclose(e_vals, e_vals[0])  # e should stay at initial


if __name__ == "__main__":
    test_rscaleUsage_basic()
    print("test_rscaleUsage_basic passed")
    test_rscaleUsage_with_prior()
    print("test_rscaleUsage_with_prior passed")
    test_rscaleUsage_partial_draws()
    print("test_rscaleUsage_partial_draws passed")
    print("All tests passed!")
