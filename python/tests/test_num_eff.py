"""
Tests for bayesm.num_eff
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from bayesm.num_eff import num_eff


def test_num_eff_iid():
    """Test numerical efficiency for IID draws"""
    np.random.seed(123)
    x = np.random.randn(1000)
    
    result = num_eff(x)
    
    # For IID, f should be close to 1
    assert result['f'] > 0.8 and result['f'] < 1.2
    
    # Stderr should be close to std(x)/sqrt(n)
    expected_stderr = np.std(x, ddof=1) / np.sqrt(len(x))
    assert_allclose(result['stderr'], expected_stderr, rtol=0.2)


def test_num_eff_autocorrelated():
    """Test numerical efficiency for autocorrelated draws"""
    np.random.seed(456)
    # Create AR(1) process with rho=0.5
    n = 1000
    x = np.zeros(n)
    x[0] = np.random.randn()
    for i in range(1, n):
        x[i] = 0.5 * x[i-1] + np.random.randn()
    
    result = num_eff(x)
    
    # For autocorrelated, f should be > 1
    assert result['f'] > 1.0
    
    # Stderr should be larger than IID case
    iid_stderr = np.std(x, ddof=1) / np.sqrt(len(x))
    assert result['stderr'] > iid_stderr


def test_num_eff_custom_m():
    """Test with custom number of lags"""
    np.random.seed(789)
    x = np.random.randn(500)
    
    result = num_eff(x, m=20)
    
    assert result['m'] == 20
    assert result['stderr'] > 0
    assert result['f'] > 0


def test_num_eff_returns_dict():
    """Test return type and keys"""
    x = np.random.randn(100)
    result = num_eff(x)
    
    assert isinstance(result, dict)
    assert 'stderr' in result
    assert 'f' in result
    assert 'm' in result
