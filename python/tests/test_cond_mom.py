"""
Tests for bayesm.cond_mom
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from bayesm.cond_mom import cond_mom


def test_cond_mom_2d():
    """Test conditional moments for 2D normal"""
    # Simple case: independent normals
    x = np.array([1.0, 2.0])
    mu = np.array([0.0, 0.0])
    sigi = np.eye(2)  # Identity = independent
    
    # Condition on first variable (i=0)
    result = cond_mom(x, mu, sigi, 0)
    
    # For independent normals, conditional mean should equal marginal mean
    assert_allclose(result['cmean'], mu[0], rtol=1e-10)
    assert_allclose(result['cvar'], 1.0, rtol=1e-10)


def test_cond_mom_correlated():
    """Test conditional moments with correlation"""
    # Covariance: [[1, 0.8], [0.8, 1]], so Sigma^-1 = [[2.778, -2.222], [-2.222, 2.778]]
    rho = 0.8
    sigma = np.array([[1, rho], [rho, 1]])
    sigi = np.linalg.inv(sigma)
    
    x = np.array([1.0, 0.5])
    mu = np.array([0.0, 0.0])
    
    # Condition on second variable (i=1)
    result = cond_mom(x, mu, sigi, 1)
    
    # Conditional variance should be 1 - rho^2
    expected_var = 1 - rho**2
    assert_allclose(result['cvar'], expected_var, rtol=1e-10)
    
    # Conditional mean should be rho * x[0] (since mu=0)
    expected_mean = rho * x[0]
    assert_allclose(result['cmean'], expected_mean, rtol=1e-10)


def test_cond_mom_3d():
    """Test conditional moments for 3D normal"""
    # Identity covariance
    mu = np.array([1.0, 2.0, 3.0])
    sigi = np.eye(3)
    x = np.array([1.5, 2.5, 3.5])
    
    result = cond_mom(x, mu, sigi, 1)
    
    # For identity covariance, conditional should equal marginal
    assert_allclose(result['cmean'], mu[1], rtol=1e-10)
    assert_allclose(result['cvar'], 1.0, rtol=1e-10)
