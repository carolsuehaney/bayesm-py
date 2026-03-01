"""
Tests for bayesm.create_x
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from bayesm.create_x import create_x


def test_create_x_basic():
    """Test basic functionality with demos only"""
    p = 3
    n = 10
    nd = 2
    Xd = np.random.randn(n, nd)
    
    X = create_x(p=p, na=None, nd=nd, Xa=None, Xd=Xd, INT=True, DIFF=False)
    
    # Should have n*p rows
    assert X.shape[0] == n * p
    # Should have (nd+1)*(p-1) columns (intercept + 2 demos, times p-1)
    assert X.shape[1] == (nd + 1) * (p - 1)


def test_create_x_attributes_only():
    """Test with attributes only"""
    p = 3
    n = 10
    na = 2
    Xa = np.random.randn(n, na * p)
    
    X = create_x(p=p, na=na, nd=None, Xa=Xa, Xd=None, INT=False, DIFF=False)
    
    assert X.shape[0] == n * p
    assert X.shape[1] == na


def test_create_x_both():
    """Test with both attributes and demos"""
    p = 3
    n = 10
    na = 2
    nd = 1
    Xa = np.random.randn(n, na * p)
    Xd = np.random.randn(n, nd)
    
    X = create_x(p=p, na=na, nd=nd, Xa=Xa, Xd=Xd, INT=True, DIFF=False)
    
    assert X.shape[0] == n * p
    # (nd+1)*(p-1) + na
    assert X.shape[1] == (nd + 1) * (p - 1) + na


def test_create_x_diff():
    """Test with differencing"""
    p = 3
    n = 5
    nd = 1
    Xd = np.ones((n, nd))
    
    X = create_x(p=p, na=None, nd=nd, Xa=None, Xd=Xd, INT=True, DIFF=True, base=p)
    
    # With DIFF, should create differences relative to base
    assert X.shape[0] == n * p
    assert X.shape[1] == (nd + 1) * (p - 1)


def test_create_x_error_both_none():
    """Test error when both Xa and Xd are None"""
    with pytest.raises(ValueError, match="both Xa and Xd None"):
        create_x(p=3, na=None, nd=None, Xa=None, Xd=None)


def test_create_x_error_dim_mismatch():
    """Test error for dimension mismatch"""
    p = 3
    n = 10
    na = 2
    Xa = np.random.randn(n, na * p + 1)  # Wrong number of columns
    
    with pytest.raises(ValueError, match="bad Xa dim"):
        create_x(p=p, na=na, nd=None, Xa=Xa, Xd=None)


def test_create_x_no_intercept():
    """Test without intercept"""
    p = 3
    n = 10
    nd = 1
    Xd = np.random.randn(n, nd)
    
    X = create_x(p=p, na=None, nd=nd, Xa=None, Xd=Xd, INT=False, DIFF=False)
    
    # Without intercept, should have nd*(p-1) columns
    assert X.shape[1] == nd * (p - 1)
