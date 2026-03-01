"""
Tests for bayesm.utilities
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from bayesm.utilities import pandterm, nmat


def test_pandterm():
    """Test error raising function"""
    with pytest.raises(ValueError, match="test error"):
        pandterm("test error")


def test_nmat_basic():
    """Test nmat with simple 2x2 covariance matrix"""
    # Covariance matrix: [[1, 0.5], [0.5, 1]]
    vec = np.array([1, 0.5, 0.5, 1])
    result = nmat(vec)
    expected = np.array([1, 0.5, 0.5, 1])  # Already a correlation matrix
    assert_allclose(result, expected, rtol=1e-10)


def test_nmat_scaled():
    """Test nmat with scaled covariance matrix"""
    # Covariance matrix: [[4, 2], [2, 4]]
    # Correlation should be: [[1, 0.5], [0.5, 1]]
    vec = np.array([4, 2, 2, 4])
    result = nmat(vec)
    expected = np.array([1, 0.5, 0.5, 1])
    assert_allclose(result, expected, rtol=1e-10)


def test_nmat_3x3():
    """Test nmat with 3x3 covariance matrix"""
    # Covariance matrix with variances [1, 4, 9]
    cov = np.array([
        [1, 1, 1.5],
        [1, 4, 3],
        [1.5, 3, 9]
    ])
    vec = cov.ravel()
    result = nmat(vec).reshape(3, 3)
    
    # Check diagonal is all 1s
    assert_allclose(np.diag(result), np.ones(3), rtol=1e-10)
    
    # Check symmetry
    assert_allclose(result, result.T, rtol=1e-10)
    
    # Check specific correlations
    assert_allclose(result[0, 1], 1/(np.sqrt(1)*np.sqrt(4)), rtol=1e-10)
