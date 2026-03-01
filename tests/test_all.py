#!/usr/bin/env python
"""Test all converted code - Phase 2, 3, and C++ bindings"""

import sys
sys.path.insert(0, '.')

import numpy as np

def test_phase2():
    """Test Phase 2 - data loaders and core utilities"""
    print("=== Phase 2: Core Utilities ===")
    
    from bayesm.data_loaders import load_data
    from bayesm.constants import BayesmConstants
    from bayesm.utilities import nmat
    from bayesm.create_x import create_x
    
    # Test data loading
    cheese = load_data('cheese')
    print(f"✓ cheese: {cheese.shape[0]} rows")
    
    camera = load_data('camera', format='lgtdata')
    print(f"✓ camera: {len(camera)} respondents")
    
    # Test constants
    assert BayesmConstants.nprint == 100
    print("✓ constants loaded")
    
    # Test nmat
    vec = np.array([4, 2, 2, 4])
    result = nmat(vec)
    assert np.allclose(result, [1, 0.5, 0.5, 1])
    print("✓ nmat works")
    
    # Test create_x
    np.random.seed(123)
    X = create_x(p=3, na=2, nd=1, Xa=np.random.randn(10, 6), Xd=np.random.randn(10, 1))
    assert X.shape == (30, 6)
    print("✓ create_x works")
    
    return True

def test_phase3():
    """Test Phase 3 - mixture functions"""
    print("\n=== Phase 3: Mixture Functions ===")
    
    from bayesm.mnl_hess import mnl_hess
    from bayesm.log_marg_den_nr import log_marg_den_nr
    
    # Test mnl_hess
    np.random.seed(123)
    n, j, k = 50, 3, 2
    X = np.random.randn(n * j, k)
    beta = np.array([0.5, -0.5])
    y = np.random.randint(0, j, n)
    H = mnl_hess(beta, y, X)
    assert H.shape == (k, k)
    assert np.allclose(H, H.T)
    print("✓ mnl_hess works")
    
    # Test log_marg_den_nr
    ll = np.array([-100, -105, -98, -102, -99])
    result = log_marg_den_nr(ll)
    assert isinstance(result, (float, np.floating))
    print(f"✓ log_marg_den_nr works: {result:.2f}")
    
    return True

def test_cpp():
    """Test C++ bindings"""
    print("\n=== C++ Bindings ===")
    
    sys.path.insert(0, 'bayesm/_cpp')
    import _bayesm_cpp as cpp
    
    # Test lndMvn
    x = np.array([0.0, 0.0])
    mu = np.array([0.0, 0.0])
    rooti = np.eye(2)
    result = cpp.lndMvn(x, mu, rooti)
    expected = -np.log(2 * np.pi)  # Log density of standard normal at 0
    assert abs(result - expected) < 0.01, f"lndMvn: got {result}, expected {expected}"
    print(f"✓ lndMvn works: {result:.4f}")
    
    # Test rwishart
    V = np.eye(2)
    W = cpp.rwishart(5.0, V)
    assert W.shape == (2, 2)
    assert np.allclose(W, W.T)  # Should be symmetric
    print(f"✓ rwishart works: shape {W.shape}")
    
    # Test rdirichlet
    alpha = np.array([1.0, 1.0, 1.0])
    d = cpp.rdirichlet(alpha)
    assert len(d) == 3
    assert abs(sum(d) - 1.0) < 1e-10  # Should sum to 1
    print(f"✓ rdirichlet works: sum={sum(d):.6f}")
    
    # Test rtrun
    result = cpp.rtrun(0.0, 1.0, -1.0, 1.0)
    assert -1.0 <= result <= 1.0
    print(f"✓ rtrun works: {result:.4f}")
    
    # Test llmnl
    beta = np.array([0.5, -0.5])
    y = np.array([0.0, 1.0, 2.0])
    X = np.random.randn(9, 2)
    ll = cpp.llmnl(beta, y, X)
    assert isinstance(ll, float)
    print(f"✓ llmnl works: {ll:.4f}")
    
    # Test breg
    y = np.random.randn(20)
    X = np.column_stack([np.ones(20), np.random.randn(20)])
    betabar = np.zeros(2)
    A = 0.01 * np.eye(2)
    beta = cpp.breg(y, X, betabar, A)
    assert len(beta) == 2
    print(f"✓ breg works: beta={beta}")
    
    return True

def main():
    print("=" * 60)
    print("bayesm Full Test Suite")
    print("=" * 60)
    
    results = {}
    
    try:
        results['Phase 2'] = test_phase2()
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        results['Phase 2'] = False
    
    try:
        results['Phase 3'] = test_phase3()
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        results['Phase 3'] = False
    
    try:
        results['C++ Bindings'] = test_cpp()
    except Exception as e:
        print(f"✗ C++ Bindings failed: {e}")
        results['C++ Bindings'] = False
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    total = sum(results.values())
    print(f"\nTotal: {total}/{len(results)} passed")
    
    return total == len(results)

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
