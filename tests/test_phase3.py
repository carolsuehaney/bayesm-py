#!/usr/bin/env python
"""
Quick test for Phase 3 functions
"""

import sys
sys.path.insert(0, '.')
import numpy as np

def test_mnl_hess():
    """Test mnl_hess"""
    from bayesm.mnl_hess import mnl_hess
    
    np.random.seed(123)
    n, j, k = 100, 3, 2
    X = np.random.randn(n * j, k)
    beta = np.array([0.5, -0.5])
    y = np.random.randint(0, j, n)
    
    H = mnl_hess(beta, y, X)
    
    # Check shape and symmetry
    assert H.shape == (k, k), f"Wrong shape: {H.shape}"
    assert np.allclose(H, H.T), "Hessian not symmetric"
    print("✓ mnl_hess works")
    return True

def test_log_marg_den_nr():
    """Test log_marg_den_nr"""
    from bayesm.log_marg_den_nr import log_marg_den_nr
    
    ll = np.array([-100, -105, -98, -102, -99])
    result = log_marg_den_nr(ll)
    
    assert isinstance(result, (float, np.floating)), "Should return scalar"
    print(f"✓ log_marg_den_nr works: {result:.2f}")
    return True

def test_mix_den():
    """Test mix_den"""
    from bayesm.mix_den import mix_den
    
    # Simple 2-component, 2-dim mixture
    comps = [
        [np.array([0, 0]), np.linalg.inv(np.linalg.cholesky(np.eye(2)).T)],
        [np.array([2, 2]), np.linalg.inv(np.linalg.cholesky(np.eye(2)).T)]
    ]
    pvec = np.array([0.5, 0.5])
    x = np.array([[0, 0], [1, 1], [2, 2]])
    
    den = mix_den(x, pvec, comps)
    
    assert den.shape == x.shape, f"Wrong shape: {den.shape}"
    assert np.all(den > 0), "Densities should be positive"
    print("✓ mix_den works")
    return True

def test_mom_mix():
    """Test mom_mix"""
    from bayesm.mom_mix import mom_mix
    
    # Create simple test data
    dim = 2
    nc = 2
    n_draws = 10
    
    compdraw = []
    probdraw = np.random.dirichlet([1, 1], n_draws)
    
    for i in range(n_draws):
        comps = []
        for j in range(nc):
            mu = np.random.randn(dim)
            rooti = np.linalg.inv(np.linalg.cholesky(np.eye(dim)).T)
            comps.append([mu, rooti])
        compdraw.append(comps)
    
    result = mom_mix(probdraw, compdraw)
    
    assert 'mu' in result, "Missing mu"
    assert 'sigma' in result, "Missing sigma"
    assert 'sd' in result, "Missing sd"
    assert 'corr' in result, "Missing corr"
    assert result['mu'].shape == (dim,), "Wrong mu shape"
    assert result['sigma'].shape == (dim, dim), "Wrong sigma shape"
    print("✓ mom_mix works")
    return True

def main():
    print("Testing Phase 3 functions...")
    print("=" * 60)
    
    tests = [
        ('mnl_hess', test_mnl_hess),
        ('log_marg_den_nr', test_log_marg_den_nr),
        ('mix_den', test_mix_den),
        ('mom_mix', test_mom_mix),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = False
    
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"\nPhase 3: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All Phase 3 functions working!")
    
    return passed == total

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
