#!/usr/bin/env python
"""
Test script for Phase 2 - Testing bayesm package

Run this from the python/ directory:
    python test_phase2.py
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    try:
        from bayesm.constants import BayesmConstants
        from bayesm.utilities import pandterm, nmat
        from bayesm.create_x import create_x
        from bayesm.cond_mom import cond_mom
        from bayesm.num_eff import num_eff
        from bayesm.data_loaders import load_data
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_constants():
    """Test constants match R values"""
    print("\nTesting constants...")
    from bayesm.constants import BayesmConstants
    
    tests = [
        (BayesmConstants.keep == 1, "keep"),
        (BayesmConstants.nprint == 100, "nprint"),
        (BayesmConstants.RRScaling == 2.38, "RRScaling"),
        (BayesmConstants.A == 0.01, "A"),
        (BayesmConstants.DPalpha == 1.0, "DPalpha"),
    ]
    
    passed = sum(1 for test, _ in tests if test)
    print(f"  {passed}/{len(tests)} constant checks passed")
    return passed == len(tests)


def test_data_loaders():
    """Test data loading"""
    print("\nTesting data loaders...")
    from bayesm.data_loaders import load_data
    
    results = {}
    
    # Test flat datasets
    for name in ['cheese', 'customerSat', 'Scotch', 'tuna']:
        try:
            df = load_data(name)
            results[name] = f"✓ {df.shape[0]} rows"
        except Exception as e:
            results[name] = f"✗ {str(e)[:50]}"
    
    # Test multi-component datasets
    for name in ['bank', 'detailing', 'margarine', 'orangeJuice']:
        try:
            data = load_data(name)
            keys = list(data.keys())
            results[name] = f"✓ {keys}"
        except Exception as e:
            results[name] = f"✗ {str(e)[:50]}"
    
    # Test camera
    try:
        df = load_data('camera', format='long')
        lgt = load_data('camera', format='lgtdata')
        results['camera'] = f"✓ {df['id'].nunique()} respondents, {len(lgt)} lgtdata"
    except Exception as e:
        results['camera'] = f"✗ {str(e)[:50]}"
    
    for name, result in results.items():
        print(f"  {name}: {result}")
    
    passed = sum(1 for r in results.values() if r.startswith('✓'))
    return passed == len(results)


def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")
    import numpy as np
    from bayesm.utilities import nmat
    
    # Test nmat
    vec = np.array([4, 2, 2, 4])  # Cov matrix with variances [4, 4]
    result = nmat(vec)
    expected = np.array([1, 0.5, 0.5, 1])  # Correlation matrix
    
    if np.allclose(result, expected, rtol=1e-10):
        print("  ✓ nmat works correctly")
        return True
    else:
        print(f"  ✗ nmat failed: got {result}, expected {expected}")
        return False


def test_create_x():
    """Test create_x function"""
    print("\nTesting create_x...")
    import numpy as np
    from bayesm.create_x import create_x
    
    try:
        np.random.seed(123)
        p, na, nd = 3, 2, 1
        n = 10
        Xa = np.random.randn(n, na * p)
        Xd = np.random.randn(n, nd)
        
        X = create_x(p=p, na=na, nd=nd, Xa=Xa, Xd=Xd, INT=True, DIFF=False)
        
        expected_rows = n * p
        expected_cols = (nd + 1) * (p - 1) + na
        
        if X.shape == (expected_rows, expected_cols):
            print(f"  ✓ create_x produces correct shape: {X.shape}")
            return True
        else:
            print(f"  ✗ create_x shape mismatch: got {X.shape}, expected ({expected_rows}, {expected_cols})")
            return False
    except Exception as e:
        print(f"  ✗ create_x failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("bayesm Phase 2 Testing")
    print("=" * 60)
    
    results = {
        'Imports': test_imports(),
        'Constants': test_constants(),
        'Data Loaders': test_data_loaders(),
        'Utilities': test_utilities(),
        'Create X': test_create_x(),
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} test groups passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Ready for Phase 3.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
