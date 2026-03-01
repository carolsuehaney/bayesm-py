"""Test rmnpGibbs - multinomial probit Gibbs sampler."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
import logging
from datetime import datetime
from bayesm import rmnpGibbs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rmnpGibbs_basic():
    """Basic functionality test."""
    np.random.seed(42)
    n = 100
    p = 3
    pm1 = p - 1
    k = 3
    
    X = np.random.randn(n * pm1, k)
    X[:, 0] = 1
    beta_true = np.array([0.5, 1.0, -0.5])
    
    w = X @ beta_true + np.random.randn(n * pm1)
    w = w.reshape(n, pm1)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if np.all(w[i, :] < 0):
            y[i] = p
        else:
            y[i] = np.argmax(w[i, :]) + 1
    
    Data = {'p': p, 'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 1}
    
    result = rmnpGibbs(Data, Mcmc=Mcmc)
    
    assert result['betadraw'].shape == (1000, k)
    assert result['sigmadraw'].shape == (1000, pm1 * pm1)
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    logger.info(f"sigmadraw shape: {result['sigmadraw'].shape}")
    logger.info("PASSED: rmnpGibbs_basic")
    return True

def test_rmnpGibbs_thinning():
    """Test thinning."""
    np.random.seed(456)
    n = 50
    p = 3
    pm1 = p - 1
    k = 2
    
    X = np.random.randn(n * pm1, k)
    X[:, 0] = 1
    beta_true = np.array([0.5, 1.0])
    
    w = X @ beta_true + np.random.randn(n * pm1)
    w = w.reshape(n, pm1)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if np.all(w[i, :] < 0):
            y[i] = p
        else:
            y[i] = np.argmax(w[i, :]) + 1
    
    Data = {'p': p, 'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 10}
    
    result = rmnpGibbs(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    assert result['betadraw'].shape[0] == 100
    logger.info("PASSED: rmnpGibbs_thinning")
    return True

def main():
    logger.info("=" * 60)
    logger.info(f"rmnpGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rmnpGibbs_basic", test_rmnpGibbs_basic),
        ("rmnpGibbs_thinning", test_rmnpGibbs_thinning),
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"Testing {name}")
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"FAILED: {name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[name] = False
    
    passed = sum(results.values())
    logger.info("=" * 60)
    logger.info(f"rmnpGibbs Tests: {passed}/{len(tests)} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()
