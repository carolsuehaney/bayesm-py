#!/usr/bin/env python
"""Tests for runireg MCMC sampler with logging."""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'test_runireg.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_runireg_basic():
    """Test runireg runs and returns correct shapes."""
    from bayesm.runireg import runireg
    
    logger.info("Testing runireg basic functionality")
    np.random.seed(42)
    
    n, k = 100, 3
    beta_true = np.array([1.0, -0.5, 0.3])
    sigmasq_true = 0.5
    
    X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
    y = X @ beta_true + np.sqrt(sigmasq_true) * np.random.randn(n)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = runireg(Data, Mcmc=Mcmc)
    
    logger.info(f"  betadraw shape: {result['betadraw'].shape}")
    logger.info(f"  sigmasqdraw shape: {result['sigmasqdraw'].shape}")
    
    assert result['betadraw'].shape == (1000, k), f"Wrong betadraw shape"
    assert result['sigmasqdraw'].shape == (1000,), f"Wrong sigmasqdraw shape"
    
    logger.info("  PASSED: runireg basic")
    return True


def test_runireg_posterior_mean():
    """Test that posterior mean is close to true values."""
    from bayesm.runireg import runireg
    
    logger.info("Testing runireg posterior mean")
    np.random.seed(123)
    
    n, k = 200, 2
    beta_true = np.array([2.0, -1.0])
    sigmasq_true = 1.0
    
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = X @ beta_true + np.sqrt(sigmasq_true) * np.random.randn(n)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 5000, 'keep': 1, 'nprint': 0}
    
    result = runireg(Data, Mcmc=Mcmc)
    
    burn = 1000
    beta_mean = result['betadraw'][burn:].mean(axis=0)
    sigmasq_mean = result['sigmasqdraw'][burn:].mean()
    
    logger.info(f"  True beta: {beta_true}")
    logger.info(f"  Posterior mean beta: {beta_mean}")
    logger.info(f"  True sigmasq: {sigmasq_true}")
    logger.info(f"  Posterior mean sigmasq: {sigmasq_mean:.4f}")
    
    assert np.allclose(beta_mean, beta_true, atol=0.3), f"Beta mean too far from truth"
    assert abs(sigmasq_mean - sigmasq_true) < 0.5, f"Sigmasq mean too far from truth"
    
    logger.info("  PASSED: runireg posterior mean")
    return True


def test_runireg_thinning():
    """Test that thinning (keep) works correctly."""
    from bayesm.runireg import runireg
    
    logger.info("Testing runireg thinning")
    np.random.seed(456)
    
    n, k = 50, 2
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = np.random.randn(n)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 10, 'nprint': 0}
    
    result = runireg(Data, Mcmc=Mcmc)
    
    logger.info(f"  R=1000, keep=10 -> expected 100 draws")
    logger.info(f"  betadraw shape: {result['betadraw'].shape}")
    
    assert result['betadraw'].shape == (100, k), f"Thinning not working"
    assert result['sigmasqdraw'].shape == (100,), f"Thinning not working"
    
    logger.info("  PASSED: runireg thinning")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"runireg Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('runireg_basic', test_runireg_basic),
        ('runireg_posterior_mean', test_runireg_posterior_mean),
        ('runireg_thinning', test_runireg_thinning),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"FAILED: {name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[name] = False
    
    logger.info("=" * 60)
    passed = sum(results.values())
    total = len(results)
    logger.info(f"runireg Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
