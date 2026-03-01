#!/usr/bin/env python
"""Tests for runiregGibbs MCMC sampler with logging."""

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
        logging.FileHandler(os.path.join(LOG_DIR, 'test_runiregGibbs.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_runiregGibbs_basic():
    """Test runiregGibbs runs and returns correct shapes."""
    from bayesm.runiregGibbs import runiregGibbs
    
    logger.info("Testing runiregGibbs basic functionality")
    np.random.seed(42)
    
    n, k = 100, 3
    beta_true = np.array([1.0, -0.5, 0.3])
    sigmasq_true = 0.5
    
    X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
    y = X @ beta_true + np.sqrt(sigmasq_true) * np.random.randn(n)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = runiregGibbs(Data, Mcmc=Mcmc)
    
    logger.info(f"  betadraw shape: {result['betadraw'].shape}")
    logger.info(f"  sigmasqdraw shape: {result['sigmasqdraw'].shape}")
    
    assert result['betadraw'].shape == (1000, k)
    assert result['sigmasqdraw'].shape == (1000,)
    
    logger.info("  PASSED: runiregGibbs basic")
    return True


def test_runiregGibbs_posterior():
    """Test that posterior mean is close to true values."""
    from bayesm.runiregGibbs import runiregGibbs
    
    logger.info("Testing runiregGibbs posterior mean")
    np.random.seed(123)
    
    n, k = 200, 2
    beta_true = np.array([2.0, -1.0])
    sigmasq_true = 1.0
    
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    y = X @ beta_true + np.sqrt(sigmasq_true) * np.random.randn(n)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 5000, 'keep': 1, 'nprint': 0}
    
    result = runiregGibbs(Data, Mcmc=Mcmc)
    
    burn = 1000
    beta_mean = result['betadraw'][burn:].mean(axis=0)
    sigmasq_mean = result['sigmasqdraw'][burn:].mean()
    
    logger.info(f"  True beta: {beta_true}")
    logger.info(f"  Posterior mean beta: {beta_mean}")
    logger.info(f"  True sigmasq: {sigmasq_true}")
    logger.info(f"  Posterior mean sigmasq: {sigmasq_mean:.4f}")
    
    assert np.allclose(beta_mean, beta_true, atol=0.3)
    assert abs(sigmasq_mean - sigmasq_true) < 0.5
    
    logger.info("  PASSED: runiregGibbs posterior mean")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"runiregGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('runiregGibbs_basic', test_runiregGibbs_basic),
        ('runiregGibbs_posterior', test_runiregGibbs_posterior),
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
    logger.info(f"runiregGibbs Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
