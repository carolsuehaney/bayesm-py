#!/usr/bin/env python
"""Tests for MNP functions (mnp_prob, llmnp) with logging."""

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
        logging.FileHandler(os.path.join(LOG_DIR, 'test_mnp.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_mnp_prob():
    """Test mnp_prob computes valid probabilities."""
    from bayesm.mnp_prob import mnp_prob
    
    logger.info("Testing mnp_prob")
    np.random.seed(42)
    
    p = 3
    k = 2
    pm1 = p - 1
    
    beta = np.array([0.5, -0.3])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.random.randn(pm1, k)
    
    prob = mnp_prob(beta, Sigma, X, r=200)
    
    logger.info(f"  beta: {beta}")
    logger.info(f"  Sigma:\n{Sigma}")
    logger.info(f"  X:\n{X}")
    logger.info(f"  Probabilities: {prob}")
    logger.info(f"  Sum of probabilities: {np.sum(prob):.6f}")
    
    assert prob.shape == (p,), f"Wrong shape: {prob.shape}"
    assert np.all(prob >= 0), f"Negative probabilities: {prob}"
    assert np.all(prob <= 1), f"Probabilities > 1: {prob}"
    assert np.abs(np.sum(prob) - 1.0) < 0.05, f"Probabilities don't sum to 1: {np.sum(prob)}"
    
    logger.info("  PASSED: mnp_prob")
    return True


def test_llmnp():
    """Test llmnp computes log-likelihood."""
    from bayesm.llmnp import llmnp
    
    logger.info("Testing llmnp")
    np.random.seed(42)
    
    n = 20
    p = 3
    k = 2
    pm1 = p - 1
    
    beta = np.array([0.5, -0.3])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.random.randn(n * pm1, k)
    y = np.random.randint(1, p + 1, n)
    
    ll = llmnp(beta, Sigma, X, y, r=100)
    
    logger.info(f"  n={n}, p={p}, k={k}")
    logger.info(f"  beta: {beta}")
    logger.info(f"  y distribution: {np.bincount(y, minlength=p+1)[1:]}")
    logger.info(f"  Log-likelihood: {ll:.4f}")
    
    assert isinstance(ll, (float, np.floating)), f"Should return scalar, got {type(ll)}"
    assert ll < 0, f"Log-likelihood should be negative: {ll}"
    assert np.isfinite(ll), f"Log-likelihood not finite: {ll}"
    
    logger.info("  PASSED: llmnp")
    return True


def test_llmnp_gradient_check():
    """Numerical gradient check for llmnp."""
    from bayesm.llmnp import llmnp
    
    logger.info("Testing llmnp gradient (numerical)")
    np.random.seed(123)
    
    n = 30
    p = 3
    k = 2
    pm1 = p - 1
    
    beta = np.array([0.5, -0.3])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    X = np.random.randn(n * pm1, k)
    y = np.random.randint(1, p + 1, n)
    
    eps = 1e-5
    grad = np.zeros(k)
    ll0 = llmnp(beta, Sigma, X, y, r=500)
    
    for i in range(k):
        beta_plus = beta.copy()
        beta_plus[i] += eps
        ll_plus = llmnp(beta_plus, Sigma, X, y, r=500)
        grad[i] = (ll_plus - ll0) / eps
    
    logger.info(f"  Numerical gradient: {grad}")
    logger.info(f"  (Gradient magnitude: {np.linalg.norm(grad):.4f})")
    
    assert np.all(np.isfinite(grad)), f"Gradient not finite: {grad}"
    
    logger.info("  PASSED: llmnp gradient check")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"MNP Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('mnp_prob', test_mnp_prob),
        ('llmnp', test_llmnp),
        ('llmnp_gradient', test_llmnp_gradient_check),
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
    logger.info(f"MNP Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
