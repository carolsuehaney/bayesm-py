#!/usr/bin/env python
"""Tests for non-homothetic logit functions (llnhlogit, simnhlogit) with logging."""

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
        logging.FileHandler(os.path.join(LOG_DIR, 'test_nhlogit.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_simnhlogit():
    """Test simnhlogit generates valid choices."""
    from bayesm.simnhlogit import simnhlogit
    
    logger.info("Testing simnhlogit")
    np.random.seed(42)
    
    n = 100
    m = 3
    d = 2
    
    alpha = np.array([0.0, 0.5, -0.5])
    k = np.array([0.0, 0.0, 0.0])
    gamma = np.array([1.0, 0.5])
    tau = 1.0
    theta = np.concatenate([alpha, k, gamma, [tau]])
    
    lnprices = np.random.randn(n, m) * 0.5
    Xexpend = np.column_stack([np.ones(n), np.random.randn(n)])
    
    result = simnhlogit(theta, lnprices, Xexpend)
    
    logger.info(f"  n={n}, m={m}, d={d}")
    logger.info(f"  theta: {theta}")
    logger.info(f"  Choice distribution: {np.bincount(result['y'], minlength=m+1)[1:]}")
    logger.info(f"  Prob row sums (should be 1): {result['prob'].sum(axis=1)[:5]}")
    
    assert result['y'].shape == (n,), f"Wrong y shape: {result['y'].shape}"
    assert np.all(result['y'] >= 1) and np.all(result['y'] <= m), "Invalid choices"
    assert result['prob'].shape == (n, m), f"Wrong prob shape: {result['prob'].shape}"
    assert np.allclose(result['prob'].sum(axis=1), 1.0), "Probabilities don't sum to 1"
    
    logger.info("  PASSED: simnhlogit")
    return True


def test_llnhlogit():
    """Test llnhlogit computes log-likelihood."""
    from bayesm.llnhlogit import llnhlogit
    from bayesm.simnhlogit import simnhlogit
    
    logger.info("Testing llnhlogit")
    np.random.seed(42)
    
    n = 100
    m = 3
    d = 2
    
    alpha = np.array([0.0, 0.5, -0.5])
    k = np.array([0.0, 0.0, 0.0])
    gamma = np.array([1.0, 0.5])
    tau = 1.0
    theta = np.concatenate([alpha, k, gamma, [tau]])
    
    lnprices = np.random.randn(n, m) * 0.5
    Xexpend = np.column_stack([np.ones(n), np.random.randn(n)])
    
    sim = simnhlogit(theta, lnprices, Xexpend)
    ll = llnhlogit(theta, sim['y'], lnprices, Xexpend)
    
    logger.info(f"  Log-likelihood at true theta: {ll:.4f}")
    
    assert isinstance(ll, (float, np.floating)), f"Should return scalar, got {type(ll)}"
    assert np.isfinite(ll), f"Log-likelihood not finite: {ll}"
    assert ll < 0, f"Log-likelihood should be negative: {ll}"
    
    logger.info("  PASSED: llnhlogit")
    return True


def test_llnhlogit_gradient():
    """Numerical gradient check for llnhlogit."""
    from bayesm.llnhlogit import llnhlogit
    from bayesm.simnhlogit import simnhlogit
    
    logger.info("Testing llnhlogit gradient (numerical)")
    np.random.seed(123)
    
    n = 50
    m = 3
    d = 2
    
    alpha = np.array([0.0, 0.5, -0.5])
    k = np.array([0.0, 0.0, 0.0])
    gamma = np.array([1.0, 0.5])
    tau = 1.0
    theta = np.concatenate([alpha, k, gamma, [tau]])
    
    lnprices = np.random.randn(n, m) * 0.5
    Xexpend = np.column_stack([np.ones(n), np.random.randn(n)])
    
    sim = simnhlogit(theta, lnprices, Xexpend)
    
    eps = 1e-5
    grad = np.zeros(len(theta))
    ll0 = llnhlogit(theta, sim['y'], lnprices, Xexpend)
    
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        ll_plus = llnhlogit(theta_plus, sim['y'], lnprices, Xexpend)
        grad[i] = (ll_plus - ll0) / eps
    
    logger.info(f"  Numerical gradient: {grad}")
    logger.info(f"  Gradient magnitude: {np.linalg.norm(grad):.4f}")
    
    assert np.all(np.isfinite(grad)), f"Gradient not finite: {grad}"
    
    logger.info("  PASSED: llnhlogit gradient check")
    return True


def test_likelihood_at_true_params():
    """Test that likelihood is higher at true params than perturbed."""
    from bayesm.llnhlogit import llnhlogit
    from bayesm.simnhlogit import simnhlogit
    
    logger.info("Testing likelihood maximized near true params")
    np.random.seed(456)
    
    n = 200
    m = 3
    d = 2
    
    alpha = np.array([0.0, 0.5, -0.5])
    k = np.array([0.0, 0.0, 0.0])
    gamma = np.array([1.0, 0.5])
    tau = 1.0
    theta_true = np.concatenate([alpha, k, gamma, [tau]])
    
    lnprices = np.random.randn(n, m) * 0.5
    Xexpend = np.column_stack([np.ones(n), np.random.randn(n)])
    
    sim = simnhlogit(theta_true, lnprices, Xexpend)
    
    ll_true = llnhlogit(theta_true, sim['y'], lnprices, Xexpend)
    
    theta_perturbed = theta_true + np.random.randn(len(theta_true)) * 0.5
    ll_perturbed = llnhlogit(theta_perturbed, sim['y'], lnprices, Xexpend)
    
    logger.info(f"  LL at true params: {ll_true:.4f}")
    logger.info(f"  LL at perturbed params: {ll_perturbed:.4f}")
    logger.info(f"  Difference: {ll_true - ll_perturbed:.4f}")
    
    assert ll_true > ll_perturbed, "Likelihood should be higher at true params"
    
    logger.info("  PASSED: likelihood comparison")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"Non-homothetic Logit Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('simnhlogit', test_simnhlogit),
        ('llnhlogit', test_llnhlogit),
        ('llnhlogit_gradient', test_llnhlogit_gradient),
        ('likelihood_comparison', test_likelihood_at_true_params),
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
    logger.info(f"NH Logit Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
