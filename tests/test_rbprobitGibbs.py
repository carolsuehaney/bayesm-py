"""Tests for rbprobitGibbs (Binary Probit Gibbs sampler)."""

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from bayesm import rbprobitGibbs

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'test_rbprobitGibbs_{datetime.now():%Y%m%d_%H%M%S}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_rbprobitGibbs_basic():
    """Test basic functionality with synthetic data."""
    np.random.seed(42)
    
    nobs = 200
    nvar = 3
    beta_true = np.array([1.0, -0.5, 0.3])
    
    X = np.random.randn(nobs, nvar)
    w = X @ beta_true + np.random.randn(nobs)
    y = (w > 0).astype(float)
    
    Data = {'y': y, 'X': X}
    Prior = {'betabar': np.zeros(nvar), 'A': 0.01 * np.eye(nvar)}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = rbprobitGibbs(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    assert result['betadraw'].shape == (1000, nvar)
    
    return True


def test_rbprobitGibbs_posterior():
    """Test that posterior mean is close to true beta."""
    np.random.seed(123)
    
    nobs = 500
    nvar = 2
    beta_true = np.array([1.5, -1.0])
    
    X = np.random.randn(nobs, nvar)
    w = X @ beta_true + np.random.randn(nobs)
    y = (w > 0).astype(float)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 2000, 'keep': 1, 'nprint': 0}
    
    result = rbprobitGibbs(Data, Mcmc=Mcmc)
    
    burnin = 500
    beta_mean = result['betadraw'][burnin:].mean(axis=0)
    
    logger.info(f"True beta: {beta_true}")
    logger.info(f"Posterior mean beta: {beta_mean}")
    
    assert np.allclose(beta_mean, beta_true, atol=0.3), f"Beta mean too far from truth"
    
    return True


def test_rbprobitGibbs_thinning():
    """Test that thinning works correctly."""
    np.random.seed(42)
    
    nobs = 100
    nvar = 2
    
    X = np.random.randn(nobs, nvar)
    y = np.random.randint(0, 2, nobs).astype(float)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 10, 'nprint': 0}
    
    result = rbprobitGibbs(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    
    assert result['betadraw'].shape[0] == 100
    
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"rbprobitGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rbprobitGibbs_basic", test_rbprobitGibbs_basic),
        ("rbprobitGibbs_posterior", test_rbprobitGibbs_posterior),
        ("rbprobitGibbs_thinning", test_rbprobitGibbs_thinning),
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"Testing {name}")
        try:
            results[name] = test_func()
            logger.info(f"PASSED: {name}")
        except Exception as e:
            results[name] = False
            logger.error(f"FAILED: {name} - {e}")
            logger.error(traceback.format_exc())
    
    passed = sum(results.values())
    total = len(results)
    logger.info("=" * 60)
    logger.info(f"rbprobitGibbs Tests: {passed}/{total} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
