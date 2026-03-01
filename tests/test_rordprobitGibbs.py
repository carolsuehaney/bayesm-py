"""Tests for rordprobitGibbs (Ordered Probit Gibbs sampler)."""

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from bayesm import rordprobitGibbs

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'test_rordprobitGibbs_{datetime.now():%Y%m%d_%H%M%S}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_rordprobitGibbs_basic():
    """Test basic functionality with synthetic data."""
    np.random.seed(42)
    
    nobs = 300
    nvar = 2
    k = 4
    beta_true = np.array([1.0, -0.5])
    cutoffs_interior = np.array([0, 1.0, 2.5])
    
    X = np.random.randn(nobs, nvar)
    z = X @ beta_true + np.random.randn(nobs)
    y = np.digitize(z, cutoffs_interior) + 1
    y = np.clip(y, 1, k).astype(float)
    
    Data = {'y': y, 'X': X, 'k': k}
    Prior = {'betabar': np.zeros(nvar), 'A': 0.01 * np.eye(nvar)}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = rordprobitGibbs(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert 'cutdraw' in result
    assert 'dstardraw' in result
    assert 'accept' in result
    
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    logger.info(f"cutdraw shape: {result['cutdraw'].shape}")
    logger.info(f"dstardraw shape: {result['dstardraw'].shape}")
    logger.info(f"accept: {result['accept']:.3f}")
    
    assert result['betadraw'].shape == (1000, nvar)
    assert result['cutdraw'].shape == (1000, k - 1)
    assert result['dstardraw'].shape == (1000, k - 2)
    assert 0 <= result['accept'] <= 1
    
    return True


def test_rordprobitGibbs_posterior():
    """Test that posterior mean is close to true parameters."""
    np.random.seed(123)
    
    nobs = 500
    nvar = 2
    k = 3
    beta_true = np.array([1.5, -1.0])
    cutoffs_interior = np.array([0, 1.5])
    
    X = np.random.randn(nobs, nvar)
    z = X @ beta_true + np.random.randn(nobs)
    y = np.digitize(z, cutoffs_interior) + 1
    y = np.clip(y, 1, k).astype(float)
    
    Data = {'y': y, 'X': X, 'k': k}
    Mcmc = {'R': 2000, 'keep': 1, 'nprint': 0}
    
    result = rordprobitGibbs(Data, Mcmc=Mcmc)
    
    burnin = 500
    beta_mean = result['betadraw'][burnin:].mean(axis=0)
    
    logger.info(f"True beta: {beta_true}")
    logger.info(f"Posterior mean beta: {beta_mean}")
    logger.info(f"Acceptance rate: {result['accept']:.3f}")
    
    assert np.allclose(beta_mean, beta_true, atol=0.4), f"Beta mean too far from truth"
    
    return True


def test_rordprobitGibbs_thinning():
    """Test that thinning works correctly."""
    np.random.seed(42)
    
    nobs = 100
    nvar = 2
    k = 3
    
    X = np.random.randn(nobs, nvar)
    y = np.random.randint(1, k + 1, nobs)
    
    Data = {'y': y, 'X': X, 'k': k}
    Mcmc = {'R': 1000, 'keep': 10, 'nprint': 0}
    
    result = rordprobitGibbs(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    
    assert result['betadraw'].shape[0] == 100
    
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"rordprobitGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rordprobitGibbs_basic", test_rordprobitGibbs_basic),
        ("rordprobitGibbs_posterior", test_rordprobitGibbs_posterior),
        ("rordprobitGibbs_thinning", test_rordprobitGibbs_thinning),
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
    logger.info(f"rordprobitGibbs Tests: {passed}/{total} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
