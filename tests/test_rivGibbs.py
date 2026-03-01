"""Tests for rivGibbs (Linear IV Gibbs sampler)."""

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from bayesm import rivGibbs

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'test_rivGibbs_{datetime.now():%Y%m%d_%H%M%S}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_rivGibbs_basic():
    """Test basic functionality with synthetic IV data."""
    np.random.seed(42)
    
    n = 200
    dimd = 2
    dimg = 1
    
    delta_true = np.array([0.5, 0.8])
    beta_true = 1.5
    gamma_true = np.array([0.3])
    
    z = np.random.randn(n, dimd)
    w = np.random.randn(n, dimg)
    
    Sigma_true = np.array([[1.0, 0.5], [0.5, 1.0]])
    L = np.linalg.cholesky(Sigma_true)
    errors = np.random.randn(n, 2) @ L.T
    e1, e2 = errors[:, 0], errors[:, 1]
    
    x = z @ delta_true + e1
    y = beta_true * x + w @ gamma_true + e2
    
    Data = {'y': y, 'x': x, 'z': z, 'w': w}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = rivGibbs(Data, Mcmc=Mcmc)
    
    assert 'deltadraw' in result
    assert 'betadraw' in result
    assert 'gammadraw' in result
    assert 'Sigmadraw' in result
    
    logger.info(f"deltadraw shape: {result['deltadraw'].shape}")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    logger.info(f"gammadraw shape: {result['gammadraw'].shape}")
    logger.info(f"Sigmadraw shape: {result['Sigmadraw'].shape}")
    
    assert result['deltadraw'].shape == (1000, dimd)
    assert result['betadraw'].shape == (1000,)
    assert result['gammadraw'].shape == (1000, dimg)
    assert result['Sigmadraw'].shape == (1000, 4)
    
    return True


def test_rivGibbs_posterior():
    """Test that posterior mean is close to true parameters."""
    np.random.seed(123)
    
    n = 500
    dimd = 2
    dimg = 1
    
    delta_true = np.array([1.0, 0.5])
    beta_true = 2.0
    gamma_true = np.array([-0.5])
    
    z = np.random.randn(n, dimd)
    w = np.random.randn(n, dimg)
    
    Sigma_true = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma_true)
    errors = np.random.randn(n, 2) @ L.T
    e1, e2 = errors[:, 0], errors[:, 1]
    
    x = z @ delta_true + e1
    y = beta_true * x + w @ gamma_true + e2
    
    Data = {'y': y, 'x': x, 'z': z, 'w': w}
    Mcmc = {'R': 2000, 'keep': 1, 'nprint': 0}
    
    result = rivGibbs(Data, Mcmc=Mcmc)
    
    burnin = 500
    delta_mean = result['deltadraw'][burnin:].mean(axis=0)
    beta_mean = result['betadraw'][burnin:].mean()
    gamma_mean = result['gammadraw'][burnin:].mean(axis=0)
    
    logger.info(f"True delta: {delta_true}")
    logger.info(f"Posterior mean delta: {delta_mean}")
    logger.info(f"True beta: {beta_true}")
    logger.info(f"Posterior mean beta: {beta_mean:.3f}")
    logger.info(f"True gamma: {gamma_true}")
    logger.info(f"Posterior mean gamma: {gamma_mean}")
    
    assert np.allclose(delta_mean, delta_true, atol=0.3), f"Delta mean too far from truth"
    assert np.abs(beta_mean - beta_true) < 0.5, f"Beta mean too far from truth"
    
    return True


def test_rivGibbs_thinning():
    """Test that thinning works correctly."""
    np.random.seed(42)
    
    n = 100
    dimd = 2
    dimg = 1
    
    z = np.random.randn(n, dimd)
    w = np.random.randn(n, dimg)
    x = np.random.randn(n)
    y = np.random.randn(n)
    
    Data = {'y': y, 'x': x, 'z': z, 'w': w}
    Mcmc = {'R': 1000, 'keep': 10, 'nprint': 0}
    
    result = rivGibbs(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"deltadraw shape: {result['deltadraw'].shape}")
    
    assert result['deltadraw'].shape[0] == 100
    
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"rivGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rivGibbs_basic", test_rivGibbs_basic),
        ("rivGibbs_posterior", test_rivGibbs_posterior),
        ("rivGibbs_thinning", test_rivGibbs_thinning),
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
    logger.info(f"rivGibbs Tests: {passed}/{total} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
