"""Tests for rmnlIndepMetrop (MNL Independence Metropolis sampler)."""

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from bayesm import rmnlIndepMetrop

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'test_rmnlIndepMetrop_{datetime.now():%Y%m%d_%H%M%S}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_rmnlIndepMetrop_basic():
    """Test basic functionality with synthetic data."""
    np.random.seed(42)
    
    p = 3
    nobs = 100
    nvar = 4
    beta_true = np.array([1.0, -0.5, 0.3, 0.2])
    
    X = np.random.randn(nobs * p, nvar)
    Xbeta = X @ beta_true
    Xbeta_mat = Xbeta.reshape((nobs, p))
    exp_Xbeta = np.exp(Xbeta_mat - Xbeta_mat.max(axis=1, keepdims=True))
    probs = exp_Xbeta / exp_Xbeta.sum(axis=1, keepdims=True)
    y = np.array([np.random.choice(p, p=probs[i]) + 1 for i in range(nobs)])
    
    Data = {'p': p, 'y': y, 'X': X}
    Prior = {'betabar': np.zeros(nvar), 'A': 0.01 * np.eye(nvar)}
    Mcmc = {'R': 500, 'keep': 1, 'nprint': 0, 'nu': 6}
    
    result = rmnlIndepMetrop(Data, Prior, Mcmc)
    
    assert 'betadraw' in result
    assert 'loglike' in result
    assert 'acceptr' in result
    
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    logger.info(f"loglike shape: {result['loglike'].shape}")
    logger.info(f"acceptance rate: {result['acceptr']:.3f}")
    
    assert result['betadraw'].shape == (500, nvar)
    assert result['loglike'].shape == (500,)
    assert 0 <= result['acceptr'] <= 1
    
    return True


def test_rmnlIndepMetrop_posterior():
    """Test that posterior mean is close to true beta with informative data."""
    np.random.seed(123)
    
    p = 2
    nobs = 500
    nvar = 2
    beta_true = np.array([2.0, -1.0])
    
    X = np.random.randn(nobs * p, nvar)
    Xbeta = X @ beta_true
    Xbeta_mat = Xbeta.reshape((nobs, p))
    exp_Xbeta = np.exp(Xbeta_mat - Xbeta_mat.max(axis=1, keepdims=True))
    probs = exp_Xbeta / exp_Xbeta.sum(axis=1, keepdims=True)
    y = np.array([np.random.choice(p, p=probs[i]) + 1 for i in range(nobs)])
    
    Data = {'p': p, 'y': y, 'X': X}
    Prior = {'betabar': np.zeros(nvar), 'A': 0.01 * np.eye(nvar)}
    Mcmc = {'R': 2000, 'keep': 1, 'nprint': 0, 'nu': 6}
    
    result = rmnlIndepMetrop(Data, Prior, Mcmc)
    
    burnin = 500
    beta_mean = result['betadraw'][burnin:].mean(axis=0)
    
    logger.info(f"True beta: {beta_true}")
    logger.info(f"Posterior mean beta: {beta_mean}")
    logger.info(f"Acceptance rate: {result['acceptr']:.3f}")
    
    assert np.allclose(beta_mean, beta_true, atol=0.5), f"Beta mean too far from truth"
    
    return True


def test_rmnlIndepMetrop_thinning():
    """Test that thinning works correctly."""
    np.random.seed(42)
    
    p = 2
    nobs = 50
    nvar = 2
    
    X = np.random.randn(nobs * p, nvar)
    y = np.random.randint(1, p + 1, nobs)
    
    Data = {'p': p, 'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 10, 'nprint': 0, 'nu': 6}
    
    result = rmnlIndepMetrop(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    
    assert result['betadraw'].shape[0] == 100
    
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"rmnlIndepMetrop Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rmnlIndepMetrop_basic", test_rmnlIndepMetrop_basic),
        ("rmnlIndepMetrop_posterior", test_rmnlIndepMetrop_posterior),
        ("rmnlIndepMetrop_thinning", test_rmnlIndepMetrop_thinning),
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
    logger.info(f"rmnlIndepMetrop Tests: {passed}/{total} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
