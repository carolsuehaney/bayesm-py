#!/usr/bin/env python
"""Tests for rhierLinearModel MCMC sampler with logging."""

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
        logging.FileHandler(os.path.join(LOG_DIR, 'test_rhierLinearModel.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_rhierLinearModel_basic():
    """Test rhierLinearModel runs and returns correct shapes."""
    from bayesm.rhierLinearModel import rhierLinearModel
    
    logger.info("Testing rhierLinearModel basic functionality")
    np.random.seed(42)
    
    nreg = 20
    nobs = 50
    nvar = 3
    
    regdata = []
    for i in range(nreg):
        X = np.column_stack([np.ones(nobs), np.random.randn(nobs, nvar-1)])
        beta = np.random.randn(nvar)
        y = X @ beta + np.random.randn(nobs)
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata}
    Mcmc = {'R': 500, 'keep': 1, 'nprint': 0}
    
    result = rhierLinearModel(Data, Mcmc=Mcmc)
    
    logger.info(f"  betadraw shape: {result['betadraw'].shape}")
    logger.info(f"  taudraw shape: {result['taudraw'].shape}")
    logger.info(f"  Deltadraw shape: {result['Deltadraw'].shape}")
    logger.info(f"  Vbetadraw shape: {result['Vbetadraw'].shape}")
    
    assert result['betadraw'].shape == (nreg, nvar, 500)
    assert result['taudraw'].shape == (500, nreg)
    assert result['Deltadraw'].shape == (500, 1 * nvar)
    assert result['Vbetadraw'].shape == (500, nvar * nvar)
    
    logger.info("  PASSED: rhierLinearModel basic")
    return True


def test_rhierLinearModel_with_Z():
    """Test rhierLinearModel with Z covariates."""
    from bayesm.rhierLinearModel import rhierLinearModel
    
    logger.info("Testing rhierLinearModel with Z covariates")
    np.random.seed(123)
    
    nreg = 30
    nobs = 40
    nvar = 2
    nz = 2
    
    Delta_true = np.array([[1.0, -0.5], [0.5, 0.3]])
    Z = np.column_stack([np.ones(nreg), np.random.randn(nreg)])
    
    regdata = []
    for i in range(nreg):
        X = np.column_stack([np.ones(nobs), np.random.randn(nobs)])
        beta = Z[i, :] @ Delta_true + np.random.randn(nvar) * 0.5
        y = X @ beta + np.random.randn(nobs)
        regdata.append({'y': y, 'X': X})
    
    Data = {'regdata': regdata, 'Z': Z}
    Mcmc = {'R': 1000, 'keep': 1, 'nprint': 0}
    
    result = rhierLinearModel(Data, Mcmc=Mcmc)
    
    burn = 500
    Deltadraw = result['Deltadraw'][burn:, :]
    Delta_mean = Deltadraw.mean(axis=0).reshape(nz, nvar)
    
    logger.info(f"  True Delta:\n{Delta_true}")
    logger.info(f"  Posterior mean Delta:\n{Delta_mean}")
    
    assert result['betadraw'].shape == (nreg, nvar, 1000)
    assert result['Deltadraw'].shape == (1000, nz * nvar)
    
    logger.info("  PASSED: rhierLinearModel with Z")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"rhierLinearModel Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('rhierLinearModel_basic', test_rhierLinearModel_basic),
        ('rhierLinearModel_with_Z', test_rhierLinearModel_with_Z),
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
    logger.info(f"rhierLinearModel Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
