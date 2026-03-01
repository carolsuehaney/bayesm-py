#!/usr/bin/env python
"""Tests for cluster_mix, plotting, and summary functions."""

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
        logging.FileHandler(os.path.join(LOG_DIR, 'test_cluster_summary.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_cluster_mix():
    """Test cluster_mix function."""
    from bayesm.cluster_mix import cluster_mix
    
    logger.info("Testing cluster_mix")
    np.random.seed(42)
    
    nobs = 50
    R = 200
    true_clusters = np.repeat([1, 2, 3], [20, 15, 15])
    
    zdraw = np.zeros((R, nobs), dtype=int)
    for r in range(R):
        zdraw[r, :] = true_clusters + np.random.choice([0, 1, -1], nobs, p=[0.8, 0.1, 0.1])
        zdraw[r, :] = np.clip(zdraw[r, :], 1, 3)
    
    result = cluster_mix(zdraw, cutoff=0.9, silent=True)
    
    logger.info(f"  clustera unique values: {np.unique(result['clustera'])}")
    logger.info(f"  clusterb unique values: {np.unique(result['clusterb'])}")
    
    assert result['clustera'].shape == (nobs,)
    assert result['clusterb'].shape == (nobs,)
    assert np.all(result['clustera'] >= 1)
    assert np.all(result['clusterb'] >= 1)
    
    logger.info("  PASSED: cluster_mix")
    return True


def test_summary_mat():
    """Test summary_mat function."""
    from bayesm.summary import summary_mat
    
    logger.info("Testing summary_mat")
    np.random.seed(42)
    
    R = 1000
    k = 3
    X = np.random.randn(R, k)
    X[:, 0] += 1.0
    X[:, 1] -= 0.5
    
    logger.info("  Calling summary_mat...")
    result = summary_mat(X, names=['beta1', 'beta2', 'beta3'], burnin=100)
    
    assert result is not None
    assert result.shape[0] == k
    
    logger.info("  PASSED: summary_mat")
    return True


def test_summary_var():
    """Test summary_var function."""
    from bayesm.summary import summary_var
    
    logger.info("Testing summary_var")
    np.random.seed(42)
    
    R = 500
    d = 2
    
    Vard = np.zeros((R, d*d))
    for r in range(R):
        V = np.eye(d) + 0.1 * np.random.randn(d, d)
        V = V @ V.T
        Vard[r, :] = V.flatten()
    
    logger.info("  Calling summary_var...")
    summary_var(Vard, names=['x1', 'x2'], burnin=50)
    
    logger.info("  PASSED: summary_var")
    return True


def main():
    logger.info("=" * 60)
    logger.info(f"Cluster/Summary Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ('cluster_mix', test_cluster_mix),
        ('summary_mat', test_summary_mat),
        ('summary_var', test_summary_var),
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
    logger.info(f"Cluster/Summary Tests: {passed}/{total} passed")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
