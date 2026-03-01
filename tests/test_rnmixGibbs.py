"""Test rnmixGibbs - normal mixture Gibbs sampler."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
import logging
from datetime import datetime
from bayesm import rnmixGibbs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rnmixGibbs_basic():
    """Basic functionality test."""
    np.random.seed(42)
    nobs = 200
    ncomp = 2
    
    # Generate mixture data
    mu1 = np.array([0, 0])
    mu2 = np.array([3, 3])
    Sigma = np.eye(2)
    
    z_true = np.random.choice([0, 1], size=nobs, p=[0.6, 0.4])
    y = np.zeros((nobs, 2))
    y[z_true == 0] = np.random.multivariate_normal(mu1, Sigma, sum(z_true == 0))
    y[z_true == 1] = np.random.multivariate_normal(mu2, Sigma, sum(z_true == 1))
    
    Data = {'y': y}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 500, 'keep': 1}
    
    result = rnmixGibbs(Data, Prior, Mcmc)
    
    assert 'probdraw' in result
    assert 'zdraw' in result
    assert 'compdraw' in result
    assert result['probdraw'].shape == (500, ncomp)
    assert result['zdraw'].shape == (500, nobs)
    assert len(result['compdraw']) == 500
    
    logger.info(f"probdraw shape: {result['probdraw'].shape}")
    logger.info(f"zdraw shape: {result['zdraw'].shape}")
    logger.info(f"compdraw length: {len(result['compdraw'])}")
    logger.info(f"Mean prob: {result['probdraw'][250:, :].mean(axis=0)}")
    logger.info("PASSED: rnmixGibbs_basic")
    return True

def test_rnmixGibbs_thinning():
    """Test thinning."""
    np.random.seed(456)
    nobs = 100
    ncomp = 2
    
    y = np.random.randn(nobs, 2)
    
    Data = {'y': y}
    Prior = {'ncomp': ncomp}
    Mcmc = {'R': 1000, 'keep': 10}
    
    result = rnmixGibbs(Data, Prior, Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"probdraw shape: {result['probdraw'].shape}")
    
    assert result['probdraw'].shape[0] == 100
    logger.info("PASSED: rnmixGibbs_thinning")
    return True

def main():
    tests = [
        ("rnmixGibbs_basic", test_rnmixGibbs_basic),
        ("rnmixGibbs_thinning", test_rnmixGibbs_thinning),
    ]
    
    logger.info("=" * 60)
    logger.info(f"rnmixGibbs Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    results = {}
    for name, test_func in tests:
        logger.info(f"Testing {name}")
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"FAILED: {name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[name] = False
    
    logger.info("=" * 60)
    passed = sum(results.values())
    logger.info(f"rnmixGibbs Tests: {passed}/{len(results)} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
