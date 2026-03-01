"""Test rnegbinRw - negative binomial RW Metropolis."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
import logging
from datetime import datetime
from bayesm import rnegbinRw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rnegbinRw_basic():
    """Basic functionality test."""
    np.random.seed(42)
    nobs = 200
    nvar = 2
    
    X = np.column_stack([np.ones(nobs), np.random.randn(nobs)])
    beta_true = np.array([1.0, 0.5])
    alpha_true = 2.0
    
    lambda_param = np.exp(X @ beta_true)
    prob = alpha_true / (alpha_true + lambda_param)
    y = np.random.negative_binomial(alpha_true, prob)
    
    Data = {'y': y.astype(float), 'X': X}
    Mcmc = {'R': 1000, 'keep': 1}
    
    result = rnegbinRw(Data, Mcmc=Mcmc)
    
    assert result['betadraw'].shape == (1000, 2)
    assert result['alphadraw'].shape == (1000,)
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    logger.info(f"alphadraw shape: {result['alphadraw'].shape}")
    logger.info(f"acceptance rate beta: {result['acceptrbeta']:.3f}")
    logger.info(f"acceptance rate alpha: {result['acceptralpha']:.3f}")
    logger.info("PASSED: rnegbinRw_basic")
    return True

def test_rnegbinRw_fixedalpha():
    """Test with fixed alpha."""
    np.random.seed(123)
    nobs = 100
    
    X = np.column_stack([np.ones(nobs), np.random.randn(nobs)])
    beta_true = np.array([1.0, 0.5])
    alpha_true = 2.0
    
    lambda_param = np.exp(X @ beta_true)
    prob = alpha_true / (alpha_true + lambda_param)
    y = np.random.negative_binomial(alpha_true, prob)
    
    Data = {'y': y.astype(float), 'X': X}
    Mcmc = {'R': 1000, 'keep': 1, 'alpha': 2.0}
    
    result = rnegbinRw(Data, Mcmc=Mcmc)
    
    assert np.all(result['alphadraw'] == 2.0)
    logger.info("All alpha draws equal to fixed value: 2.0")
    logger.info("PASSED: rnegbinRw_fixedalpha")
    return True

def test_rnegbinRw_thinning():
    """Test thinning."""
    np.random.seed(456)
    nobs = 100
    X = np.column_stack([np.ones(nobs), np.random.randn(nobs)])
    y = np.random.negative_binomial(2, 0.5, size=nobs).astype(float)
    
    Data = {'y': y, 'X': X}
    Mcmc = {'R': 1000, 'keep': 10}
    
    result = rnegbinRw(Data, Mcmc=Mcmc)
    
    logger.info(f"R=1000, keep=10 -> expected 100 draws")
    logger.info(f"betadraw shape: {result['betadraw'].shape}")
    assert result['betadraw'].shape[0] == 100
    logger.info("PASSED: rnegbinRw_thinning")
    return True

def main():
    logger.info("=" * 60)
    logger.info(f"rnegbinRw Test Suite - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    tests = [
        ("rnegbinRw_basic", test_rnegbinRw_basic),
        ("rnegbinRw_fixedalpha", test_rnegbinRw_fixedalpha),
        ("rnegbinRw_thinning", test_rnegbinRw_thinning),
    ]
    
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
    
    passed = sum(results.values())
    logger.info("=" * 60)
    logger.info(f"rnegbinRw Tests: {passed}/{len(tests)} passed")
    for name, result in results.items():
        logger.info(f"  {name}: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()
