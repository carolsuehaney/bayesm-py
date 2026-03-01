import numpy as np
from bayesm import rDPGibbs


def test_rDPGibbs_basic():
    """Test basic DP Gibbs sampler functionality."""
    np.random.seed(42)
    n = 100
    dimy = 2
    
    # Generate mixture data
    y1 = np.random.randn(n // 2, dimy) + np.array([2, 2])
    y2 = np.random.randn(n // 2, dimy) + np.array([-2, -2])
    y = np.vstack([y1, y2])
    
    Data = {'y': y}
    Prior = {}
    Mcmc = {'R': 100, 'keep': 2}
    
    result = rDPGibbs(Data, Prior, Mcmc)
    
    assert 'alphadraw' in result
    assert 'Istardraw' in result
    assert 'adraw' in result
    assert 'nudraw' in result
    assert 'vdraw' in result
    assert 'nmix' in result
    assert len(result['alphadraw']) == 50
    assert len(result['Istardraw']) == 50


def test_rDPGibbs_univariate():
    """Test DP Gibbs with univariate data."""
    np.random.seed(123)
    n = 80
    
    y = np.concatenate([np.random.randn(40) - 3, np.random.randn(40) + 3]).reshape(-1, 1)
    
    Data = {'y': y}
    Prior = {'Prioralpha': {'Istarmin': 1, 'Istarmax': 10}}
    Mcmc = {'R': 50, 'keep': 1, 'SCALE': True}
    
    result = rDPGibbs(Data, Prior, Mcmc)
    
    assert len(result['alphadraw']) == 50
    assert result['nmix']['zdraw'].shape == (50, n)
