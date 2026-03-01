import numpy as np
from bayesm import rivDP


def test_rivDP_basic():
    """Test basic IV with DP functionality."""
    np.random.seed(42)
    n = 100
    
    # Generate IV data
    delta_true = np.array([0.5, 1.0])
    beta_true = 2.0
    
    z = np.random.randn(n, 2)
    e1 = np.random.randn(n) * 0.5
    e2 = np.random.randn(n) * 0.5
    x = z @ delta_true + e1
    y = beta_true * x + e2
    
    Data = {'y': y, 'x': x, 'z': z}
    Prior = {}
    Mcmc = {'R': 100, 'keep': 2}
    
    result = rivDP(Data, Prior, Mcmc)
    
    assert 'deltadraw' in result
    assert 'betadraw' in result
    assert 'alphadraw' in result
    assert 'Istardraw' in result
    assert result['deltadraw'].shape == (50, 2)
    assert len(result['betadraw']) == 50


def test_rivDP_with_exogenous():
    """Test IV with DP and exogenous variables."""
    np.random.seed(123)
    n = 80
    
    delta_true = np.array([1.0])
    beta_true = 1.5
    gamma_true = np.array([0.5, -0.3])
    
    z = np.random.randn(n, 1)
    w = np.random.randn(n, 2)
    e1 = np.random.randn(n) * 0.3
    e2 = np.random.randn(n) * 0.3
    x = z @ delta_true + e1
    y = beta_true * x + w @ gamma_true + e2
    
    Data = {'y': y, 'x': x, 'z': z, 'w': w}
    Prior = {}
    Mcmc = {'R': 50, 'keep': 1}
    
    result = rivDP(Data, Prior, Mcmc)
    
    assert 'deltadraw' in result
    assert 'betadraw' in result
    assert 'gammadraw' in result
    assert result['gammadraw'].shape == (50, 2)
