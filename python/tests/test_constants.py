"""
Tests for bayesm.constants
"""

import pytest
from bayesm.constants import BayesmConstants


def test_mcmc_constants():
    """Test MCMC constants match R values"""
    assert BayesmConstants.keep == 1
    assert BayesmConstants.nprint == 100
    assert BayesmConstants.RRScaling == 2.38
    assert BayesmConstants.w == 0.1


def test_prior_constants():
    """Test prior constants match R values"""
    assert BayesmConstants.A == 0.01
    assert BayesmConstants.nuInc == 3
    assert BayesmConstants.a == 5
    assert BayesmConstants.nu_e == 3.0
    assert BayesmConstants.nu == 3.0
    assert BayesmConstants.agammaprior == 0.5
    assert BayesmConstants.bgammaprior == 0.1


def test_dp_constants():
    """Test Dirichlet Process constants match R values"""
    assert BayesmConstants.DPalimdef == [0.01, 10]
    assert BayesmConstants.DPnulimdef == [0.01, 3]
    assert BayesmConstants.DPvlimdef == [0.1, 4]
    assert BayesmConstants.DPIstarmin == 1
    assert BayesmConstants.DPpower == 0.8
    assert BayesmConstants.DPalpha == 1.0
    assert BayesmConstants.DPmaxuniq == 200
    assert BayesmConstants.DPSCALE is True
    assert BayesmConstants.DPgridsize == 20


def test_math_constants():
    """Test mathematical constants"""
    assert abs(BayesmConstants.gamma - 0.5772156649015328606) < 1e-15


def test_blp_constants():
    """Test BayesBLP constants"""
    assert BayesmConstants.BLPVOmega == [[1, 0.5], [0.5, 1]]
    assert BayesmConstants.BLPtol == 1e-6
