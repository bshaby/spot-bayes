"""Tests for missing-data imputation (Phase 6)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import missing_data as md  # noqa: E402
import covariance as cov  # noqa: E402
from config import ModelConfig  # noqa: E402


def _setup(n=12, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, (n, 2))
    K = cov.matern_wang(cov.dist_matrix(coords), rho=3.0, sigmasq=1.0, nu=1.0)
    return coords, K, rng


def test_dense_conditional_matches_visgp_on_fullgraph():
    coords, K, rng = _setup(12, 1)
    Q = np.linalg.inv(K)
    obs = list(range(8)); miss = [8, 9, 10, 11]
    Z = rng.multivariate_normal(np.zeros(12), K)
    m_d, c_d = md.conditional_gaussian_dense(K, obs, miss, Z[obs])
    m_q, c_q = md.conditional_gaussian_visgp(Q, obs, miss, Z[obs])
    assert np.allclose(m_d, m_q, atol=1e-7)
    assert np.allclose(c_d, c_q, atol=1e-7)


def test_imputed_Z_marginals_match_conditional_variance():
    """Empirical variance of many conditional draws matches cov diagonal."""
    coords, K, rng = _setup(10, 2)
    obs = list(range(7)); miss = [7, 8, 9]
    Z = rng.multivariate_normal(np.zeros(10), K)
    mean, cov_mm = md.conditional_gaussian_dense(K, obs, miss, Z[obs])
    draws = md.draw_Z_missing(mean, cov_mm, rng, n_draws=20000)
    emp_var = draws.var(axis=1)
    assert np.allclose(emp_var, np.diag(cov_mm), rtol=0.1)
    assert np.allclose(draws.mean(axis=1), mean, atol=0.05)


def test_impute_Y_runs_and_respects_threshold():
    coords, K, rng = _setup(10, 3)
    obs = list(range(7)); miss = [7, 8, 9]
    Z = rng.multivariate_normal(np.zeros(10), K)
    nm = len(miss)
    Z_miss, Y_miss = md.impute_Y_missing(
        ModelConfig(model_type="shi"), K, obs, miss, Z[obs],
        R_miss=np.full(nm, 2.0), phi_miss=np.full(nm, 0.5),
        gamma_bar_miss=np.full(nm, 1.0), alpha0=5.0,
        p_cens=0.95, u_miss=np.full(nm, 60.0),
        sigma_miss=np.full(nm, 20.0), xi_miss=np.full(nm, 0.15),
        rng=rng)
    assert Y_miss.shape == (nm,)
    assert np.all(Y_miss >= 60.0)  # censored at or above threshold
