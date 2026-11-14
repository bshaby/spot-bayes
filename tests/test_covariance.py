"""Tests for VisGP covariance (Phase 4)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import covariance as cov  # noqa: E402


def _coords(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 10, size=(n, 2))


def test_visgp_fullgraph_equals_dense_inverse():
    """Single clique = all nodes, no separators -> Q == inv(Matern)."""
    coords = _coords(8)
    rho, nu = 3.0, 1.0
    D = cov.dist_matrix(coords)
    Sigma = cov.matern_wang(D, rho, 1.0, nu)
    Q = cov.visgp_precision([list(range(8))], [], coords, rho, nu)
    rel = np.linalg.norm(Q - np.linalg.inv(Sigma)) / np.linalg.norm(np.linalg.inv(Sigma))
    assert rel < 1e-6


def test_visgp_precision_positive_definite():
    # A small chain junction tree: cliques {0,1,2},{2,3,4} sep {2}.
    coords = _coords(5, seed=1)
    Q = cov.visgp_precision([[0, 1, 2], [2, 3, 4]], [[2]], coords, 4.0, 1.0)
    eig = np.linalg.eigvalsh(Q)
    assert np.all(eig > 0)


def test_visgp_logprob_matches_scipy_on_fullgraph():
    coords = _coords(6, seed=2)
    rho, nu = 3.0, 1.0
    Sigma = cov.matern_wang(cov.dist_matrix(coords), rho, 1.0, nu)
    Q = np.linalg.inv(Sigma)
    rng = np.random.default_rng(3)
    Z = rng.standard_normal(6)
    import scipy.stats
    expected = scipy.stats.multivariate_normal.logpdf(Z, mean=np.zeros(6), cov=Sigma)
    assert abs(cov.visgp_logprob(Z, Q) - expected) < 1e-8


def test_sparse_Q_and_cached_logdet_match_dense():
    """Sparse assembly + cached logdet give the same logprob as the dense path."""
    import scipy.sparse as sp
    coords = _coords(12, seed=8)
    cliques = [[0, 1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10, 11]]
    seps = [[5, 6]]
    Q_dense = cov.visgp_precision(cliques, seps, coords, 4.0, 1.0)
    Q_sp, ld = cov.visgp_precision(cliques, seps, coords, 4.0, 1.0,
                                   sparse=True, return_logdet=True)
    assert sp.issparse(Q_sp)
    assert np.allclose(np.asarray(Q_sp.todense()), Q_dense, atol=1e-10)
    # cached logdet equals slogdet of dense Q
    _, ld_dense = np.linalg.slogdet(Q_dense)
    assert abs(ld - ld_dense) < 1e-8
    # logprob: dense vs sparse+cached agree
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((12, 4))
    lp_dense = cov.visgp_logprob(Z, Q_dense)
    lp_sparse = cov.visgp_logprob(Z, Q_sp, logdet=ld)
    assert abs(lp_dense - lp_sparse) < 1e-8


def test_visgp_conditional_sparse_matches_dense():
    coords = _coords(10, seed=9)
    cliques = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]]
    seps = [[4, 5]]
    Q_dense = cov.visgp_precision(cliques, seps, coords, 4.0, 1.0)
    Q_sp = cov.visgp_precision(cliques, seps, coords, 4.0, 1.0, sparse=True)
    rng = np.random.default_rng(1)
    Z_obs = rng.standard_normal(6)
    obs, pred = [0, 1, 2, 3, 4, 5], [6, 7, 8, 9]
    m_d, c_d = cov.visgp_conditional(Q_dense, obs, pred, Z_obs)
    m_s, c_s = cov.visgp_conditional(Q_sp, obs, pred, Z_obs)
    assert np.allclose(m_d, m_s, atol=1e-10)
    assert np.allclose(c_d, c_s, atol=1e-10)


def test_visgp_conditional_matches_dense():
    """Conditional mean/cov from precision == dense Gaussian conditioning."""
    coords = _coords(10, seed=4)
    rho, nu = 3.0, 1.0
    Sigma = cov.matern_wang(cov.dist_matrix(coords), rho, 1.0, nu)
    Q = np.linalg.inv(Sigma)
    obs = [0, 1, 2, 3, 4, 5, 6]
    pred = [7, 8, 9]
    rng = np.random.default_rng(5)
    Z = rng.multivariate_normal(np.zeros(10), Sigma)
    Z_obs = Z[obs]

    mean_q, cov_q = cov.visgp_conditional(Q, obs, pred, Z_obs)

    # dense conditioning
    S_oo = Sigma[np.ix_(obs, obs)]
    S_po = Sigma[np.ix_(pred, obs)]
    S_pp = Sigma[np.ix_(pred, pred)]
    mean_d = S_po @ np.linalg.solve(S_oo, Z_obs)
    cov_d = S_pp - S_po @ np.linalg.solve(S_oo, S_po.T)

    assert np.allclose(mean_q, mean_d, atol=1e-8)
    assert np.allclose(cov_q, cov_d, atol=1e-8)
