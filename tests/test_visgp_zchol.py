"""Regression: ModelState.zchol() for visgp must give a correct, robust square root
of the marginal covariance Sigma = Q^{-1} WITHOUT forming Q^{-1}.

The exp2 Wang MPI run hung at startup because the old `cholesky(inv(Q))` threw "not
positive definite" when Q (the visgp precision) was extremely ill-conditioned at
large rho -- Q stays PD but the explicit inverse goes numerically non-PD (and the
threshold is BLAS-dependent, so it surfaced on the cluster). zchol now factors Q
directly, Q = R Rᵀ, and uses M = R^{-T} (M Mᵀ = Sigma): cheaper, and stable for any
PD Q since it never inverts a near-singular matrix.
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcmc_updates as mu  # noqa: E402


def _zchol_visgp(Q):
    """The exact sequence sampler.ModelState.zchol() uses for visgp: M = R^{-T}
    with R = chol(Q). M is UPPER triangular and M Mᵀ = Q^{-1}."""
    Qd = np.asarray(Q)
    _, R = mu._ensure_pd(0.5 * (Qd + Qd.T))
    return solve_triangular(R, np.eye(Qd.shape[0]), lower=True).T


def test_zchol_visgp_square_root_is_correct():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 40))
    Q = A @ A.T + 40 * np.eye(40)                 # well-conditioned PD precision
    M = _zchol_visgp(Q)
    assert np.allclose(M, np.triu(M))             # upper triangular (M = R^{-T})
    assert np.max(np.abs(M @ M.T - np.linalg.inv(Q))) < 1e-9   # M Mᵀ = Q^{-1}
    # whiten(unwhiten(eta)) == eta, where unwhiten = M@eta, whiten = solve(M, ., upper)
    eta = rng.standard_normal(40)
    back = solve_triangular(M, M @ eta, lower=False)
    assert np.max(np.abs(back - eta)) < 1e-9


def test_zchol_visgp_survives_extreme_ill_conditioning():
    """Eigenvalues spanning ~16 orders of magnitude -- the regime where the old
    chol(inv(Q)) RAISED 'not positive definite'. Factoring Q directly must instead
    return a finite, valid (upper-triangular, positive-diagonal) factor and not
    raise. (Accuracy is not meaningful at cond ~1e16 -- inv(Q) itself is garbage --
    and the nu cap / rho prior keep the sampler far from this regime anyway.)"""
    rng = np.random.default_rng(1)
    n = 60
    Qr, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q = (Qr * np.logspace(-8, 8, n)) @ Qr.T        # cond ~1e16
    M = _zchol_visgp(Q)
    assert np.all(np.isfinite(M))
    assert np.allclose(M, np.triu(M))              # upper triangular, no crash
    assert np.all(np.diag(M) > 0)


def test_zchol_visgp_accurate_at_moderate_conditioning():
    """At a realistic conditioning (~1e10) the factor must still reproduce Q^{-1}."""
    rng = np.random.default_rng(2)
    n = 60
    Qr, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q = (Qr * np.logspace(-5, 5, n)) @ Qr.T        # cond ~1e10
    M = _zchol_visgp(Q)
    rel = np.max(np.abs(M @ M.T - np.linalg.inv(Q))) / np.max(np.abs(np.linalg.inv(Q)))
    assert rel < 1e-6
