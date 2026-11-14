"""Tests for the standalone stage-1 GPD-regression model (marginal_model.py).

Covers: (1) analytic gradient vs central finite differences (incl. the xi->0 branch),
(2) the value equals a from-scratch _gp_logpdf recompute over exceedances + priors,
(3) NUTS recovers a known coefficient vector from simulated exceedances.
"""
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import marginal as mg          # noqa: E402
import utilities as u          # noqa: E402
import marginal_model as mm    # noqa: E402


# --------------------------------------------------------------------------- #
#  helpers                                                                      #
# --------------------------------------------------------------------------- #
def _simulate(Ns=40, Nt=30, p_sp_sigma=2, p_sp_xi=1, seed=0, exc_frac=0.4,
              beta_sigma=None, beta_xi=None):
    """Simulate a (Ns, Nt) field of GPD exceedances with spatial covariates.

    Returns (MarginalData, beta_sigma_true, beta_xi_true). Non-exceedance cells are
    placed just below the (unit) threshold; only exceedances enter the model."""
    rng = np.random.default_rng(seed)
    u_vec = np.full(Ns, 10.0)
    Xs_sigma = rng.standard_normal((Ns, p_sp_sigma))
    Xs_xi = rng.standard_normal((Ns, p_sp_xi))
    if beta_sigma is None:
        beta_sigma = np.concatenate([[np.log(3.0)], 0.4 * rng.standard_normal(p_sp_sigma)])
    if beta_xi is None:
        beta_xi = np.concatenate([[mg.eta_from_xi(0.15)], 0.3 * rng.standard_normal(p_sp_xi)])
    sig_site = mg.sigma_from_eta(beta_sigma[0] + Xs_sigma @ beta_sigma[1:])   # (Ns,)
    xi_site = mg.xi_from_eta(beta_xi[0] + Xs_xi @ beta_xi[1:])                # (Ns,)

    Y = np.full((Ns, Nt), u_vec[0] - 1.0)                # below threshold by default
    is_exc = rng.random((Ns, Nt)) < exc_frac
    for s in range(Ns):
        n = int(is_exc[s].sum())
        if n:
            draws = scipy.stats.genpareto.rvs(c=xi_site[s], scale=sig_site[s],
                                              size=n, random_state=rng)
            Y[s, is_exc[s]] = u_vec[s] + draws

    sd = mg.MarginalDesign(Ns, Nt, spatial=Xs_sigma)
    xd = mg.MarginalDesign(Ns, Nt, spatial=Xs_xi)
    data = mm.MarginalData(Y=Y, u_vec=u_vec, sigma_design=sd, xi_design=xd)
    return data, beta_sigma, beta_xi


# --------------------------------------------------------------------------- #
#  tests                                                                        #
# --------------------------------------------------------------------------- #
def test_gradient_matches_finite_difference():
    data, b_sig, b_xi = _simulate(seed=1)
    rng = np.random.default_rng(11)
    theta0 = np.concatenate([b_sig, b_xi])
    for _ in range(4):
        theta = theta0 + 0.3 * rng.standard_normal(theta0.size)
        logp, grad = data.value_grad(theta)
        assert np.isfinite(logp)
        eps = 1e-6
        fd = np.empty_like(grad)
        for j in range(theta.size):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            fd[j] = (data.value_grad(tp)[0] - data.value_grad(tm)[0]) / (2 * eps)
        assert np.allclose(grad, fd, rtol=1e-4, atol=1e-5), \
            f"grad mismatch: max|Δ|={np.max(np.abs(grad - fd)):.2e}"


def test_gradient_finite_difference_near_xi_zero():
    """The xi->0 removable-singularity branch is continuous and correct."""
    data, b_sig, b_xi = _simulate(seed=2, p_sp_xi=1)
    # force the xi intercept to make xi(s,t) ~ 0 (eta such that scaled-logit -> 0)
    b_xi = b_xi.copy(); b_xi[0] = mg.eta_from_xi(1e-9); b_xi[1] = 1e-3
    theta = np.concatenate([b_sig, b_xi])
    logp, grad = data.value_grad(theta)
    assert np.isfinite(logp)
    eps = 1e-6
    fd = np.empty_like(grad)
    for j in range(theta.size):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        fd[j] = (data.value_grad(tp)[0] - data.value_grad(tm)[0]) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-5)


def test_value_matches_gp_logpdf_recompute():
    data, b_sig, b_xi = _simulate(seed=3)
    theta = np.concatenate([b_sig, b_xi])
    logp, _ = data.value_grad(theta)

    # from-scratch: full-grid eta -> sigma/xi, GP logpdf on the exceedance mask
    sig = mg.sigma_from_eta(data.sigma_design.eta(b_sig))
    xi = mg.xi_from_eta(data.xi_design.eta(b_xi))
    exc = np.isfinite(data.Y) & (data.Y > data.u_vec[:, None])
    yy = data.Y[exc]
    ld = u._gp_logpdf_np(yy, data.u_vec[:, None].repeat(data.Y.shape[1], 1)[exc],
                         sig[exc], xi[exc])
    ll = ld.sum()
    ll += (-0.5 * (b_sig / data.sigma_beta_sd) ** 2 - np.log(data.sigma_beta_sd)).sum()
    ll += (-0.5 * (b_xi / data.xi_beta_sd) ** 2 - np.log(data.xi_beta_sd)).sum()
    ll += -0.5 * theta.size * np.log(2 * np.pi)
    assert abs(logp - ll) < 1e-9


def test_out_of_support_returns_neg_inf():
    data, b_sig, b_xi = _simulate(seed=4)
    # a very negative xi with a large exceedance forces 1 + xi*z <= 0 somewhere
    bad = np.concatenate([b_sig, np.array([mg.eta_from_xi(-0.49)] + [0.0] * (b_xi.size - 1))])
    # shrink sigma hugely so z is enormous -> support violated
    bad[0] = np.log(1e-3)
    logp, grad = data.value_grad(bad)
    assert not np.isfinite(logp)
    assert np.allclose(grad, 0.0)


def test_nuts_recovers_known_beta():
    # bigger data for a tighter posterior; modest covariate effects
    data, b_sig, b_xi = _simulate(Ns=80, Nt=120, p_sp_sigma=2, p_sp_xi=1,
                                  seed=5, exc_frac=0.5)
    res = mm.fit(data, b_sig, b_xi, n_iters=800, warmup=400, thin=1, seed=7)
    assert res["accept"][-200:].mean() > 0.5
    bs, bx = res["beta_sigma"][400:], res["beta_xi"][400:]     # post-warmup
    for k in range(b_sig.size):
        lo, hi = np.percentile(bs[:, k], [1, 99])
        assert lo <= b_sig[k] <= hi, f"beta_sigma[{k}]={b_sig[k]} outside [{lo},{hi}]"
    for k in range(b_xi.size):
        lo, hi = np.percentile(bx[:, k], [1, 99])
        assert lo <= b_xi[k] <= hi, f"beta_xi[{k}]={b_xi[k]} outside [{lo},{hi}]"


def test_surfaces_shape():
    data, b_sig, b_xi = _simulate(seed=6)
    assert data.sigma_surface(b_sig).shape == data.Y.shape
    assert data.xi_surface(b_xi).shape == data.Y.shape
    assert np.all(data.sigma_surface(b_sig) > 0)


def test_checkpoint_resume_reproduces_single_shot(tmp_path):
    """A run split across a checkpoint (100 then resume to 200) reproduces the single-shot
    200-iter run BIT-FOR-BIT (same RNG stream + restored NUTS adapter)."""
    data, b_sig, b_xi = _simulate(Ns=30, Nt=20, seed=9)
    ck = str(tmp_path / "m.ckpt.pkl")
    full = mm.fit(data, b_sig, b_xi, n_iters=200, warmup=50, thin=2, seed=3)
    snaps = []
    mm.fit(data, b_sig, b_xi, n_iters=100, warmup=50, thin=2, seed=3,
           checkpoint_path=ck, checkpoint_every=40, on_checkpoint=lambda r: snaps.append(r))
    assert len(snaps) >= 1                      # periodic snapshot fired
    r2 = mm.fit(data, b_sig, b_xi, n_iters=200, warmup=50, thin=2, seed=3,
                checkpoint_path=ck, checkpoint_every=40)
    assert r2["beta_sigma"].shape == full["beta_sigma"].shape
    assert np.array_equal(r2["beta_sigma"], full["beta_sigma"])
    assert np.array_equal(r2["beta_xi"], full["beta_xi"])
    assert np.array_equal(r2["accept"], full["accept"])
    # re-running an already-complete checkpoint is an idempotent no-op
    r3 = mm.fit(data, b_sig, b_xi, n_iters=200, warmup=50, thin=2, seed=3,
                checkpoint_path=ck, checkpoint_every=40)
    assert np.array_equal(r3["beta_sigma"], full["beta_sigma"])
