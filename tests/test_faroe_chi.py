"""chi on the FITTED-CDF scale (faroe_diagnostics): margins, the qRW-threshold
shortcut, and the per-draw chi_assess driver.

The load-bearing claim is that thresholding the raw field at qRW_i(u) is EXACTLY
chi on pRW-uniforms -- that is what buys the ~700x over evaluating pRW on every
replicate. It is tested here against the real C pRW/qRW, not a stub.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import genpareto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "faroe_analysis", "scripts"))

import simulate as sm                                            # noqa: E402
import utilities as u                                           # noqa: E402
from config import ModelConfig                                  # noqa: E402
from sampler import (SamplerData, SamplerSettings, ModelState, run_mcmc,  # noqa: E402
                     save_run, load_run)
import faroe_diagnostics as D                                   # noqa: E402


def _sim(seed=11, Ns=30, Nt=40):
    out = sm.simulate(ModelConfig("shi"), sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"])
    return out, data


# --------------------------------------------------------------------------
# marginal transform
# --------------------------------------------------------------------------
def test_pCGP_censors_below_and_matches_gpd_above():
    p = 0.95
    u_vec = np.array([2.0, 5.0])
    sigma = np.array([1.5, 3.0])
    xi = np.array([0.1, -0.05])
    Y = np.array([[1.0, 2.0, 4.0, np.nan],      # site 0: below, AT thresh, above, NaN
                  [4.9, 5.0, 9.0, 100.0]])      # site 1
    U = D.pCGP(Y, p, u_vec, sigma, xi)

    # at or below the site threshold the margin is censored -> exactly p
    assert U[0, 0] == p and U[0, 1] == p
    assert U[1, 0] == p and U[1, 1] == p
    # NaN propagates
    assert np.isnan(U[0, 3])
    # above: p + (1-p) * conditional GPD CDF
    assert U[0, 2] == pytest.approx(
        p + (1 - p) * genpareto.cdf(4.0, xi[0], loc=u_vec[0], scale=sigma[0]))
    assert U[1, 2] == pytest.approx(
        p + (1 - p) * genpareto.cdf(9.0, xi[1], loc=u_vec[1], scale=sigma[1]))
    # a proper CDF: in [p, 1], nondecreasing along the exceedances of a site
    fin = np.isfinite(U)
    assert np.all(U[fin] >= p) and np.all(U[fin] <= 1.0)
    assert U[1, 3] >= U[1, 2]


def test_pCGP_is_calibrated_at_the_true_margin():
    """At the TRUE margin, pCGP(Y) is uniform above p: P(U>u)/(1-u) -> 1.

    This is the empirical side's calibration check, and the reason chi is no longer
    capped at 1 -- that ratio is now ESTIMATED, not 1 by construction as with ranks.
    """
    out, data = _sim(Ns=25, Nt=4000)
    U = D.pCGP(data.Y, out["p_cens"], data.u_vec, out["sigma_mat"][:, 0],
               out["xi_mat"][:, 0])
    for uu in (0.96, 0.98):
        ratio = np.nanmean(U > uu, axis=1) / (1.0 - uu)
        assert np.allclose(ratio.mean(), 1.0, atol=0.15)


def test_draw_sigma_xi_matches_sitewise():
    rng = np.random.default_rng(3)
    Ns, ps, px = 12, 4, 3
    sig_sp = rng.normal(size=(Ns, ps))
    xi_sp = rng.normal(size=(Ns, px))
    draw = {"beta_sigma": rng.normal(size=ps + 1) * 0.2,
            "beta_xi": rng.normal(size=px + 1) * 0.2}
    sigma, xi = D.draw_sigma_xi(draw, sig_sp, xi_sp)
    for i in range(Ns):
        s_i, x_i = D._site_sigma_xi(draw, sig_sp[i], xi_sp[i])
        assert sigma[i] == pytest.approx(s_i)
        assert xi[i] == pytest.approx(x_i)


# --------------------------------------------------------------------------
# the qRW-threshold shortcut == chi on pRW uniforms
# --------------------------------------------------------------------------
def test_threshold_chi_equals_pRW_uniform_chi():
    """X > qRW_i(u)  <=>  pRW_i(X) > u  (pRW monotone), so the two chi agree exactly."""
    out, data = _sim()
    state = ModelState(ModelConfig("shi"), data, SamplerSettings(nu=out["nu"]))
    rng = np.random.default_rng(0)
    X = D.simulate_field(state, 300, rng)
    us = np.array([0.96, 0.98, 0.99])

    thr = D.model_thresholds(state, us)
    assert thr.shape == (us.size, state.Ns)
    assert np.all(np.diff(thr, axis=0) > 0)          # thresholds increase with u

    # reference path: full pRW transform of every replicate
    Ns, R = X.shape
    U = u.pRW(X.ravel(), np.repeat(state.phi_vec, R), np.repeat(state.gamma_bar, R),
              state.alpha0).reshape(Ns, R)
    for ri in (0, Ns // 2):
        fast = D.chi_vs_reference_thresh(X, thr, ri, us)
        slow = D.chi_vs_reference(U, ri, us)
        assert np.allclose(fast, slow, atol=1e-12, equal_nan=True)


def test_chi_self_is_one():
    """chi of the reference against ITSELF is 1 at every level (sanity on the ratio)."""
    out, data = _sim()
    state = ModelState(ModelConfig("shi"), data, SamplerSettings(nu=out["nu"]))
    X = D.simulate_field(state, 4000, np.random.default_rng(1))
    us = np.array([0.96, 0.98])
    thr = D.model_thresholds(state, us)
    chi = D.chi_vs_reference_thresh(X, thr, 0, us)
    # P(U_ref>u & U_ref>u)/(1-u) = P(U_ref>u)/(1-u) -> 1 as the fitted margin is exact
    assert np.allclose(chi[0], 1.0, atol=0.15)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def test_chi_levels_below_p_cens_rejected():
    out, data = _sim()
    with pytest.raises(ValueError, match="must all exceed p_cens"):
        D.chi_assess({"phi_k": np.zeros((4, 1)), "beta_sigma": np.zeros((4, 2)),
                      "beta_xi": np.zeros((4, 2))},
                     ModelConfig("shi"), data, SamplerSettings(nu=out["nu"]),
                     data.Y, data.u_vec, np.zeros((data.Y.shape[0], 1)),
                     np.zeros((data.Y.shape[0], 1)), 0.95,
                     [0], [0.90, 0.98])


def test_chi_assess_end_to_end(tmp_path):
    """Real (tiny) fit -> chi_assess: shapes, bands bracket the mean, chi in [0, 1].

    Goes through save_run/load_run, as Task 7 does: build_draws needs the scalar
    geometry metadata that only the saved run carries.
    """
    out, data = _sim()
    cfg = ModelConfig("shi")
    st = SamplerSettings(n_iters=120, adapt_size=25, thin=5, nu=out["nu"], seed=1,
                         save_latent=True, update_GPD_sigma=True, update_GPD_xi=True)
    TR = run_mcmc(cfg, data, st)
    rp = str(tmp_path / "fit.npz")
    save_run(rp, TR, cfg, st, data)
    run = load_run(rp)
    if "beta_sigma" not in run:
        pytest.skip("fit did not record GPD betas; nothing to test on the empirical side")

    Ns = data.Y.shape[0]
    rng = np.random.default_rng(5)
    sig_sp = rng.normal(size=(Ns, run["beta_sigma"].shape[1] - 1)) * 0.1
    xi_sp = rng.normal(size=(Ns, run["beta_xi"].shape[1] - 1)) * 0.1
    us = [0.96, 0.98]
    ref_idx = [0, Ns // 2]

    res = D.chi_assess(run, cfg, data, st, data.Y, data.u_vec,
                       sig_sp, xi_sp, data.p_cens, ref_idx, us,
                       n_draws=6, n_rep=200, burn=0.5)

    shape = (len(ref_idx), Ns, len(us))
    # NB chi is NOT capped at 1 on the fitted-CDF scale. With ranks, P(U_ref>u)=1-u
    # held by construction and chi<=1 followed; with a fitted margin that probability
    # is estimated (MC noise on the model side, marginal misfit on the data side), so
    # chi can exceed 1. The only hard bound is P(both>u)<=1 => chi <= 1/(1-u).
    hard_max = 1.0 / (1.0 - min(us))
    for k in ("model_mean", "model_lo", "model_hi", "emp_mean", "emp_lo", "emp_hi"):
        assert res[k].shape == shape, k
        assert np.all(np.isfinite(res[k])), k
        assert np.all((res[k] >= 0) & (res[k] <= hard_max + 1e-9)), k
    assert np.all(res["model_lo"] <= res["model_mean"] + 1e-9)
    assert np.all(res["model_mean"] <= res["model_hi"] + 1e-9)
    assert np.all(res["emp_lo"] <= res["emp_mean"] + 1e-9)
    assert np.all(res["emp_mean"] <= res["emp_hi"] + 1e-9)
    # The MODEL side is self-consistent by construction (the field is simulated at the
    # same draw the qRW thresholds come from), so a reference site is ~perfectly
    # dependent with itself. MC-noisy at n_rep=200: ~8 exceedances set the u=0.96
    # denominator.
    for r, ri in enumerate(ref_idx):
        assert np.allclose(res["model_mean"][r, ri], 1.0, atol=0.35)
    # The EMPIRICAL side is checked for WIRING only (it produced signal), not for
    # calibration or magnitude -- two fixture limits make magnitudes meaningless here:
    # the covariates are FABRICATED (the simulator has no GPD regression), so the
    # reconstructed margin exp(b0 + junk.beta) is deliberately miscalibrated -- badly
    # enough that some sites' uniforms never clear 0.96 at all; and at Nt=40 a site
    # exceeds u=0.96 only ~1.6 times, so most empirical chi is an honest 0. pCGP's
    # calibration at a CORRECT margin is test_pCGP_is_calibrated_at_the_true_margin.
    assert np.any(res["emp_mean"] > 0)
