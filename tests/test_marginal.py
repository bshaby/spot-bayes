"""Tests for GPD-marginal covariate regression (sigma, xi)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import marginal as mg  # noqa: E402
import simulate as sm  # noqa: E402
import utilities as u  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState, run_mcmc  # noqa: E402


def test_link_roundtrips_and_ranges():
    s = np.array([0.1, 1.0, 25.0])
    assert np.allclose(mg.sigma_from_eta(mg.eta_from_sigma(s)), s)
    assert np.all(mg.sigma_from_eta(np.linspace(-5, 5, 50)) > 0)
    xi = np.array([-0.49, -0.2, 0.0, 0.3, 0.95])
    assert np.allclose(mg.xi_from_eta(mg.eta_from_xi(xi)), xi)
    mapped = mg.xi_from_eta(np.linspace(-8, 8, 100))
    assert np.all(mapped > -0.5) and np.all(mapped < 1.0)


def test_design_eta_layout():
    rng = np.random.default_rng(0)
    Ns, Nt = 5, 4
    sp = rng.standard_normal((Ns, 2))
    tp = rng.standard_normal((Nt, 1))
    st = rng.standard_normal((Ns, Nt, 1))
    d = mg.MarginalDesign(Ns, Nt, spatial=sp, temporal=tp, spatiotemporal=st)
    assert d.n_params == 1 + 2 + 1 + 1
    beta = rng.standard_normal(d.n_params)
    eta = d.eta(beta)
    # check a couple of entries against the explicit formula
    for s in (0, 3):
        for t in (1, 2):
            expect = (beta[0] + sp[s] @ beta[1:3] + tp[t] @ beta[3:4]
                      + st[s, t] @ beta[4:5])
            assert abs(eta[s, t] - expect) < 1e-12


def _setup(seed=7):
    sc = sm.SimConfig(Ns=50, Nt=15, seed=seed, Beta_logsigma=(3.0, 0.4))
    out = sm.simulate(ModelConfig("shi"), sc)
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"], sigma_spatial=out["elevations"][:, None])
    return out, data


def test_beta_update_path_is_exact():
    """The optimized GPD-beta step (exceedance patch + sigma_col override) equals
    a full from-scratch recompute."""
    out, data = _setup()
    st = SamplerSettings(nu=out["nu"], seed=1, update_GPD_sigma=True)
    state = ModelState(ModelConfig("shi"), data, st)
    qv, dv = state.qd_RW_all()
    beta_prop = state.beta_sigma + np.array([0.2, 0.3])
    sig_prop = mg.sigma_from_eta(state.sigma_design.eta(beta_prop))
    pY_prop = u.pCGP(state.Yfilled, state.data.p_cens,
                     state.data.u_vec[:, None], sig_prop, state.xi_mat)
    qvp, dvp = state.qd_exc_patch(qv, dv, pY_prop)
    ll_opt = sum(state.loglik_t(t, qv=qvp[:, t], dv=dvp[:, t], sigma_col=sig_prop[:, t])
                 for t in range(state.Nt))
    # ground truth
    state.sigma_mat = sig_prop; state.refresh_pY()
    qvF, dvF = state.qd_RW_all()
    ll_full = sum(state.loglik_t(t, qv=qvF[:, t], dv=dvF[:, t])
                  for t in range(state.Nt))
    assert np.max(np.abs(qvp - qvF)) == 0.0
    assert abs(ll_opt - ll_full) < 1e-9


def test_fixed_marginal_unaffected_when_update_off():
    out, data = _setup()
    # no update flag -> sigma_mat is the broadcast of sigma_vec, constant in time
    state = ModelState(ModelConfig("shi"), data, SamplerSettings(nu=out["nu"]))
    assert not state.update_sigma and not state.update_xi
    assert np.allclose(state.sigma_mat, data.sigma_vec[:, None])
    assert np.allclose(state.xi_mat, data.xi_vec[:, None])


def test_sigma_regression_chain_runs_and_records_beta():
    out, data = _setup()
    st = SamplerSettings(n_iters=200, adapt_size=25, thin=5, nu=out["nu"], seed=1,
                         update_GPD_sigma=True)
    TR = run_mcmc(ModelConfig("shi"), data, st)
    assert TR["beta_sigma"].shape == (len(range(0, 200, 5)), 2)
    assert np.all(np.isfinite(TR["ll"]))
    assert np.all(np.isfinite(TR["beta_sigma"]))


def test_holdout_scoring_uses_fitted_marginal():
    """Holdout sigma at new sites is derived from the fitted beta + holdout
    covariates per draw, not from the (here deliberately wrong) fixed fallback."""
    import tempfile, os
    from sampler import save_run, load_run, build_draws
    import predictive as pr
    out, data = _setup()
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "run.npz")
    st = SamplerSettings(n_iters=60, adapt_size=20, thin=10, nu=out["nu"], seed=1,
                         update_GPD_sigma=True, save_latent=True)
    save_run(path, run_mcmc(ModelConfig("shi"), data, st), ModelConfig("shi"), st, data)
    run = load_run(path); draws = build_draws(run, burn=1)
    assert "beta_sigma" in draws[0]
    rng = np.random.default_rng(0)
    new = rng.uniform(0, 10, (3, 2)); new_elev = rng.standard_normal((3, 1))
    geom = pr.NewSiteGeometry(
        new_coords=new, train_coords=run["sites_xy"], knots_S=run["knots_S"],
        knots_phi=run["knots_phi"], knots_rho=run["knots_rho"],
        radius_S=float(run["radius_S"]), eff_range_phi=float(run["eff_range_phi"]),
        eff_range_rho=float(run["eff_range_rho"]), config=run["config"],
        water_poly=None, sigma_spatial=new_elev)
    hold = pr.HoldoutData(Y=60 + np.abs(rng.standard_normal((3, 12))) * 5,
                          u=np.full(3, 60.0), sigma=np.full(3, 1e6), xi=np.full(3, 0.15))
    s0, _ = pr._holdout_marginal(geom, draws[0], 12, hold.sigma, hold.xi)
    s1, _ = pr._holdout_marginal(geom, draws[-1], 12, hold.sigma, hold.xi)
    assert np.all(s0 < 1e5)               # uses fitted beta, not the 1e6 fallback
    assert not np.allclose(s0, s1)        # varies across draws
    assert not np.allclose(s0[:, 0], s0[0, 0])  # varies across sites (covariate)
    ll = pr.predictive_loglik_new_sites(geom, draws, hold, M=5)
    assert np.isfinite(ll["total"])


def test_holdout_covariate_width_mismatch_raises():
    """A holdout covariate group whose width != the fitted beta block errors
    clearly instead of silently mis-aligning."""
    import pytest
    import predictive as pr
    out, data = _setup()
    geom = pr.NewSiteGeometry(
        new_coords=np.array([[1.0, 1.0], [5.0, 5.0]]), train_coords=data.sites_xy,
        knots_S=data.knots_S, knots_phi=data.knots_phi, knots_rho=data.knots_rho,
        radius_S=3.0, eff_range_phi=3.0, eff_range_rho=3.0, config=ModelConfig("shi"),
        water_poly=None, sigma_spatial=np.zeros((2, 2)))   # 2 spatial cols -> 3 params
    draw = {"beta_sigma": np.array([3.0, 0.4])}            # fitted beta has only 2
    with pytest.raises(ValueError, match="same layout"):
        pr._holdout_marginal(geom, draw, 4, np.array([20.0, 30.0]), np.array([0.1, 0.1]))


def test_holdout_fallback_when_margin_fixed():
    """When the run held the marginal fixed (no beta_* in the draw), scoring uses
    the fixed HoldoutData sigma/xi."""
    import predictive as pr
    out, data = _setup()
    geom = pr.NewSiteGeometry(
        new_coords=np.array([[1.0, 1.0], [5.0, 5.0]]), train_coords=data.sites_xy,
        knots_S=data.knots_S, knots_phi=data.knots_phi, knots_rho=data.knots_rho,
        radius_S=3.0, eff_range_phi=3.0, eff_range_rho=3.0,
        config=ModelConfig("shi"), water_poly=None)
    draw = {}                                   # marginal held fixed -> no beta_*
    sig, xi = pr._holdout_marginal(geom, draw, 4, np.array([20.0, 30.0]), np.array([0.1, 0.1]))
    assert np.allclose(sig, np.array([[20.0] * 4, [30.0] * 4]))  # fixed fallback used


def test_holdout_intercept_only_uses_fitted_beta():
    """An intercept-only marginal fit (beta has length 1, no covariates) is used
    at the holdout sites: constant exp(beta0)/xi_from_eta(beta0), NOT the fixed
    fallback."""
    import marginal as mg
    import predictive as pr
    out, data = _setup()
    geom = pr.NewSiteGeometry(
        new_coords=np.array([[1.0, 1.0], [5.0, 5.0]]), train_coords=data.sites_xy,
        knots_S=data.knots_S, knots_phi=data.knots_phi, knots_rho=data.knots_rho,
        radius_S=3.0, eff_range_phi=3.0, eff_range_rho=3.0,
        config=ModelConfig("shi"), water_poly=None)
    b0_sig, b0_xi = 3.0, 0.4
    draw = {"beta_sigma": np.array([b0_sig]), "beta_xi": np.array([b0_xi])}
    sig, xi = pr._holdout_marginal(geom, draw, 4, np.array([20.0, 30.0]),
                                   np.array([0.1, 0.1]))
    assert np.allclose(sig, mg.sigma_from_eta(b0_sig))   # constant, ignores fallback
    assert np.allclose(xi, mg.xi_from_eta(b0_xi))


def test_xi_regression_respects_support():
    sc = sm.SimConfig(Ns=40, Nt=12, seed=3)
    out = sm.simulate(ModelConfig("shi"), sc)
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"], xi_spatial=out["elevations"][:, None])
    st = SamplerSettings(n_iters=150, adapt_size=25, thin=5, nu=out["nu"], seed=1,
                         update_GPD_xi=True)
    TR = run_mcmc(ModelConfig("shi"), data, st)
    assert "beta_xi" in TR and np.all(np.isfinite(TR["beta_xi"]))
    # all visited xi stay strictly inside (-0.5, 1)
    for b in TR["beta_xi"]:
        xi = mg.xi_from_eta(b[0])
        assert -0.5 < xi < 1.0
