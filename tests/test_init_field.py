"""Tests for the plug-in ("X-inversion") starting values in init_field.py.

The contract this module has to keep is narrow but load-bearing:
  * the emitted snapshot is a VALID ModelState snapshot for all three model types,
  * it is DETERMINISTIC (every MPI rank must build the identical one),
  * the implied Z field is marginally ~N(0,1) (the self-calibration actually works),
  * it beats the cold start on the data log-likelihood -- which is the whole point,
  * the margin ladder picks the rung it claims to, including the no-guess fallback.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
import init_field as ini  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState  # noqa: E402


def _data_from_sim(out):
    return SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"],
        u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
        xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
        knots_S=out["knots_S"], knots_phi=out["knots_phi"], knots_rho=out["knots_rho"])


def _case(model="shi", Ns=40, Nt=60, seed=7):
    cfg = ModelConfig(model_type=model, phi_domain="positive")   # production domain
    out = sm.simulate(cfg, sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = _data_from_sim(out)
    st = SamplerSettings(n_iters=10, thin=5, seed=1, nu=out["nu"])
    return cfg, data, st, out


def _total_ll(state):
    qv, dv = state.qd_RW_all()
    return float(sum(state.loglik_t(t, qv=qv[:, t], dv=dv[:, t])
                     for t in range(state.Nt)))


def _state_at(cfg, data, st, snap):
    state = ModelState(cfg, data, st)
    state.restore(snap)
    return state


@pytest.mark.parametrize("model", ["shi", "huser-wadsworth"])
def test_snapshot_is_valid_and_restorable(model):
    cfg, data, st, _ = _case(model)
    snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    state = _state_at(cfg, data, st, snap)
    assert state.S.shape == (state.k_S, state.Nt)
    assert state.Z.shape == (state.Ns, state.Nt)
    assert np.all(state.S > 0) and np.all(np.isfinite(state.Z))
    assert np.all(np.isfinite(state.R)) and np.all(state.R > 0)
    assert np.isfinite(_total_ll(state))
    assert info["single_knot"] == (model == "huser-wadsworth")


def test_deterministic_across_calls():
    """Every MPI rank builds its own snapshot; they must be bit-identical."""
    cfg, data, st, _ = _case()
    a = ini.plugin_init_snapshot(cfg, data, st)
    b = ini.plugin_init_snapshot(cfg, data, st)
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, b[k]), k
        else:
            assert v == b[k], k


@pytest.mark.parametrize("model", ["shi", "huser-wadsworth"])
def test_z_is_marginally_standard_normal(model):
    """The default "auto" map targets sd(Z) = 1 either way: the scalar solve hits it,
    or the rank fallback makes it exact by construction.

    The tolerance is loose because these simulations are tiny: with one knot (HW) and
    only Nt=60 blocks the sub-threshold rank grid is coarse enough that the residual
    has heavy TIES, and tied normal scores cannot spread to a full unit sd. Real
    datasets (NZ: 309 x 3080) hit 1.000 -- see the module docstring."""
    cfg, data, st, _ = _case(model)
    _snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    assert info["z_sd"] == pytest.approx(1.0, abs=0.12)
    assert 0.0 <= info["lam"] <= 1.0
    assert info["z_map"] in ("cdf", "rank")


def test_cdf_map_calibrates_via_a_scalar_knob():
    """With the fallback disabled, the lam/offset solve must still get close."""
    cfg, data, st, _ = _case()
    _snap, info = ini.plugin_init_snapshot(cfg, data, st, z_map="cdf",
                                           return_info=True)
    assert info["z_map"] == "cdf"
    assert info["z_sd_knob"] in ("lam", "offset")
    assert info["z_sd"] == pytest.approx(1.0, abs=0.3)
    assert info["clamped_frac"] < 0.35


def test_rank_map_gives_exactly_standard_normal_Z():
    cfg, data, st, _ = _case()
    snap, info = ini.plugin_init_snapshot(cfg, data, st, z_map="rank",
                                          return_info=True)
    assert info["z_map"] == "rank"
    assert snap["Z"].std() == pytest.approx(1.0, abs=0.01)
    assert snap["Z"].mean() == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("model", ["shi", "huser-wadsworth"])
def test_beats_cold_start_on_data_loglik(model):
    cfg, data, st, _ = _case(model)
    cold = ModelState(cfg, data, st)                       # S=1, Z=0, phi=1, alpha0=5
    snap = ini.plugin_init_snapshot(cfg, data, st)
    assert _total_ll(_state_at(cfg, data, st, snap)) > _total_ll(cold)


def test_explicit_scalars_are_honoured():
    cfg, data, st, _ = _case()
    snap, info = ini.plugin_init_snapshot(
        cfg, data, st, phi0=0.7, alpha0=2.3, nu=1.25, rho=2.5, return_info=True)
    assert snap["alpha0"] == pytest.approx(2.3)
    assert np.exp(snap["phi_k"]) == pytest.approx(0.7)
    assert snap["nu"] == pytest.approx(1.25)
    assert np.allclose(snap["rho_k"], 2.5)
    assert "nu_fit" not in info                       # both given -> no correlogram fit


def test_nu_rho_estimated_when_absent():
    cfg, data, st, _ = _case()
    _snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    lo, hi = ini.nu_prior_support()
    assert lo <= info["nu_fit"] <= hi
    assert info["rho_fit"] > 0.0
    # nu adopts the fit only when it was not censored (see the fallback tests below)
    assert info["nu"] == pytest.approx(1.5 if info["nu_censored"] else info["nu_fit"])
    assert info["rho"] == pytest.approx(info["rho_fit"])


def test_fit_nu_rho_off_leaves_defaults():
    cfg, data, st, _ = _case()
    _snap, info = ini.plugin_init_snapshot(cfg, data, st, fit_nu_rho=False,
                                           return_info=True)
    assert "nu_fit" not in info
    assert info["rho"] == pytest.approx(1.0)          # ModelState default rho_k


# -- the margin ladder ------------------------------------------------------- #
def test_margin_rung1_given_betas():
    cfg, data, st0, out = _case()
    Ns = data.Y.shape[0]
    data.sigma_spatial = np.column_stack([out["sites_xy"][:, 0]])
    data.xi_spatial = np.column_stack([out["sites_xy"][:, 1]])
    st = SamplerSettings(n_iters=10, thin=5, seed=1, nu=out["nu"],
                         update_GPD_sigma=True, update_GPD_xi=True)
    bs, bx = np.array([0.3, 0.02]), np.array([-1.0, 0.01])
    snap, info = ini.plugin_init_snapshot(cfg, data, st, beta_sigma=bs, beta_xi=bx,
                                          return_info=True)
    assert "sigma:given-betas" in info["margins"]
    assert "xi:given-betas" in info["margins"]
    assert np.allclose(snap["beta_sigma"], bs) and np.allclose(snap["beta_xi"], bx)
    assert Ns == snap["Z"].shape[0]


def test_margin_rung2_fixed_data_vectors():
    """update_GPD off -> margins come from data.sigma_vec/xi_vec, betas are ignored."""
    cfg, data, st, _ = _case()
    _snap, info = ini.plugin_init_snapshot(
        cfg, data, st, beta_sigma=np.array([0.1]), beta_xi=np.array([-1.0]),
        return_info=True)
    assert "sigma:data-vec" in info["margins"]
    assert "update_GPD_sigma=False" in info["margins"]


def test_margin_rung3_no_guess_fallback():
    """No betas at all still initialises -- ModelState's intercept-only default."""
    cfg, data, st0, out = _case()
    data.sigma_spatial = np.column_stack([out["sites_xy"][:, 0]])
    data.xi_spatial = np.column_stack([out["sites_xy"][:, 1]])
    st = SamplerSettings(n_iters=10, thin=5, seed=1, nu=out["nu"],
                         update_GPD_sigma=True, update_GPD_xi=True)
    snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    assert "sigma:default-intercept" in info["margins"]
    assert np.isfinite(_total_ll(_state_at(cfg, data, st, snap)))


def test_wrong_beta_length_is_rejected():
    cfg, data, st0, out = _case()
    data.sigma_spatial = np.column_stack([out["sites_xy"][:, 0]])
    st = SamplerSettings(n_iters=10, thin=5, seed=1, nu=out["nu"],
                         update_GPD_sigma=True)
    with pytest.raises(ValueError, match="beta_sigma has"):
        ini.plugin_init_snapshot(cfg, data, st, beta_sigma=np.zeros(5))


def test_alpha0_below_wall_is_rejected():
    cfg, data, st, _ = _case()
    with pytest.raises(ValueError, match="alpha0"):
        ini.plugin_init_snapshot(cfg, data, st, alpha0=1.5)


# -- pieces ------------------------------------------------------------------ #
def test_uniformize_ranks_below_threshold():
    cfg, data, st, _ = _case()
    state = ModelState(cfg, data, st)
    V = ini.uniformize(state)
    assert V.shape == data.Y.shape
    assert np.all((V > 0) & (V < 1))
    exc = (data.Y > data.u_vec[:, None]) & ~np.isnan(data.Y)
    assert np.all(V[exc] >= data.p_cens)
    below = (~exc) & ~np.isnan(data.Y) & (data.Y <= data.u_vec[:, None])
    assert np.all(V[below] < data.p_cens)
    # within a site, the sub-threshold order matches the data order
    s = int(np.argmax(below.sum(1)))
    m = below[s]
    assert np.array_equal(np.argsort(V[s, m]), np.argsort(data.Y[s, m]))


def test_matern_correlogram_recovers_planted_parameters():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 10, size=(60, 2))
    nu_t, rho_t = 1.0, 3.0
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    K = ini._matern_corr(d, nu_t, rho_t) + 1e-8 * np.eye(60)
    L = np.linalg.cholesky(K)
    Z = L @ rng.standard_normal((60, 4000))
    nu_h, rho_h, rms, mid, cor = ini.fit_matern_correlogram(Z, xy)
    assert rho_h == pytest.approx(rho_t, rel=0.25)
    assert nu_h == pytest.approx(nu_t, rel=0.5)
    assert rms < 0.05
    assert mid.size == cor.size > 5


def test_solve_scalar_falls_back_without_a_sign_change():
    x, solved = ini._solve_scalar(lambda v: v ** 2 + 1.0, 0.0, 1.0)
    assert not solved and x == pytest.approx(0.0)
    x, solved = ini._solve_scalar(lambda v: v - 0.3, 0.0, 1.0)
    assert solved and x == pytest.approx(0.3, abs=1e-3)


# -- nu must respect the sampler's own prior support -------------------------- #
def test_nu_bounds_track_the_sampler_prior():
    """Hardcoding the range would let it drift from priors.nu_logprior."""
    import priors as pr
    loc, scale = pr.nu_logprior.__defaults__
    lo, hi = ini.nu_prior_support(margin=0.0)
    assert (lo, hi) == (loc, loc + scale)
    lo, hi = ini.nu_prior_support()               # default inset
    assert loc < lo < hi < loc + scale


@pytest.mark.parametrize("model", ["shi", "huser-wadsworth"])
def test_emitted_nu_is_inside_the_prior_support(model):
    """Outside it the log-prior is -inf, which freezes the RWMH nu block and
    desynchronises the NUTS one -- both silently."""
    import priors as pr
    cfg, data, st, _ = _case(model)
    snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    assert np.isfinite(pr.nu_logprior(snap["nu"])), snap["nu"]
    assert np.isfinite(pr.nu_logprior(info["nu"]))


def test_out_of_support_nu_is_rejected_not_silently_used():
    cfg, data, st, _ = _case()
    with pytest.raises(ValueError, match="outside the sampler's prior support"):
        ini.plugin_init_snapshot(cfg, data, st, nu=3.0)
    with pytest.raises(ValueError, match="outside the sampler's prior support"):
        ini.plugin_init_snapshot(cfg, data, st, nu=0.3)


def test_censored_nu_is_reported():
    """A fit that ran into a bound means nu was clamped, not identified -- callers
    need to be able to tell the difference."""
    cfg, data, st, _ = _case()
    _snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    assert "nu_censored" in info
    assert info["nu_censored"] == ini._nu_is_censored(info["nu_fit"],
                                                      ini.nu_prior_support())


def test_censored_fit_falls_back_to_an_interior_nu():
    """A bound is not an estimate. When the correlogram runs into one, use an interior
    default rather than starting nu pinned against the wall it cannot easily leave."""
    import priors as pr
    cfg, data, st, _ = _case("huser-wadsworth")     # single-knot R basis -> censors high
    snap, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    lo, hi = ini.nu_prior_support()
    if info["nu_censored"]:
        assert snap["nu"] == pytest.approx(1.5)
        assert lo < snap["nu"] < hi
        assert not ini._nu_is_censored(snap["nu"], (lo, hi))
    assert np.isfinite(pr.nu_logprior(snap["nu"]))


def test_nu_fallback_is_configurable_and_only_used_when_censored():
    cfg, data, st, _ = _case("huser-wadsworth")
    _s, info = ini.plugin_init_snapshot(cfg, data, st, nu_fallback=1.1,
                                        return_info=True)
    if info["nu_censored"]:
        assert info["nu"] == pytest.approx(1.1)
    # an explicit nu always wins, censored or not
    s2, i2 = ini.plugin_init_snapshot(cfg, data, st, nu=0.8, nu_fallback=1.1,
                                      return_info=True)
    assert s2["nu"] == pytest.approx(0.8)


def test_uncensored_fit_keeps_its_estimate():
    cfg, data, st, _ = _case("shi")
    _s, info = ini.plugin_init_snapshot(cfg, data, st, return_info=True)
    if not info["nu_censored"]:
        assert info["nu"] == pytest.approx(info["nu_fit"])
