"""Tests for the stage-2 recursive-Bayes beta update (recursive_bayes.py).

The core check reconstructs the exact M-H log-ratio from public pieces (full dependence
loglik + stage-1 marginal loglik + both coefficient priors) and confirms
update_beta_recursive feeds that same value to mh_accept. A second test drives the update
and checks the accepted state stays self-consistent (caches match a from-scratch recompute).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import marginal as mg          # noqa: E402
import mcmc_updates as mu      # noqa: E402
import simulate as sm          # noqa: E402
import utilities as u          # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState  # noqa: E402
from mcmc_updates import Ctx   # noqa: E402
import recursive_bayes as rb   # noqa: E402


def _state(seed=7):
    """Serial ModelState with BOTH margins active (sigma & xi regressions on)."""
    sc = sm.SimConfig(Ns=45, Nt=16, seed=seed, Beta_logsigma=(3.0, 0.4))
    out = sm.simulate(ModelConfig("shi"), sc)
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"], sigma_spatial=out["elevations"][:, None],
        xi_spatial=out["elevations"][:, None])
    st = SamplerSettings(nu=out["nu"], seed=1, update_GPD_sigma=True, update_GPD_xi=True,
                         beta_sampler="recursive")
    state = ModelState(ModelConfig("shi"), data, st)
    rng = np.random.default_rng(0)
    ctx = Ctx(range(state.Nt), rng, reduce=None, supports_compact=True,
              covers_all_times=True)
    caches = mu.init_caches(state, ctx.times)
    return state, ctx, caches, st


def _full_loglik_at(state, ctx, sig_mat, xi_mat):
    """From-scratch full dependence loglik at the given sigma/xi surfaces."""
    qvc = state.qd_RW_cens()
    ll = {}
    for t in ctx.times:
        pYc = u.pCGP(state.Yfilled[:, t], state.data.p_cens, state.data.u_vec,
                     sig_mat[:, t], xi_mat[:, t])
        q, d = state.qd_RW_one(t, qvc, pY_col=pYc)
        ll[t] = state.loglik_t(t, qv=q, dv=d, sigma_col=sig_mat[:, t], xi_col=xi_mat[:, t])
    return ctx.total(ll)


def test_recursive_log_ratio_matches_from_scratch():
    state, ctx, caches, st = _state()
    # a single fixed stage-1 "draw" -> propose() always returns this beta*
    b_sig_star = state.beta_sigma + np.array([0.15, -0.1])
    b_xi_star = state.beta_xi + np.array([0.1, 0.05])
    state._recursive_prior = rb.RecursivePrior(
        b_sig_star[None, :], b_xi_star[None, :], sigma_beta_sd=4.0, xi_beta_sd=3.0)

    # expected log-ratio, assembled independently
    sig_p = mg.sigma_from_eta(state.sigma_design.eta(b_sig_star))
    xi_p = mg.xi_from_eta(state.xi_design.eta(b_xi_star))
    lfull_p = _full_loglik_at(state, ctx, sig_p, xi_p)
    lfull_c = ctx.total(caches.Yll)
    lmarg_p = rb._marginal_exc_loglik(state, ctx, sig_p, xi_p)
    lmarg_c = rb._marginal_exc_loglik(state, ctx, state.sigma_mat, state.xi_mat)
    lp2_p = rb._normal_logprior(b_sig_star, b_xi_star, st.sigma_beta_sd, st.xi_beta_sd)
    lp2_c = rb._normal_logprior(state.beta_sigma, state.beta_xi,
                                st.sigma_beta_sd, st.xi_beta_sd)
    lp1_p = state._recursive_prior.logprior(b_sig_star, b_xi_star)
    lp1_c = state._recursive_prior.logprior(state.beta_sigma, state.beta_xi)
    expected = ((lfull_p + lp2_p) - (lfull_c + lp2_c)) - ((lmarg_p + lp1_p) - (lmarg_c + lp1_c))

    # capture the log-ratio update_beta_recursive hands to mh_accept
    captured = {}
    orig = mu.mh_accept
    mu.mh_accept = lambda log_r, rng: (captured.setdefault("log_r", log_r), orig(log_r, rng))[1]
    try:
        rb.update_beta_recursive(state, ctx, caches, st)
    finally:
        mu.mh_accept = orig
    assert "log_r" in captured
    assert abs(captured["log_r"] - expected) < 1e-8, \
        f"log_r {captured['log_r']} != expected {expected}"


def test_recursive_update_keeps_caches_consistent():
    """Drive many recursive moves; whenever a move is accepted the cached full loglik must
    equal a from-scratch recompute at the new betas, and betas must be a stage-1 draw."""
    state, ctx, caches, st = _state(seed=11)
    rng = np.random.default_rng(3)
    draws_sig = state.beta_sigma + 0.25 * rng.standard_normal((40, state.beta_sigma.size))
    draws_xi = state.beta_xi + 0.20 * rng.standard_normal((40, state.beta_xi.size))
    state._recursive_prior = rb.RecursivePrior(draws_sig, draws_xi)

    n_accept = 0
    for _ in range(60):
        b0 = state.beta_sigma.copy()
        rb.update_beta_recursive(state, ctx, caches, st)
        if not np.array_equal(state.beta_sigma, b0):
            n_accept += 1
            # accepted beta must be one of the stage-1 draws
            assert np.any(np.all(np.isclose(draws_sig, state.beta_sigma), axis=1))
            # cached full loglik consistent with a from-scratch recompute
            recomputed = _full_loglik_at(state, ctx, state.sigma_mat, state.xi_mat)
            assert abs(ctx.total(caches.Yll) - recomputed) < 1e-8
    assert n_accept > 0, "no recursive move ever accepted (proposal too far?)"


def test_recursive_prior_from_marginal_fit(tmp_path):
    """from_marginal_fit drops warmup//thin draws and reads the stage-1 prior sds."""
    import json
    npz = tmp_path / "marginal_fit.npz"
    K = 20
    np.savez(npz, beta_sigma=np.arange(K * 2).reshape(K, 2).astype(float),
             beta_xi=np.arange(K).reshape(K, 1).astype(float))
    with open(str(npz)[:-4] + ".config.json", "w") as fh:
        json.dump(dict(warmup=50, thin=5, niters=100, sigma_beta_sd=4.0, xi_beta_sd=3.0), fh)
    rp = rb.RecursivePrior.from_marginal_fit(str(npz))
    assert rp.K == K - (50 // 5)          # burn = warmup//thin = 10
    assert rp.sigma_beta_sd == 4.0 and rp.xi_beta_sd == 3.0
