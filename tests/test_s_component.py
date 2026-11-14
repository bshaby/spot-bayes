"""Componentwise S update (s_sampler="component").

The move is only valid because a Wendland knot has COMPACT SUPPORT and R enters
neither qRW nor dRW: the MH ratio is a difference of subset likelihoods, which must
equal the difference of full ones. Nothing else in the codebase computes a delta
likelihood, so that identity is the first thing checked here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcmc_updates as mu  # noqa: E402


def _state(Ns=30, Nt=6, seed=4):
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState
    cfg = ModelConfig(model_type="shi")
    o = sm.simulate(cfg, sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = SamplerData(Y=o["Y_missing"], sites_xy=o["sites_xy"],
                       u_vec=o["u_mat"][:, 0], sigma_vec=o["sigma_mat"][:, 0],
                       xi_vec=o["xi_mat"][:, 0], p_cens=o["p_cens"],
                       knots_S=o["knots_S"], knots_phi=o["knots_phi"],
                       knots_rho=o["knots_rho"])
    st = SamplerSettings(nu=o["nu"], seed=1, n_iters=20, s_sampler="component")
    state = ModelState(cfg, data, st)
    ctx = mu.Ctx(range(state.Nt), np.random.default_rng(0), reduce=None,
                 supports_compact=True, covers_all_times=True)
    return state, st, ctx, mu.init_caches(state, ctx.times)


def test_subset_delta_equals_full_delta():
    """THE correctness condition: moving knot j changes the full log-likelihood by
    exactly the change in the subset over that knot's support."""
    state, st, ctx, caches = _state()
    rng = np.random.default_rng(2)
    W = state.W_wend_S
    for t in (0, 3):
        qv, dv = caches.qd[t]
        ll_full_before = state.loglik_t(t, qv=qv, dv=dv)
        for j in range(state.k_S):
            sj = np.flatnonzero(W[:, j] > 0.0)
            if sj.size == 0:
                continue
            logS = np.log(state.S[:, t])
            lp = logS[j] + 0.3 * rng.standard_normal()
            R_cur = state.R[sj, t].copy()
            R_new = R_cur + W[sj, j] * (np.exp(lp) - np.exp(logS[j]))
            if not (np.all(R_new > 0) and np.all(np.isfinite(R_new))):
                continue
            d_sub = (state.loglik_subset_S(t, sj, R_new, qv, dv)
                     - state.loglik_subset_S(t, sj, R_cur, qv, dv))
            R_full = state.R[:, t].copy()
            R_full[sj] = R_new
            d_full = state.loglik_t(t, R=R_full, qv=qv, dv=dv) - ll_full_before
            assert abs(d_sub - d_full) <= 1e-9 * max(1.0, abs(d_full)), (
                f"subset delta {d_sub} != full delta {d_full} at t={t}, knot={j}")


def test_component_update_keeps_state_and_cache_consistent():
    """R must stay equal to W @ S, and the Yll cache must match a fresh likelihood."""
    state, st, ctx, caches = _state()
    for _ in range(6):
        mu.update_S_component(state, {t: [mu._mk(1, 1.0) for _ in range(state.k_S)]
                                      for t in ctx.times}, ctx, caches, st)
    assert np.all(state.S > 0) and np.all(np.isfinite(state.S))
    assert np.allclose(state.R, state.W_wend_S @ state.S, rtol=1e-12, atol=1e-12)
    for t in ctx.times:
        q, d = caches.qd[t]
        assert np.isclose(caches.Yll[t], state.loglik_t(t, qv=q, dv=d), rtol=1e-10)


def test_component_actually_moves_every_knot():
    """A knot with empty support would be silently frozen; assert all of them move."""
    state, st, ctx, caches = _state()
    ad = {t: [mu._mk(1, 1.0) for _ in range(state.k_S)] for t in ctx.times}
    S0 = state.S.copy()
    for _ in range(25):
        mu.update_S_component(state, ad, ctx, caches, st)
    moved = np.any(state.S != S0, axis=1)
    assert moved.all(), f"knots never moved: {np.flatnonzero(~moved)}"
    props = np.array([[a.n_prop for a in ad[t]] for t in ctx.times])
    assert (props == 25).all(), "every knot must be proposed once per sweep per time"


def test_component_adapters_are_wired_through_make_adaptives():
    """make_adaptives must hand back per-knot lists, adapt_all must walk them, and
    acceptance_summary must still report a pooled S rate (it flattens one level)."""
    from config import ModelConfig
    from sampler import SamplerSettings
    state, st, ctx, caches = _state()
    cfg = ModelConfig(model_type="shi")
    ad = mu.make_adaptives(state, st, ctx.times, cfg)
    assert isinstance(ad["S"][ctx.times[0]], list)
    assert len(ad["S"][ctx.times[0]]) == state.k_S
    mu.update_S(state, ad["S"], ctx, caches, st)
    mu.adapt_all(ad, 1, 1)                      # must not raise on the nested lists
    rate = mu.acceptance_summary(ad).get("S")
    assert rate is not None and 0.0 <= rate <= 1.0

    st_joint = SamplerSettings(nu=st.nu, seed=1, n_iters=20)     # default s_sampler
    adj = mu.make_adaptives(state, st_joint, ctx.times, cfg)
    assert not isinstance(adj["S"][ctx.times[0]], list)
    assert adj["S"][ctx.times[0]].dim == state.k_S


def test_joint_arm_is_unchanged_by_the_dispatch():
    """The default path must still be the original block, byte-for-byte."""
    from sampler import SamplerSettings
    state, st, ctx, caches = _state()
    st_joint = SamplerSettings(nu=st.nu, seed=1, n_iters=20)
    S_before = state.S.copy()
    ad = {t: mu._mk(state.k_S, 1.0) for t in ctx.times}
    g1 = np.random.default_rng(5)
    c1 = mu.Ctx(range(state.Nt), g1, reduce=None, supports_compact=True,
                covers_all_times=True)
    mu.update_S(state, ad, c1, caches, st_joint)
    via_dispatch = state.S.copy()

    state.S = S_before.copy()
    state.R = state.W_wend_S @ state.S
    caches2 = mu.init_caches(state, ctx.times)
    ad2 = {t: mu._mk(state.k_S, 1.0) for t in ctx.times}
    c2 = mu.Ctx(range(state.Nt), np.random.default_rng(5), reduce=None,
                supports_compact=True, covers_all_times=True)
    mu.update_S_joint(state, ad2, c2, caches2, st_joint)
    assert np.array_equal(via_dispatch, state.S)


def test_s_monitor_trace_is_recorded_and_scoreable():
    """The A/B needs S mixing observable WITHOUT the 1.1 GB latent stream, so a
    small log-S trace rides along. Check it is populated, finite, and actually
    varying (a constant trace would score as ESS ~ n and look fantastic)."""
    import warnings
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, run_mcmc
    cfg = ModelConfig(model_type="shi")
    o = sm.simulate(cfg, sm.SimConfig(Ns=25, Nt=9, seed=4))
    data = SamplerData(Y=o["Y_missing"], sites_xy=o["sites_xy"],
                       u_vec=o["u_mat"][:, 0], sigma_vec=o["sigma_mat"][:, 0],
                       xi_vec=o["xi_mat"][:, 0], p_cens=o["p_cens"],
                       knots_S=o["knots_S"], knots_phi=o["knots_phi"],
                       knots_rho=o["knots_rho"])
    st = SamplerSettings(nu=o["nu"], seed=3, n_iters=30, thin=1,
                         s_sampler="component", s_trace_times=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR = run_mcmc(cfg, data, st)
    S = TR["S_mon"]
    assert S.shape == (30, TR["state"].k_S, 4)
    assert np.isfinite(S).all()
    assert (S.std(axis=0) > 0).any(), "S monitor never varies"
    assert list(TR["S_mon_times"]) == [0, 1, 2, 3]

    st0 = SamplerSettings(nu=o["nu"], seed=3, n_iters=30, thin=1)   # default: off
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR0 = run_mcmc(cfg, data, st0)
    assert "S_mon" not in TR0, "the monitor must stay off by default"


def _sblock_pieces(state, caches, t):
    ci, ei = state.cens_exc(t)
    cens = np.zeros((state.Ns, 1), dtype=bool)
    exc = np.zeros((state.Ns, 1), dtype=bool)
    cens[ci, 0] = True
    exc[ei, 0] = True
    logsf = mu._norm_logsf_np(np.ascontiguousarray(state.Z[:, t:t + 1]))
    qv, dv = caches.qd[t]
    return qv, dv, logsf, cens, exc


def test_s_gradient_matches_finite_differences():
    state, st, ctx, caches = _state()
    qv, dv, logsf, cens, exc = _sblock_pieces(state, caches, 0)
    l0 = np.log(state.S[:, 0])
    rng = np.random.default_rng(4)
    for _ in range(3):
        l = l0 + 0.4 * rng.standard_normal(l0.size)
        _, g = mu._sblock_value_grad(state, 0, qv, dv, logsf, cens, exc, l)
        fd = np.empty_like(g)
        eps = 1e-6
        for j in range(l.size):
            a = l.copy(); a[j] += eps
            b = l.copy(); b[j] -= eps
            fd[j] = (mu._sblock_value_grad(state, 0, qv, dv, logsf, cens, exc, a)[0]
                     - mu._sblock_value_grad(state, 0, qv, dv, logsf, cens, exc, b)[0]) / (2 * eps)
        assert np.max(np.abs(g - fd)) <= 1e-5 * max(1.0, np.max(np.abs(fd)))


def test_s_gradient_potential_differs_from_production_by_a_CONSTANT():
    """The fused kernel drops S-independent terms, exactly as the Z one does. That is
    fine for NUTS only because the omission is CONSTANT in S -- constants cancel in
    the Hamiltonian differences. If it ever varied with S the sampler would target
    the wrong distribution while every gradient check still passed."""
    import priors as pr
    state, st, ctx, caches = _state()
    qv, dv, logsf, cens, exc = _sblock_pieces(state, caches, 0)
    rng = np.random.default_rng(5)
    l0 = np.log(state.S[:, 0])
    offs = []
    for _ in range(8):
        l = l0 + 0.5 * rng.standard_normal(l0.size)
        mine = mu._sblock_value_grad(state, 0, qv, dv, logsf, cens, exc, l)[0]
        R = state.W_wend_S @ np.exp(l)
        prod = (state.loglik_t(0, R=R, qv=qv, dv=dv)
                + pr.S_logprior_1t(l, state.gamma_k))
        offs.append(prod - mine)
    assert np.ptp(np.asarray(offs)) < 1e-8, "the dropped term is not constant in S"


def test_levy_log_and_grad_matches_the_production_prior():
    import priors as pr
    rng = np.random.default_rng(6)
    for _ in range(4):
        l = rng.normal(0.5, 1.0, size=9)
        g = rng.uniform(0.5, 3.0, size=9)
        val, grad = mu._levy_log_and_grad(l, g)
        assert abs(val - pr.S_logprior_1t(l, g)) <= 1e-9 * max(1.0, abs(val))
        eps = 1e-6
        fd = np.array([(pr.S_logprior_1t(np.where(np.arange(9) == j, l + eps, l), g)
                        - pr.S_logprior_1t(np.where(np.arange(9) == j, l - eps, l), g))
                       / (2 * eps) for j in range(9)])
        assert np.max(np.abs(grad - fd)) <= 1e-5 * max(1.0, np.max(np.abs(fd)))


def test_s_nuts_runs_and_keeps_state_consistent():
    from sampler import SamplerSettings
    state, _, ctx, caches = _state()
    st = SamplerSettings(nu=1.0, seed=1, n_iters=20, s_sampler="nuts",
                         nuts_warmup=5)
    ad = {t: mu._mk(state.k_S, 1.0) for t in ctx.times}
    S0 = state.S.copy()
    for _ in range(8):
        mu.update_S_nuts(state, ad, ctx, caches, st)
    assert np.all(state.S > 0) and np.all(np.isfinite(state.S))
    assert not np.array_equal(state.S, S0)
    assert np.allclose(state.R, state.W_wend_S @ state.S, rtol=1e-12, atol=1e-12)
    for t in ctx.times:
        q, d = caches.qd[t]
        assert np.isclose(caches.Yll[t], state.loglik_t(t, qv=q, dv=d), rtol=1e-10)


def test_all_s_arms_target_the_same_conditional():
    """Every S arm must sample the SAME p(S | rest). The independence arm is the one
    that could silently fail: its acceptance omits the prior entirely (it cancels
    against the proposal), so a sign slip or a missed Jacobian would leave a sampler
    that runs happily and targets the wrong distribution. Everything but S is frozen
    here, so the arms are directly comparable."""
    from sampler import SamplerSettings
    N = 6000
    out = {}
    for arm in ("joint", "component", "indep", "both"):
        state, _, ctx, caches = _state(Ns=22, Nt=2, seed=11)
        st = SamplerSettings(nu=1.0, seed=1, n_iters=10, s_sampler=arm,
                             s_prior_mix=1.0)
        ad = ({t: mu._mk(state.k_S, 1.0) for t in ctx.times} if arm == "joint"
              else {t: [mu._mk(1, 1.0) for _ in range(state.k_S)] for t in ctx.times})
        c = mu.Ctx(range(state.Nt), np.random.default_rng(5), reduce=None,
                   supports_compact=True, covers_all_times=True)
        keep = np.empty((N, state.k_S))
        for i in range(N):
            mu.update_S(state, ad, c, caches, st)
            keep[i] = np.log(state.S[:, 0])
            if (i + 1) % 50 == 0:
                for t in c.times:
                    for a in (ad[t] if isinstance(ad[t], list) else [ad[t]]):
                        a.adapt(50, (i + 1) // 50)
        out[arm] = keep[N // 3:]

    ref = out["component"]                       # the best-mixed reference
    for arm in ("joint", "indep", "both"):
        d = out[arm]
        # log S is heavy-tailed, so compare robust location/scale, and allow for the
        # joint arm's own poor mixing by scaling the tolerance by its MC error
        for j in range(ref.shape[1]):
            lo = abs(np.median(d[:, j]) - np.median(ref[:, j]))
            scale = np.subtract(*np.percentile(ref[:, j], [75, 25])) + 1e-12
            assert lo < 0.8 * scale, (
                f"{arm} knot {j}: median off by {lo:.3f} vs IQR {scale:.3f}")
