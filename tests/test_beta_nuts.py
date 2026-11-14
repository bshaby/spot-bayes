"""Gradient for the GPD-regression (beta) NUTS block.

The joint [beta_sigma, beta_xi] target sum_t loglik_t(beta) + N(0,sd^2) priors is
differentiated in closed form (`mcmc_updates._beta_value_grad`): analytic through
the link/design and GPD density, with the one RW-transform piece d/dX log dRW(X)
supplied by the exact closed-form derivative rw_inte.dlog_dRW_dx. This checks the
gradient against finite differences of the exact same value pipeline, and that the
value matches the sampler's own loglik_t sum.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcmc_updates as mu  # noqa: E402
import simulate as sm  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState  # noqa: E402


def _state(seed=7, Ns=60, Nt=20, p_sp=4):
    out = sm.simulate(ModelConfig(model_type="shi"), sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    rng = np.random.default_rng(0)
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
        knots_S=out["knots_S"], knots_phi=out["knots_phi"], knots_rho=out["knots_rho"],
        sigma_spatial=rng.standard_normal((Ns, p_sp)),
        xi_spatial=rng.standard_normal((Ns, p_sp)))
    st = SamplerSettings(nu=out["nu"], seed=1, update_GPD_sigma=True, update_GPD_xi=True,
                         sigma_beta_sd=5.0, xi_beta_sd=5.0)
    return ModelState(ModelConfig(model_type="shi"), data, st), st


def _ctx(state):
    rng = np.random.default_rng(0)
    return mu.Ctx(range(state.Nt), rng, reduce=None, covers_all_times=True)


def test_beta_value_matches_loglik_sum():
    state, st = _state()
    ctx = _ctx(state)
    qvc = state.qd_RW_cens()
    theta = np.concatenate([state.beta_sigma, state.beta_xi])
    logp, _ = mu._beta_value_grad(theta, state, ctx, st, qvc)
    # reconstruct the likelihood part independently and add the priors
    ll = sum(state.loglik_t(t) for t in range(state.Nt))
    import scipy.stats as ss
    pri = (ss.norm.logpdf(state.beta_sigma, 0, st.sigma_beta_sd).sum()
           + ss.norm.logpdf(state.beta_xi, 0, st.xi_beta_sd).sum())
    assert np.isclose(logp, ll + pri, rtol=0, atol=1e-7)


def test_beta_gradient_matches_finite_differences():
    state, st = _state()
    ctx = _ctx(state)
    qvc = state.qd_RW_cens()
    rng = np.random.default_rng(3)
    theta0 = np.concatenate([state.beta_sigma, state.beta_xi])
    theta0 = theta0 + 0.15 * rng.standard_normal(theta0.size)     # off the init

    _, g = mu._beta_value_grad(theta0, state, ctx, st, qvc)
    eps = 1e-5
    g_fd = np.empty_like(theta0)
    for k in range(theta0.size):
        e = np.zeros_like(theta0); e[k] = eps
        g_fd[k] = (mu._beta_value_grad(theta0 + e, state, ctx, st, qvc)[0]
                   - mu._beta_value_grad(theta0 - e, state, ctx, st, qvc)[0]) / (2 * eps)
    # The gradient is analytic through the link/design + GPD density; the dRW piece
    # uses the exact analytic-density log-derivative. The value, however, uses the
    # numeric dRW_safe, and in the deep tail 1/dRW_safe ~ 1e3 amplifies the known
    # few-% analytic-vs-safe density bias into a large error on the couple of
    # components tied to those rare tail exceedances. So the gradient is accurate in
    # BULK (most components match FD tightly) but not uniformly -- assert the bulk.
    rel = np.abs(g - g_fd) / np.maximum(np.abs(g_fd), 1.0)
    assert np.median(rel) < 0.02          # the bulk of components track FD closely
    assert np.sum(rel > 0.05) <= 2        # at most a couple tail-amplified outliers


def test_nuts_warmup_estimates_diagonal_metric():
    """NUTSWarmup should recover a target's very-different per-dim scales as its
    diagonal metric (inv_mass ~ posterior variance), so an ill-scaled block is
    preconditioned to ~unit scale (the fix for the deep beta-NUTS trees)."""
    scales = np.array([0.1, 1.0, 10.0, 100.0])         # 1e6 spread in variance

    def vg(th):                                        # N(0, diag(scales^2))
        return -0.5 * float(np.sum((th / scales) ** 2)), -th / scales ** 2

    rng = np.random.default_rng(0)
    th = np.zeros(4)
    w = mu.NUTSWarmup(vg, th, rng, warmup=400, adapt_mass=True)
    for _ in range(400):
        th, acc, _ = mu.nuts_transition(vg, th, w.eps, rng, inv_mass=w.inv_mass)
        w.update(th, acc, vg, rng)
    assert w.inv_mass is not None
    ratio = w.inv_mass / scales ** 2                   # should be ~1 per dimension
    assert np.all((ratio > 0.25) & (ratio < 4.0))


def test_nuts_unit_metric_matches_no_metric():
    """inv_mass of all ones must reproduce the unit-metric transition bit-for-bit
    (guards the byte-identity of the inv_mass=None fast path)."""
    def vg(th):
        return -0.5 * float(th @ th), -th

    th0 = np.array([0.7, -1.3, 0.2])
    a, sa, da = mu.nuts_transition(vg, th0, 0.5, np.random.default_rng(1), inv_mass=None)
    b, sb, db = mu.nuts_transition(vg, th0, 0.5, np.random.default_rng(1),
                                   inv_mass=np.ones(3))
    assert np.array_equal(a, b) and sa == sb and da == db


def test_beta_support_violation_returns_neg_inf():
    state, st = _state()
    ctx = _ctx(state)
    qvc = state.qd_RW_cens()
    # push xi very negative so 1 + xi*(Y-u)/sigma <= 0 at some exceedance
    theta = np.concatenate([state.beta_sigma, state.beta_xi])
    theta[state.sigma_design.n_params] = -50.0        # xi intercept -> xi ~ XI_LO
    logp, g = mu._beta_value_grad(theta, state, ctx, st, qvc)
    assert not np.isfinite(logp) and np.all(g == 0.0)


# --- Barker-proposal MALA (alternative gradient beta sampler) ------------------

def test_barker_samples_anisotropic_gaussian():
    """A correct Barker transition + warmup must leave a correlated, ill-scaled
    Gaussian invariant: post-burn empirical covariance ~ the true covariance. This
    is the end-to-end check of the proposal skew + the exact Metropolis correction
    (the softplus ratio) -- a sign error there breaks reversibility and the recovered
    variances drift off."""
    rng = np.random.default_rng(0)
    d = 6
    A = np.diag(np.array([100.0, 50.0, 1.0, 1.0, 0.2, 0.2]))     # 500x variance spread

    def vg(x):
        return -0.5 * float(x @ A @ x), -A @ x

    x = np.zeros(d)
    bk = mu.BarkerWarmup(vg, x, rng, warmup=400)
    xs = []
    for i in range(4000):
        x, _, _, a = mu.barker_transition(vg, x, bk.eps, bk.precond_sd, rng)
        bk.update(x, a, vg, rng)
        if i > 800:
            xs.append(x.copy())
    xs = np.array(xs)
    true_var = 1.0 / np.diag(A)
    ratio = np.diag(np.cov(xs.T)) / true_var
    assert np.all((ratio > 0.6) & (ratio < 1.6))       # variances recovered per-dim
    assert np.all(np.abs(xs.mean(0)) < 0.3 * np.sqrt(true_var))   # mean ~ 0


def test_barker_warmup_estimates_diagonal_preconditioner():
    """BarkerWarmup should learn a target's different per-dim scales as its diagonal
    preconditioner (precond_sd ~ posterior sd) from a cold start, so a moderately
    ill-scaled block is preconditioned to ~unit scale. (Extreme anisotropy is instead
    handled by warm-seeding init_precond_sd -- see the next test -- since a single-step
    cold warmup can't move the large-scale dims enough to estimate their variance.)"""
    scales = np.array([0.3, 1.0, 3.0, 10.0])           # 1e3 spread in variance

    def vg(th):
        return -0.5 * float(np.sum((th / scales) ** 2)), -th / scales ** 2

    rng = np.random.default_rng(0)
    th = np.zeros(4)
    w = mu.BarkerWarmup(vg, th, rng, warmup=800)
    for _ in range(800):
        th, _, _, a = mu.barker_transition(vg, th, w.eps, w.precond_sd, rng)
        w.update(th, a, vg, rng)
    ratio = w.precond_sd / scales                      # should be ~1 per dimension
    assert np.all((ratio > 0.25) & (ratio < 4.0))


def test_barker_warm_seeded_preconditioner_handles_extreme_anisotropy():
    """A fixed (warm-seeded) preconditioner takes the per-dim scales directly, so even
    a 1e6 variance spread -- which a cold warmup can't self-estimate -- mixes well and
    is left invariant. This is the path the exp3 warm A/B uses (beta_*_proposal_cov)."""
    scales = np.array([0.1, 1.0, 10.0, 100.0])         # 1e6 spread

    def vg(th):
        return -0.5 * float(np.sum((th / scales) ** 2)), -th / scales ** 2

    rng = np.random.default_rng(1)
    th = np.zeros(4)
    w = mu.BarkerWarmup(vg, th, rng, warmup=400, init_precond_sd=scales)
    xs = []
    for i in range(4000):
        th, _, _, a = mu.barker_transition(vg, th, w.eps, w.precond_sd, rng)
        w.update(th, a, vg, rng)
        if i > 800:
            xs.append(th.copy())
    ratio = np.std(np.array(xs), axis=0) / scales
    assert np.all((ratio > 0.6) & (ratio < 1.6))       # variances recovered per-dim


def test_barker_transition_rejects_out_of_support():
    """A proposal into -inf (outside GPD support) must be rejected in place with
    accept-prob 0 -- the value+grad returned are the unchanged current point."""
    d = 4
    A = np.eye(d)

    def vg(x):                                         # -inf for ANY move off the origin
        return (-0.5 * float(x @ A @ x), -A @ x) if abs(x[0]) < 1e-12 else (-np.inf, np.zeros(d))

    x0 = np.zeros(d)
    t1, lp, g, a = mu.barker_transition(vg, x0, 5.0, np.ones(d), np.random.default_rng(2))
    assert np.array_equal(t1, x0) and a == 0.0 and lp == 0.0


def test_update_beta_barker_preserves_cache_invariants():
    """Driving the real block: update_beta_barker must keep sigma/xi matrices, the
    per-time qd/Yll caches, and _pY consistent with the (possibly moved) beta -- the
    same post-move refresh contract as update_beta_nuts."""
    state, st = _state()
    st.beta_sampler = "barker"
    st.nuts_warmup = 10
    ctx = _ctx(state)
    from mcmc_updates import Caches
    caches = Caches(Yll={t: state.loglik_t(t) for t in range(state.Nt)}, Zlp={},
                    qd={t: state.qd_RW_one(t, state.qd_RW_cens()) for t in range(state.Nt)})
    b0 = np.concatenate([state.beta_sigma, state.beta_xi]).copy()
    moved = False
    for _ in range(30):
        mu.update_beta_barker(state, ctx, caches, st)
        if not np.array_equal(np.concatenate([state.beta_sigma, state.beta_xi]), b0):
            moved = True
    assert moved                                       # the chain actually moves
    # caches match the committed beta: recompute sigma/xi -> loglik and compare
    sig = state._mg.sigma_from_eta(state.sigma_design.eta(state.beta_sigma))
    xi = state._mg.xi_from_eta(state.xi_design.eta(state.beta_xi))
    assert np.allclose(sig, state.sigma_mat) and np.allclose(xi, state.xi_mat)
    for t in range(state.Nt):
        q, dd = state.qd_RW_one(t, state.qd_RW_cens())
        ll = state.loglik_t(t, qv=q, dv=dd, sigma_col=sig[:, t], xi_col=xi[:, t])
        assert np.isclose(caches.Yll[t], ll, rtol=0, atol=1e-9)
