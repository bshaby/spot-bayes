"""phi_sampler="nuts": does the gradient block target the same posterior as the RWMH
block it replaces, and does it leave the chain in a consistent state?"""
import warnings

import numpy as np
import pytest

import simulate as sm
from config import ModelConfig
from sampler import SamplerData, SamplerSettings, ModelState, run_mcmc
import mcmc_updates as mu
import phi_gradient as pg


def build(domain, Ns=25, Nt=8, seed=4):
    cfg = ModelConfig(model_type="shi", phi_domain=domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = sm.simulate(cfg, sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    d = SamplerData(Y=o["Y_missing"], sites_xy=o["sites_xy"], u_vec=o["u_mat"][:, 0],
                    sigma_vec=o["sigma_mat"][:, 0], xi_vec=o["xi_mat"][:, 0],
                    p_cens=o["p_cens"], knots_S=o["knots_S"],
                    knots_phi=o["knots_phi"], knots_rho=o["knots_rho"])
    return cfg, d, o


@pytest.mark.parametrize("domain", ["positive", "unit"])
def test_value_matches_the_rwmh_target(domain):
    """The NUTS value must BE the RWMH acceptance target: sum_t loglik_t + prior.

    If these differ the block still runs and still mixes -- it just samples the wrong
    distribution, which no acceptance rate or trace plot would reveal."""
    cfg, d, o = build(domain)
    st = SamplerSettings(nu=o["nu"], seed=1, n_iters=10)
    state = ModelState(cfg, d, st)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val, _ = pg.phi_value_grad(state, state.phi_k.copy())
        qvc = state.qd_RW_cens()
        ref = sum(state.loglik_t(t, qv=state.qd_RW_one(t, qvc)[0],
                                 dv=state.qd_RW_one(t, qvc)[1])
                  for t in range(state.Nt))
        ref += float(mu.phi_logprior(state, state.phi_k))
    assert abs(val - ref) < 1e-8 * max(1.0, abs(ref))


@pytest.mark.parametrize("domain", ["positive", "unit"])
def test_time_split_reproduces_the_full_sum(domain):
    """MPI decomposition: a rank sums its own columns, the Allreduce does the rest."""
    cfg, d, o = build(domain)
    st = SamplerSettings(nu=o["nu"], seed=1, n_iters=10)
    state = ModelState(cfg, d, st)
    pk = state.phi_k.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, g = pg.phi_value_grad(state, pk)
        parts = [pg.phi_value_grad(state, pk, times=list(range(r, state.Nt, 3)),
                                   areduce=lambda b: b) for r in range(3)]
        prior = float(mu.phi_logprior(state, pk))
        pgrad = pg._phi_logprior_grad(state, pk)
    # each partial adds the prior once; a real Allreduce sums the buffer BEFORE the
    # prior is added, so here the 2 extra copies are removed by hand
    assert abs(v - (sum(p[0] for p in parts) - 2 * prior)) < 1e-8
    assert np.max(np.abs(g - (sum(p[1] for p in parts) - 2 * pgrad))) < 1e-8


@pytest.mark.parametrize("domain", ["positive", "unit"])
def test_nuts_runs_and_moves_phi(domain):
    cfg, d, o = build(domain)
    kw = dict(nu=o["nu"], seed=3, n_iters=60, thin=1, nuts_warmup=20,
              update_nu=False, update_rho=False, phi_sampler="nuts")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR = run_mcmc(cfg, d, SamplerSettings(**kw), verbose=False)
    phi = TR["phi_k"]
    assert np.all(np.isfinite(TR["ll"])), "non-finite ll under phi NUTS"
    assert np.ptp(phi) > 0, "phi never moved"
    if domain == "unit":                      # the hard support wall must hold
        cfgu = ModelConfig(model_type="shi", phi_domain="unit")
        stu = SamplerSettings(**kw)
        s2 = ModelState(cfgu, d, stu)
        fld = np.array([s2._phi_field(p) for p in phi])
        assert fld.min() > 0.0 and fld.max() < 1.0, "phi(s) left (0,1)"


def test_caches_are_consistent_after_the_move():
    """update_phi_nuts refreshes qd/Yll itself. If it did not, every LATER block would
    score against a phi that is no longer current -- and the chain would still run, with
    entirely ordinary-looking traces. Drive the real block and check the invariant."""
    cfg, d, o = build("positive")
    st = SamplerSettings(nu=o["nu"], seed=5, n_iters=40, nuts_warmup=10,
                         phi_sampler="nuts")
    state = ModelState(cfg, d, st)
    ctx = mu.Ctx(range(state.Nt), np.random.default_rng(0), reduce=None,
                 covers_all_times=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qvc = state.qd_RW_cens()
        caches = mu.Caches(
            Yll={t: state.loglik_t(t) for t in range(state.Nt)}, Zlp={},
            qd={t: state.qd_RW_one(t, qvc) for t in range(state.Nt)})
        p0 = state.phi_k.copy()
        moved = False
        for _ in range(12):
            mu.update_phi_nuts(state, None, ctx, caches, st)
            moved |= not np.array_equal(state.phi_k, p0)
        assert moved, "phi never moved"
        # phi_vec must track phi_k, and the caches must match the committed phi
        assert np.allclose(state.phi_vec, state._phi_field(state.phi_k))
        qvc = state.qd_RW_cens()
        for t in range(state.Nt):
            q, dd = state.qd_RW_one(t, qvc)
            assert np.allclose(q, caches.qd[t][0]) and np.allclose(dd, caches.qd[t][1])
            assert abs(state.loglik_t(t, qv=q, dv=dd) - caches.Yll[t]) < 1e-9


# --------------------------------------------------------------------------- barker

def test_cached_transform_matches_recomputing():
    """The Barker current-point shortcut: passing caches.qd must give EXACTLY the
    value/gradient that recomputing the transform gives. If it drifts, the block
    samples a slightly wrong target and nothing anywhere would report it."""
    cfg, d, o = build("positive")
    st = SamplerSettings(nu=o["nu"], seed=1, n_iters=10)
    state = ModelState(cfg, d, st)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qvc = state.qd_RW_cens()
        qd = {t: state.qd_RW_one(t, qvc) for t in range(state.Nt)}
        v0, g0 = pg.phi_value_grad(state, state.phi_k.copy())
        v1, g1 = pg.phi_value_grad(state, state.phi_k.copy(), qd_cached=qd)
    assert abs(v0 - v1) < 1e-10 * max(1.0, abs(v0))
    assert np.max(np.abs(g0 - g1)) < 1e-10


@pytest.mark.parametrize("domain", ["positive", "unit"])
def test_barker_runs_and_moves_phi(domain):
    cfg, d, o = build(domain)
    kw = dict(nu=o["nu"], seed=3, n_iters=80, thin=1, nuts_warmup=25,
              update_nu=False, update_rho=False, phi_sampler="barker")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR = run_mcmc(cfg, d, SamplerSettings(**kw), verbose=False)
    assert np.all(np.isfinite(TR["ll"]))
    assert np.ptp(TR["phi_k"]) > 0, "phi never moved under barker"


def test_barker_caches_are_consistent_after_the_move():
    cfg, d, o = build("positive")
    st = SamplerSettings(nu=o["nu"], seed=5, n_iters=40, nuts_warmup=10,
                         phi_sampler="barker")
    state = ModelState(cfg, d, st)
    ctx = mu.Ctx(range(state.Nt), np.random.default_rng(0), reduce=None,
                 covers_all_times=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qvc = state.qd_RW_cens()
        caches = mu.Caches(
            Yll={t: state.loglik_t(t) for t in range(state.Nt)}, Zlp={},
            qd={t: state.qd_RW_one(t, qvc) for t in range(state.Nt)})
        p0 = state.phi_k.copy()
        moved = False
        for _ in range(15):
            mu.update_phi_barker(state, None, ctx, caches, st)
            moved |= not np.array_equal(state.phi_k, p0)
        assert moved, "phi never moved"
        assert np.allclose(state.phi_vec, state._phi_field(state.phi_k))
        qvc = state.qd_RW_cens()
        for t in range(state.Nt):
            q, dd = state.qd_RW_one(t, qvc)
            assert np.allclose(q, caches.qd[t][0]) and np.allclose(dd, caches.qd[t][1])
            assert abs(state.loglik_t(t, qv=q, dv=dd) - caches.Yll[t]) < 1e-9


def test_barker_requires_an_array_allreduce_under_mpi():
    """A partial-times ctx with no reduce_arr must RAISE, not silently sample a
    per-rank posterior."""
    cfg, d, o = build("positive")
    st = SamplerSettings(nu=o["nu"], seed=5, n_iters=20, phi_sampler="barker")
    state = ModelState(cfg, d, st)
    ctx = mu.Ctx([0, 2], np.random.default_rng(0), reduce=None,
                 covers_all_times=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        caches = mu.Caches(Yll={}, Zlp={}, qd={})
        with pytest.raises(RuntimeError, match="Allreduce"):
            mu.update_phi_barker(state, None, ctx, caches, st)
