"""Prototype NUTS update for the Shi rho range knots (CP conditional p(rho_k|Z,nu)).

Covers the two things that must hold before the block is worth A/B-ing:
  1. the closed-form gradient in `_rho_cp_value_grad` matches finite differences
     (the Matern-recurrence dK/drho derivation -- no dK_nu/dnu), and
  2. `update_rho_nuts` runs end-to-end on a real Shi state, leaves rho_k positive,
     keeps the log-likelihood finite, and keeps the Zlp cache consistent with K.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcmc_updates as mu  # noqa: E402
import utilities as u  # noqa: E402
import priors as pr  # noqa: E402


def test_rho_cp_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    Ns, Nt, k_rho, nu = 12, 5, 3, 1.3
    coords = rng.uniform(0, 5, size=(Ns, 2))
    dist_cache = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dist_cache['d1_sq'] + dist_cache['d2_sq'])
    W = rng.uniform(0.1, 1.0, size=(Ns, k_rho))
    Z = rng.standard_normal((Ns, Nt))
    psi0 = np.log(rng.uniform(0.5, 2.0, size=k_rho))

    def vg(psi):
        return mu._rho_cp_value_grad(psi, W, nu, dist_cache, Dmat, Z, Nt, 10.0)

    _, g = vg(psi0)
    eps = 1e-6
    g_fd = np.array([(vg(psi0 + eps * e)[0] - vg(psi0 - eps * e)[0]) / (2 * eps)
                     for e in np.eye(k_rho)])
    assert np.max(np.abs(g - g_fd)) < 1e-5


def test_closedform_target_matches_ns_cov_cached():
    """The logp uses the production ns_cov_cached; sanity-check the Gaussian value
    against an independent slogdet/solve at the same rho."""
    rng = np.random.default_rng(1)
    Ns, Nt, k_rho, nu = 10, 4, 2, 0.8
    coords = rng.uniform(0, 4, size=(Ns, 2))
    dist_cache = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dist_cache['d1_sq'] + dist_cache['d2_sq'])
    W = rng.uniform(0.2, 1.0, size=(Ns, k_rho))
    Z = rng.standard_normal((Ns, Nt))
    psi0 = np.log(np.array([1.2, 0.7]))

    logp, _ = mu._rho_cp_value_grad(psi0, W, nu, dist_cache, Dmat, Z, Nt, 10.0)
    rho_vec = W @ np.exp(psi0)
    r = rho_vec ** 2 / (8.0 * nu)
    K = u.ns_cov_cached(r, np.ones(Ns), dist_cache, kappa=nu, cov_model="matern")
    sign, logdet = np.linalg.slogdet(K)
    quad = float(np.sum(Z * np.linalg.solve(K, Z)))
    ref = (-0.5 * Nt * logdet - 0.5 * quad - 0.5 * Nt * Ns * np.log(2 * np.pi)
           + pr.rho_logprior(np.exp(psi0), scale=10.0) + float(np.sum(psi0)))
    assert sign > 0
    assert abs(logp - ref) < 1e-8


def test_nu_rho_cp_gradient_matches_finite_differences():
    """Joint (nu, rho) CP gradient in [z_nu, log rho_k] coords -- exercises the nu
    order channel (ported dK_nu/dnu), the nu range channel, and the logit Jacobian."""
    rng = np.random.default_rng(0)
    Ns, Nt, k_rho = 14, 6, 3
    nu_loc, nu_scale = 0.5, 2.5
    coords = rng.uniform(0, 5, size=(Ns, 2))
    dc = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dc['d1_sq'] + dc['d2_sq'])
    W = rng.uniform(0.1, 1.0, size=(Ns, k_rho))
    Z = rng.standard_normal((Ns, Nt))
    iu = np.triu_indices(Ns, k=1)
    sig0 = (1.3 - nu_loc) / nu_scale
    theta = np.concatenate(([np.log(sig0 / (1 - sig0))],
                            np.log(rng.uniform(0.5, 2.0, size=k_rho))))

    def vg(th):
        return mu._nu_rho_cp_value_grad(th, W, dc, Dmat, Z, Nt, 10.0,
                                        nu_loc, nu_scale, iu)

    _, g = vg(theta)
    eps = 1e-6
    g_fd = np.array([(vg(theta + eps * e)[0] - vg(theta - eps * e)[0]) / (2 * eps)
                     for e in np.eye(len(theta))])
    assert np.max(np.abs(g - g_fd)) < 1e-5


def test_update_nu_rho_nuts_runs_and_keeps_state_consistent():
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState

    sc = sm.SimConfig(Ns=25, Nt=6, seed=4)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
                       u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
                       xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
                       knots_S=out["knots_S"], knots_phi=out["knots_phi"],
                       knots_rho=out["knots_rho"])
    st = SamplerSettings(nu=out["nu"], seed=1, update_nu=True,
                         nu_rho_sampler="nuts", nuts_warmup=5, n_iters=20)
    state = ModelState(ModelConfig(model_type="shi"), data, st)

    rng = np.random.default_rng(0)
    ctx = mu.Ctx(range(state.Nt), rng, reduce=None, supports_compact=True,
                 covers_all_times=True)
    caches = mu.init_caches(state, ctx.times)

    nu0, rho0 = state.nu, state.rho_k.copy()
    for _ in range(15):
        mu.update_nu_rho_nuts(state, ctx, caches, st)
        assert 0.5 <= state.nu <= 3.0          # stays inside the uniform support
        assert np.all(state.rho_k > 0) and np.all(np.isfinite(state.rho_k))

    assert (state.nu != nu0) or (not np.array_equal(state.rho_k, rho0))
    for t in ctx.times:
        assert np.isclose(caches.Zlp[t], state.Z_logprior(state.Z[:, t]))
        assert np.isfinite(caches.Yll[t])


def test_nu_rho_nuts_every_fires_on_schedule():
    """The amortized schedule fires on sweeps 0, k, 2k, ... and the k == 1 default
    leaves no counter on the state (so existing runs snapshot unchanged)."""
    from sampler import SamplerSettings

    class _S:                      # only _nu_rho_sweep is touched
        pass

    st1 = SamplerSettings(nu_rho_sampler="nuts")
    s1 = _S()
    assert all(mu._nu_rho_nuts_due(s1, st1) for _ in range(5))
    assert not hasattr(s1, "_nu_rho_sweep")

    st4 = SamplerSettings(nu_rho_sampler="nuts", nu_rho_nuts_every=4)
    s4 = _S()
    fired = [mu._nu_rho_nuts_due(s4, st4) for _ in range(12)]
    assert fired == [i % 4 == 0 for i in range(12)]
    assert s4._nu_rho_sweep == 12          # phase, not just tally, is carried


def test_nu_rho_nuts_every_holds_nu_and_survives_resume(monkeypatch):
    """In the real dispatch: the block fires ceil(n/k) times, nu/rho are HELD (not
    RWMH-updated) on the skipped sweeps, and a snapshot/restore keeps the phase."""
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState

    cfg = ModelConfig(model_type="shi")
    sc = sm.SimConfig(Ns=25, Nt=6, seed=4)
    out = sm.simulate(cfg, sc)
    data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
                       u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
                       xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
                       knots_S=out["knots_S"], knots_phi=out["knots_phi"],
                       knots_rho=out["knots_rho"])
    st = SamplerSettings(nu=out["nu"], seed=1, update_nu=True, nu_rho_sampler="nuts",
                         nu_rho_nuts_every=3, nuts_warmup=5, n_iters=20)
    state = ModelState(cfg, data, st)

    rng = np.random.default_rng(0)
    ctx = mu.Ctx(range(state.Nt), rng, reduce=None, supports_compact=True,
                 covers_all_times=True)
    caches = mu.init_caches(state, ctx.times)
    ad = mu.make_adaptives(state, st, ctx.times, cfg)

    calls = []
    real = mu.update_nu_rho_nuts
    monkeypatch.setattr(mu, "update_nu_rho_nuts",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    nu_seen = []
    for _ in range(9):
        nu_before, rho_before = state.nu, state.rho_k.copy()
        mu.run_global_updates(state, ad, ctx, caches, st, cfg)
        nu_seen.append((len(calls), state.nu == nu_before,
                        np.array_equal(state.rho_k, rho_before)))
    assert len(calls) == 3                       # sweeps 0, 3, 6 of 9
    # every sweep that did NOT fire the block left nu and rho untouched
    for i, (_, nu_held, rho_held) in enumerate(nu_seen):
        if i % 3:
            assert nu_held and rho_held, f"sweep {i} moved nu/rho without the block"

    snap = state.snapshot()
    assert int(snap["nu_rho_sweep"]) == 9
    state._nu_rho_sweep = 0                      # simulate a fresh process
    state.restore(snap)
    assert state._nu_rho_sweep == 9              # resume continues the cycle
    state._nu_rho_sweep = 0
    state.restore(snap, nuts=False)              # warm start: cold adapters, cold cycle
    assert state._nu_rho_sweep == 0


_MPI_EVERY_CHILD = '''
import sys, warnings
import numpy as np
import simulate as sm, mcmc_updates as mu
from config import ModelConfig
from sampler import SamplerData, SamplerSettings
from sampler_mpi import run_mcmc_mpi
from mpi4py import MPI

EVERY, NIT = 3, 12
cfg = ModelConfig(model_type="shi", phi_domain="unit")
out = sm.simulate(cfg, sm.SimConfig(Ns=18, Nt=6, seed=7))
knots = sm.make_isometric_knots(0, 10, 0, 10, 4)
data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
    u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
    xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"], knots_S=knots,
    knots_phi=knots, knots_rho=knots)
st = SamplerSettings(n_iters=NIT, thin=1, seed=12345, adapt_size=10, update_nu=True,
                     nu_rho_sampler="nuts", nu_rho_nuts_every=EVERY, nuts_warmup=4)

fired = []
real = mu.update_nu_rho_nuts
mu.update_nu_rho_nuts = lambda *a, **k: (fired.append(1), real(*a, **k))[1]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    TR = run_mcmc_mpi(cfg, data, st)

comm = MPI.COMM_WORLD
counts = comm.allgather(len(fired))
has = TR is not None and "nu" in TR
nus = comm.allgather(np.asarray(TR["nu"], float).tolist() if has else None)
if comm.Get_rank() == 0:
    assert len(set(counts)) == 1, "ranks fired the block different numbers of times: %r" % counts
    assert counts[0] == -(-NIT // EVERY), "expected %d calls, got %d" % (-(-NIT // EVERY), counts[0])
    seen = [x for x in nus if x is not None]
    assert seen and all(x == seen[0] for x in seen), "nu trace differs across ranks"
    moves = int(np.count_nonzero(np.diff(np.asarray(seen[0], float))))
    assert moves <= counts[0], "nu moved %d times but the block fired %d" % (moves, counts[0])
    open(sys.argv[1], "w").write("ok %d" % counts[0])
'''


def test_nu_rho_nuts_every_is_lockstep_under_mpi(tmp_path):
    """The amortized schedule must fire on the SAME sweeps on every rank.

    The block Allreduces its gradient across ranks, so a rank that skipped a sweep
    while the others called would deadlock (or, worse, pair up mismatched partial
    sums). The counter lives on each rank's own state, so nothing enforces this but
    the fact that run_global_updates is called once per sweep everywhere -- which is
    exactly what this checks, end to end under a real launcher."""
    import subprocess
    import pytest
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_sampler_golden import _find_mpi_launcher

    launcher, bindir = _find_mpi_launcher()
    if launcher is None:
        pytest.skip("no MPI launcher (mpiexec/mpirun) available")
    child = tmp_path / "_mpi_every_child.py"
    child.write_text(_MPI_EVERY_CHILD)
    flag = str(tmp_path / "ok.txt")
    env = dict(os.environ)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                         + os.pathsep + env.get("PYTHONPATH", ""))
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    try:
        subprocess.run([launcher, "-n", "2", sys.executable, str(child), flag],
                       check=True, env=env, cwd=str(tmp_path),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"MPI child failed: {e.stderr.decode()[-2000:]}")
    if not os.path.exists(flag):
        pytest.skip("launcher/mpi4py mismatch: ranks ran as independent size-1 jobs")


def test_update_rho_nuts_runs_and_keeps_state_consistent():
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState

    sc = sm.SimConfig(Ns=25, Nt=6, seed=4)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
                       u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
                       xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
                       knots_S=out["knots_S"], knots_phi=out["knots_phi"],
                       knots_rho=out["knots_rho"])
    st = SamplerSettings(nu=out["nu"], seed=1, rho_sampler="nuts",
                         nuts_warmup=5, n_iters=20)
    state = ModelState(ModelConfig(model_type="shi"), data, st)
    assert state.k_rho > 1, "need a vector-rho (Shi) model to exercise the block"

    rng = np.random.default_rng(0)
    ctx = mu.Ctx(range(state.Nt), rng, reduce=None, supports_compact=True,
                 covers_all_times=True)
    caches = mu.init_caches(state, ctx.times)

    rho_before = state.rho_k.copy()
    for _ in range(15):
        mu.update_rho_nuts(state, ctx, caches, st)
        assert np.all(state.rho_k > 0)
        assert np.all(np.isfinite(state.rho_k))

    # the chain moved off its start
    assert not np.array_equal(state.rho_k, rho_before)
    # Zlp cache is consistent with the committed K (the block must refresh it)
    for t in ctx.times:
        assert np.isclose(caches.Zlp[t], state.Z_logprior(state.Z[:, t]))
    # Yll (data likelihood) is untouched by this CP move
    for t in ctx.times:
        assert np.isfinite(caches.Yll[t])


def test_fast_nu_rho_gradient_matches_legacy():
    """The fast joint gradient must be a PURE performance change.

    It removes redundant work (one-pass covariance build, Bessel on unique pairs,
    streamed order-channel quadrature) but must not change the target, so it is held
    to the legacy implementation -- which is the finite-difference-validated one --
    at several points along a plausible trajectory."""
    rng = np.random.default_rng(3)
    Ns, Nt, k_rho = 40, 9, 4
    coords = rng.uniform(0, 5, size=(Ns, 2))
    dc = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dc["d1_sq"] + dc["d2_sq"])
    W = rng.uniform(0.1, 1.0, size=(Ns, k_rho))
    Z = rng.standard_normal((Ns, Nt))
    iu = np.triu_indices(Ns, k=1)
    nu_loc, nu_scale, rho_scale = 0.5, 2.5, 10.0

    for z_nu in (-1.5, -0.4, 0.0, 0.7, 1.6):
        th = np.concatenate(([z_nu], np.log(rng.uniform(0.6, 2.5, size=k_rho))))
        l0, g0 = mu._nu_rho_cp_value_grad(th, W, dc, Dmat, Z, Nt, rho_scale,
                                          nu_loc, nu_scale, iu)
        l1, g1 = mu._nu_rho_cp_value_grad_fast(th, W, Dmat, Z, Nt, rho_scale,
                                               nu_loc, nu_scale, iu)
        assert abs(l1 - l0) <= 1e-8 * max(1.0, abs(l0)), f"logp drift at z_nu={z_nu}"
        assert np.max(np.abs(g1 - g0)) <= 1e-6 * max(1.0, np.max(np.abs(g0))), \
            f"gradient drift at z_nu={z_nu}"


def test_order_channel_chunking_is_invariant():
    """The streamed contraction must not depend on the block size it is streamed in
    -- that is the whole basis for calling it the same arithmetic."""
    rng = np.random.default_rng(5)
    n = 5000
    x = rng.uniform(0.05, 12.0, size=n)
    nu = 1.4
    c = 2.0 ** (1 - nu) / __import__("scipy.special", fromlist=["gamma"]).gamma(nu)
    xnu = x ** nu
    Mu = c * xnu * __import__("scipy.special", fromlist=["kv"]).kv(nu, x)
    w = rng.standard_normal(n)
    vals = [mu._order_channel(nu, x, xnu, Mu, w, 128, -8.0, 3.0, c, chunk=ch)
            for ch in (256, 1024, 4096, n, 2 * n)]
    assert max(abs(v - vals[0]) for v in vals) <= 1e-9 * abs(vals[0])


def test_order_channel_is_ill_conditioned():
    """Guard the reason the quadrature grid must not be economised: this contraction
    cancels hard, so dM error is amplified. If a future change makes the contraction
    well-conditioned, the comment on _NU_RHO_QUAD_DEFAULT needs revisiting."""
    rng = np.random.default_rng(5)
    Ns, Nt = 60, 8
    coords = rng.uniform(0, 5, size=(Ns, 2))
    dc = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dc["d1_sq"] + dc["d2_sq"])
    Z = rng.standard_normal((Ns, Nt))
    iu = np.triu_indices(Ns, k=1)
    core = mu._cp_gaussian_core_fast(1.6, np.full(Ns, 2.5), Dmat, Z, Nt, iu)
    assert core is not None
    w = (core["Gm"] * core["S"])[iu[0], iu[1]]
    import bessel_nu
    terms = w * bessel_nu.dmatern_dnu(1.6, core["x"], n=128, M=core["Mu"])
    cond = float(np.sum(np.abs(terms)) / abs(np.sum(terms)))
    assert cond > 5.0, f"contraction is better conditioned than documented ({cond:.1f})"


def _zblock_fixture(Ns=48, seed=2, upper=False):
    """A realistic single-column Z block: high censoring, a real Matern factor."""
    from scipy.linalg import cholesky
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 6, size=(Ns, 2))
    dc = u.ns_cov_precompute(coords)
    K = u.ns_cov_cached(np.full(Ns, 2.0 ** 2 / (8 * 1.2)), np.ones(Ns), dc,
                        kappa=1.2, cov_model="matern")
    L = cholesky(K + 1e-8 * np.eye(Ns), lower=not upper)
    eta = rng.standard_normal((Ns, 1))
    Z = L @ eta
    phi = np.full(Ns, 0.4)
    logR = rng.standard_normal((Ns, 1)) * 0.5
    logX = phi[:, None] * logR - mu._norm_logsf_np(Z) + rng.standard_normal((Ns, 1)) * .3
    cens = np.zeros((Ns, 1), dtype=bool)
    exc = np.zeros((Ns, 1), dtype=bool)
    idx = rng.permutation(Ns)
    cens[idx[:int(.85 * Ns)], 0] = True
    exc[idx[int(.85 * Ns):], 0] = True
    base = dict(L=L, logX=logX, logR=logR, phi=phi, alpha0=1.4, cens=cens, exc=exc)
    trmv = dict(base, Lf=np.asfortranarray(L), lower=not upper)
    return base, trmv, eta


def test_zblock_trmv_gradient_matches_numpy():
    """The triangular-BLAS Z gradient must be the same arithmetic, not an
    approximation -- checked at several points, for BOTH factor orientations, since
    VisGP's factor is UPPER triangular and a wrong `lower` flag returns a plausible
    but wrong gradient rather than raising."""
    for upper in (False, True):
        base, trmv, eta = _zblock_fixture(upper=upper)
        rng = np.random.default_rng(9)
        for _ in range(5):
            e = eta + 0.4 * rng.standard_normal(eta.shape)
            l0, g0 = mu.zblock_value_grad_np(base, e)
            l1, g1 = mu.zblock_value_grad_trmv(trmv, e)
            assert abs(l1 - l0) <= 1e-9 * max(1.0, abs(l0)), f"logp, upper={upper}"
            assert np.max(np.abs(g1 - g0)) <= 1e-9 * max(1.0, np.max(np.abs(g0))), \
                f"grad, upper={upper}"


def test_zblock_trmv_wrong_triangle_is_detectably_wrong():
    """Guard the flag: if `lower` is set from the wrong end, the result must NOT
    silently agree -- this is the failure mode the VisGP path would hit."""
    base, trmv, eta = _zblock_fixture(upper=False)
    bad = dict(trmv, lower=False)
    _, g_ok = mu.zblock_value_grad_np(base, eta)
    _, g_bad = mu.zblock_value_grad_trmv(bad, eta)
    assert np.max(np.abs(g_bad - g_ok)) > 1e-6 * np.max(np.abs(g_ok))


def test_order_channel_warns_only_when_the_grid_is_inadequate():
    """The cancellation guard must stay quiet where the model is actually fitted and
    fire where the quadrature grid is known to fail. Faroe sits at nu~1.28, rho~3.1
    (contracted error 2e-3); the grid breaks down near nu~2 with long ranges."""
    import warnings
    Ns, Nt = 200, 12
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 10, size=(Ns, 2))
    dc = u.ns_cov_precompute(coords)
    Dmat = np.sqrt(dc["d1_sq"] + dc["d2_sq"])
    Z = rng.standard_normal((Ns, Nt))
    iu = np.triu_indices(Ns, k=1)

    def fired(nu, rho):
        mu.reset_order_cond_monitor()
        core = mu._cp_gaussian_core_fast(nu, np.full(Ns, rho), Dmat, Z, 500, iu)
        assert core is not None
        w = (core["Gm"] * core["S"])[iu[0], iu[1]]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            mu._order_channel(nu, core["x"], core["xnu"], core["Mu"], w,
                              128, -8.0, 3.0, core["c"])
        return len(rec) > 0

    assert not fired(1.28, 3.09), "warned at the (nu, rho) Faroe is actually fitted at"
    assert fired(1.95, 6.0), "did not warn where the grid is known to be inadequate"
    mu.reset_order_cond_monitor()


def test_order_channel_guard_does_not_change_the_value():
    """Adding the cancellation monitor must not perturb the contraction it monitors."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(0.05, 12.0, size=n)
    nu = 1.3
    from scipy.special import gamma as _g, kv as _k
    c = 2.0 ** (1 - nu) / _g(nu)
    xnu = x ** nu
    Mu = c * xnu * _k(nu, x)
    w = rng.standard_normal(n)
    import bessel_nu
    dM = bessel_nu.dmatern_dnu(nu, x, n=128, M=Mu)
    assert abs(mu._order_channel(nu, x, xnu, Mu, w, 128, -8.0, 3.0, c)
               - float(np.sum(w * dM))) <= 1e-9 * abs(float(np.sum(w * dM)))
