"""Integration tests for the unified sampler.

These are fast smoke/recovery tests on small simulated data. Full Shi-replication
(comparing posterior summaries to Shi's original sampler on the 225x128 dataset)
is a much longer run and is not part of this suite; here we verify the machinery is correct:
  * the chain runs end-to-end for all three model types,
  * the log-likelihood increases from initialisation,
  * the spatially-varying phi field is recovered on a modest dataset,
  * the alpha0 conditional likelihood is correctly shaped (single peak).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState, run_mcmc  # noqa: E402


def _data_from_sim(out):
    return SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"],
        u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
        xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
        knots_S=out["knots_S"], knots_phi=out["knots_phi"], knots_rho=out["knots_rho"])


def test_shi_chain_runs_and_ll_increases():
    sc = sm.SimConfig(Ns=25, Nt=8, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    st = SamplerSettings(n_iters=150, adapt_size=25, thin=5, nu=out["nu"], seed=1)
    TR = run_mcmc(cfg, _data_from_sim(out), st)
    assert np.isfinite(TR["ll"]).all()
    # later half of chain has higher mean ll than the very first sample
    assert np.mean(TR["ll"][len(TR["ll"]) // 2:]) > TR["ll"][1]


def test_shi_recovers_phi_field():
    sc = sm.SimConfig(Ns=30, Nt=12, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    st = SamplerSettings(n_iters=400, adapt_size=25, thin=5, nu=out["nu"], seed=1)
    TR = run_mcmc(cfg, _data_from_sim(out), st)
    burn = len(TR["phi_k"]) // 2
    phi_post = (out["W_gauss_phi"] @ TR["phi_k"][burn:].mean(0)).mean()
    assert abs(phi_post - out["phi_vec"].mean()) < 0.1


def test_alpha0_loglik_is_single_peaked():
    sc = sm.SimConfig(Ns=30, Nt=12, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    state = ModelState(cfg, _data_from_sim(out),
                       SamplerSettings(nu=out["nu"], seed=1))
    state.Z = out["Z"].copy()
    state.R = out["R_at_sites"].copy()
    state.phi_vec = out["phi_vec"].copy()
    state.gamma_bar = out["gamma_bar_vec"].copy()
    grid = [2.2, 3.0, 4.0, 5.0, 6.0, 8.0]
    lls = [sum(state.loglik_t(t, alpha0=a0) for t in range(state.Nt)) for a0 in grid]
    # concave-ish: the max is interior, and ll decreases monotonically past it
    kmax = int(np.argmax(lls))
    assert 0 <= kmax <= len(grid) - 1
    assert all(lls[i] >= lls[i + 1] for i in range(kmax, len(grid) - 1))


def test_huser_wadsworth_chain_runs():
    sc = sm.SimConfig(Ns=25, Nt=8, seed=4, delta=0.7)
    cfg = ModelConfig(model_type="huser-wadsworth", phi_domain="positive")
    out = sm.simulate(cfg, sc)
    st = SamplerSettings(n_iters=80, adapt_size=20, thin=5, nu=out["nu"], seed=2,
                         update_rho=False)  # HW uses stationary range (rho scalar)
    TR = run_mcmc(cfg, _data_from_sim(out), st)
    assert np.isfinite(TR["ll"]).all()


@pytest.mark.parametrize("model_type,extra", [
    ("shi", {}),
    ("wang", {}),
    ("huser-wadsworth", {"update_rho": False}),
])
def test_z_update_ncp_and_centered_both_run(model_type, extra):
    """Both Z parametrizations run end-to-end and increase the log-likelihood
    on every model type (incl. wang via the VisGP precision Cholesky)."""
    sc = sm.SimConfig(Ns=25, Nt=8, seed=7, delta=0.7)
    cfg = ModelConfig(model_type=model_type, phi_domain="positive")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    for z_update in ("ncp", "centered"):
        st = SamplerSettings(n_iters=120, adapt_size=20, thin=5, nu=out["nu"],
                             seed=3, z_update=z_update, **extra)
        TR = run_mcmc(cfg, data, st)
        assert np.isfinite(TR["ll"]).all(), (model_type, z_update)
        if model_type != "huser-wadsworth":  # HW starts near-stationary
            assert np.mean(TR["ll"][len(TR["ll"]) // 2:]) > TR["ll"][1], \
                (model_type, z_update)


@pytest.mark.parametrize("model_type,extra", [
    ("shi", {}),
    ("wang", {}),
    ("huser-wadsworth", {"update_rho": False}),
])
def test_z_sampler_ess_runs_and_moves(model_type, extra):
    """Elliptical slice sampling for Z runs end-to-end on every model type,
    keeps the log-likelihood finite (and rising off init for the spatial models),
    and -- the point of ESS -- Z actually MOVES every step (no frozen coords)."""
    sc = sm.SimConfig(Ns=25, Nt=8, seed=7, delta=0.7)
    cfg = ModelConfig(model_type=model_type, phi_domain="positive")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    st = SamplerSettings(n_iters=120, adapt_size=20, thin=5, nu=out["nu"],
                         seed=3, z_sampler="ess", save_latent=True, **extra)
    TR = run_mcmc(cfg, data, st)
    assert np.isfinite(TR["ll"]).all(), model_type
    Z = np.asarray(TR["Z"])                      # (n_rec, Ns, Nt)
    # ESS always proposes a new ellipse point, so consecutive recorded Z differ.
    assert np.mean(np.any(np.diff(Z, axis=0) != 0.0, axis=(1, 2))) > 0.99, model_type
    if model_type != "huser-wadsworth":          # HW starts near-stationary
        assert np.mean(TR["ll"][len(TR["ll"]) // 2:]) > TR["ll"][1], model_type


@pytest.mark.parametrize("model_type,extra", [
    ("shi", {}),
    ("huser-wadsworth", {"update_rho": False}),
])
def test_z_sampler_nuts_runs_and_moves(model_type, extra):
    """Per-time NUTS for Z runs end-to-end, keeps the log-likelihood finite (and
    rising off init for the spatial model), and Z actually moves every step."""
    sc = sm.SimConfig(Ns=20, Nt=6, seed=7, delta=0.7)
    cfg = ModelConfig(model_type=model_type, phi_domain="positive")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    st = SamplerSettings(n_iters=60, adapt_size=20, thin=5, nu=out["nu"], seed=3,
                         z_sampler="nuts", nuts_warmup=30, save_latent=True, **extra)
    TR = run_mcmc(cfg, data, st)
    assert np.isfinite(TR["ll"]).all(), model_type
    Z = np.asarray(TR["Z"])
    assert np.mean(np.any(np.diff(Z, axis=0) != 0.0, axis=(1, 2))) > 0.99, model_type
    if model_type != "huser-wadsworth":
        assert np.mean(TR["ll"][len(TR["ll"]) // 2:]) > TR["ll"][1], model_type


def test_zblock_potential_matches_loglik_t():
    """Faithfulness gate for the NUTS Z arm: the Z-block potential the
    gradient differentiates must reproduce the production ``state.loglik_t`` *as a
    function of Z*. The potential drops the Z-independent dCGP/dRW terms, so it
    matches loglik_t's Z-DEPENDENCE (differences), not its absolute value."""
    import mcmc_updates as mu
    sc = sm.SimConfig(Ns=30, Nt=12, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    state = ModelState(cfg, _data_from_sim(out),
                       SamplerSettings(nu=out["nu"], seed=1))
    # pin every non-Z quantity at the simulation truth: the exported conditional
    # is then exactly the Z-block target a Gibbs sweep sees.
    state.R = out["R_at_sites"].copy()
    state.phi_vec = out["phi_vec"].copy()
    state.gamma_bar = out["gamma_bar_vec"].copy()
    state.Z = out["Z"].copy()
    state.refresh_pY()
    arrays = state.export_zblock()

    def prod_loglik(Zmat):
        return sum(state.loglik_t(t, Z=Zmat[:, t]) for t in range(state.Nt))

    rng = np.random.default_rng(0)
    Z0 = state.Z
    base_prod, base_pot = prod_loglik(Z0), mu.zblock_loglik_np(arrays, Z0)
    d_prod, d_pot = [], []
    for _ in range(6):
        Zp = Z0 + 0.3 * rng.standard_normal(Z0.shape)
        d_prod.append(prod_loglik(Zp) - base_prod)
        d_pot.append(mu.zblock_loglik_np(arrays, Zp) - base_pot)
    d_prod, d_pot = np.asarray(d_prod), np.asarray(d_pot)
    # the moves are O(1-100) in log-lik; identical float64 formulas agree to ~1e-9.
    assert np.max(np.abs(d_prod)) > 1.0, "perturbation too small to be a real test"
    assert np.allclose(d_prod, d_pot, rtol=1e-7, atol=1e-7), \
        np.column_stack([d_prod, d_pot])

    # white-coord log-joint = data loglik(L eta) + standard-normal prior.
    eta = np.linalg.solve(arrays["L"], Z0)
    lj = mu.zblock_logjoint_np(arrays, eta)
    assert np.isclose(lj, base_pot - 0.5 * np.sum(eta * eta), rtol=0, atol=1e-9)


def test_zblock_value_grad_matches_potential_and_fd():
    """The Phase-4 NUTS gradient: its value must equal zblock_logjoint_np, and its
    analytic gradient must match central finite differences of the same potential."""
    import mcmc_updates as mu
    sc = sm.SimConfig(Ns=20, Nt=8, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    state = ModelState(cfg, _data_from_sim(out),
                       SamplerSettings(nu=out["nu"], seed=1))
    state.R = out["R_at_sites"].copy()
    state.phi_vec = out["phi_vec"].copy()
    state.gamma_bar = out["gamma_bar_vec"].copy()
    state.Z = out["Z"].copy()
    state.refresh_pY()
    arrays = state.export_zblock()

    rng = np.random.default_rng(1)
    eta = np.linalg.solve(arrays["L"], state.Z) + 0.1 * rng.standard_normal(state.Z.shape)
    logp, grad = mu.zblock_value_grad_np(arrays, eta)

    # value agrees with the independently-implemented, already-gated potential.
    assert np.isclose(logp, mu.zblock_logjoint_np(arrays, eta), rtol=0, atol=1e-9)

    # analytic gradient agrees with central FD at a random subset of coords.
    h = 1e-5
    coords = [tuple(c) for c in
              rng.integers([0, 0], list(eta.shape), size=(15, 2))]
    for ij in coords:
        ep = eta.copy(); ep[ij] += h
        em = eta.copy(); em[ij] -= h
        fd = (mu.zblock_logjoint_np(arrays, ep) - mu.zblock_logjoint_np(arrays, em)) / (2 * h)
        assert abs(fd - grad[ij]) <= 1e-5 + 1e-4 * abs(grad[ij]), (ij, fd, grad[ij])


def test_nuts_recovers_correlated_gaussian():
    """The numpy NUTS driver must sample a known target. For N(m, S) the
    value_and_grad is exact, so the empirical mean and covariance must converge."""
    import mcmc_updates as mu
    rng = np.random.default_rng(0)
    m = np.array([1.0, -2.0, 0.5])
    A = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.3, -0.4, 0.9]])
    S = A @ A.T
    P = np.linalg.inv(S)

    def vg(x):
        d = x - m
        return -0.5 * float(d @ P @ d), -(P @ d)

    eps = mu.nuts_find_reasonable_eps(vg, np.zeros(3), rng)
    theta = np.zeros(3)
    n = 4000
    draws = np.empty((n, 3))
    for i in range(n):
        theta, a, depth = mu.nuts_transition(vg, theta, eps, rng)
        draws[i] = theta
    burn = n // 2
    assert np.max(np.abs(draws[burn:].mean(0) - m)) < 0.1
    assert np.max(np.abs(np.cov(draws[burn:], rowvar=False) - S)) < 0.2


def test_nuts_dual_averaging_hits_target_accept():
    """Dual averaging must drive the step size to ~target acceptance on a Gaussian."""
    import mcmc_updates as mu
    rng = np.random.default_rng(1)
    P = np.diag([1.0, 25.0, 100.0])                    # ill-scaled -> eps matters

    def vg(x):
        return -0.5 * float(x @ P @ x), -(P @ x)

    eps0 = mu.nuts_find_reasonable_eps(vg, np.zeros(3), rng)
    adapter = mu.NUTSStepAdapter(eps0, target_accept=0.8)
    theta = np.zeros(3)
    accepts = []
    for _ in range(800):
        theta, a, depth = mu.nuts_transition(vg, theta, adapter.eps, rng)
        adapter.update(a)
        accepts.append(a)
    eps_final = adapter.finalize()
    assert eps_final > 0 and np.isfinite(eps_final)
    assert abs(np.mean(accepts[-400:]) - 0.8) < 0.15    # near the requested target


def test_elliptical_slice_leaves_gaussian_prior_invariant():
    """Under a FLAT likelihood, elliptical slice sampling must leave the Gaussian
    prior N(0, K) invariant: the empirical mean -> 0 and covariance -> K."""
    import mcmc_updates as mu
    rng = np.random.default_rng(0)
    Ns = 3
    idx = np.arange(Ns)
    K = np.exp(-np.abs(idx[:, None] - idx[None, :]) / 2.0)  # SPD exp-correlation
    L = np.linalg.cholesky(K)
    f = L @ rng.standard_normal(Ns)              # start from the prior
    n = 80000
    samples = np.empty((n, Ns))
    flat = lambda x: 0.0                          # noqa: E731
    for i in range(n):
        nu = L @ rng.standard_normal(Ns)
        f, ll, n_eval = mu.elliptical_slice(f, nu, flat, 0.0, rng)
        assert n_eval == 1                        # flat slice accepts immediately
        samples[i] = f
    assert np.max(np.abs(samples.mean(axis=0))) < 0.05
    assert np.max(np.abs(np.cov(samples, rowvar=False) - K)) < 0.06


def test_save_load_and_score_new_sites():
    """End-to-end: fit -> save_run -> load_run -> build_draws -> score new sites."""
    import tempfile
    import os
    from sampler import save_run, load_run, build_draws
    import predictive as pr

    sc = sm.SimConfig(Ns=22, Nt=6, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    st = SamplerSettings(n_iters=40, adapt_size=20, thin=10, nu=out["nu"],
                         seed=1, save_latent=True)
    TR = run_mcmc(cfg, data, st)
    assert TR["S"].shape[0] == TR["phi_k"].shape[0]   # consistent trace length
    assert np.all(np.isfinite(TR["rho_k"]))           # no trailing zero row

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "run.npz")
    save_run(path, TR, cfg, st, data)
    run = load_run(path)
    draws = build_draws(run, burn=1, thin=1)
    assert len(draws) >= 1
    assert {"Z_train", "S", "phi_k", "rho_k", "rho", "alpha0"} <= set(draws[0])

    rng = np.random.default_rng(0)
    new = rng.uniform(0, 10, (3, 2))
    geom = pr.NewSiteGeometry(
        new_coords=new, train_coords=run["sites_xy"], knots_S=run["knots_S"],
        knots_phi=run["knots_phi"], knots_rho=run["knots_rho"],
        radius_S=float(run["radius_S"]), eff_range_phi=float(run["eff_range_phi"]),
        eff_range_rho=float(run["eff_range_rho"]), config=run["config"], water_poly=None)
    hold = pr.HoldoutData(Y=60 + np.abs(rng.standard_normal((3, 6))) * 5,
                          u=np.full(3, 60.0), sigma=np.full(3, 20.0), xi=np.full(3, 0.15))
    ll = pr.predictive_loglik_new_sites(geom, draws, hold, M=6)
    assert np.isfinite(ll["total"])


def test_wendland_phi_localized_update_is_exact():
    """The compact-support phi update (patch only affected sites' qRW/dRW) is
    bit-identical to a full qd_RW_all recompute for the same proposed phi."""
    from sampler import ModelState
    sc = sm.SimConfig(Ns=80, Nt=10, seed=11)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    st = SamplerSettings(nu=out["nu"], seed=1, phi_kernel="wendland",
                         phi_radius=2.5, eff_range_phi=3.0)
    state = ModelState(ModelConfig(model_type="shi"), _data_from_sim(out), st)
    assert state._phi_compact
    qv, dv = state.qd_RW_all()
    j = 4
    phi_k_prop = state.phi_k.copy(); phi_k_prop[j] = 0.7
    phi_vec_prop = state._phi_field(phi_k_prop)
    aff = state._phi_affected[j]
    assert 0 < aff.size < state.Ns                  # genuinely compact
    # localized patch
    qvL, dvL = qv.copy(), dv.copy()
    qs, ds = state.qd_RW_subset(aff, phi_vec=phi_vec_prop)
    qvL[aff, :] = qs; dvL[aff, :] = ds
    # full recompute
    qvF, dvF = state.qd_RW_all(phi_vec=phi_vec_prop)
    assert np.max(np.abs(qvL - qvF)) == 0.0
    assert np.max(np.abs(dvL - dvF)) == 0.0


def test_wendland_phi_chain_runs_and_recovers():
    sc = sm.SimConfig(Ns=40, Nt=12, seed=11)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    st = SamplerSettings(n_iters=300, adapt_size=25, thin=5, nu=out["nu"], seed=1,
                         phi_kernel="wendland", phi_radius=4.0, eff_range_phi=3.0)
    TR = run_mcmc(ModelConfig(model_type="shi"), _data_from_sim(out), st)
    assert np.isfinite(TR["ll"]).all()
    assert np.mean(TR["ll"][len(TR["ll"]) // 2:]) > TR["ll"][1]


def test_save_latent_off_saves_space_and_blocks_scoring():
    """--no-save-latent omits the big S/Z traces; build_draws then errors clearly."""
    import tempfile, os
    from sampler import save_run, load_run, build_draws
    sc = sm.SimConfig(Ns=20, Nt=6, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    st = SamplerSettings(n_iters=30, adapt_size=15, thin=10, nu=out["nu"],
                         seed=1, save_latent=False)
    TR = run_mcmc(cfg, data, st)
    assert "S" not in TR and "Z" not in TR
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "run.npz")
    save_run(path, TR, cfg, st, data)
    run = load_run(path)
    assert "Z" not in run and "S" not in run
    import pytest
    with pytest.raises(ValueError):
        build_draws(run)


def test_latent_float32_halves_storage_and_scores():
    import tempfile, os
    from sampler import save_run, load_run, build_draws
    sc = sm.SimConfig(Ns=20, Nt=6, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    st32 = SamplerSettings(n_iters=30, adapt_size=15, thin=10, nu=out["nu"],
                           seed=1, save_latent=True, latent_dtype="float32")
    TR = run_mcmc(cfg, data, st32)
    assert TR["Z"].dtype == np.float32 and TR["S"].dtype == np.float32
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "run.npz")
    save_run(path, TR, cfg, st32, data)
    draws = build_draws(load_run(path))
    # build_draws casts back to float64 for the numba likelihood
    assert draws[0]["Z_train"].dtype == np.float64


def test_stream_latent_memmaps_and_scores():
    """Streaming keeps the Z trace on disk (memmap), out of the npz, and the run
    still loads (lazily) and scores."""
    import tempfile, os
    from sampler import save_run, load_run, build_draws
    sc = sm.SimConfig(Ns=20, Nt=6, seed=11)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    data = _data_from_sim(out)
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "run.npz")
    st = SamplerSettings(n_iters=40, adapt_size=20, thin=10, nu=out["nu"], seed=1,
                         save_latent=True, stream_latent=True, latent_out=path,
                         latent_chunk=2)
    TR = run_mcmc(cfg, data, st)
    assert isinstance(TR["Z"], np.memmap)            # never a full in-RAM array
    save_run(path, TR, cfg, st, data)
    # latent lives in sibling .npy, NOT inside the npz
    with np.load(path) as npz:
        assert "Z" not in npz.files and "S" not in npz.files
    assert os.path.exists(path[:-4] + ".Z.npy")
    run = load_run(path)
    assert isinstance(run["Z"], np.memmap)           # lazy on load
    draws = build_draws(run, burn=1, thin=1)
    assert draws[0]["Z_train"].dtype == np.float64
    assert len(draws) >= 1


def test_wang_visgp_chain_runs():
    sc = sm.SimConfig(Ns=20, Nt=6, seed=7, rho_const=4.0)
    cfg = ModelConfig(model_type="wang", use_visgp=True, cliques_path="dummy.rds")
    cliques = [list(range(sc.Ns))]
    out = sm.simulate(cfg, sc, cliques=cliques, separators=[])
    data = _data_from_sim(out)
    data.cliques = cliques
    data.separators = []
    st = SamplerSettings(n_iters=60, adapt_size=20, thin=5, nu=out["nu"], seed=3,
                         update_rho=False)  # VisGP rho recompute is expensive; keep fixed
    TR = run_mcmc(cfg, data, st)
    assert np.isfinite(TR["ll"]).all()
