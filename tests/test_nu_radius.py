"""Tests for the nu (Matern smoothness) and radius_S (Wendland support) updates."""
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
import covariance as cov  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState, run_mcmc  # noqa: E402


def _data(seed=11, Ns=40, Nt=12):
    out = sm.simulate(ModelConfig("shi"), sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"])
    return out, data


def test_nu_loglik_identified_at_truth():
    """Z-logprior is maximised at the true nu (clean sweep, rho fixed)."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (40, 2)); D = cov.dist_matrix(coords)
    K = cov.matern_wang(D, 3.0, 1.0, 1.0)
    Z = rng.multivariate_normal(np.zeros(40), K, size=300).T
    grid = [0.5, 0.8, 1.0, 1.3, 1.8, 2.5]
    lls = [scipy.stats.multivariate_normal.logpdf(
        Z.T, mean=np.zeros(40), cov=cov.matern_wang(D, 3.0, 1.0, nu),
        allow_singular=True).sum() for nu in grid]
    assert grid[int(np.argmax(lls))] == 1.0


def test_wendland_rebuild_matches_at_same_radius():
    out, data = _data()
    state = ModelState(ModelConfig("shi"), data, SamplerSettings(nu=out["nu"], radius_S=3.0))
    assert np.allclose(state.wendland_for_radius(3.0), state.W_wend_S)
    assert state._min_radius_S > 0


def test_nu_chain_runs_in_bounds():
    out, data = _data()
    st = SamplerSettings(n_iters=200, adapt_size=25, thin=5, nu=1.0, seed=1,
                         update_nu=True)
    TR = run_mcmc(ModelConfig("shi"), data, st)
    assert "nu" in TR and TR["nu"].shape == (len(range(0, 200, 5)),)
    assert np.all((TR["nu"] >= 0.5) & (TR["nu"] <= 3.0))
    assert np.all(np.isfinite(TR["ll"]))


def test_radius_S_chain_stays_covering():
    out, data = _data()
    state0 = ModelState(ModelConfig("shi"), data, SamplerSettings(nu=out["nu"]))
    st = SamplerSettings(n_iters=200, adapt_size=25, thin=5, nu=out["nu"], seed=1,
                         radius_S=3.0, update_radius_S=True)
    TR = run_mcmc(ModelConfig("shi"), data, st)
    assert "radius_S" in TR
    assert np.all(TR["radius_S"] > state0._min_radius_S)   # never loses coverage
    assert np.all(np.isfinite(TR["ll"]))


def test_nu_and_rho_update_for_huser_wadsworth():
    """nu (Matern smoothness) and rho (stationary range) are updatable for HW."""
    cfg = ModelConfig("huser-wadsworth")
    out = sm.simulate(cfg, sm.SimConfig(Ns=40, Nt=12, seed=4, delta=0.7, rho_const=4.0))
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"])
    st = SamplerSettings(n_iters=200, adapt_size=25, thin=5, nu=1.0, seed=1,
                         update_nu=True, update_rho=True)
    TR = run_mcmc(cfg, data, st)
    assert "nu" in TR
    assert TR["nu"].std() > 1e-4                       # nu actually moves for HW
    assert np.all((TR["nu"] >= 0.5) & (TR["nu"] <= 3.0))
    assert TR["rho_k"].std() > 1e-4                    # HW stationary range moves
    assert np.all(np.isfinite(TR["ll"]))


def test_anisotropy_updates_shi_and_wang():
    """aniso_ratio/angle are sampled for shi and wang; ratio stays in (0,1)."""
    for mt in ("shi", "wang"):
        if mt == "wang":
            cfg = ModelConfig(model_type="wang", use_anisotropy=True,
                              use_visgp=True, cliques_path="x")
            sc = sm.SimConfig(Ns=24, Nt=6, seed=3)
            cl = [list(range(sc.Ns))]
            o = sm.simulate(cfg, sc, cliques=cl, separators=[])
            extra = dict(cliques=cl, separators=[])
        else:
            cfg = ModelConfig(model_type="shi", use_anisotropy=True)
            sc = sm.SimConfig(Ns=24, Nt=6, seed=3)
            o = sm.simulate(cfg, sc)
            extra = {}
        data = SamplerData(
            Y=o["Y_missing"], sites_xy=o["sites_xy"], u_vec=o["u_mat"][:, 0],
            sigma_vec=o["sigma_mat"][:, 0], xi_vec=o["xi_mat"][:, 0], p_cens=o["p_cens"],
            knots_S=o["knots_S"], knots_phi=o["knots_phi"], knots_rho=o["knots_rho"], **extra)
        st = SamplerSettings(n_iters=120, adapt_size=20, thin=5, nu=o["nu"], seed=1,
                             update_anisotropy=True, update_rho=(mt == "shi"))
        TR = run_mcmc(cfg, data, st)
        ar = TR["aniso_ratio"]
        assert ar.std() > 1e-4 and np.all((ar > 0) & (ar < 1))
        assert np.all((TR["aniso_angle"] >= 0) & (TR["aniso_angle"] < np.pi))
        assert np.all(np.isfinite(TR["ll"]))


def test_geometry_slack_jacobian():
    """The joint-geometry proposal samples u = (log(r-r_min), logit(ratio), angle);
    its change-of-variables Jacobian must be |J| = (r-r_min)*ratio*(1-ratio) -- a
    wrong Jacobian silently biases the geometry marginals. Validate the analytic
    formula against a finite-difference Jacobian of the forward map at points where
    r_min genuinely depends on (ratio, angle) (the off-diagonal terms that must drop
    out of the determinant)."""
    from scipy.special import expit, logit
    out, data = _data()
    st = ModelState(ModelConfig("shi", use_anisotropy=True), data,
                    SamplerSettings(nu=out["nu"]))

    def fwd(u):
        rho = expit(u[1]); th = u[2] % np.pi
        return np.array([st.covering_radius(rho, th) + np.exp(u[0]), rho, th])

    rng = np.random.default_rng(0)
    for _ in range(5):
        rho = rng.uniform(0.3, 0.8); th = rng.uniform(0.3, 2.8)
        slack = rng.uniform(0.05, 0.5)
        u = np.array([np.log(slack), logit(rho), th])
        J = np.zeros((3, 3)); h = 1e-6
        for j in range(3):
            up = u.copy(); um = u.copy(); up[j] += h; um[j] -= h
            J[:, j] = (fwd(up) - fwd(um)) / (2 * h)
        analytic = slack * rho * (1 - rho)
        assert abs(abs(np.linalg.det(J)) - analytic) / analytic < 1e-4


def test_joint_geometry_updates_shi_and_wang():
    """joint_geometry drives radius_S/ratio/angle in one block: all three move and
    stay in-bounds (3-D, anisotropy on), and the 1-D radius-only case still mixes."""
    for mt in ("shi", "wang"):
        if mt == "wang":
            cfg = ModelConfig(model_type="wang", use_anisotropy=True,
                              use_visgp=True, cliques_path="x")
            sc = sm.SimConfig(Ns=24, Nt=6, seed=3)
            cl = [list(range(sc.Ns))]
            o = sm.simulate(cfg, sc, cliques=cl, separators=[])
            extra = dict(cliques=cl, separators=[])
        else:
            cfg = ModelConfig(model_type="shi", use_anisotropy=True)
            sc = sm.SimConfig(Ns=24, Nt=6, seed=3)
            o = sm.simulate(cfg, sc)
            extra = {}
        data = SamplerData(
            Y=o["Y_missing"], sites_xy=o["sites_xy"], u_vec=o["u_mat"][:, 0],
            sigma_vec=o["sigma_mat"][:, 0], xi_vec=o["xi_mat"][:, 0], p_cens=o["p_cens"],
            knots_S=o["knots_S"], knots_phi=o["knots_phi"], knots_rho=o["knots_rho"], **extra)
        st = SamplerSettings(n_iters=150, adapt_size=20, thin=5, nu=o["nu"], seed=1,
                             update_radius_S=True, update_anisotropy=True,
                             joint_geometry=True, update_rho=(mt == "shi"))
        TR = run_mcmc(cfg, data, st)
        rad, ar, ang = TR["radius_S"], TR["aniso_ratio"], TR["aniso_angle"]
        assert np.all(np.isfinite(TR["ll"]))
        assert np.all((ar > 0) & (ar < 1)) and np.all((ang >= 0) & (ang < np.pi))
        assert rad.std() > 1e-5 and ar.std() > 1e-5 and ang.std() > 1e-5

    # 1-D path: radius_S only (anisotropy off) -> the joint block is just radius.
    out, data = _data()
    st = SamplerSettings(n_iters=150, adapt_size=20, thin=5, nu=out["nu"], seed=1,
                         radius_S=3.0, update_radius_S=True, joint_geometry=True)
    TR = run_mcmc(ModelConfig("shi"), data, st)
    assert np.all(np.isfinite(TR["ll"])) and TR["radius_S"].std() > 1e-5
    assert "aniso_ratio" not in TR


def test_nu_radius_persist_and_per_draw():
    """nu/radius_S traces persist and build_draws yields per-draw values."""
    import tempfile
    from sampler import save_run, load_run, build_draws
    out, data = _data()
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "run.npz")
    st = SamplerSettings(n_iters=120, adapt_size=20, thin=10, nu=1.0, radius_S=3.0,
                         seed=1, update_nu=True, update_radius_S=True, save_latent=True)
    save_run(path, run_mcmc(ModelConfig("shi"), data, st), ModelConfig("shi"), st, data)
    run = load_run(path)
    assert "nu_trace" in run and "radius_S_trace" in run
    draws = build_draws(run, burn=1)
    nus = [d["nu"] for d in draws]; rads = [d["radius_S"] for d in draws]
    assert len(set(np.round(nus, 6))) > 1 or len(set(np.round(rads, 6))) > 1  # vary
