"""Tests for predictive scoring (Phase 8)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predictive as pr  # noqa: E402
import covariance as cov  # noqa: E402


def _make_draws_and_holdout(B=4, n_obs=8, n_h=3, T=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_obs + n_h
    coords = rng.uniform(0, 10, (n, 2))
    K = cov.matern_wang(cov.dist_matrix(coords), 3.0, 1.0, 1.0)
    obs_idx = list(range(n_obs))
    holdout_idx = list(range(n_obs, n))

    Z_full = rng.multivariate_normal(np.zeros(n), K, size=T).T  # (n, T)
    # holdout responses: just exceedances above threshold for a concrete test
    Y_h = 60.0 + np.abs(rng.standard_normal((n_h, T))) * 5.0
    holdout = pr.HoldoutData(Y=Y_h, u=np.full(n_h, 60.0),
                             sigma=np.full(n_h, 20.0), xi=np.full(n_h, 0.1))

    draws = []
    for b in range(B):
        draws.append(dict(
            Z_obs=Z_full[obs_idx] + 0.01 * rng.standard_normal((n_obs, T)),
            phi_holdout=np.full(n_h, 0.5),
            gamma_bar_holdout=np.full(n_h, 1.0),
            R_holdout=np.full((n_h, T), 2.0),
            alpha0=5.0,
        ))
    return draws, holdout, obs_idx, holdout_idx, K


def test_predictive_loglik_finite_shi():
    draws, holdout, obs_idx, holdout_idx, K = _make_draws_and_holdout()
    res = pr.predictive_loglik(
        draws, holdout, obs_idx=obs_idx, holdout_idx=holdout_idx,
        cov_for_draw=lambda d: K, use_visgp=False, M=10, seed=1)
    assert np.isfinite(res["total"])
    assert res["per_draw"].shape == (len(draws),)
    assert np.all(np.isfinite(res["per_draw"]))


def test_predictive_loglik_visgp_path():
    draws, holdout, obs_idx, holdout_idx, K = _make_draws_and_holdout(seed=2)
    Q = np.linalg.inv(K)
    res = pr.predictive_loglik(
        draws, holdout, obs_idx=obs_idx, holdout_idx=holdout_idx,
        cov_for_draw=lambda d: Q, use_visgp=True, M=10, seed=3)
    assert np.isfinite(res["total"])


def test_crps_against_known_value():
    # CRPS of samples vs an observation; compare to a brute-force estimate.
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 50000)
    y = 0.5
    crps = pr.crps_empirical(x, y)
    # brute force E|X-y| - 0.5 E|X-X'|
    e1 = np.mean(np.abs(x - y))
    xp = rng.normal(0, 1, 50000)
    e2 = np.mean(np.abs(x - xp))
    assert abs(crps - (e1 - 0.5 * e2)) < 1e-2


def _twcrps_integral(samples, y, w):
    """Reference: integral (F(z)-1{y<=z})^2 w(z) dz on a fine grid."""
    samples = np.sort(samples)
    lo = min(samples.min(), y) - 1.0
    hi = max(samples.max(), y) + 1.0
    z = np.linspace(lo, hi, 20000)
    F = np.searchsorted(samples, z, side="right") / samples.size
    ind = (y <= z).astype(float)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return trapz((F - ind) ** 2 * w(z), z)


def test_chaining_matches_threshold_weighted_integral():
    """The chaining-function CRPS equals the integral form for vanilla, hard,
    and smooth weights."""
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 4000)
    y = 2.3
    # vanilla: w=1
    assert abs(pr.weighted_crps_empirical(x, y, pr.chain_identity)
               - _twcrps_integral(x, y, lambda z: np.ones_like(z))) < 1e-2
    # hard threshold t=2.5
    t = 2.5
    assert abs(pr.weighted_crps_empirical(x, y, pr.chain_threshold(t))
               - _twcrps_integral(x, y, lambda z: (z > t).astype(float))) < 1e-2
    # smooth weight Phi((z-t)/s)
    import scipy.stats
    s = 0.4
    assert abs(pr.weighted_crps_empirical(x, y, pr.chain_smooth(t, s))
               - _twcrps_integral(x, y, lambda z: scipy.stats.norm.cdf((z - t) / s))) < 1e-2


def test_twcrps_le_vanilla_for_unit_weight():
    # weight in [0,1] -> tail-weighted CRPS <= vanilla CRPS
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    y = 0.7
    v = pr.weighted_crps_empirical(x, y, pr.chain_identity)
    tw_hard = pr.weighted_crps_empirical(x, y, pr.chain_threshold(0.5))
    tw_soft = pr.weighted_crps_empirical(x, y, pr.chain_smooth(0.5, 0.3))
    assert tw_hard <= v + 1e-9
    assert tw_soft <= v + 1e-9


def _newsite_setup(model_type="wang", seed=5):
    import simulate as sm
    from config import ModelConfig
    sc = sm.SimConfig(Ns=24, Nt=5, seed=seed, rho_const=4.0)
    if model_type == "wang":
        cfg = ModelConfig(model_type="wang", use_visgp=True, cliques_path="x.rds")
        cliques = [list(range(24))]
        out = sm.simulate(cfg, sc, cliques=cliques, separators=[])
    else:
        cfg = ModelConfig(model_type=model_type)
        cliques = None
        out = sm.simulate(cfg, sc)
    rng = np.random.default_rng(0)
    new = rng.uniform(0, 10, (3, 2))
    T = sc.Nt
    k_S = out["knots_S"].shape[0]; k_phi = out["knots_phi"].shape[0]
    k_rho = out["knots_rho"].shape[0]
    draws = [dict(Z_train=out["Z"] + 0.01 * rng.standard_normal(out["Z"].shape),
                  S=np.abs(rng.standard_normal((k_S, T))) + 0.5,
                  phi_k=np.full(k_phi, 0.5), gamma_k=np.ones(k_S),
                  rho_k=np.full(k_rho, 4.0), rho=4.0, nu=out["nu"],
                  alpha0=out["alpha0"]) for _ in range(3)]
    geom = pr.NewSiteGeometry(
        new_coords=new, train_coords=out["sites_xy"], knots_S=out["knots_S"],
        knots_phi=out["knots_phi"], knots_rho=out["knots_rho"], radius_S=3.0,
        eff_range_phi=3.0, eff_range_rho=3.0, config=cfg,
        cliques=cliques, separators=[], water_poly=None)
    return geom, draws, T


def test_predict_Y_new_sites_wang_and_shi():
    for mt in ("wang", "shi"):
        geom, draws, T = _newsite_setup(mt)
        Yp = pr.predict_Y_new_sites(geom, draws, u_new=np.full(3, 60.0),
                                    sigma_new=np.full(3, 20.0), xi_new=np.full(3, 0.15),
                                    M=6, seed=1)
        assert Yp.shape == (3, T, len(draws) * 6)
        assert np.all(np.isfinite(Yp))


def test_predictive_loglik_and_crps_new_sites():
    geom, draws, T = _newsite_setup("wang")
    hold = pr.HoldoutData(Y=60 + np.abs(np.random.default_rng(2).standard_normal((3, T))) * 5,
                          u=np.full(3, 60.0), sigma=np.full(3, 20.0), xi=np.full(3, 0.15))
    ll = pr.predictive_loglik_new_sites(geom, draws, hold, M=6, seed=3)
    assert np.isfinite(ll["total"]) and ll["per_draw"].shape == (3,)
    cr = pr.crps_holdout_new_sites(geom, draws, hold, M=6, seed=4,
                                   tail_threshold=60.0, tail_smooth=5.0)
    assert np.isfinite(cr["vanilla_mean"]) and np.isfinite(cr["twcrps_mean"])
    assert cr["twcrps_mean"] <= cr["vanilla_mean"] + 1e-9


def test_crps_holdout_all_models():
    draws, holdout, obs_idx, holdout_idx, K = _make_draws_and_holdout(seed=9)
    # shi / HW path (dense covariance)
    res = pr.crps_holdout(draws, holdout, obs_idx=obs_idx, holdout_idx=holdout_idx,
                          cov_for_draw=lambda d: K, use_visgp=False, M=8, seed=1,
                          tail_threshold=60.0, tail_smooth=5.0)
    assert np.isfinite(res["vanilla_mean"])
    assert np.isfinite(res["twcrps_mean"])
    assert res["vanilla"].shape == holdout.Y.shape
    # wang path (VisGP precision)
    Q = np.linalg.inv(K)
    resw = pr.crps_holdout(draws, holdout, obs_idx=obs_idx, holdout_idx=holdout_idx,
                           cov_for_draw=lambda d: Q, use_visgp=True, M=8, seed=2,
                           tail_threshold=60.0)
    assert np.isfinite(resw["twcrps_mean"])
