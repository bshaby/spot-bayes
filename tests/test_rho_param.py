"""Tests for the canonical rho parameterisation shared by all three models.

rho is exposed on Wang's effective-range scale for every model. For shi it is
converted to ns_cov's internal range via rho^2/(8 nu); the resulting covariance
must equal Wang's matern_wang at the same canonical rho.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import covariance as cov  # noqa: E402
import utilities as u  # noqa: E402


def test_canonical_mapping_roundtrip():
    for nu in [0.5, 1.0, 1.5, 2.0]:
        rho = np.array([0.5, 1.0, 3.0, 7.0])
        internal = cov.canonical_to_ns_range(rho, nu)
        back = cov.ns_range_to_canonical(internal, nu)
        assert np.allclose(rho, back)


def test_canonical_rho_makes_shi_equal_wang_covariance():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (18, 2))
    for nu in [0.5, 1.0, 1.5]:
        for rho in [0.8, 2.0, 5.0]:
            Kw = cov.matern_wang(cov.dist_matrix(coords), rho, 1.0, nu)
            internal = cov.canonical_to_ns_range(rho, nu)
            Ks = u.ns_cov(range_vec=np.full(18, internal), sigsq_vec=np.ones(18),
                          coords=coords, kappa=nu, cov_model="matern")
            assert np.max(np.abs(Kw - Ks)) < 1e-9


def test_all_models_same_rho_same_covariance():
    """A given canonical rho yields the same Z covariance for shi, wang, HW."""
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState

    sc = sm.SimConfig(Ns=20, Nt=4, seed=3)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
                       u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
                       xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
                       knots_S=out["knots_S"], knots_phi=out["knots_phi"],
                       knots_rho=out["knots_rho"])
    nu, rho0 = out["nu"], 2.5
    st = SamplerSettings(nu=nu, seed=1)
    Ns = data.Y.shape[0]

    shi = ModelState(ModelConfig(model_type="shi"), data, st)
    shi.rho_vec = np.full(Ns, rho0); shi.rebuild_cov()

    hw = ModelState(ModelConfig(model_type="huser-wadsworth"), data, st)
    hw.rho_k = np.array([rho0])
    hw.rho_vec = hw.W_gauss_rho @ hw.rho_k   # HW rebuild reads rho_vec (canonical)
    hw.rebuild_cov()

    # shi (nonstationary machinery, constant canonical rho) == HW (stationary
    # matern at canonical rho)
    assert np.allclose(shi.K, hw.K, atol=1e-8)
