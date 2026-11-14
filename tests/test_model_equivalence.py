"""Check 2: with no barrier and a stationary covariance, the Wang model reduces
to the Shi model.

A full visibility graph (single clique = all sites) makes the VisGP precision
exactly the inverse of the dense stationary Matern (Gilbert & Datta), and with
no barrier / no anisotropy the Wendland kernels are identical to Shi's. Hence,
given the same stationary covariance matrix, the Wang and Shi code paths must
produce an identical log-posterior target: identical Wendland weights, identical
gamma_bar, identical Z log-prior, and identical per-time Y log-likelihood.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
import covariance as cov  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, ModelState  # noqa: E402


def _common_setup(seed=11, Ns=24, Nt=6):
    sc = sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"],
        u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
        xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"],
        knots_S=out["knots_S"], knots_phi=out["knots_phi"], knots_rho=out["knots_rho"])
    return out, data


def test_wang_fullgraph_stationary_equals_shi():
    out, data = _common_setup()
    Ns, Nt = data.Y.shape
    nu, rho0 = out["nu"], 1.5

    # shared stationary covariance
    DistMtx = cov.dist_matrix(data.sites_xy)
    K_common = cov.matern_wang(DistMtx, rho0, 1.0, nu)

    st = SamplerSettings(nu=nu, seed=1)

    # --- Shi state: set the SAME canonical rho (no manual K) -------------
    shi = ModelState(ModelConfig(model_type="shi"), data, st)
    shi.rho_vec = np.full(Ns, rho0)   # canonical effective-range
    shi.rebuild_cov()                 # converts canonical->ns_cov internally
    # the canonical mapping should reproduce the shared stationary covariance
    assert np.allclose(shi.K, K_common, atol=1e-8)

    # --- Wang state: full-visibility graph + same canonical rho ----------
    data_w = SamplerData(**{**data.__dict__})
    data_w.cliques = [list(range(Ns))]
    data_w.separators = []
    wang_cfg = ModelConfig(model_type="wang", use_visgp=True, cliques_path="x.rds")
    wang = ModelState(wang_cfg, data_w, st)
    wang.rho_vec = np.full(Ns, rho0)
    wang.rebuild_cov()   # Q = inv(matern_wang(rho0, nu)) == inv(K_common)

    # 1) identical Wendland weights and gamma_bar (no barrier / no aniso)
    assert np.allclose(shi.W_wend_S, wang.W_wend_S)
    assert np.allclose(shi.gamma_bar, wang.gamma_bar)
    # 2) Wang's full-graph precision is the inverse of the shared covariance
    import scipy.sparse as sp
    Qw = np.asarray(wang.Q.todense()) if sp.issparse(wang.Q) else wang.Q
    assert np.allclose(np.linalg.inv(Qw), K_common, atol=1e-6)

    # set identical latent state in both
    rng = np.random.default_rng(0)
    Ztest = rng.standard_normal((Ns, Nt))
    for stt in (shi, wang):
        stt.Z = Ztest.copy()
        stt.S = out["W_wend_S"].T @ np.ones(0) if False else np.full((stt.k_S, Nt), 1.3)
        stt.R = stt.W_wend_S @ stt.S
        stt.phi_k = np.full(stt.k_phi, 0.5)
        stt.phi_vec = stt._phi_field(stt.phi_k)
        stt.alpha0 = 4.0

    # 3) identical Z log-prior (dense MVN vs full-graph VisGP)
    for t in range(Nt):
        lp_shi = shi.Z_logprior(shi.Z[:, t])
        lp_wang = wang.Z_logprior(wang.Z[:, t])
        assert abs(lp_shi - lp_wang) < 1e-6

    # 4) identical per-time Y log-likelihood
    for t in range(Nt):
        ll_shi = shi.loglik_t(t)
        ll_wang = wang.loglik_t(t)
        assert abs(ll_shi - ll_wang) < 1e-8


def test_no_barrier_kernels_equal_standard():
    """build_wendland_weights with the wang config but no barrier/aniso equals
    Shi's standard isotropic kernel exactly."""
    import kernels
    out, data = _common_setup()
    shi_W = kernels.build_wendland_weights(
        data.sites_xy, data.knots_S, np.repeat(3.0, data.knots_S.shape[0]),
        config=ModelConfig(model_type="shi"))
    wang_W = kernels.build_wendland_weights(
        data.sites_xy, data.knots_S, np.repeat(3.0, data.knots_S.shape[0]),
        config=ModelConfig(model_type="wang", use_visgp=True, cliques_path="x.rds"))
    assert np.array_equal(shi_W, wang_W)
