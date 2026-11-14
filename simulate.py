"""
Forward simulation from all three model types.

The generative model (shared GPD + multiplicative Log-Laplace nugget tail):

    Z(s,t) ~ N(0, K)            K = Matern (shi/HW) or VisGP^{-1} (wang)
    W(s,t) = g(Z)               norm -> standard Pareto
    R(s,t) = (W_wend @ S)[s,t]  S_k(t) ~ Levy(gamma_k)   (shi/wang)
             R(t) = S(t)        single global Levy scale  (huser-wadsworth)
    X*(s,t) = R(s,t)^phi(s) * W(s,t)     phi(s) field (shi/wang) or scalar (HW)
    eps(s,t) ~ Log-Laplace(alpha0)
    X(s,t) = eps * X*(s,t)
    p = pRW(X; phi, gamma_bar, alpha0)
    Y = qCGP(p; p_cens, u, sigma, xi)

For model_type='shi' with default settings this reproduces Shi's
simulate_data_multip.py generative process exactly (validated in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.stats

import utilities as u
import kernels
import covariance as cov
from config import ModelConfig, MODEL_SHI, MODEL_WANG, MODEL_HW, PHI_UNIT
from rw_inte import pRW


def make_isometric_knots(minX, maxX, minY, maxY, N_outer_grid):
    """Outer+inner interleaved isometric grid clipped to the open domain.

    Identical to Shi's make_isometric_knots / utilities._iso_grid.
    """
    return u._iso_grid(minX, maxX, minY, maxY, N_outer_grid)


@dataclass
class SimConfig:
    """Parameters of a synthetic dataset (defaults mirror Shi's small study)."""
    Ns: int = 225
    Nt: int = 128
    domain: tuple = (0.0, 10.0, 0.0, 10.0)
    miss_prob: float = 0.05
    grid_S: int = 9
    grid_phi: int = 9
    grid_rho: int = 9
    radius_S: float = 3.0
    eff_range_phi: float = 3.0
    eff_range_rho: float = 3.0
    gamma_at_knots_val: float = 1.0
    alpha_stable: float = 0.5
    nu: float = 1.0
    sigsq_Z: float = 1.0
    threshold: float = 60.0
    Beta_logsigma: tuple = (3.0,)
    Beta_xi: tuple = (0.15,)
    p_cens: float = 0.95
    alpha0: float = 5.0
    # tail-exponent surface (knot values). For HW this is a single scalar delta.
    phi_at_knots: Optional[np.ndarray] = None   # length k_phi (shi/wang)
    delta: float = 0.5                          # scalar exponent (HW)
    rho_at_knots: Optional[np.ndarray] = None   # length k_rho (shi/wang)
    rho_const: float = 1.5                       # stationary range (wang/HW)
    seed: int = 2345


def _default_phi_knots(knots_phi):
    """Shi's 'nonstatsc3' phi surface."""
    mvn1 = scipy.stats.multivariate_normal.pdf(
        knots_phi, mean=np.array([2.5, 3.0]), cov=2 * np.array([[1, 0.2], [0.2, 1]]))
    mvn2 = scipy.stats.multivariate_normal.pdf(
        knots_phi, mean=np.array([7.0, 7.5]), cov=2 * np.array([[1, -0.2], [-0.2, 1]]))
    return 0.37 + 5.0 * (mvn1 + mvn2)


def _default_rho_knots(knots_rho):
    """Shi's 'nonstat' rho surface."""
    xk, yk = knots_rho[:, 0], knots_rho[:, 1]
    return np.sqrt(0.3 * xk + 0.4 * yk) / 2.0


def simulate(model: ModelConfig, sim: SimConfig,
             sites_xy: Optional[np.ndarray] = None,
             cliques=None, separators=None, water_poly=None,
             aniso_ratio: float = 1.0, aniso_angle: float = 0.0,
             sigma_mat_override: Optional[np.ndarray] = None,
             xi_mat_override: Optional[np.ndarray] = None) -> dict:
    """Simulate a dataset; returns a dict of arrays (the data contract).

    For wang with use_visgp, pass `cliques`/`separators`; for barrier kernels
    pass `water_poly`. Anisotropy via aniso_ratio/aniso_angle (shi/wang).

    `sigma_mat_override` / `xi_mat_override`: optional (Ns, Nt) GPD scale/shape
    surfaces used in place of the intercept-only margins the SimConfig Beta would
    build (e.g. spatially-varying margins from an external covariate design, as in
    Experiment 3). The PIT X -> Y then uses these surfaces directly.
    """
    rng = np.random.default_rng(sim.seed)
    np.random.seed(sim.seed)
    minX, maxX, minY, maxY = sim.domain

    if sites_xy is None:
        sites_xy = (np.random.random((sim.Ns, 2)) * [maxX - minX, maxY - minY]
                    + [minX, minY])
    Ns = sites_xy.shape[0]
    Nt = sim.Nt

    knots_S = make_isometric_knots(minX, maxX, minY, maxY, sim.grid_S)
    knots_phi = make_isometric_knots(minX, maxX, minY, maxY, sim.grid_phi)
    knots_rho = make_isometric_knots(minX, maxX, minY, maxY, sim.grid_rho)
    k_S = knots_S.shape[0]

    # --- Wendland weights for R(s) (shi/wang) -------------------------------
    W_wend_S = kernels.build_wendland_weights(
        sites_xy, knots_S, np.repeat(sim.radius_S, k_S), config=model,
        aniso_ratio=aniso_ratio, aniso_angle=aniso_angle, water_poly=water_poly)

    # --- Gaussian smoothing weights for phi / rho ---------------------------
    bw_phi = sim.eff_range_phi ** 2 / 6.0
    bw_rho = sim.eff_range_rho ** 2 / 6.0
    W_gauss_phi = u.build_weight_matrix(sites_xy, knots_phi, kernel="gaussian",
                                        radius=sim.radius_S, bandwidth=bw_phi, cutoff=False)
    W_gauss_rho = u.build_weight_matrix(sites_xy, knots_rho, kernel="gaussian",
                                        radius=sim.radius_S, bandwidth=bw_rho, cutoff=False)

    # --- tail exponent phi(s) (or scalar delta for HW) ----------------------
    if model.model_type == MODEL_HW:
        phi_vec = np.full(Ns, sim.delta)
    else:
        phi_at_knots = (sim.phi_at_knots if sim.phi_at_knots is not None
                        else _default_phi_knots(knots_phi))
        phi_vec = W_gauss_phi @ phi_at_knots

    # --- covariance for Z(s) ------------------------------------------------
    if model.use_visgp:
        if cliques is None:
            raise ValueError("VisGP simulation requires cliques.")
        rho = sim.rho_const
        Q = cov.visgp_precision(cliques, separators or [], sites_xy, rho, sim.nu,
                                sigmasq=sim.sigsq_Z, n_cores=model.n_cores)
        K = np.linalg.inv(Q)
    elif model.model_type == MODEL_HW:
        # stationary isotropic Matern
        D = cov.dist_matrix(sites_xy)
        K = cov.matern_wang(D, sim.rho_const, sim.sigsq_Z, sim.nu)
    else:
        rho_at_knots = (sim.rho_at_knots if sim.rho_at_knots is not None
                        else _default_rho_knots(knots_rho))
        rho_vec = W_gauss_rho @ rho_at_knots   # canonical effective-range scale
        range_internal = cov.canonical_to_ns_range(rho_vec, sim.nu)
        K = u.ns_cov(range_vec=range_internal, sigsq_vec=np.repeat(sim.sigsq_Z, Ns),
                     coords=sites_xy, kappa=sim.nu, cov_model="matern")

    # --- random scale R(s,t) ------------------------------------------------
    gamma_at_knots = np.repeat(sim.gamma_at_knots_val, k_S)
    Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(Ns), cov=K, size=Nt).T
    if Z.ndim == 1:
        Z = Z.reshape(Ns, Nt)
    Wf = u.g(Z)

    if model.model_type == MODEL_HW:
        # single global Levy scale R(t), broadcast across sites
        S_t = scipy.stats.levy.rvs(loc=0.0, scale=sim.gamma_at_knots_val, size=Nt)
        R_at_sites = np.tile(S_t, (Ns, 1))
        S_at_knots = S_t[None, :]                 # one global scale, shaped (1, Nt)
        gamma_bar_vec = np.full(Ns, sim.gamma_at_knots_val)
    else:
        S_at_knots = scipy.stats.levy.rvs(loc=0.0, scale=gamma_at_knots[:, None],
                                          size=(k_S, Nt))
        R_at_sites = W_wend_S @ S_at_knots
        gamma_bar_vec = kernels.gamma_bar_from_weights(
            W_wend_S, gamma_at_knots, alpha=sim.alpha_stable)

    X_star = np.power(R_at_sites, phi_vec[:, None]) * Wf
    eps = u.rloglaplace(size=X_star.shape, alpha0=sim.alpha0)
    X = eps * X_star

    # --- marginal CGP params ------------------------------------------------
    elev_gen = scipy.stats.multivariate_normal(mean=[0, 5], cov=np.diag([15.0, 30.0]))
    elevations = elev_gen.pdf(sites_xy)
    elevations = (elevations - np.mean(elevations)) / np.std(elevations)
    Beta_logsigma = np.asarray(sim.Beta_logsigma)
    Beta_xi = np.asarray(sim.Beta_xi)
    C0 = np.ones((Ns, Nt))
    C1 = np.tile(elevations, (Nt, 1)).T
    C_logsigma = np.stack([C0, C1], axis=0)
    C_xi = np.stack([C0, C1], axis=0)
    sigma_mat = np.exp((C_logsigma[:len(Beta_logsigma)].T @ Beta_logsigma).T)
    xi_mat = (C_xi[:len(Beta_xi)].T @ Beta_xi).T
    if sigma_mat_override is not None:
        sigma_mat = np.asarray(sigma_mat_override, dtype=float)
    if xi_mat_override is not None:
        xi_mat = np.asarray(xi_mat_override, dtype=float)
    assert sigma_mat.shape == (Ns, Nt) and xi_mat.shape == (Ns, Nt)
    u_mat = np.full((Ns, Nt), sim.threshold)

    # --- PIT X -> Y ---------------------------------------------------------
    Y = np.empty((Ns, Nt))
    for t in range(Nt):
        pX = pRW(X[:, t], phi_vec, gamma_bar_vec, sim.alpha0)
        cens = pX <= sim.p_cens
        Y[cens, t] = u_mat[cens, t]
        exc = ~cens
        if np.any(exc):
            Y[exc, t] = u.qCGP(pX[exc], sim.p_cens, u_mat[exc, t],
                               sigma_mat[exc, t], xi_mat[exc, t])

    # --- missingness --------------------------------------------------------
    miss_mat = rng.random((Ns, Nt)) < sim.miss_prob
    Y_missing = Y.copy()
    Y_missing[miss_mat] = np.nan

    return dict(
        S_at_knots=S_at_knots,
        Y=Y, Y_missing=Y_missing, miss_mat=miss_mat,
        sites_xy=sites_xy, knots_S=knots_S, knots_phi=knots_phi, knots_rho=knots_rho,
        u_mat=u_mat, sigma_mat=sigma_mat, xi_mat=xi_mat, elevations=elevations,
        Z=Z, W=Wf, R_at_sites=R_at_sites, X_star=X_star, X=X,
        phi_vec=phi_vec, gamma_bar_vec=gamma_bar_vec, gamma_at_knots=gamma_at_knots,
        W_wend_S=W_wend_S, W_gauss_phi=W_gauss_phi, W_gauss_rho=W_gauss_rho,
        K=K, alpha0=sim.alpha0, Beta_logsigma=Beta_logsigma, Beta_xi=Beta_xi,
        p_cens=sim.p_cens, nu=sim.nu,
    )
