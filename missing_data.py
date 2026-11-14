"""
Missing-data handling: impute the latent field at unobserved sites and draw
predictive responses there.

Two covariance backends, dispatched by ModelConfig:
  * dense Matern (shi / huser-wadsworth): standard Gaussian conditioning using a
    Cholesky of K_oo (numerically stable, matching Shi's imputation).
  * VisGP precision (wang): sparse-precision conditioning via
    covariance.visgp_conditional. This is the non-convex prediction path the
    prompt flags as tricky; the clique structure is already baked into Q, so the
    conditional formula is exact for sites included in the graph.

The same conditional machinery is reused by predictive.py for holdout scoring.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Optional

import numpy as np
import scipy.linalg

import utilities as u
import covariance as cov
from rw_inte import pRW


def conditional_gaussian_dense(K: np.ndarray, obs_idx: Sequence[int],
                               miss_idx: Sequence[int], Z_obs: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Z[miss] | Z[obs] ~ N(mean, cov) for a dense covariance K.

    mean = K_mo K_oo^{-1} Z_obs ;  cov = K_mm - K_mo K_oo^{-1} K_om.
    Uses a Cholesky solve on K_oo. Z_obs may be (n_obs,) or (n_obs, T).
    """
    obs_idx = list(obs_idx)
    miss_idx = list(miss_idx)
    K_oo = K[np.ix_(obs_idx, obs_idx)]
    K_mo = K[np.ix_(miss_idx, obs_idx)]
    K_mm = K[np.ix_(miss_idx, miss_idx)]
    L = scipy.linalg.cho_factor(K_oo, lower=True)
    sol = scipy.linalg.cho_solve(L, np.asarray(Z_obs))  # K_oo^{-1} Z_obs
    mean = K_mo @ sol
    cov_mm = K_mm - K_mo @ scipy.linalg.cho_solve(L, K_mo.T)
    return mean, cov_mm


def conditional_gaussian_visgp(Q: np.ndarray, obs_idx: Sequence[int],
                               miss_idx: Sequence[int], Z_obs: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Z[miss] | Z[obs] from a VisGP precision Q (see covariance.visgp_conditional)."""
    return cov.visgp_conditional(Q, obs_idx, miss_idx, Z_obs)


def predict_Z_new_sites(new_coords, train_coords, Q, rho, nu, water_poly,
                        Z_train, sigmasq: float = 1.0, n_cores: int = 1):
    """Conditional latent field at NEW sites absent from the original VisGP graph.

    Convenience wrapper: computes the visibility of each new site to the training
    sites and applies the VisGP Markov-blanket predictor
    (covariance.visgp_predict_new). Use this for holdout data at locations the
    fitting MCMC never saw -- the in-graph path (conditional_Z) only handles
    sites that were already in the precision Q (masked via miss_mat).

    Returns (mean, var): mean is (n_new,) or (n_new, T); var is (n_new,).
    """
    import domain
    visible = domain.visible_from_new(new_coords, train_coords, water_poly, n_cores)
    return cov.visgp_predict_new(new_coords, train_coords, Q, rho, nu,
                                 Z_train, visible, sigmasq=sigmasq)


def predict_Z_new_sites_joint(new_coords, train_coords, Q, rho, nu, water_poly,
                              Z_train, sigmasq: float = 1.0, n_cores: int = 1):
    """JOINT conditional (mean, cov) of Z at new sites given the training field.

    Like predict_Z_new_sites but returns a full (n_new, n_new) covariance that
    captures dependence *among* the new sites (via covariance.visgp_predict_new_joint),
    suitable for drawing coherent joint realisations with draw_Z_missing.
    """
    import domain
    visible_train = domain.visible_from_new(new_coords, train_coords, water_poly, n_cores)
    visible_new = domain.visible_among_new(new_coords, water_poly, n_cores)
    return cov.visgp_predict_new_joint(new_coords, train_coords, Q, rho, nu,
                                       Z_train, visible_train, visible_new,
                                       sigmasq=sigmasq)


def conditional_Z(config, cov_obj: np.ndarray, obs_idx, miss_idx, Z_obs):
    """Dispatch conditional draw parameters based on whether cov_obj is a dense
    covariance K (shi/HW) or a VisGP precision Q (wang, config.use_visgp).
    """
    if getattr(config, "use_visgp", False):
        return conditional_gaussian_visgp(cov_obj, obs_idx, miss_idx, Z_obs)
    return conditional_gaussian_dense(cov_obj, obs_idx, miss_idx, Z_obs)


def draw_Z_missing(mean: np.ndarray, cov_mm: np.ndarray, rng: np.random.Generator,
                   n_draws: int = 1) -> np.ndarray:
    """Draw from N(mean, cov_mm). Supports mean shaped (m,) or (m, T).

    Returns shape (m,) [n_draws==1, mean 1-D], (m, T), or (m, T, n_draws).
    """
    cov_mm = 0.5 * (cov_mm + cov_mm.T)              # symmetrise
    L = np.linalg.cholesky(cov_mm + 1e-10 * np.eye(cov_mm.shape[0]))
    m = mean.shape[0]
    if mean.ndim == 1:
        if n_draws == 1:
            return mean + L @ rng.standard_normal(m)
        return mean[:, None] + L @ rng.standard_normal((m, n_draws))
    T = mean.shape[1]
    if n_draws == 1:
        return mean + L @ rng.standard_normal((m, T))
    out = np.empty((m, T, n_draws))
    for k in range(n_draws):
        out[:, :, k] = mean + L @ rng.standard_normal((m, T))
    return out


def impute_Y_missing(config, cov_obj, obs_idx, miss_idx, Z_obs, *,
                     R_miss, phi_miss, gamma_bar_miss, alpha0,
                     p_cens, u_miss, sigma_miss, xi_miss,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Impute Z and Y at missing sites for a single time slice.

    Draws Z_miss from the conditional Gaussian, forms
    X = eps * R_miss^phi * g(Z_miss) with eps ~ Log-Laplace(alpha0), maps to a
    probability via pRW, and inverts the censored-GPD margin to get Y_miss.
    Returns (Z_miss, Y_miss).
    """
    mean, cov_mm = conditional_Z(config, cov_obj, obs_idx, miss_idx, Z_obs)
    Z_miss = draw_Z_missing(mean, cov_mm, rng)
    W_miss = u.g(Z_miss)
    eps = u.rloglaplace(size=Z_miss.shape, alpha0=alpha0)
    X_miss = eps * (np.asarray(R_miss) ** np.asarray(phi_miss)) * W_miss
    pX = pRW(X_miss, phi_miss, gamma_bar_miss, alpha0)
    Y_miss = np.where(pX <= p_cens, u_miss,
                      u.qCGP(pX, p_cens, u_miss, sigma_miss, xi_miss))
    return Z_miss, Y_miss
