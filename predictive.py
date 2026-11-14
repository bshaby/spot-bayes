"""
Predictive log-likelihood (and CRPS) on holdout sets, for all three models.

Generalises Shi's plot_predictive_loglik.py post-processing. The estimator is a
two-level Monte Carlo integral over (a) B posterior draws and (b) M conditional
draws of the latent field at the holdout sites:

    log p(Y_holdout) ~= sum_t logsumexp_{b,m} [ Y_loglik_1t(Y_h_t | theta_b, Z_h^{b,m}) ]
                        - log(B * M)

Only two pieces are model-dependent, both already built:
  1. conditional Z at holdout sites:
       shi / huser-wadsworth -> dense Matern conditional (missing_data)
       wang                  -> VisGP-precision conditional (missing_data/covariance)
  2. the tail likelihood + qRW/dRW, which are model-agnostic (utilities/rw_inte),
     parameterised by phi_domain and the chosen R(s) construction.

`predictive_loglik` takes per-draw parameter arrays (the "traces") plus the
holdout data and returns the per-draw scores and the aggregate. It is agnostic
to how the traces were produced, so it can be unit-tested with synthetic draws.

`crps_holdout` similarly scores any of the three models with both the ordinary
(vanilla) CRPS and a tail-weighted CRPS (threshold-weighted CRPS via a chaining
function; hard or smooth upper-tail weight), drawing posterior-predictive Y at
the holdout sites through the same conditional-Z + forward-map machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Callable

import warnings

import numpy as np
import scipy.special
import scipy.stats

import utilities as u
import missing_data as md
import covariance as cov
import kernels
from config import ModelConfig, PHI_UNIT
from rw_inte import pRW, qRW
from rw_numeric import dRW_safe as dRW


@dataclass
class HoldoutData:
    """A holdout set for scoring."""
    Y: np.ndarray            # (n_holdout, T) observed responses at holdout sites
    u: np.ndarray            # (n_holdout,) thresholds
    sigma: np.ndarray        # (n_holdout,) or (n_holdout, T) GPD scale
    xi: np.ndarray           # (n_holdout,) or (n_holdout, T) GPD shape
    p_cens: float = 0.95


def _as_2d(a, T):
    a = np.asarray(a)
    return a if a.ndim == 2 else np.repeat(a[:, None], T, axis=1)


def predictive_loglik(draws: Sequence[dict], holdout: HoldoutData, *,
                      obs_idx: Sequence[int], holdout_idx: Sequence[int],
                      cov_for_draw: Callable[[dict], np.ndarray],
                      use_visgp: bool = False,
                      M: int = 20, seed: int = 0) -> dict:
    """Two-level Monte Carlo predictive log-likelihood.

    Parameters
    ----------
    draws : list of B dicts, each with keys
        'Z_obs'        : (n_obs, T) latent field at observed (training) sites
        'phi_holdout'  : (n_holdout,) tail exponent at holdout sites
        'gamma_bar_holdout' : (n_holdout,)
        'R_holdout'    : (n_holdout, T) random scale at holdout sites
        'alpha0'       : scalar
    obs_idx, holdout_idx : indices into the full site set defining the partition.
    cov_for_draw : callable(draw) -> dense covariance K (shi/HW) or VisGP
        precision Q (wang) for that draw's covariance parameters.
    use_visgp : whether cov_for_draw returns a precision (wang) vs covariance.
    M : number of conditional latent draws per posterior draw.

    Returns dict with 'per_draw' (B,) and 'total' (scalar) predictive log-lik.
    """
    rng = np.random.default_rng(seed)
    B = len(draws)
    Y = holdout.Y
    n_h, T = Y.shape
    sigma = _as_2d(holdout.sigma, T)
    xi = _as_2d(holdout.xi, T)
    uvec = np.asarray(holdout.u)
    p = holdout.p_cens

    # config-like shim so missing_data.conditional_Z dispatches correctly
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.use_visgp = use_visgp

    loglik_BTM = np.full((B, T, M), -np.inf)

    for b, draw in enumerate(draws):
        cov_obj = cov_for_draw(draw)
        Z_obs = draw["Z_obs"]                       # (n_obs, T)
        phi_h = np.asarray(draw["phi_holdout"])     # (n_h,)
        gbar_h = np.asarray(draw["gamma_bar_holdout"])
        R_h = np.asarray(draw["R_holdout"])         # (n_h, T)
        alpha0 = float(draw["alpha0"])

        # conditional latent params at holdout sites given observed
        mean, cov_mm = md.conditional_Z(cfg, cov_obj, obs_idx, holdout_idx, Z_obs)
        # draw M conditional realisations: (n_h, T, M)
        Z_draws = md.draw_Z_missing(mean, cov_mm, rng, n_draws=M)
        if Z_draws.ndim == 2:   # M==1 edge case
            Z_draws = Z_draws[:, :, None]

        # precompute per-(site,time) transformed quantile/density (indep. of m)
        # qRW depends only on (Y, phi, gamma_bar, alpha0), not on Z.
        for t in range(T):
            pY_t = u.pCGP(Y[:, t], p, uvec, sigma[:, t], xi[:, t])
            qRW_t = qRW(pY_t, phi_h, gbar_h, alpha0)
            dRW_t = dRW(qRW_t, phi_h, gbar_h, alpha0)
            cens = np.flatnonzero(Y[:, t] <= uvec)
            exc = np.flatnonzero(Y[:, t] > uvec)
            f = np.ascontiguousarray
            for m in range(Z_draws.shape[2]):
                Zc = Z_draws[:, t, m]
                ll = u.Y_loglik_1t_best(
                    f(Y[:, t]), float(p), f(uvec), f(sigma[:, t]), f(xi[:, t]),
                    f(R_h[:, t]), f(Zc), None, f(phi_h), f(gbar_h), float(alpha0),
                    f(qRW_t), f(dRW_t), f(cens).astype(np.int64),
                    f(exc).astype(np.int64))
                loglik_BTM[b, t, m] = ll

    # integrate over (b, m) via logsumexp, sum over t
    flat = loglik_BTM.transpose(1, 0, 2).reshape(T, B * M)
    per_time = scipy.special.logsumexp(flat, axis=1) - np.log(B * M)
    total = float(np.sum(per_time))
    # also per-draw (integrate over m only)
    per_draw = scipy.special.logsumexp(loglik_BTM, axis=2) - np.log(M)  # (B,T)
    per_draw = np.sum(per_draw, axis=1)
    return {"total": total, "per_draw": per_draw, "per_time": per_time,
            "loglik_BTM": loglik_BTM}


# ---------------------------------------------------------------------------
# (Tail-weighted) CRPS via chaining functions
# ---------------------------------------------------------------------------
# Threshold-weighted CRPS (Gneiting & Ranjan 2011; Allen et al. 2023):
#   twCRPS_w(F, y) = integral (F(z) - 1{y<=z})^2 w(z) dz
#                  = E|v(X) - v(y)| - 0.5 E|v(X) - v(X')|
# where the "chaining function" v is an antiderivative of the weight w (v' = w).
# w == 1 gives the ordinary (vanilla) CRPS; a tail weight focuses scoring on
# exceedances. Both a hard threshold and a smooth weight are provided.

def chain_identity(x):
    """Vanilla CRPS: weight w(z) = 1, chaining v(z) = z."""
    return np.asarray(x, dtype=float)


def chain_threshold(t: float):
    """Hard upper-tail weight w(z) = 1{z > t}; chaining v(z) = max(z - t, 0)."""
    def v(x):
        return np.maximum(np.asarray(x, dtype=float) - t, 0.0)
    return v


def chain_smooth(t: float, s: float):
    """Smooth upper-tail weight w(z) = Phi((z - t)/s); chaining

        v(z) = (z - t) Phi((z - t)/s) + s phi((z - t)/s).

    `s` controls how sharply the weight ramps up around the threshold t
    (s -> 0 recovers the hard threshold chain_threshold(t)).
    """
    def v(x):
        z = (np.asarray(x, dtype=float) - t) / s
        return (np.asarray(x, dtype=float) - t) * scipy.stats.norm.cdf(z) \
            + s * scipy.stats.norm.pdf(z)
    return v


def weighted_crps_empirical(pred_samples: np.ndarray, y_obs: float,
                            chain=chain_identity) -> float:
    """(Threshold-weighted) CRPS of an empirical predictive distribution.

    With chain=chain_identity this is the ordinary CRPS
    E|X - y| - 0.5 E|X - X'|. With a tail chaining function it is the
    threshold-weighted CRPS. Estimated from samples via the sorted-order
    identity applied to the transformed values v(X).
    """
    vx = np.sort(np.asarray(chain(pred_samples), dtype=float))
    vy = float(chain(np.asarray([y_obs], dtype=float))[0])
    n = vx.size
    term1 = np.mean(np.abs(vx - vy))
    i = np.arange(1, n + 1)
    term2 = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * vx)
    return float(term1 - 0.5 * term2)


# backward-compatible alias
def crps_empirical(pred_samples, y_obs):
    return weighted_crps_empirical(pred_samples, y_obs, chain=chain_identity)


# ---------------------------------------------------------------------------
# Holdout CRPS for all three models
# ---------------------------------------------------------------------------
def posterior_predictive_Y_holdout(draws, holdout: HoldoutData, *,
                                   obs_idx, holdout_idx, cov_for_draw,
                                   use_visgp: bool = False, M: int = 20,
                                   seed: int = 0) -> np.ndarray:
    """Draw posterior-predictive Y at holdout sites for every model type.

    For each of the B posterior draws, draws M conditional latent realisations
    at the holdout sites (dense Matern for shi/HW, VisGP precision for wang),
    pushes them through the forward map
        X = eps * R^phi * g(Z),  p = pRW(X; phi, gamma_bar, alpha0),
        Y = qCGP(p; p_cens, u, sigma, xi)
    and returns an array of shape (n_holdout, T, B*M) of predictive responses.
    """
    rng = np.random.default_rng(seed)
    n_h, T = holdout.Y.shape
    sigma = _as_2d(holdout.sigma, T)
    xi = _as_2d(holdout.xi, T)
    uvec = np.asarray(holdout.u)
    p = holdout.p_cens

    class _Cfg:
        pass
    cfg = _Cfg(); cfg.use_visgp = use_visgp

    B = len(draws)
    Y_pred = np.empty((n_h, T, B * M))
    for b, draw in enumerate(draws):
        cov_obj = cov_for_draw(draw)
        Z_obs = draw["Z_obs"]
        phi_h = np.asarray(draw["phi_holdout"])
        gbar_h = np.asarray(draw["gamma_bar_holdout"])
        R_h = np.asarray(draw["R_holdout"])      # (n_h, T)
        alpha0 = float(draw["alpha0"])
        mean, cov_mm = md.conditional_Z(cfg, cov_obj, obs_idx, holdout_idx, Z_obs)
        Zd = md.draw_Z_missing(mean, cov_mm, rng, n_draws=M)   # (n_h, T, M)
        if Zd.ndim == 2:
            Zd = Zd[:, :, None]
        for m in range(M):
            Wf = u.g(Zd[:, :, m])
            eps = u.rloglaplace(size=Wf.shape, alpha0=alpha0)
            X = eps * (R_h ** phi_h[:, None]) * Wf
            for t in range(T):
                pX = pRW(X[:, t], phi_h, gbar_h, alpha0)
                col = np.where(pX <= p, uvec,
                               u.qCGP(pX, p, uvec, sigma[:, t], xi[:, t]))
                Y_pred[:, t, b * M + m] = col
    return Y_pred


def crps_holdout(draws, holdout: HoldoutData, *, obs_idx, holdout_idx,
                 cov_for_draw, use_visgp: bool = False, M: int = 20,
                 seed: int = 0, tail_threshold: Optional[float] = None,
                 tail_smooth: Optional[float] = None) -> dict:
    """Vanilla and tail-weighted CRPS on a holdout set, for any of the 3 models.

    Returns per-(site,time) CRPS arrays and aggregate means for:
      * 'vanilla'  : unweighted CRPS,
      * 'twcrps'   : tail-weighted CRPS (only if tail_threshold is given).
    The tail weight is a hard upper-threshold by default, or a smooth
    Gaussian-ramp weight Phi((z - t)/s) when tail_smooth (s) is provided.
    """
    Y_pred = posterior_predictive_Y_holdout(
        draws, holdout, obs_idx=obs_idx, holdout_idx=holdout_idx,
        cov_for_draw=cov_for_draw, use_visgp=use_visgp, M=M, seed=seed)
    n_h, T, _ = Y_pred.shape

    chains = {"vanilla": chain_identity}
    if tail_threshold is not None:
        chains["twcrps"] = (chain_smooth(tail_threshold, tail_smooth)
                            if tail_smooth is not None
                            else chain_threshold(tail_threshold))

    out = {}
    for name, chain in chains.items():
        grid = np.empty((n_h, T))
        for i in range(n_h):
            for t in range(T):
                grid[i, t] = weighted_crps_empirical(
                    Y_pred[i, t, :], holdout.Y[i, t], chain=chain)
        out[name] = grid
        out[name + "_mean"] = float(np.mean(grid))
        out[name + "_by_site"] = grid.mean(axis=1)
    out["Y_pred"] = Y_pred
    return out


# ---------------------------------------------------------------------------
# Full prediction pipeline at NEW sites (not in the original graph)
# ---------------------------------------------------------------------------
@dataclass
class NewSiteGeometry:
    """Static geometry / model config for predicting at new locations.

    `new_coords` are the prediction locations; `train_coords` the original MCMC
    sites. The fitted knot grids and kernel settings let us evaluate the random
    scale R, tail exponent phi and gamma_bar at the new coords. For the wang
    model, `cliques`/`separators` rebuild the VisGP precision per draw and
    `water_poly` supplies the visibility (barriers); leaving water_poly None
    means full visibility (dense / no-barrier kriging).
    """
    new_coords: np.ndarray
    train_coords: np.ndarray
    knots_S: np.ndarray
    knots_phi: np.ndarray
    radius_S: float
    eff_range_phi: float
    config: ModelConfig
    knots_rho: Optional[np.ndarray] = None   # required for the shi (nonstationary) path
    eff_range_rho: float = 3.0
    cliques: Optional[list] = None
    separators: Optional[list] = None
    water_poly: object = None
    sigsq_Z: float = 1.0
    n_cores: int = 1
    # GPD-marginal covariates AT THE HOLDOUT SITES (same groups/format as the
    # training fit). When provided and the draw carries beta_sigma/beta_xi, the
    # holdout sigma/xi are derived from the fitted marginal per draw instead of
    # being taken as fixed (HoldoutData.sigma/xi).
    sigma_spatial: Optional[np.ndarray] = None
    sigma_temporal: Optional[np.ndarray] = None
    sigma_spatiotemporal: Optional[np.ndarray] = None
    xi_spatial: Optional[np.ndarray] = None
    xi_temporal: Optional[np.ndarray] = None
    xi_spatiotemporal: Optional[np.ndarray] = None

    def _has_sigma_cov(self):
        return any(c is not None for c in (self.sigma_spatial, self.sigma_temporal,
                                           self.sigma_spatiotemporal))

    def _has_xi_cov(self):
        return any(c is not None for c in (self.xi_spatial, self.xi_temporal,
                                           self.xi_spatiotemporal))


def _holdout_marginal(geom: "NewSiteGeometry", draw: dict, T: int,
                      sigma_fallback, xi_fallback):
    """sigma/xi (n_new, T) at the holdout sites for one draw.

    Derived from the fitted marginal beta + holdout covariates when available;
    otherwise the supplied fixed values (broadcast over time).
    """
    import marginal as mg
    n_new = np.atleast_2d(geom.new_coords).shape[0]
    # Use the fitted marginal whenever the draw carries beta_* (i.e. the run
    # estimated it). With no holdout covariates this is a clean intercept-only
    # marginal -- the design has just the intercept, giving a constant
    # exp(beta0) / xi_from_eta(beta0). Fall back to the supplied fixed values
    # only when the marginal was held fixed during fitting (no beta_* present).
    if "beta_sigma" in draw:
        d = mg.MarginalDesign(n_new, T, geom.sigma_spatial, geom.sigma_temporal,
                              geom.sigma_spatiotemporal)
        _check_design_matches_beta(d, draw["beta_sigma"], "sigma")
        sigma = mg.sigma_from_eta(d.eta(draw["beta_sigma"]))
    else:
        sigma = _as_2d(sigma_fallback, T)
    if "beta_xi" in draw:
        d = mg.MarginalDesign(n_new, T, geom.xi_spatial, geom.xi_temporal,
                              geom.xi_spatiotemporal)
        _check_design_matches_beta(d, draw["beta_xi"], "xi")
        xi = mg.xi_from_eta(d.eta(draw["beta_xi"]))
    else:
        xi = _as_2d(xi_fallback, T)
    return sigma, xi


def _check_design_matches_beta(design, beta, which):
    """Guard against silently mis-aligned holdout covariates: the holdout design
    must have the same number of parameters (1 + p_sp + p_tp + p_st) as the
    fitted beta, i.e. the same covariate groups in the same column order."""
    nb = np.asarray(beta).shape[0]
    if design.n_params != nb:
        raise ValueError(
            f"Holdout {which} covariates imply {design.n_params} coefficients "
            f"(1 intercept + {design.p_sp} spatial + {design.p_tp} temporal + "
            f"{design.p_st} spatiotemporal), but the fitted beta_{which} has {nb}. "
            f"Supply the holdout covariate groups in the same layout/column order "
            f"as the training fit.")


def _newsite_visibility(geom: NewSiteGeometry):
    """(visible_train, visible_new) booleans; full visibility if no water_poly."""
    import domain
    n_new = np.atleast_2d(geom.new_coords).shape[0]
    n_tr = geom.train_coords.shape[0]
    if geom.water_poly is None:
        vt = np.ones((n_new, n_tr), bool)
        vn = np.ones((n_new, n_new), bool); np.fill_diagonal(vn, False)
        return vt, vn
    vt = domain.visible_from_new(geom.new_coords, geom.train_coords,
                                 geom.water_poly, geom.n_cores)
    vn = domain.visible_among_new(geom.new_coords, geom.water_poly, geom.n_cores)
    return vt, vn


def _newsite_components(geom: NewSiteGeometry, draw: dict, vt, vn):
    """Per-draw new-site quantities: joint Z mean/cov, R, phi, gamma_bar, alpha0.

    `draw` keys: Z_train (d,T), S (k_S,T), phi_k, gamma_k, rho (scalar canonical
    range), nu, alpha0, and optionally aniso_ratio/aniso_angle.
    """
    cfg = geom.config
    new = np.atleast_2d(geom.new_coords)
    nu = float(draw["nu"])
    Z_train = np.asarray(draw["Z_train"])
    n_new = new.shape[0]

    if cfg.use_visgp:
        # wang: joint VisGP prediction (scalar canonical range)
        rho = float(draw["rho"])
        Q = cov.visgp_precision(geom.cliques, geom.separators or [],
                                geom.train_coords, rho, nu, sigmasq=geom.sigsq_Z,
                                sparse=True, n_cores=geom.n_cores)
        mean_Z, cov_Z = cov.visgp_predict_new_joint(
            new, geom.train_coords, Q, rho, nu, Z_train, vt, vn, sigmasq=geom.sigsq_Z)
    elif cfg.model_type == "shi":
        # nonstationary Matern: exact joint kriging on the augmented coords using
        # the fitted rho field interpolated to the new sites.
        rho_k = np.asarray(draw["rho_k"])
        bw = geom.eff_range_rho ** 2 / 6.0
        W_tr = u.build_weight_matrix(geom.train_coords, geom.knots_rho,
                                     kernel="gaussian", radius=geom.radius_S,
                                     bandwidth=bw, cutoff=False)
        W_nw = u.build_weight_matrix(new, geom.knots_rho, kernel="gaussian",
                                     radius=geom.radius_S, bandwidth=bw, cutoff=False)
        rho_aug = np.concatenate([W_tr @ rho_k, W_nw @ rho_k])      # canonical
        aug = np.vstack([geom.train_coords, new])
        rng_int = cov.canonical_to_ns_range(rho_aug, nu)
        K_aug = u.ns_cov(range_vec=rng_int, sigsq_vec=np.full(aug.shape[0], geom.sigsq_Z),
                         coords=aug, kappa=nu, cov_model="matern")
        d = geom.train_coords.shape[0]
        mean_Z, cov_Z = md.conditional_gaussian_dense(
            K_aug, list(range(d)), list(range(d, d + n_new)), Z_train)
    else:
        # huser-wadsworth: stationary Matern, exact joint kriging
        rho = float(draw["rho"])
        K = cov.matern_wang(cov.dist_matrix(geom.train_coords), rho, geom.sigsq_Z, nu)
        Q = np.linalg.inv(K)
        mean_Z, cov_Z = cov.visgp_predict_new_joint(
            new, geom.train_coords, Q, rho, nu, Z_train, vt, vn, sigmasq=geom.sigsq_Z)

    # new-site kernel quantities from the fitted knot params (per-draw radius_S
    # if it was sampled, else the fixed geometry radius)
    if cfg.model_type == "huser-wadsworth":
        # single global Levy scale: R(s) = S(t) for every site (all-ones kernel)
        W_wend = np.ones((new.shape[0], geom.knots_S.shape[0]))
    else:
        ar = float(draw.get("aniso_ratio", 1.0))
        aa = float(draw.get("aniso_angle", 0.0))
        radius_S = float(draw.get("radius_S", geom.radius_S))
        W_wend = kernels.build_wendland_weights(
            new, geom.knots_S, np.repeat(radius_S, geom.knots_S.shape[0]),
            config=cfg, aniso_ratio=ar, aniso_angle=aa, water_poly=geom.water_poly)
    S = np.asarray(draw["S"])
    R_new = W_wend @ S                                   # (n_new, T)
    gamma_bar_new = kernels.gamma_bar_from_weights(W_wend, np.asarray(draw["gamma_k"]))

    bw = geom.eff_range_phi ** 2 / 6.0
    W_gphi = u.build_weight_matrix(new, geom.knots_phi, kernel="gaussian",
                                   radius=geom.radius_S, bandwidth=bw, cutoff=False)
    phi_k = np.asarray(draw["phi_k"])
    phi_new = (W_gphi @ phi_k if cfg.phi_domain == PHI_UNIT
               else np.exp(W_gphi @ phi_k))
    return mean_Z, cov_Z, R_new, phi_new, gamma_bar_new, float(draw["alpha0"])


def predict_Y_new_sites(geom: NewSiteGeometry, draws, *, u_new, sigma_new,
                        xi_new, p_cens=0.95, M: int = 20, seed: int = 0):
    """Posterior-predictive Y at new sites (joint over the new locations).

    For each posterior draw, jointly draws M latent realisations at the new
    sites (covariance.visgp_predict_new_joint -> coherent across new sites and
    time), maps them through X = eps R^phi g(Z) -> pRW -> qCGP. Returns
    Y_pred of shape (n_new, T, B*M).
    """
    rng = np.random.default_rng(seed)
    new = np.atleast_2d(geom.new_coords)
    n_new = new.shape[0]
    vt, vn = _newsite_visibility(geom)
    T = np.asarray(draws[0]["Z_train"]).shape[1]
    u_new = np.asarray(u_new)
    B = len(draws)
    Y_pred = np.empty((n_new, T, B * M))
    for b, draw in enumerate(draws):
        mean_Z, cov_Z, R_new, phi_new, gbar_new, alpha0 = _newsite_components(
            geom, draw, vt, vn)
        # per-draw holdout marginal (from fitted beta + holdout covariates if any)
        sigma_b, xi_b = _holdout_marginal(geom, draw, T, sigma_new, xi_new)
        Zd = md.draw_Z_missing(mean_Z, cov_Z, rng, n_draws=M)   # (n_new, T, M)
        if Zd.ndim == 2:
            Zd = Zd[:, :, None]
        for m in range(M):
            Wf = u.g(Zd[:, :, m])
            eps = u.rloglaplace(size=Wf.shape, alpha0=alpha0, rng=rng)
            X = eps * (R_new ** phi_new[:, None]) * Wf
            for t in range(T):
                pX = pRW(X[:, t], phi_new, gbar_new, alpha0)
                Y_pred[:, t, b * M + m] = np.where(
                    pX <= p_cens, u_new,
                    u.qCGP(pX, p_cens, u_new, sigma_b[:, t], xi_b[:, t]))
    return Y_pred


def predictive_loglik_new_sites(geom: NewSiteGeometry, draws, holdout: HoldoutData,
                                *, M: int = 20, seed: int = 0):
    """Two-level Monte-Carlo predictive log-likelihood at NEW sites.

    Mirrors predictive_loglik but draws the latent field jointly at the new
    locations via the VisGP new-site predictor. Returns total / per_draw scores.
    """
    rng = np.random.default_rng(seed)
    new = np.atleast_2d(geom.new_coords)
    Y = holdout.Y
    n_new, T = Y.shape
    uvec = np.asarray(holdout.u); p = holdout.p_cens
    vt, vn = _newsite_visibility(geom)
    B = len(draws)
    loglik_BTM = np.full((B, T, M), -np.inf)
    f = np.ascontiguousarray
    for b, draw in enumerate(draws):
        mean_Z, cov_Z, R_new, phi_new, gbar_new, alpha0 = _newsite_components(
            geom, draw, vt, vn)
        sigma, xi = _holdout_marginal(geom, draw, T, holdout.sigma, holdout.xi)
        Zd = md.draw_Z_missing(mean_Z, cov_Z, rng, n_draws=M)
        if Zd.ndim == 2:
            Zd = Zd[:, :, None]
        for t in range(T):
            pY_t = u.pCGP(Y[:, t], p, uvec, sigma[:, t], xi[:, t])
            qRW_t = qRW(pY_t, phi_new, gbar_new, alpha0)
            dRW_t = dRW(qRW_t, phi_new, gbar_new, alpha0)
            cens = np.flatnonzero(Y[:, t] <= uvec).astype(np.int64)
            exc = np.flatnonzero(Y[:, t] > uvec).astype(np.int64)
            for m in range(Zd.shape[2]):
                loglik_BTM[b, t, m] = u.Y_loglik_1t_best(
                    f(Y[:, t]), float(p), f(uvec), f(sigma[:, t]), f(xi[:, t]),
                    f(R_new[:, t]), f(Zd[:, t, m]), None, f(phi_new), f(gbar_new),
                    float(alpha0), f(qRW_t), f(dRW_t), cens, exc)
    flat = loglik_BTM.transpose(1, 0, 2).reshape(T, B * M)
    per_time = scipy.special.logsumexp(flat, axis=1) - np.log(B * M)
    per_draw = np.sum(scipy.special.logsumexp(loglik_BTM, axis=2) - np.log(M), axis=1)
    return {"total": float(np.sum(per_time)), "per_draw": per_draw,
            "per_time": per_time}


def crps_holdout_new_sites(geom: NewSiteGeometry, draws, holdout: HoldoutData, *,
                           M: int = 20, seed: int = 0,
                           tail_threshold: Optional[float] = None,
                           tail_smooth: Optional[float] = None):
    """Vanilla + tail-weighted CRPS at NEW sites (joint predictive draws)."""
    Y_pred = predict_Y_new_sites(geom, draws, u_new=holdout.u,
                                 sigma_new=holdout.sigma, xi_new=holdout.xi,
                                 p_cens=holdout.p_cens, M=M, seed=seed)
    n_new, T, _ = Y_pred.shape
    chains = {"vanilla": chain_identity}
    if tail_threshold is not None:
        chains["twcrps"] = (chain_smooth(tail_threshold, tail_smooth)
                            if tail_smooth is not None
                            else chain_threshold(tail_threshold))
    out = {}
    for name, chain in chains.items():
        grid = np.empty((n_new, T))
        for i in range(n_new):
            for t in range(T):
                grid[i, t] = weighted_crps_empirical(Y_pred[i, t, :],
                                                     holdout.Y[i, t], chain=chain)
        out[name] = grid
        out[name + "_mean"] = float(np.mean(grid))
    out["Y_pred"] = Y_pred
    return out


def score_holdout_all(geom: NewSiteGeometry, draws, holdout: HoldoutData, *,
                      M: int = 20, seed: int = 0,
                      tail_threshold: Optional[float] = None,
                      tail_smooths=(), progress=None, comm=None, regions=None):
    """Fused new-site scoring: predictive log-lik AND CRPS (vanilla + hard- and
    any number of smooth-threshold TWCRPS) in a SINGLE pass over posterior draws.

    The O(n^3) new-site kriging (`_newsite_components`) is computed ONCE per draw
    and reused for both the log-likelihood (conditioning on the holdout Y) and the
    forward posterior-predictive Y draws used by CRPS -- ~3x fewer kriging solves
    than calling predictive_loglik_new_sites + crps_holdout_new_sites twice (as
    the old score_holdout did).

    MPI: pass an mpi4py `comm` to parallelise the (expensive) per-draw kriging
    across ranks. Draws are assigned strided (rank, rank+nranks, ...); the log-lik
    is reduced with a cross-rank logsumexp and the forward-predictive Y_pred slices
    are gathered to rank 0, which scores the empirical CRPS over the full sample
    pool. Each global draw is seeded independently (default_rng([seed, b]) for the
    log-lik field, [seed+1, b] for the predictive field), so the result is
    BIT-FOR-BIT independent of the rank count (up to floating-point reduction
    order) -- size-1 comm == comm=None == size-64 comm. The empirical CRPS is
    permutation-invariant in the sample pool, so the gather order is irrelevant.
    Off rank 0 the returned crps_* fields are the rank-0 values (bcast).

    NOTE: per-draw seeding differs from the old single-stream serial RNG, so
    absolute scores shift slightly vs. pre-MPI runs (no test pins them); the
    estimator and its Monte-Carlo error are unchanged.

    `tail_smooths`: iterable of smoothing bandwidths -> one smooth TWCRPS each.
    `progress(b, B)`: optional callback invoked after each LOCAL draw (rank 0).
    Returns dict: loglik {total,per_draw,per_time}, crps_vanilla_mean,
    twcrps_hard_mean, twcrps_smooth_mean (dict keyed by bandwidth), n_draws.

    `regions`: optional {name: local site indices into the holdout arrays}. Scores
    EVERY named region from this ONE pass and returns {"by_region": {name: <the dict
    above>}, "n_draws": B}. Omit it and the return contract above is unchanged.

    Why this exists. The log-score is a JOINT density over a region's sites, so
    region scores are NOT additive: log p(Y_all) != sum of the block log p's,
    because the Monte-Carlo logsumexp over latent-field draws does not distribute
    over the site-sum. Callers therefore used to re-run the whole pass per region,
    doubling the cost of a multi-day job. But the blocks are EXTRACTABLE even though
    they are not combinable: conditional on Z the sites ARE independent, so the
    per-site log-densities this pass already computes can be summed over any index
    set, and the marginal of a joint Z draw over a subset is that subset's correct
    marginal. So one pass yields the pooled score and every block score, using
    common random numbers.
    """
    if comm is not None:
        rank, nranks = comm.Get_rank(), comm.Get_size()
    else:
        rank, nranks = 0, 1
    Y = holdout.Y
    n_new, T = Y.shape
    uvec = np.asarray(holdout.u); p = holdout.p_cens
    vt, vn = _newsite_visibility(geom)
    B = len(draws)
    f = np.ascontiguousarray
    my_idx = list(range(rank, B, nranks))     # strided global draw indices for this rank
    Bloc = len(my_idx)
    # Region axis. regions=None keeps ONE region spanning every site AND the original
    # summed (numba, when available) kernel, so that path stays exactly as it was.
    # With regions we switch to the pointwise kernel -- which sums to the very same
    # value -- and project onto the regions with an indicator matrix. The matrix (not
    # a partition-based bincount) is what lets regions OVERLAP, e.g. a pooled "all"
    # alongside the blocks it contains.
    reg_names = list(regions) if regions else None
    if reg_names:
        A = np.zeros((len(reg_names), n_new))
        for r, nm in enumerate(reg_names):
            A[r, np.asarray(regions[nm], dtype=int)] = 1.0
        n_reg = len(reg_names)
    else:
        A, n_reg = None, 1
    loglik_BTM = np.full((Bloc, T, M, n_reg), -np.inf)
    Y_pred = np.empty((n_new, T, Bloc * M))
    for j, b in enumerate(my_idx):
        draw = draws[b]
        rng_ll = np.random.default_rng([seed, b])         # per-draw -> rank-independent
        rng_pred = np.random.default_rng([seed + 1, b])
        mean_Z, cov_Z, R_new, phi_new, gbar_new, alpha0 = _newsite_components(
            geom, draw, vt, vn)
        # --- log-likelihood (condition on holdout Y) ---
        sig_h, xi_h = _holdout_marginal(geom, draw, T, holdout.sigma, holdout.xi)
        Zll = md.draw_Z_missing(mean_Z, cov_Z, rng_ll, n_draws=M)
        if Zll.ndim == 2:
            Zll = Zll[:, :, None]
        for t in range(T):
            pY_t = u.pCGP(Y[:, t], p, uvec, sig_h[:, t], xi_h[:, t])
            qRW_t = qRW(pY_t, phi_new, gbar_new, alpha0)
            dRW_t = dRW(qRW_t, phi_new, gbar_new, alpha0)
            cens = np.flatnonzero(Y[:, t] <= uvec).astype(np.int64)
            exc = np.flatnonzero(Y[:, t] > uvec).astype(np.int64)
            for m in range(Zll.shape[2]):
                if A is None:
                    loglik_BTM[j, t, m, 0] = u.Y_loglik_1t_best(
                        f(Y[:, t]), float(p), f(uvec), f(sig_h[:, t]), f(xi_h[:, t]),
                        f(R_new[:, t]), f(Zll[:, t, m]), None, f(phi_new), f(gbar_new),
                        float(alpha0), f(qRW_t), f(dRW_t), cens, exc)
                else:
                    pw = u.Y_loglik_1t_pointwise(
                        f(Y[:, t]), float(p), f(uvec), f(sig_h[:, t]), f(xi_h[:, t]),
                        f(R_new[:, t]), f(Zll[:, t, m]), None, f(phi_new), f(gbar_new),
                        float(alpha0), f(qRW_t), f(dRW_t), cens, exc)
                    loglik_BTM[j, t, m, :] = A @ pw
        # --- forward predictive Y (for CRPS) ---
        Zpr = md.draw_Z_missing(mean_Z, cov_Z, rng_pred, n_draws=M)
        if Zpr.ndim == 2:
            Zpr = Zpr[:, :, None]
        for m in range(M):
            Wf = u.g(Zpr[:, :, m])
            eps = u.rloglaplace(size=Wf.shape, alpha0=alpha0, rng=rng_pred)
            X = eps * (R_new ** phi_new[:, None]) * Wf
            for t in range(T):
                pX = pRW(X[:, t], phi_new, gbar_new, alpha0)
                Y_pred[:, t, j * M + m] = np.where(
                    pX <= holdout.p_cens, uvec,
                    u.qCGP(pX, holdout.p_cens, uvec, sig_h[:, t], xi_h[:, t]))
        if progress is not None and rank == 0:
            progress(j + 1, Bloc)
    # --- aggregate log-likelihood (rank-independent) ---
    # per_time = logsumexp over ALL B*M values per t, minus log(B*M); combine the
    # per-rank partial logsumexps across ranks (logsumexp is associative).
    # (n_reg, T): every region reduced in one go, so the MPI collectives below stay
    # at one call each regardless of how many regions are being scored.
    if Bloc > 0:
        part_t = scipy.special.logsumexp(
            loglik_BTM.transpose(3, 1, 0, 2).reshape(n_reg, T, Bloc * M), axis=2)
    else:
        part_t = np.full((n_reg, T), -np.inf)
    if comm is not None:
        from mpi4py import MPI
        parts = np.array(comm.allgather(part_t))                # (nranks, n_reg, T)
        per_time = scipy.special.logsumexp(parts, axis=0) - np.log(B * M)
    else:
        per_time = part_t - np.log(B * M)
    # per_draw in global draw order (only `total` is used downstream, but keep the
    # contract): each global index is owned by exactly one rank -> Allreduce(SUM).
    per_draw = np.zeros((n_reg, B))
    if Bloc > 0:
        pd_loc = np.sum(scipy.special.logsumexp(loglik_BTM, axis=2) - np.log(M),
                        axis=1)                                  # (Bloc, n_reg)
        per_draw[:, my_idx] = pd_loc.T
    if comm is not None:
        tot = np.zeros_like(per_draw)
        comm.Allreduce(per_draw, tot, op=MPI.SUM)
        per_draw = tot
    # --- aggregate CRPS variants: gather the predictive pool to rank 0 ---
    chains = {"vanilla": chain_identity}
    if tail_threshold is not None:
        chains["twcrps_hard"] = chain_threshold(tail_threshold)
        for bw in tail_smooths:
            chains[f"twcrps_smooth_{bw:g}"] = chain_smooth(tail_threshold, bw)
    if comm is not None:
        gathered = comm.gather(Y_pred, root=0)            # list (rank0) or None
        Y_pool = np.concatenate(gathered, axis=2) if rank == 0 else None
    else:
        Y_pool = Y_pred
    # CRPS is a MEAN of per-site scores, so unlike the log-score it needs no special
    # treatment: score every site once, then average within each region.
    crps, crps_t = {}, {}
    if rank == 0:
        for name, chain in chains.items():
            grid = np.empty((n_new, T))
            for i in range(n_new):
                for t in range(T):
                    grid[i, t] = weighted_crps_empirical(Y_pool[i, t, :], Y[i, t],
                                                         chain=chain)
            sets = ([np.asarray(regions[nm], dtype=int) for nm in reg_names]
                    if reg_names else [np.arange(n_new)])
            crps[name] = [float(np.mean(grid[ix])) for ix in sets]
            # Mean over SITES for each time replicate. This is what a standard error
            # needs: the (site, time) cells are NOT independent -- within a time they
            # share one latent field -- so treating all n*T of them as independent
            # units understates the error badly. The T time replicates are the
            # exchangeable unit, and se = sd(crps_t)/sqrt(T).
            crps_t[name] = [grid[ix].mean(axis=0) for ix in sets]
    if comm is not None:
        crps = comm.bcast(crps, root=0)
        crps_t = comm.bcast(crps_t, root=0)

    def _pack(r):
        return {"loglik": {"total": float(np.sum(per_time[r])),
                           "per_draw": per_draw[r], "per_time": per_time[r]},
                "crps_vanilla_mean": crps["vanilla"][r],
                "twcrps_hard_mean": (crps["twcrps_hard"][r]
                                     if "twcrps_hard" in crps else None),
                "twcrps_smooth_mean": {bw: crps[f"twcrps_smooth_{bw:g}"][r]
                                       for bw in tail_smooths},
                # per-TIME contributions, kept so a standard error (and, far more
                # usefully, a PAIRED difference between two models) can be formed
                # afterwards without re-running the pass
                "crps_per_time": {k: v[r] for k, v in crps_t.items()},
                "n_draws": B}

    if reg_names:
        return {"by_region": {nm: _pack(r) for r, nm in enumerate(reg_names)},
                "n_draws": B}
    return _pack(0)


# ---------------------------------------------------------------------------
# WAIC (conditional, per-(site,time))
# ---------------------------------------------------------------------------
def _train_forward(draw, geom):
    """Deterministic per-draw forward pieces at `geom.train_coords` (no latent
    sampling): R = W_wend @ S (the draw's S), the phi field, and gamma_bar."""
    coords = geom.train_coords
    cfg = geom.config
    n = coords.shape[0]
    bw_phi = geom.eff_range_phi ** 2 / 6.0
    W_phi = u.build_weight_matrix(coords, geom.knots_phi, kernel="gaussian",
                                  radius=geom.radius_S, bandwidth=bw_phi, cutoff=False)
    phi = W_phi @ np.asarray(draw["phi_k"])
    if cfg.phi_domain != PHI_UNIT:
        phi = np.exp(phi)
    gamma_k = np.asarray(draw["gamma_k"])
    kS = geom.knots_S.shape[0]
    if cfg.model_type == "huser-wadsworth":
        W_wend = np.ones((n, kS))
    else:
        W_wend = kernels.build_wendland_weights(
            coords, geom.knots_S,
            np.repeat(float(draw.get("radius_S", geom.radius_S)), kS),
            config=cfg, aniso_ratio=float(draw.get("aniso_ratio", 1.0)),
            aniso_angle=float(draw.get("aniso_angle", 0.0)), water_poly=geom.water_poly)
    R = W_wend @ np.asarray(draw["S"])                       # (n, T)
    gbar = kernels.gamma_bar_from_weights(W_wend, gamma_k)
    return R, phi, gbar


# Per-cell p_waic above which a (site,time) cell is deemed to make WAIC unreliable
# (pointwise log-lik varies by > ~10 nats across draws). See waic_conditional.
_PWAIC_CELL_MAX = 100.0


def waic_conditional(geom: NewSiteGeometry, draws, train: HoldoutData, *,
                     seed: int = 0, progress=None, comm=None) -> dict:
    """Per-(site,time) CONDITIONAL WAIC on the training data.

    Observation unit = an observed (site, time) cell; the pointwise likelihood
    conditions on each draw's FITTED latent field:
        ll[s,(i,t)] = log p(Y_{i,t} | Z^s_{i,t}, R^s_{i,t}, phi^s, alpha0^s).
    Conditioning on the fitted field makes this both tractable (no latent
    integration -> no Monte-Carlo blow-up) AND sensitive to the dependence
    structure (a smoother covariance yields a smoother fitted Z). Then
        lppd   = sum_{i,t} logmeanexp_s ll[s,i,t]
        p_WAIC = sum_{i,t} var_s( ll[s,i,t] )
        WAIC   = -2 (lppd - p_WAIC).
    Pure POST-PROCESSING from the saved latent (Z, S) + traces -- no effect on
    fitting, nothing extra saved. Missing cells are excluded. (Caveat: a
    conditional criterion targets leave-one-cell-out GIVEN the field, not a new
    replicate -- see the holdout predictive scores for the new-site task.)

    MPI: draws are split strided across `comm`; the per-cell streaming
    log-sum-exp and moments are combined with Allreduce(MAX/SUM), so the result
    is independent of the rank count.
    Returns: waic, lppd, p_waic, elpd_waic (=lppd-p_waic), n_obs, n_draws.
    """
    if comm is not None:
        rank, nranks = comm.Get_rank(), comm.Get_size()
    else:
        rank, nranks = 0, 1
    Y = train.Y
    n, T = Y.shape
    sigma = _as_2d(train.sigma, T); xi = _as_2d(train.xi, T)
    uvec = np.asarray(train.u); p = train.p_cens
    obs = ~np.isnan(Y)
    f = np.ascontiguousarray
    B = len(draws)
    my_idx = list(range(rank, B, nranks))
    # streaming accumulators over draws, per (site, time) cell
    run_max = np.full((n, T), -np.inf)                       # running max of ll
    run_se = np.zeros((n, T))                                # running sum exp(ll-max)
    s1 = np.zeros((n, T)); s2 = np.zeros((n, T)); cnt = 0    # for the variance
    for jj, b in enumerate(my_idx):
        draw = draws[b]
        Z = np.asarray(draw["Z_train"], dtype=float)         # (n, T) FITTED field
        alpha0 = float(draw["alpha0"])
        R, phi, gbar = _train_forward(draw, geom)
        L = np.zeros((n, T))                                  # pointwise ll (obs cells)
        for t in range(T):
            ot = np.flatnonzero(obs[:, t])
            if ot.size == 0:
                continue
            Yt = Y[ot, t]; ut = uvec[ot]
            pY = u.pCGP(Yt, p, ut, sigma[ot, t], xi[ot, t])
            qv = qRW(pY, phi[ot], gbar[ot], alpha0)
            dv = dRW(qv, phi[ot], gbar[ot], alpha0)
            cens = np.flatnonzero(Yt <= ut).astype(np.int64)
            exc = np.flatnonzero(Yt > ut).astype(np.int64)
            L[ot, t] = u.Y_loglik_1t_pointwise(
                f(Yt), float(p), f(ut), f(sigma[ot, t]), f(xi[ot, t]),
                f(R[ot, t]), f(Z[ot, t]), None, f(phi[ot]), f(gbar[ot]),
                float(alpha0), f(qv), f(dv), cens, exc)
        nm = np.maximum(run_max, L)
        run_se = run_se * np.exp(run_max - nm) + np.exp(L - nm)
        run_max = nm
        s1 += L; s2 += L * L; cnt += 1
        if progress is not None and rank == 0:
            progress(jj + 1, len(my_idx))
    # combine across ranks (rank-independent): global log-sum-exp via MAX+SUM
    if comm is not None:
        from mpi4py import MPI
        gmax = np.empty_like(run_max)
        comm.Allreduce(run_max, gmax, op=MPI.MAX)
        contrib = np.where(np.isfinite(gmax), run_se * np.exp(run_max - gmax), 0.0)
        gse = np.empty_like(contrib); comm.Allreduce(contrib, gse, op=MPI.SUM)
        lse = gmax + np.log(gse)
        gs1 = np.empty_like(s1); comm.Allreduce(s1, gs1, op=MPI.SUM); s1 = gs1
        gs2 = np.empty_like(s2); comm.Allreduce(s2, gs2, op=MPI.SUM); s2 = gs2
        Btot = comm.allreduce(cnt, op=MPI.SUM)
    else:
        lse = run_max + np.log(run_se); Btot = cnt
    lppd_it = lse - np.log(Btot)
    var_it = (s2 - s1 ** 2 / Btot) / (Btot - 1)
    m = obs
    pw = var_it[m]
    lppd = float(lppd_it[m].sum()); pwaic = float(pw.sum())
    elpd = lppd - pwaic
    # Robustness diagnostic. p_WAIC is a SUM of per-cell posterior variances of the
    # pointwise log-lik, so a handful of cells with a PATHOLOGICAL variance can
    # dominate and make WAIC untrustworthy -- the signature of an unstable deep-tail
    # density flip-flopping across draws (e.g. the historical _TINY floor put log p
    # at ~ -708, giving per-cell variances ~1e5). We deliberately do NOT use the
    # textbook p_waic_i > 0.4 flag: with heavy-tailed extremes over ~2e4 (site,time)
    # cells, MANY cells legitimately sit above 0.4 (healthy models show a per-cell
    # max ~5), so 0.4 fires constantly and is useless here. Instead flag cells whose
    # pointwise log-lik swings by more than ~10 nats across draws (variance > 100) --
    # ~20x the healthy max, but far below a density-cliff blow-up -- and expose the
    # single worst-cell variance so a regression announces itself.
    pw_max = float(pw.max()) if pw.size else 0.0
    n_bad = int(np.count_nonzero(pw > _PWAIC_CELL_MAX))
    if n_bad:
        warnings.warn(
            f"WAIC: {n_bad}/{pw.size} cells have pointwise p_waic > {_PWAIC_CELL_MAX:g} "
            f"(max {pw_max:.4g}); p_waic is dominated by unstable cells and is likely "
            f"unreliable -- check the deep-tail density evaluation.", stacklevel=2)
    # per-time elpd contributions, grouped the same way as the holdout scores so the
    # two are comparable and a standard error can be formed later: cells within a time
    # share a latent field, so the TIME is the exchangeable unit, not the cell.
    elpd_it = np.where(m, lppd_it - var_it, 0.0)
    return {"waic": -2.0 * elpd, "lppd": lppd, "p_waic": pwaic, "elpd_waic": elpd,
            "n_obs": int(m.sum()), "n_draws": int(Btot),
            "elpd_per_time": elpd_it.sum(axis=0),
            "p_waic_max": pw_max, "n_pwaic_bad": n_bad}
