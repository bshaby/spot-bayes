#!/usr/bin/env python3
"""
Model-assessment diagnostics for the NZ analysis (used by Task 7).

Two families, ported from the shi-2026b / wang-2026a / GPD_ScaleMixture reference
schemes but NOT present in spot-bayes:

1. Extremal dependence chi(u), relative to reference locations (wang-2026a Fig. 2).
   * empirical  : rank-transform the observed block-maxima series to uniform per
                  site, then chi_ij(u) = P(U_i>u, U_j>u)/(1-u).
   * model-based: for each posterior draw, SIMULATE fresh replicates from the fitted
                  scale-mixture process (shi-2026b scheme), rank-transform to uniform
                  per site, compute chi, and average over draws.
   Both are reported as chi of every site RELATIVE TO a small set of reference
   locations (no moving window), for a few thresholds u.

2. Site-wise QQ plots against a standard Gumbel margin (shi-2026b). For each chosen
   site the observed exceedances are pushed through the posterior-AVERAGED censored
   -GPD CDF and mapped to the Gumbel scale (-log(-log F)); binomial 95% confidence
   bands come from the Beta(k, n-k+1) law of the uniform order statistics.

Reuses: sampler.ModelState (exact model geometry incl. VisGP / barrier kernels /
anisotropy), utilities.g / rloglaplace, marginal.xi_from_eta, scipy GPD.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nz_analysis/scripts
import numpy as np                                     # noqa: E402
import scipy.stats                                     # noqa: E402
from scipy.stats import rankdata, genpareto, beta as beta_dist   # noqa: E402
import utilities as u                                  # noqa: E402
import kernels                                          # noqa: E402
import marginal as mg                                  # noqa: E402
from sampler import ModelState, build_draws            # noqa: E402


# ==========================================================================
# chi(u): extremal dependence relative to reference locations
# ==========================================================================
def uniforms_by_rank(X, axis=1):
    """Rank-transform to uniform margins along `axis` (NaN-aware, per row/col)."""
    X = np.asarray(X, float)
    U = np.full_like(X, np.nan)
    if axis == 1:
        for i in range(X.shape[0]):
            row = X[i]
            fin = np.isfinite(row)
            n = int(fin.sum())
            if n > 1:
                U[i, fin] = rankdata(row[fin]) / (n + 1.0)
    else:
        return uniforms_by_rank(X.T, axis=1).T
    return U


def chi_vs_reference(U, ref_idx, us):
    """chi_ij(u) of every site j relative to reference site `ref_idx`.

    U : (Ns, R) uniform margins (per-site). Returns (Ns, len(us)):
        chi_j(u) = P(U_ref > u & U_j > u) / (1 - u), estimated over the R columns
        (using only columns where both are finite).
    """
    us = np.atleast_1d(np.asarray(us, float))
    Uref = U[ref_idx]
    Ns = U.shape[0]
    out = np.full((Ns, us.size), np.nan)
    for j in range(Ns):
        both = np.isfinite(Uref) & np.isfinite(U[j])
        if both.sum() < 2:
            continue
        a = Uref[both]
        b = U[j][both]
        for k, uu in enumerate(us):
            denom = 1.0 - uu
            out[j, k] = np.mean((a > uu) & (b > uu)) / denom if denom > 0 else np.nan
    return out


def empirical_chi(Y, ref_idx, us):
    """Empirical chi of every site vs reference from the observed series Y (Ns, Nt)."""
    U = uniforms_by_rank(Y, axis=1)
    return chi_vs_reference(U, ref_idx, us)


# -- model-based chi: simulate replicates from posterior draws ---------------
def _apply_draw(state, draw, alpha_stable):
    """Mutate a ModelState's geometry to a posterior draw's parameters (mirrors
    exp1_setup.warm_snapshot). Rebuilds the Wendland weights only when the geometry
    was sampled, then the covariance/precision."""
    state.phi_k = np.asarray(draw["phi_k"], float)
    state.phi_vec = state._phi_field(state.phi_k)
    state.rho_k = np.asarray(draw["rho_k"], float)
    state.rho_vec = state.W_gauss_rho @ state.rho_k
    state.alpha0 = float(draw["alpha0"])
    state.nu = float(draw.get("nu", state.nu))
    changed = False
    if draw.get("radius_S") is not None:
        state.radius_S = float(draw["radius_S"]); changed = True
    if "aniso_ratio" in draw:
        state.aniso_ratio = float(draw["aniso_ratio"])
        state.aniso_angle = float(draw["aniso_angle"]); changed = True
    if changed:
        state.W_wend_S = state.build_wendland(state.radius_S, state.aniso_ratio,
                                              state.aniso_angle)
        state.gamma_bar = kernels.gamma_bar_from_weights(
            state.W_wend_S, state.gamma_k, alpha=alpha_stable)
    state.rebuild_cov()


def simulate_field(state, n_rep, rng):
    """MARGINAL field: simulate n_rep FRESH replicates of the scale-mixture process
    X(s) at the current ModelState parameters -- both latent fields are redrawn from
    their conditional priors (Z ~ N(0, Sigma), S ~ Levy). Returns X (Ns, n_rep).

    This integrates the latent (Z, S) out, so the resulting chi is the model's
    *implied* extremal dependence given only the hyperparameters. Contrast with
    simulate_field_conditional, which reuses the inferred latent field."""
    Ns = state.Ns
    L = state.zchol()                                   # cov = L L^T (both branches)
    eta = rng.standard_normal((Ns, n_rep))
    Z = L @ eta
    Wf = u.g(Z)
    S_knots = scipy.stats.levy.rvs(loc=0.0, scale=state.gamma_k[:, None],
                                   size=(state.k_S, n_rep), random_state=rng)
    R = state.W_wend_S @ S_knots
    X_star = np.power(R, state.phi_vec[:, None]) * Wf
    eps = u.rloglaplace(size=X_star.shape, alpha0=state.alpha0, rng=rng)
    return eps * X_star


def simulate_field_conditional(state, draw, rng):
    """CONDITIONAL field: reconstruct the fitted process X(s, t) at the OBSERVED
    times from a posterior draw's inferred latent Z and S (NOT resimulated),
    redrawing only the log-Laplace nugget. Returns X (Ns, T).

    This is the GPD_ScaleMixture-style chi construction: it conditions on the
    inferred latent field rather than integrating it out, so it reflects the
    dependence actually present in the fitted field (and tends to track the
    empirical chi more closely, since Z/S were fit to the data)."""
    Z = np.asarray(draw["Z_train"], dtype=np.float64)   # (Ns, T)
    S = np.asarray(draw["S"], dtype=np.float64)          # (k_S, T)
    Wf = u.g(Z)
    R = state.W_wend_S @ S                                # (Ns, T)
    X_star = np.power(R, state.phi_vec[:, None]) * Wf
    eps = u.rloglaplace(size=X_star.shape, alpha0=state.alpha0, rng=rng)
    return eps * X_star


def model_chi(run, cfg, data, settings, ref_idx, us, *, n_draws=150, n_rep=2000,
              burn=0.5, seed=0, comm=None, progress=None, mode="both"):
    """Posterior-mean model-based chi of every site vs each reference location.

    `mode` selects the field construction (see simulate_field* for the distinction):
        "marginal"    -- fresh Z & S per draw (model-implied dependence; n_rep reps)
        "conditional" -- reuse the draw's inferred Z & S at the observed times
        "both"        -- compute both in a single draw loop (default)

    Returns a dict {mode: array (len(ref_idx), Ns, len(us))}, each averaged over
    posterior draws. MPI: draws split strided over `comm`, summed with Allreduce
    (rank-independent). The "conditional" mode requires the latent Z/S traces.
    """
    modes = ["marginal", "conditional"] if mode == "both" else [mode]
    if "conditional" in modes and ("Z" not in run or "S" not in run):
        raise ValueError("conditional chi needs latent Z/S traces (refit with "
                         "save_latent=True)")
    nrec = run["phi_k"].shape[0]
    lo = int(burn * nrec)
    avail = nrec - lo
    thin = max(1, int(np.ceil(avail / n_draws)))
    draws = build_draws(run, burn=lo, thin=thin)
    state = ModelState(cfg, data, settings)
    us = np.atleast_1d(np.asarray(us, float))
    R = len(ref_idx)
    acc = {m: np.zeros((R, state.Ns, us.size)) for m in modes}
    cnt = 0
    rank, nranks = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    for b in range(rank, len(draws), nranks):
        _apply_draw(state, draws[b], settings.alpha_stable)
        for m in modes:
            # independent RNG stream per (draw, mode) so the two are not coupled
            rng = np.random.default_rng(seed * 100003 + b * 7 + (m == "conditional"))
            X = (simulate_field(state, n_rep, rng) if m == "marginal"
                 else simulate_field_conditional(state, draws[b], rng))
            U = uniforms_by_rank(X, axis=1)
            for r, ri in enumerate(ref_idx):
                acc[m][r] += chi_vs_reference(U, ri, us)
        cnt += 1
        if progress is not None and rank == 0:
            progress(cnt, len(range(rank, len(draws), nranks)))
    if comm is not None:
        from mpi4py import MPI
        for m in modes:
            tot = np.empty_like(acc[m]); comm.Allreduce(acc[m], tot, op=MPI.SUM)
            acc[m] = tot
        cnt = comm.allreduce(cnt, op=MPI.SUM)
    return {m: acc[m] / max(cnt, 1) for m in modes}


# ==========================================================================
# Site-wise Gumbel QQ plots
# ==========================================================================
def _site_sigma_xi(draw, sigma_row, xi_row):
    """Per-draw (sigma, xi) at one site from its covariate row + the draw's betas."""
    bsig = np.asarray(draw["beta_sigma"], float)
    bxi = np.asarray(draw["beta_xi"], float)
    eta_s = bsig[0] + sigma_row @ bsig[1:]
    eta_x = bxi[0] + xi_row @ bxi[1:]
    return float(np.exp(eta_s)), float(mg.xi_from_eta(np.array([eta_x]))[0])


def gumbel_qq_site(run, site, Y, u_vec, sigma_spatial, xi_spatial, *,
                   p_cens=0.95, burn=0.5, n_draws=300):
    """QQ data for one site on the standard-Gumbel scale.

    Pushes the observed exceedances (Y > u) through the posterior-AVERAGED
    CONDITIONAL GPD CDF, G(y) = P(Y <= y | Y > u), which is standard uniform if the
    marginal fits, and maps to Gumbel via -log(-log G). Compared against standard
    Gumbel plotting positions with binomial 95% bands (uniform order statistics).
    """
    y = Y[site]
    uu = float(u_vec[site])
    exc = np.isfinite(y) & (y > uu)
    yv = np.sort(y[exc])
    n = yv.size
    if n < 5:
        return None
    sig_row = sigma_spatial[site]
    xi_row = xi_spatial[site]
    if "beta_sigma" in run and "beta_xi" in run:
        # margins were sampled: posterior-averaged CONDITIONAL GPD CDF over draws
        nrec = run["beta_sigma"].shape[0]
        lo = int(burn * nrec)
        idx = np.arange(lo, nrec)
        if idx.size > n_draws:
            idx = idx[np.linspace(0, idx.size - 1, n_draws).astype(int)]
        cdf_acc = np.zeros(n)
        for i in idx:
            draw = {"beta_sigma": run["beta_sigma"][i], "beta_xi": run["beta_xi"][i]}
            s, x = _site_sigma_xi(draw, sig_row, xi_row)
            cdf_acc += genpareto.cdf(yv, x, loc=uu, scale=s)
        cdf = cdf_acc / idx.size
    else:
        # margins were held FIXED (no beta trace): use the run's per-site sigma/xi
        s = float(run["sigma_vec"][site])
        x = float(run["xi_vec"][site])
        cdf = genpareto.cdf(yv, x, loc=uu, scale=s)
    cdf = np.clip(cdf, 1e-9, 1 - 1e-12)
    z_data = -np.log(-np.log(cdf))                      # fitted -> Gumbel (y-axis)
    k = np.arange(1, n + 1)
    p_theory = k / (n + 1.0)
    z_theory = -np.log(-np.log(p_theory))               # theoretical Gumbel (x-axis)
    p_lo = beta_dist.ppf(0.025, k, n - k + 1)
    p_hi = beta_dist.ppf(0.975, k, n - k + 1)
    z_lo = -np.log(-np.log(np.clip(p_lo, 1e-9, 1 - 1e-12)))
    z_hi = -np.log(-np.log(np.clip(p_hi, 1e-9, 1 - 1e-12)))
    return dict(z_theory=z_theory, z_data=z_data, z_lo=z_lo, z_hi=z_hi,
                n_exceed=n, u=uu)


# ==========================================================================
# helpers
# ==========================================================================
def nearest_site(sites_xy, lonlat):
    """Index of the site nearest a target (lon, lat)."""
    d = np.hypot(sites_xy[:, 0] - lonlat[0], sites_xy[:, 1] - lonlat[1])
    return int(np.argmin(d))
