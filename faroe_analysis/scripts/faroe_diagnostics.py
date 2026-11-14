#!/usr/bin/env python3
"""
Model-assessment diagnostics for the Faroe analysis (used by Task 7).

Two families, ported from the shi-2026b / wang-2026a / GPD_ScaleMixture reference
schemes but NOT present in spot-bayes:

1. Extremal dependence chi(u), relative to reference locations (wang-2026a Fig. 2).
   Both sides use the FITTED marginal CDF (GPD_ScaleMixture scheme), NOT a rank
   transform, and both are recomputed at every posterior draw so that marginal
   uncertainty propagates into 2.5/97.5% bands:
   * empirical  : push the observed block maxima through the draw's censored-GPD CDF
                  pCGP, then chi_ij(u) = P(U_i>u, U_j>u)/(1-u).
   * model-based: SIMULATE fresh replicates from the fitted scale-mixture process
                  (shi-2026b scheme) and threshold them at the draw's site-specific
                  qRW_i(u) -- equivalent to pRW(X) > u, but ~700x cheaper.
   Because the margins are fitted rather than ranked, a MARGINAL misfit now shows up
   as chi bias; with ranks it could not. That is the point of the change (it is also
   what lets u exceed 1 - 1/Nt), but it means chi is no longer a pure dependence
   diagnostic -- read it alongside the QQ plots below.
   Levels u must exceed the censoring probability p_cens: below it the censored-GPD
   margin only identifies F <= p, so chi there is meaningless.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # faroe_analysis/scripts
import numpy as np                                     # noqa: E402
import scipy.stats                                     # noqa: E402
from scipy.stats import genpareto, beta as beta_dist            # noqa: E402
import utilities as u                                  # noqa: E402
import kernels                                          # noqa: E402
import marginal as mg                                  # noqa: E402
from sampler import ModelState, build_draws            # noqa: E402


# ==========================================================================
# chi(u): extremal dependence relative to reference locations
# ==========================================================================
def pCGP(Y, p, u_vec, sigma_vec, xi_vec):
    """Censored-GPD marginal CDF F(y)=P(Y<=y) on the (Ns, Nt) field Y.

    Above the site threshold u_i,  F(y) = p + (1-p) * G_i(y)  with G_i the
    GPD(loc=u_i, scale=sigma_i, shape=xi_i) CDF; this is the same marginal the
    likelihood censors at (utilities.Y_loglik_1t_*).  Below u_i the model only
    identifies F <= p, so we return exactly p -- harmless because chi is defined
    here only for levels u > p (see chi_vs_reference).  NaN in Y propagates.
    """
    Y = np.asarray(Y, float)
    uu = np.asarray(u_vec, float)[:, None]
    sg = np.asarray(sigma_vec, float)[:, None]
    xi = np.asarray(xi_vec, float)[:, None]
    U = np.where(np.isfinite(Y), float(p), np.nan)
    exc = np.isfinite(Y) & (Y > uu)
    if exc.any():
        i, _ = np.nonzero(exc)
        G = genpareto.cdf(Y[exc], xi[i, 0], loc=uu[i, 0], scale=sg[i, 0])
        U[exc] = p + (1.0 - p) * G
    return U


def draw_sigma_xi(draw, sigma_spatial, xi_spatial):
    """Per-site (sigma, xi) vectors for one posterior draw (vector _site_sigma_xi)."""
    bsig = np.asarray(draw["beta_sigma"], float)
    bxi = np.asarray(draw["beta_xi"], float)
    sigma = np.exp(bsig[0] + sigma_spatial @ bsig[1:])
    xi = mg.xi_from_eta(bxi[0] + xi_spatial @ bxi[1:])
    return sigma, xi


def chi_vs_reference(U, ref_idx, us):
    """chi_ij(u) of every site j relative to reference site `ref_idx`.

    U : (Ns, R) fitted-CDF uniform margins (per-site). Returns (Ns, len(us)):
        chi_j(u) = P(U_ref > u & U_j > u) / (1 - u), estimated over the R columns
        (using only columns where both are finite).

    NB the margins are FITTED, not ranks: P(U_ref > u) equals 1-u only if the
    marginal fits, so a marginal misfit shows up here as chi bias (by design --
    that is the diagnostic). Levels u must exceed the censoring probability p.
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


def chi_vs_reference_thresh(X, thr, ref_idx, us):
    """chi of every site vs `ref_idx` from a RAW field X (Ns, R) and site-specific
    thresholds thr (len(us), Ns).

    Equivalent to chi_vs_reference(pRW(X), ...) because pRW is monotone -- but it
    costs len(us)*Ns qRW root-finds per draw instead of Ns*R pRW evaluations, which
    for (Ns, R) = (432, 2000) is ~700x less work.
    """
    us = np.atleast_1d(np.asarray(us, float))
    Ns = X.shape[0]
    fin = np.isfinite(X)
    ref_fin = fin[ref_idx]
    n_both = (fin & ref_fin).sum(axis=1)
    out = np.full((Ns, us.size), np.nan)
    for k, uu in enumerate(us):
        E = fin & (X > thr[k][:, None])
        joint = (E & E[ref_idx]).sum(axis=1)
        ok = n_both >= 2
        out[ok, k] = joint[ok] / n_both[ok] / (1.0 - uu)
    return out


def model_thresholds(state, us):
    """Fitted-model site thresholds qRW_i(u) at the current draw. (len(us), Ns)."""
    us = np.atleast_1d(np.asarray(us, float))
    return np.stack([u.qRW(np.full(state.Ns, uu), state.phi_vec, state.gamma_bar,
                           state.alpha0) for uu in us])


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
    """Simulate n_rep fresh replicates of the scale-mixture process X(s) at the
    current ModelState parameters. Returns X (Ns, n_rep)."""
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

    Conditions on the inferred latent field rather than integrating it out, so it
    reflects the dependence actually present in the fitted field (and tends to
    track the empirical chi more closely, since Z/S were fit to the data)."""
    Z = np.asarray(draw["Z_train"], dtype=np.float64)   # (Ns, T)
    S = np.asarray(draw["S"], dtype=np.float64)          # (k_S, T)
    Wf = u.g(Z)
    R = state.W_wend_S @ S                                # (Ns, T)
    X_star = np.power(R, state.phi_vec[:, None]) * Wf
    eps = u.rloglaplace(size=X_star.shape, alpha0=state.alpha0, rng=rng)
    return eps * X_star

def simulate_field_prior_z(state, draw, rng):
    """COMPROMISE field: keep the draw's INFERRED S but resample the Gaussian field Z
    FRESH from its prior N(0, Sigma) at the current draw's covariance; redraws the
    log-Laplace nugget. Returns X (Ns, T), T = #observed times (matched to S).

    Sits between the marginal field (fresh Z AND fresh S) and the conditional field
    (inferred Z AND inferred S): it isolates the contribution of the FITTED Z
    dependence -- contrast with the conditional chi to see how much of the fitted
    dependence is carried by Z versus by the shared scale mixture S."""
    S = np.asarray(draw["S"], dtype=np.float64)          # (k_S, T)
    T = S.shape[1]
    L = state.zchol()                                    # cov = L L^T
    Z = L @ rng.standard_normal((state.Ns, T))           # fresh prior Z, T replicates
    Wf = u.g(Z)
    R = state.W_wend_S @ S                                # inferred S
    X_star = np.power(R, state.phi_vec[:, None]) * Wf
    eps = u.rloglaplace(size=X_star.shape, alpha0=state.alpha0, rng=rng)
    return eps * X_star

def chi_assess(run, cfg, data, settings, Y, u_vec, sigma_spatial, xi_spatial, p_cens,
               ref_idx, us, *, sigma_fixed=None, xi_fixed=None,
               n_draws=150, n_rep=2000, burn=0.5, seed=0, comm=None, progress=None):
    """Model-based AND empirical chi vs each reference location, on the FITTED-CDF
    uniform scale (GPD_ScaleMixture scheme), with posterior bands.

    Three model constructions are computed side by side, all thresholded at the same
    per-draw qRW_i(u) so they share a scale (see simulate_field* for the contrast):
      * model (marginal)    : simulate n_rep FRESH replicates X (Z ~ N(0,Sigma),
                              S ~ Levy) -> the model's *implied* dependence given only
                              the hyperparameters.
      * model (prior Z)     : keep the draw's INFERRED S but resample Z fresh from its
                              prior N(0,Sigma) -> a compromise isolating the fitted Z
                              field's contribution to the dependence.
      * model (conditional) : reuse the draw's INFERRED Z/S at the observed times,
                              redrawing only the nugget -> reflects the dependence in
                              the fitted field (tends to track empirical, since Z/S
                              were fit to the data). Needs the latent traces (already
                              required by build_draws).
      * empirical           : push the OBSERVED Y through the draw's censored-GPD CDF
                              pCGP.

    Levels `us` must all exceed p_cens -- below the censoring threshold the fitted
    margin is not identified (F <= p there) and chi would be meaningless.

    Returns dict of (len(ref_idx), Ns, len(us)) arrays: model_mean/lo/hi,
    model_priorz_mean/lo/hi, model_cond_mean/lo/hi, emp_mean/lo/hi, plus n_draws_used.
    MPI: draws strided over `comm`; each rank fills only its own slots of the
    zero-initialised per-draw arrays, so a single Allreduce(SUM) reconstructs them
    rank-independently.
    """
    us = np.atleast_1d(np.asarray(us, float))
    if np.any(us <= p_cens):
        raise ValueError(f"chi levels {us.tolist()} must all exceed p_cens={p_cens}: "
                         "the censored-GPD margin is unidentified below it.")
    # A fixed-margin fit (--fix-margins) records no beta traces: sampler.py only
    # allocates them when update_sigma/update_xi. Then the margin is CONSTANT across
    # draws, and the empirical bands correctly collapse -- there is no marginal
    # uncertainty to propagate.
    sampled_margins = "beta_sigma" in run and "beta_xi" in run
    if not sampled_margins and (sigma_fixed is None or xi_fixed is None):
        raise ValueError("Run has no beta_sigma/beta_xi traces (margins were fixed); "
                         "pass sigma_fixed/xi_fixed so the empirical chi can use the "
                         "margin the fit actually used.")
    nrec = run["phi_k"].shape[0]
    lo = int(burn * nrec)
    avail = nrec - lo
    thin = max(1, int(np.ceil(avail / n_draws)))
    draws = build_draws(run, burn=lo, thin=thin)
    state = ModelState(cfg, data, settings)
    B, R = len(draws), len(ref_idx)
    mod = np.zeros((B, R, state.Ns, us.size))          # model chi, marginal (fresh Z/S)
    mod_pz = np.zeros((B, R, state.Ns, us.size))        # model chi, prior Z + post. S
    mod_c = np.zeros((B, R, state.Ns, us.size))         # model chi, conditional (post. Z/S)
    emp = np.zeros((B, R, state.Ns, us.size))
    cnt = 0
    rank, nranks = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    mine = range(rank, B, nranks)
    for b in mine:
        rng = np.random.default_rng(seed * 100003 + b + 1)
        _apply_draw(state, draws[b], settings.alpha_stable)
        X = simulate_field(state, n_rep, rng)
        # prior-Z compromise + conditional field: each gets an independent nugget/Z
        # RNG so the marginal stream above (and its numbers) stay unchanged.
        rng_pz = np.random.default_rng(seed * 100003 + b + 1 + 900007)
        X_pz = simulate_field_prior_z(state, draws[b], rng_pz)
        rng_c = np.random.default_rng(seed * 100003 + b + 1 + 500009)
        X_c = simulate_field_conditional(state, draws[b], rng_c)
        thr = model_thresholds(state, us)               # same qRW_i(u) for all fields
        if sampled_margins:
            sigma, xi = draw_sigma_xi(draws[b], sigma_spatial, xi_spatial)
        else:
            sigma, xi = sigma_fixed, xi_fixed
        U_dat = pCGP(Y, p_cens, u_vec, sigma, xi)
        for r, ri in enumerate(ref_idx):
            mod[b, r] = chi_vs_reference_thresh(X, thr, ri, us)
            mod_pz[b, r] = chi_vs_reference_thresh(X_pz, thr, ri, us)
            mod_c[b, r] = chi_vs_reference_thresh(X_c, thr, ri, us)
            emp[b, r] = chi_vs_reference(U_dat, ri, us)
        cnt += 1
        if progress is not None and rank == 0:
            progress(cnt, len(mine))
    if comm is not None:
        from mpi4py import MPI
        for a in (mod, mod_pz, mod_c, emp):
            tot = np.empty_like(a); comm.Allreduce(a, tot, op=MPI.SUM); a[...] = tot
    q = [0.025, 0.975]
    mod_lo, mod_hi = np.nanquantile(mod, q, axis=0)
    modpz_lo, modpz_hi = np.nanquantile(mod_pz, q, axis=0)
    modc_lo, modc_hi = np.nanquantile(mod_c, q, axis=0)
    emp_lo, emp_hi = np.nanquantile(emp, q, axis=0)
    return dict(model_mean=np.nanmean(mod, axis=0), model_lo=mod_lo, model_hi=mod_hi,
                model_priorz_mean=np.nanmean(mod_pz, axis=0),
                model_priorz_lo=modpz_lo, model_priorz_hi=modpz_hi,
                model_cond_mean=np.nanmean(mod_c, axis=0),
                model_cond_lo=modc_lo, model_cond_hi=modc_hi,
                emp_mean=np.nanmean(emp, axis=0), emp_lo=emp_lo, emp_hi=emp_hi,
                n_draws_used=B)


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
                   p_cens=0.95, burn=0.5, n_draws=300,
                   sigma_fixed=None, xi_fixed=None):
    """QQ data for one site on the standard-Gumbel scale.

    Pushes the observed exceedances (Y > u) through the posterior-AVERAGED
    CONDITIONAL GPD CDF, G(y) = P(Y <= y | Y > u), which is standard uniform if the
    marginal fits, and maps to Gumbel via -log(-log G). Compared against standard
    Gumbel plotting positions with binomial 95% bands (uniform order statistics).

    A --fix-margins fit records no beta traces; pass sigma_fixed/xi_fixed (the
    per-site surfaces the fit used) and the CDF is evaluated at that single margin
    (no posterior averaging -- there is no marginal posterior to average over).
    """
    y = Y[site]
    uu = float(u_vec[site])
    exc = np.isfinite(y) & (y > uu)
    yv = np.sort(y[exc])
    n = yv.size
    if n < 5:
        return None
    if "beta_sigma" not in run:                 # margins were FIXED at the fit
        if sigma_fixed is None or xi_fixed is None:
            raise ValueError("Run has no beta traces (fixed margins); pass "
                             "sigma_fixed/xi_fixed.")
        cdf = genpareto.cdf(yv, float(xi_fixed[site]), loc=uu,
                            scale=float(sigma_fixed[site]))
    else:
        nrec = run["beta_sigma"].shape[0]
        lo = int(burn * nrec)
        idx = np.arange(lo, nrec)
        if idx.size > n_draws:
            idx = idx[np.linspace(0, idx.size - 1, n_draws).astype(int)]
        sig_row = sigma_spatial[site]
        xi_row = xi_spatial[site]
        # posterior-averaged CONDITIONAL GPD CDF at each (sorted) exceedance
        cdf_acc = np.zeros(n)
        for i in idx:
            draw = {"beta_sigma": run["beta_sigma"][i], "beta_xi": run["beta_xi"][i]}
            s, x = _site_sigma_xi(draw, sig_row, xi_row)
            cdf_acc += genpareto.cdf(yv, x, loc=uu, scale=s)
        cdf = cdf_acc / idx.size
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
