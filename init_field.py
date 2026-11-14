#!/usr/bin/env python3
"""Plug-in ("X-inversion") starting values for the spatial-extremes sampler.

Dataset-generic: this module knows nothing about Faroe / NZ / any particular
analysis. It takes the same (ModelConfig, SamplerData, SamplerSettings) triple the
sampler takes and returns a ``ModelState.snapshot()`` dict suitable for
``run_mcmc(_mpi)(..., init_snapshot=snap)``.

WHY
---
The cold init (S=1, Z=0, phi=1, alpha0=alpha0_min+3, rho=1) puts the chain at the
Z-PRIOR MODE with a hopeless data fit, so the transient is spent inflating a
(Ns x Nt) Gaussian field out to its typical set before rho/nu are identified at
all. Measured on the NZ dataset (309 sites x 3080 blocks, fixed margins):

    cold default                                   data loglik  -2,230,008
    scalars tuned (phi .5, alpha0 2.05), S=1, Z=0                  -730,053
    + plug-in S/Z field                                            ~-132,000

and the plug-in Z lands INSIDE the Gaussian typical set (quadratic form 412k vs
the expected Ns*Nt/2 = 476k) rather than at the mode.

HOW
---
The observation model is

    X = R^phi * g(Z) * eps,   g(Z) = 1/Phibar(Z),   eps ~ log-Laplace(alpha0)
    X = qRW(pCGP(Y); phi, gamma_bar, alpha0)

so once the margins are fixed X is COMPUTABLE, and the initial field can be read
off by inverting the decomposition at eps = 1 (the log-Laplace mode):

  1. uniformize   V = pCGP(Y) at exceedances; below the threshold, the WITHIN-SITE
                  empirical rank of Y rescaled to (0, p_cens). The censored
                  likelihood discards the sub-threshold values, but they are still
                  perfectly good for building a starting configuration and they
                  carry the real spatial structure that a random Unif(0, p_cens)
                  draw would not.
  2. transform    X = qRW(V; phi, gamma_bar, alpha0)
  3. split        log X = phi*log R + r,  with r = log g(Z) ~ Exp(1). log S is the
                  least-squares projection of (log X - offset)/phi onto the Wendland
                  basis, shrunk by lam toward spatially constant (rows of W_wend_S
                  sum to 1, so lam=0 gives R constant in space -- which is also
                  exactly what the single-knot HW model can represent).
  4. read off     Z = Phi^{-1}(1 - e^{-r}), i.e. Phibar(Z) = e^{-r} = 1/g(Z).
  5. calibrate    solve the ONE free scalar (lam for a multi-knot S basis, else the
                  offset) so that sd(Z) = 1 -- Z must be marginally N(0,1) under the
                  model, which makes the split self-calibrating rather than tuned.
                  When neither knob can reach it (small or heavily censored data),
                  fall back to normal scores of r, which give the right Z marginal
                  by construction at the cost of eps != 1. See `z_map`.
  6. estimate     fit a Matern correlogram to the resulting Z field to get (nu, rho).
                  With Nt replicates this is a full-rank estimate, and it is the only
                  point in the pipeline where rho/nu see data BEFORE Z has inflated.

MARGIN SOURCES (the ladder, highest priority first)
---------------------------------------------------
Step 1 needs sigma/xi surfaces. In descending order of preference:

  1. explicit ``beta_sigma`` / ``beta_xi`` -- the STAGE-1 posterior mean in a
     two-stage run, or a workflow's initial-guess coefficients when there is no
     stage 1 (a full joint dependence+marginal fit still almost always has some
     initial guess to hand);
  2. ``data.sigma_vec`` / ``data.xi_vec`` -- per-site margins already fixed on the
     SamplerData (what ``update_GPD_*=False`` runs use);
  3. ModelState's own intercept-only default, built from the medians of those same
     vectors. Always available, so there is always a fallback.

``plugin_init_snapshot`` reports which rung it used via the returned ``info`` dict.

MPI
---
This module is MPI-free and fully DETERMINISTIC (no rng: the sub-threshold
imputation is a rank, not a draw), so every rank building the snapshot
independently gets a bit-identical result -- the same contract the other
``init_snapshot`` builders honour. Cost is dominated by one qRW sweep over all
(Ns x Nt) cells (~30 s at 1e6 cells).
"""
from __future__ import annotations

import numpy as np
import scipy.spatial.distance
from scipy.special import gamma as _gammafn, kv as _kv
from scipy.stats import norm, rankdata

from rw_inte import qRW
from sampler import ModelState

__all__ = ["plugin_init_snapshot", "plugin_field", "fit_matern_correlogram",
           "uniformize"]

# qRW's Brent root-find loses its bracket below ~1e-5 and returns NaN; the affected
# cells are the very bottom of a heavy left tail and carry no information about the
# latent field, so clipping there is harmless.
_V_FLOOR = 1e-4
# log g(Z) = -log Phibar(Z) is strictly positive, so the residual must be floored
# before the inverse transform. r = 0.02 maps to Z = -2.06 (a plausible field value,
# not an outlier), which is why the floor does not distort the Z marginal.
_R_FLOOR = 0.02


# --------------------------------------------------------------------------- #
#  Step 1: uniformize                                                          #
# --------------------------------------------------------------------------- #
def uniformize(state) -> np.ndarray:
    """(Ns, Nt) uniform scores V in (0, 1) for the plug-in transform.

    Exceedances get the model's own ``pCGP(Y)`` (already cached on the state as
    ``_pY``). Sub-threshold observed cells get the within-site empirical rank of Y
    rescaled to ``(0, p_cens)`` -- they are censored in the LIKELIHOOD but their
    values still order correctly, which is what the initial field needs. Missing
    cells (censored with no value) get the midpoint ``p_cens/2``.
    """
    Y, u_vec = state.data.Y, np.asarray(state.data.u_vec, float)
    p_cens = float(state.data.p_cens)
    miss = state.miss
    V = np.asarray(state._pY, float).copy()          # exact at exceedances
    for s in range(state.Ns):
        below = (~miss[s]) & (Y[s] <= u_vec[s])
        n = int(below.sum())
        if n:
            V[s, below] = p_cens * rankdata(Y[s, below]) / (n + 1.0)
    V[miss] = 0.5 * p_cens
    return np.clip(V, _V_FLOOR, 1.0 - 1e-9)


# --------------------------------------------------------------------------- #
#  Steps 2-5: the split                                                        #
# --------------------------------------------------------------------------- #
def _solve_scalar(f, lo, hi, n_grid=11, n_refine=14):
    """Root of `f` on [lo, hi] with no exceptions: coarse grid, then bisection on
    the first sign-changing pair; if `f` never changes sign, return the grid point
    with the smallest |f|. `f` is cheap (a few array ops), so this is ~25 evals."""
    xs = np.linspace(lo, hi, n_grid)
    fs = np.array([f(x) for x in xs])
    sign = np.signbit(fs)
    k = np.flatnonzero(sign[:-1] != sign[1:])
    if k.size == 0:
        return float(xs[int(np.argmin(np.abs(fs)))]), False
    a, b = xs[k[0]], xs[k[0] + 1]
    fa = fs[k[0]]
    for _ in range(n_refine):
        m = 0.5 * (a + b)
        fm = f(m)
        if np.signbit(fm) == np.signbit(fa):
            a, fa = m, fm
        else:
            b = m
    return float(0.5 * (a + b)), True


def plugin_field(state, *, lam=None, offset=1.0, target_z_sd=1.0, r_floor=_R_FLOOR,
                 z_map="auto", sd_tol=0.05, verbose=False):
    """Build and INSTALL the plug-in (S, Z, R) field on `state`.

    Assumes the scalars that enter the transform (``phi_vec``, ``gamma_bar``,
    ``alpha0``) and the margins (``sigma_mat``/``xi_mat`` + ``refresh_pY()``) are
    already set on `state`.

    lam : shrinkage of log S toward spatially constant, in [0, 1]. ``None`` (the
          default) SOLVES for it so that sd(Z) = ``target_z_sd``. For a single-knot
          S basis (HW) lam has no effect -- there the `offset` is solved instead.

    z_map : how the residual r becomes Z.
        "cdf"  -- the model-exact inverse transform Z = Phi^{-1}(1 - e^{-r}), which
                  sets eps = X/X* to exactly 1 (the log-Laplace mode). Correct when
                  r really is Exp(1); its Z marginal is otherwise only as good as
                  the one free scalar can make it.
        "rank" -- normal scores of r, Z = Phi^{-1}(rank(r)/(N+1)). Gives an exactly
                  N(0,1) Z marginal and needs no floor, at the cost of eps != 1.
        "auto" -- (default) "cdf", falling back to "rank" when the scalar solve
                  cannot bring sd(Z) within `sd_tol` of the target. Small or heavily
                  censored datasets can put sd(Z)=1 out of reach of both knobs
                  (the r_floor caps how much spread the cdf map can produce), and a
                  wrong Z marginal is worse than a nugget that is not exactly 1.

    Returns a dict of diagnostics.
    """
    Ns, Nt = state.Ns, state.Nt
    phi = np.asarray(state.phi_vec, float)[:, None]
    V = uniformize(state)
    gb = np.asarray(state.gamma_bar, float)[:, None]
    X = qRW(V.ravel(),
            np.broadcast_to(phi, (Ns, Nt)).ravel().copy(),
            np.broadcast_to(gb, (Ns, Nt)).ravel().copy(),
            float(state.alpha0)).reshape(Ns, Nt)
    if not np.all(np.isfinite(X)) or np.any(X <= 0.0):
        bad = int((~np.isfinite(X)).sum() + (X <= 0.0).sum())
        raise ValueError(f"plug-in transform produced {bad} non-positive/non-finite "
                         f"X values; check phi/gamma_bar/alpha0.")
    logX = np.log(X)

    W = np.asarray(state.W_wend_S, float)                     # (Ns, k_S), rows sum to 1
    single_knot = W.shape[1] == 1
    Wp = np.linalg.pinv(W)                                    # (k_S, Ns)

    def split(lam_, offset_):
        # log S: least-squares projection of the log-R target onto the S basis,
        # shrunk toward its spatial mean (lam_=0 -> R constant in space).
        b_ls = Wp @ ((logX - offset_) / phi)                   # (k_S, Nt)
        level = (W @ b_ls).mean(0)                             # (Nt,)
        b = level[None, :] + lam_ * (b_ls - level[None, :])
        Sf = np.exp(b)
        R = W @ Sf
        return Sf, R, logX - phi * np.log(R)

    def to_Z(r, how):
        if how == "rank":
            return norm.ppf(rankdata(r, method="average").reshape(r.shape)
                            / (r.size + 1.0))
        return norm.ppf(-np.expm1(-np.maximum(r, r_floor)))

    def build(lam_, offset_, how="cdf"):
        Sf, R, r = split(lam_, offset_)
        return Sf, R, to_Z(r, how), r

    # Two knobs can hit sd(Z) = target, and they are tried in that order because they
    # cost different things. `lam` moves variation between R and Z at a FIXED total
    # (the split we actually care about); `offset` shifts the whole residual r, which
    # also changes how much of it the r_floor clamps. Prefer lam; fall back to offset
    # when the target is unreachable over lam's whole [0, 1] range -- which is always
    # the case for the single-knot (HW) basis, where lam does nothing at all.
    lam_, knob, solved = (0.0 if single_knot else 0.5), "none", True
    if lam is not None:
        lam_ = float(lam)
    else:
        if not single_knot:
            lam_, solved = _solve_scalar(
                lambda l: build(l, offset)[2].std() - target_z_sd, 0.0, 1.0)
            knob = "lam"
        if single_knot or not solved:
            offset, solved = _solve_scalar(
                lambda c: build(lam_, c)[2].std() - target_z_sd, 0.1, 6.0)
            knob = "offset"

    how = z_map
    if z_map == "auto":
        how = "cdf"
        if abs(build(lam_, offset, "cdf")[2].std() - target_z_sd) > sd_tol:
            how = "rank"
    Sf, R, Z, r = build(lam_, offset, how)

    state.S, state.R, state.Z = Sf, R, Z
    info = dict(lam=float(lam_), offset=float(offset), z_sd=float(Z.std()),
                z_map=how, z_sd_knob=knob, z_sd_solved=bool(solved),
                single_knot=bool(single_knot),
                clamped_frac=float((r < r_floor).mean()) if how == "cdf" else 0.0,
                logX_mean=float(logX.mean()), S_median=float(np.median(Sf)))
    if verbose:
        print(f"[init] plug-in field: lam={info['lam']:.3f} offset={info['offset']:.2f} "
              f"map={how} sd(Z)={info['z_sd']:.3f} (via {knob}"
              f"{'' if solved else '; no root, grid best'}) "
              f"floor-clamped {info['clamped_frac']:.3%}", flush=True)
    return info


# --------------------------------------------------------------------------- #
#  Step 6: (nu, rho) from the implied Z field                                  #
# --------------------------------------------------------------------------- #
def _matern_corr(d, nu, rho):
    d = np.asarray(d, float)
    out = np.ones_like(d)
    m = d > 0
    t = np.sqrt(2.0 * nu) * d[m] / rho
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out[m] = (2.0 ** (1.0 - nu) / _gammafn(nu)) * t ** nu * _kv(nu, t)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def nu_prior_support(margin=0.02):
    """(lo, hi) for nu, inset from the SAMPLER's own uniform prior support.

    priors.nu_logprior is Uniform[loc, loc+scale] = [0.5, 2.0], a HARD bound: outside
    it the log-prior is -inf. Two consequences make it essential that a starting value
    respects it, neither of which announces itself:

      * the RWMH path (mcmc_updates.update_nu, used by hw/wang) rejects any proposal
        outside the support, so a chain STARTED outside can only escape by proposing
        clear across the gap -- from an adaptive scale that begins small. nu then sits
        frozen at its initial value, which reads exactly like the slow mixing nu is
        already notorious for;
      * the NUTS path (update_nu_rho_nuts, used by shi) maps nu through
        clip((nu - loc)/scale, 1e-6, 1-1e-6), so an out-of-support nu is SILENTLY
        reinterpreted as the boundary value while state.nu keeps the original for
        rebuild_cov and the Z prior -- the two disagree for the rest of the run.

    Read from the prior rather than hardcoded so the two cannot drift apart (this is
    the same __defaults__ trick update_nu_rho_nuts uses). `margin` insets by that
    fraction of the range so the fit never lands exactly on a boundary.
    """
    import priors as pr
    loc, scale = pr.nu_logprior.__defaults__
    return loc + margin * scale, loc + (1.0 - margin) * scale


def fit_matern_correlogram(Z, sites_xy, n_bins=25, nu_bounds=None,
                           rho_bounds=None, min_pairs=30):
    """Least-squares Matern (nu, rho) fit to the binned empirical correlogram of the
    (Ns, Nt) field `Z`, using its Nt replicates. Returns (nu, rho, rms, mid, cor).

    This is the ONE point in the pipeline where the covariance parameters see the
    data before Z has inflated out of the prior mode: at Z = 0 the data likelihood
    is exactly flat in (nu, rho), which is why a cold chain identifies them so
    slowly.
    """
    from scipy.optimize import least_squares
    if nu_bounds is None:
        nu_bounds = nu_prior_support()
    xy = np.asarray(sites_xy, float)
    D = scipy.spatial.distance.cdist(xy, xy)
    iu = np.triu_indices(xy.shape[0], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(np.asarray(Z, float))
    d, c = D[iu], C[iu]
    ok = np.isfinite(c)
    d, c = d[ok], c[ok]
    if d.size < min_pairs:
        raise ValueError("too few finite site pairs to fit a correlogram")
    edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    mid, cor = [], []
    for i in range(n_bins):
        sel = (d >= edges[i]) & (d < edges[i + 1] if i < n_bins - 1 else d <= edges[i + 1])
        if sel.sum() >= min_pairs:
            mid.append(d[sel].mean()); cor.append(c[sel].mean())
    mid, cor = np.array(mid), np.array(cor)
    if rho_bounds is None:
        rho_bounds = (0.05 * float(D.max()), 0.8 * float(D.max()))
    lo = np.log([nu_bounds[0], rho_bounds[0]])
    hi = np.log([nu_bounds[1], rho_bounds[1]])
    x0 = np.clip(np.log([1.0, 0.25 * float(D.max())]), lo + 1e-6, hi - 1e-6)
    fit = least_squares(lambda p: _matern_corr(mid, *np.exp(p)) - cor,
                        x0=x0, bounds=(lo, hi))
    nu_h, rho_h = np.exp(fit.x)
    return float(nu_h), float(rho_h), float(np.sqrt(np.mean(fit.fun ** 2))), mid, cor


def _nu_is_censored(nu, nu_bounds, tol=1e-3):
    """True when the correlogram fit ran into a bound -- nu was NOT identified, it was
    clamped. Common and expected: nu is an artifact of how the plug-in split apportions
    smooth structure between R and Z, so a single-knot R basis (HW) pushes ALL spatial
    structure into Z and wants a very smooth field, while a wide Wendland basis (Wang)
    absorbs it and leaves Z rough. rho is stable across the split; nu is not."""
    lo, hi = nu_bounds
    return bool(nu <= lo * (1 + tol) or nu >= hi * (1 - tol))


# --------------------------------------------------------------------------- #
#  Top level                                                                   #
# --------------------------------------------------------------------------- #
def _apply_margins(state, beta_sigma, beta_xi):
    """Install the margin surfaces and report which rung of the ladder was used.

    Rung 1 needs the state to actually MODEL the margins (``update_GPD_*``); when the
    margins are fixed on the SamplerData instead, ModelState has already broadcast
    ``data.sigma_vec``/``xi_vec`` into sigma_mat/xi_mat (rung 2), and explicit betas
    would have nothing to drive. Rung 3 is ModelState's intercept-only default.
    """
    used = []
    for name, beta, upd, design in (
            ("sigma", beta_sigma, state.update_sigma, state.sigma_design),
            ("xi", beta_xi, state.update_xi, state.xi_design)):
        if beta is None:
            used.append(f"{name}:{'data-vec' if not upd else 'default-intercept'}")
            continue
        b = np.asarray(beta, float)
        if not upd:
            used.append(f"{name}:data-vec(betas ignored: update_GPD_{name}=False)")
            continue
        if b.size != design.n_params:
            raise ValueError(f"beta_{name} has {b.size} entries but the {name} design "
                             f"has {design.n_params} parameters")
        setattr(state, f"beta_{name}", b.copy())
        mat = (state._mg.sigma_from_eta if name == "sigma" else state._mg.xi_from_eta)
        setattr(state, f"{name}_mat", mat(design.eta(b)))
        used.append(f"{name}:given-betas")
    state.refresh_pY()
    return ", ".join(used)


def plugin_init_snapshot(cfg, data, settings, *, phi0=0.5, alpha0=None, nu=None,
                         rho=None, beta_sigma=None, beta_xi=None, lam=None,
                         offset=1.0, z_map="auto", fit_nu_rho=True,
                         nu_bounds=None, nu_fallback=1.5, return_info=False,
                         verbose=False):
    """A ``ModelState.snapshot()`` warm-started by the plug-in decomposition.

    phi0      : starting tail exponent (scalar, applied at every phi knot). 0.5 is the
                asymptotic-independence/dependence boundary -- a start that commits to
                neither regime. phi cannot be identified from the split itself (larger
                phi simply moves variation from Z into R), so it is a choice, not a fit.
    alpha0    : nugget tail parameter; default ``settings.alpha0_min + 0.05``. This is
                the single largest scalar lever (see the module docstring) -- the cold
                default sits 3.0 above the wall.
    nu, rho   : covariance parameters. Given values are used as-is; otherwise they are
                ESTIMATED from the plug-in Z correlogram when ``fit_nu_rho`` (default),
                else left at the ModelState defaults.
    nu_fallback: nu to use when the correlogram fit is CENSORED (runs into the prior
                bound). nu is not identified by this procedure -- it is an artifact of
                how the split apportions smooth structure between R and Z, so a
                single-knot R basis (HW) wants a very smooth field and a wide Wendland
                basis (Wang) a rough one, while rho stays put. A bound is a bad start
                for the parameter whose mixing is already the bottleneck, so prefer an
                interior value. Ignored when `nu` is given explicitly.
    beta_*    : rung-1 margin coefficients (see the module docstring's ladder).
    lam       : S-basis shrinkage; ``None`` solves for sd(Z) = 1.

    Returns the snapshot dict, or ``(snapshot, info)`` with ``return_info=True``.
    """
    state = ModelState(cfg, data, settings)
    info = dict(margins=_apply_margins(state, beta_sigma, beta_xi))

    a0 = float(settings.alpha0_min + 0.05 if alpha0 is None else alpha0)
    if a0 <= settings.alpha0_min:
        raise ValueError(f"alpha0={a0} must exceed alpha0_min={settings.alpha0_min}")
    state.alpha0 = a0
    phi0 = float(phi0)
    if not 0.0 < phi0 < (1.0 if state.phi_domain != "positive" else np.inf):
        raise ValueError(f"phi0={phi0} outside the {state.phi_domain} phi domain")
    state.phi_k = np.full(state.k_phi,
                          np.log(phi0) if state.phi_domain == "positive" else phi0)
    state.phi_vec = state._phi_field(state.phi_k)
    if rho is not None:
        state.rho_k = np.full(state.k_rho, float(rho))
        state.rho_vec = state.W_gauss_rho @ state.rho_k
    if nu is not None:
        state.nu = float(nu)
    state.rebuild_cov()
    info.update(phi0=phi0, alpha0=a0)

    info.update(plugin_field(state, lam=lam, offset=offset, z_map=z_map,
                             verbose=verbose))

    if fit_nu_rho and (nu is None or rho is None):
        nb = nu_prior_support() if nu_bounds is None else nu_bounds
        nu_h, rho_h, rms, _mid, _cor = fit_matern_correlogram(
            state.Z, data.sites_xy, nu_bounds=nb)
        censored = _nu_is_censored(nu_h, nb)
        if nu is None:
            # A censored fit did not identify nu, it hit a wall -- and a bound is a bad
            # place to leave the parameter whose mixing is already the bottleneck. Fall
            # back to an interior default instead of pretending the bound is an estimate.
            state.nu = nu_fallback if censored else nu_h
        if rho is None:
            state.rho_k = np.full(state.k_rho, rho_h)
            state.rho_vec = state.W_gauss_rho @ state.rho_k
        state.rebuild_cov()
        info.update(nu_fit=nu_h, rho_fit=rho_h, corr_rms=rms, nu_censored=censored)
        if verbose:
            note = (f"  [nu fit hit the prior bound -- NOT identified by the "
                    f"correlogram; using the interior fallback {nu_fallback:g}]"
                    if censored and nu is None else "")
            print(f"[init] correlogram Matern fit: nu={nu_h:.3f} rho={rho_h:.3f} "
                  f"(rms {rms:.4f}) -> nu={state.nu:.3f} rho={state.rho_k[0]:.3f}"
                  + note, flush=True)

    # Hard invariant: an out-of-support nu freezes the RWMH nu block and desynchronises
    # the NUTS one (see nu_prior_support). Never emit one, whatever its source.
    import priors as _pr
    if not np.isfinite(_pr.nu_logprior(state.nu)):
        lo, hi = _pr.nu_logprior.__defaults__[0], sum(_pr.nu_logprior.__defaults__)
        raise ValueError(f"starting nu={state.nu:g} is outside the sampler's prior "
                         f"support [{lo:g}, {hi:g}], where the log-prior is -inf: the "
                         f"nu block would freeze (RWMH) or silently disagree with "
                         f"state.nu (NUTS). Pass nu= inside the support.")
    info.update(nu=float(state.nu), rho=float(state.rho_k[0]),
                radius_S=float(state.radius_S))
    if verbose:
        print(f"[init] plug-in snapshot: phi={phi0} alpha0={a0} nu={state.nu:.3f} "
              f"rho={state.rho_k[0]:.3f} radius_S={state.radius_S:.3f} "
              f"margins[{info['margins']}]", flush=True)
    snap = state.snapshot()
    return (snap, info) if return_info else snap
