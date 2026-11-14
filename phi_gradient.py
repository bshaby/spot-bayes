"""Analytic gradient of the phi conditional log-posterior (the Laplace/L2-phi mode
solve). Fully analytic via the fused C RW derivatives (rw_inte.dpRW_dphi,
dlog_dRW_dphi) -- no finite differences, no qRW re-inversion. See
docs/phi_gradient_l2phi.md; the promoted form of the validated prototype.

F(phi_k) = sum_t loglik_t + phi_logprior, with phi(s) = exp(W_gauss_phi @ phi_k)
(POSITIVE domain). Per exceedance cell the likelihood is
  -log X* + llpdf(X/X*) - log dRW(X; phi) + log_gY     (log_gY is phi-independent),
per censored cell llcdf(x0/X*), with X* = R^phi g(Z), X = qRW(pCGP(Y); phi). The
gradient pieces: dX/dphi = -dpRW_dphi(X)/dRW(X) (avoids re-inverting qRW),
d/dphi log dRW|_X = dlog_dRW_dphi(X), dlog_dRW_dx (all C), dX*/dphi = X* ln R, the
log-Laplace copula derivatives (elementary), and the knot chain
d phi_i/d phi_k[j] = phi_i W_gauss_phi[i,j].
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from rw_inte import dRW, dlog_dRW_dx, dpRW_dphi, dlog_dRW_dphi
from mcmc_updates import phi_logprior
from config import PHI_POSITIVE, PHI_UNIT

_SQRT2 = np.sqrt(2.0)


def _gZ(Z):
    return 1.0 / (0.5 * erfc(np.asarray(Z) / _SQRT2))


def _dloglaplace_logpdf_de(eps, a0):
    return np.where(eps < 1.0, (a0 - 1.0) / eps, -(a0 + 1.0) / eps)


def _dloglaplace_logcdf_de(eps, a0):
    return np.where(eps < 1.0, a0 / eps,
                    (0.5 * a0 * eps ** (-a0 - 1.0)) / (1.0 - 0.5 * eps ** (-a0)))


def _phi_logprior_grad(state, phi_k):
    """d/d phi_k of phi_logprior. POSITIVE domain: N(0,1) on phi_k -> -phi_k.
    Other domains: central finite difference (cheap; no qRW)."""
    if state.phi_domain == PHI_POSITIVE:
        return -np.asarray(phi_k, float)
    g = np.zeros(phi_k.size)
    for j in range(phi_k.size):
        d = np.zeros(phi_k.size); d[j] = 1e-6
        g[j] = (phi_logprior(state, phi_k + d) - phi_logprior(state, phi_k - d)) / 2e-6
    return g


def _accum_one_time(state, t, qcol, dcol, x0s, dx0, a0, dF):
    """Add time ``t``'s contribution to the per-SITE gradient ``dF`` (in place).

    Split out of phi_value_grad so the sum can run over an arbitrary set of times.
    That is what makes the block MPI-capable: phi is GLOBAL, so without it every rank
    would have to evaluate every time replicate to get the value, which at Faroe means
    32 ranks each doing all 1656 columns -- 32x the necessary work on what is already
    the most expensive block in the sampler."""
    ci, ei = state.cens_exc(t)
    phiv, gb = state.phi_vec, state.gamma_bar
    R, Z = state.R, state.Z
    if ei.size:                       # exceedances: fully analytic via the C derivs
        X = qcol[ei]; dX = dcol[ei]; ph = phiv[ei]; gg = gb[ei]
        Re = R[ei, t]; lnR = np.log(Re)
        rat = X / (Re ** ph * _gZ(Z[ei, t]))
        dXdphi = -dpRW_dphi(X, ph, gg, a0) / dX        # implicit fn: no re-inversion
        drat = rat * (dXdphi / X - lnR)
        term = (-lnR + _dloglaplace_logpdf_de(rat, a0) * drat
                - (dlog_dRW_dx(X, ph, gg, a0) * dXdphi
                   + dlog_dRW_dphi(X, ph, gg, a0)))
        np.add.at(dF, ei, term)
    if ci.size:                       # censored: the threshold quantile is per-SITE
        Xc = x0s[ci]; Rc = R[ci, t]; lnRc = np.log(Rc)
        ratc = Xc / (Rc ** phiv[ci] * _gZ(Z[ci, t]))
        np.add.at(dF, ci, _dloglaplace_logcdf_de(ratc, a0)
                  * (ratc * (dx0[ci] / Xc - lnRc)))


def phi_value_grad(state, phi_k=None, times=None, areduce=None, qguess=None,
                   qd_cached=None):
    """``(F, dF/dphi_k)`` at ``phi_k`` (default: the state's current phi_k).

    ``qd_cached`` maps t -> the (qv, dv) pair ALREADY computed for exactly this phi
    (i.e. caches.qd when phi_k is the current value). The transform is then skipped
    outright rather than warm-started -- worth ~35% of the call, which matters for
    Barker, where one of the two gradient evaluations per sweep is always at the
    current point. The caller owns the invariant that the cache matches this phi.

    ``qguess`` maps t -> the last accepted qv column, warm-starting the exceedance
    root-finds (a stale guess is safe: qRW_warm falls back to the cold solve). Within a
    trajectory phi moves in small steps, so this is the cheap-bracket case.

    ``times`` restricts the sum to a subset of replicates and ``areduce`` combines the
    partial (gradient, value) across MPI ranks, so a worker evaluates only its own
    columns. Serial callers pass neither and get the full sum.

    Mutates ``state.phi_k`` / ``phi_vec`` when ``phi_k`` is given, matching what a
    proposal evaluates. pY is margin-based and phi-INDEPENDENT, so it is not
    refreshed."""
    if phi_k is not None:
        state.phi_k = np.asarray(phi_k, float)
        state.phi_vec = state._phi_field(state.phi_k)
    a0 = float(state.alpha0)
    phiv, gb = state.phi_vec, state.gamma_bar
    ts = range(state.Nt) if times is None else list(times)

    # censored threshold quantile: time-independent, so one pass for all times
    x0s = state.qd_RW_cens()
    dx0 = -dpRW_dphi(x0s, phiv, gb, a0) / dRW(x0s, phiv, gb, a0)

    dF = np.zeros(state.Ns)
    val = 0.0
    for t in ts:
        if qd_cached is not None:
            qcol, dcol = qd_cached[t]
        else:
            qcol, dcol = state.qd_RW_one(
                t, x0s, qe_guess=None if qguess is None else qguess.get(t))
        val += state.loglik_t(t, qv=qcol, dv=dcol)
        _accum_one_time(state, t, qcol, dcol, x0s, dx0, a0, dF)

    if areduce is not None:                 # one reduction: gradient + value together
        buf = np.empty(state.Ns + 1)
        buf[:state.Ns] = dF
        buf[-1] = val
        buf = areduce(buf)
        dF, val = buf[:state.Ns], float(buf[-1])

    # priors are GLOBAL: added once, after the reduction, never per rank
    val += float(phi_logprior(state, state.phi_k))
    # Chain to the knots; the factor DEPENDS ON THE DOMAIN:
    #   positive: phi(s) = exp(W phi_k) -> dphi_i/dphi_k[j] = phi_i * W[i,j]
    #   unit:     phi(s) =     W phi_k  -> dphi_i/dphi_k[j] =         W[i,j]
    dphi_dk = 1.0 if state.phi_domain == PHI_UNIT else phiv
    grad = state.W_gauss_phi.T @ (dF * dphi_dk) + _phi_logprior_grad(state, state.phi_k)
    return float(val), grad
