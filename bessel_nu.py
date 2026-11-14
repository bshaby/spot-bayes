"""Derivative of the modified Bessel function K_nu(x) with respect to its ORDER nu.

This is the one piece the Matern covariance needs that scipy/JAX do not expose:
fitting the Matern smoothness nu by gradient (HMC/NUTS) requires d/dnu of K_nu, and
"all existing software packages that estimate nu use finite-difference estimates"
(Geoga, Marin, Schanen & Stein 2022, who instead make K_nu autodiff-able -- the
BesselK.jl approach). Here we take the equivalent exact route via the integral
representation, which is trivially differentiable under the integral sign:

    K_nu(x)        = ∫_0^∞ e^{-x cosh t} cosh(nu t) dt          (x > 0)
    d/dnu K_nu(x)  = ∫_0^∞ e^{-x cosh t} t sinh(nu t) dt.

Substituting t = e^u turns it into a double-exponentially decaying integrand on
(-inf, inf) -- F(u) = t^2 sinh(nu t) e^{-x cosh t}, t = e^u -- for which the uniform
TRAPEZOIDAL rule is spectrally accurate. One fixed u-grid integrates every x at
once; the only x-dependent factor is e^{-x cosh t}, so the t-dependent pieces are
precomputed and the per-call cost is a single (n_pts x n_node) exp + matmul.

`dkv_dnu` is the fast vectorized estimate; `dkv_dnu_ref` is an mpmath complex-step
reference (exact to working precision) for validation. See tests/test_bessel_nu.py.
"""
from __future__ import annotations

import numpy as np

# Cache the (x-independent) trapezoidal node factors per (nu, grid) so a NUTS
# trajectory at fixed nu pays the sinh/cosh setup once, not per leapfrog.
_TRAP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, float]] = {}


def _trap_nodes(nu: float, n: int, u_min: float, u_max: float):
    key = (float(nu), int(n), float(u_min), float(u_max))
    out = _TRAP_CACHE.get(key)
    if out is None:
        u = np.linspace(u_min, u_max, n)
        h = float(u[1] - u[0])
        t = np.exp(u)
        A = t * t * np.sinh(nu * t)                 # node weight (x-independent)
        C = np.cosh(t)                              # enters only via e^{-x C}
        if len(_TRAP_CACHE) > 64:
            _TRAP_CACHE.clear()
        out = _TRAP_CACHE[key] = (A, C, h)
    return out


def dkv_dnu(nu: float, x, n: int = 160, u_min: float = -8.0,
            u_max: float = 3.0) -> np.ndarray:
    """Vectorized d/dnu K_nu(x) for scalar ``nu`` and array-like ``x`` (x > 0).

    Trapezoidal-in-log(t) quadrature of the integral representation (see module
    docstring). Worst relative error vs the mpmath reference over nu in [0.25, 3.5],
    x in [0.02, 8]: ~1e-5 at the default n=160 (~9e-7 at n=200). Far tighter than
    HMC gradients need; raise ``n`` if you want the value itself to higher accuracy.
    """
    x = np.asarray(x, dtype=float)
    A, C, h = _trap_nodes(nu, n, u_min, u_max)
    return h * (np.exp(-x[..., None] * C) @ A)


def dkv_dnu_ref(nu: float, x, dps: int = 50, h: float = 1e-30):
    """mpmath complex-step reference: d/dnu K_nu(x) = Im[K_{nu+ih}(x)]/h (no
    subtractive cancellation -> exact to working precision). Scalar/looped, slow;
    for validation only. Returns a float (scalar x) or np.ndarray (array x)."""
    import mpmath as mp
    mp.mp.dps = dps

    def one(xx):
        return float(mp.im(mp.besselk(mp.mpf(nu) + 1j * mp.mpf(h), mp.mpf(xx))) / h)

    xa = np.asarray(x, dtype=float)
    if xa.ndim == 0:
        return one(float(xa))
    return np.array([one(float(v)) for v in xa.ravel()]).reshape(xa.shape)


def dmatern_dnu(nu: float, x, n: int = 160, M=None):
    """d/dnu of the Matern correlation M(x;nu) = (2^{1-nu}/Gamma(nu)) x^nu K_nu(x),
    the quantity the nonstationary Matern covariance actually needs. Uses
    `dkv_dnu` for the d/dnu K_nu(x) term; the rest is elementary:

        dM/dnu = M(x) [ -ln2 - psi(nu) + ln x ] + (2^{1-nu}/Gamma(nu)) x^nu dKnu/dnu.

    ``M`` may be passed when the caller already has the correlation (e.g. M = K/S
    from the covariance build) -- this skips the redundant scipy.kv(nu, x) eval.

    Accurate to ~1e-5 (relative) for x >~ 0.1. For x -> 0 the two terms above are
    large and nearly cancel (the result -> 0 as M -> 1 and dKnu/dnu blows up), so
    near-coincident pairs lose relative precision -- harmless for HMC (a few
    near-diagonal entries with tiny weight; the Metropolis accept uses the exact
    density from scipy.kv). A small-x series would be needed for high accuracy there.
    """
    from scipy.special import kv, gamma, psi
    x = np.asarray(x, dtype=float)
    c = 2.0 ** (1 - nu) / gamma(nu)
    if M is None:
        M = c * x ** nu * kv(nu, x)
    return M * (-np.log(2.0) - psi(nu) + np.log(x)) + c * x ** nu * dkv_dnu(nu, x, n)
