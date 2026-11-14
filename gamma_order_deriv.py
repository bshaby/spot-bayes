"""Derivative of the incomplete gamma functions with respect to their ORDER a.

This is the one special-function piece the analytic ``d/dphi`` of the RW density
needs and that scipy/GSL do not expose. The RW density
(``cpp/rw_inte_cpp.cpp:dRW_standard_Pareto_multiplicative_nugget_C``) is built from
four incomplete gammas whose ORDERS depend on phi -- gamma(1/2, lam),
Gamma(1/2 - phi, lam), gamma(1/2 + a0 phi, lam), Gamma(1/2 - a0 phi, lam) -- so
``d(logf)/dphi`` requires ``d/da`` of gamma(a, z) / Gamma(a, z), the exact analogue
of the Bessel order-derivative in ``bessel_nu.py`` (which powers the Matern-nu
gradient). Same route: an integral representation that is trivially differentiable,
evaluated by a fixed high-order quadrature and validated against mpmath.

    dGammaU_da(a, z) = d/da Gamma(a, z) = int_z^inf t^{a-1} e^{-t} ln t dt
    dGammaL_da(a, z) = d/da gamma(a, z) = int_0^z  t^{a-1} e^{-t} ln t dt

Numerics. The integrand spikes at t=z for a<1 (t^{a-1} -> inf as t->z^+ with small
z), which fixed-node rules on t miss by orders of magnitude. The substitution
``t = z e^{+/-s}`` removes the spike, factoring the entire magnitude into a clean
``z^a`` prefactor and leaving a smooth, analytic integrand on a finite interval:

    dGammaU_da(a, z) = z^a int_0^Smax e^{ a s} e^{-z e^{ s}} (ln z + s) ds   (any a)
    dGammaL_da(a, z) = z^a int_0^Smax e^{-a s} e^{-z e^{-s}} (ln z - s) ds   (a > 0)

  * Upper: the ``e^{-z e^s}`` factor decays double-exponentially, so the form is
    valid for ANY order a, including the NEGATIVE orders (down to ~-6) the RW
    density hits at large alpha0 * phi. Computed DIRECTLY (not via the identity
    dGammaL + dGammaU = Gamma(a) psi(a), which cancels catastrophically at small z).
  * Lower: needs a > 0 for convergence -- and the lower RW orders (1/2, 1/2+a0 phi)
    ARE always positive.

Because the integrand is smooth and analytic on ``[0, Smax]``, GAUSS-LEGENDRE is
spectrally accurate: n=128 nodes give ~1e-13 relative error vs mpmath across
a in [-6, 3], z in [1e-3, 8] (the whole reachable RW regime), at ~0.01 ms/scalar
call and fully vectorized over a field of cells. ``Smax`` is the point past which
the integrand is < e^{-cutoff}; the truncation error is ~e^{-cutoff}.

``dGammaU_da`` / ``dGammaL_da`` are the fast estimates; ``*_ref`` are mpmath
references (exact to working precision) for validation -- see
tests/test_gamma_order_deriv.py.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl(n: int):
    """Cached Gauss-Legendre nodes/weights on [-1, 1]."""
    if n not in _GL_CACHE:
        _GL_CACHE[n] = leggauss(n)
    return _GL_CACHE[n]


def dGammaU_da(a, z, n: int = 128, cutoff: float = 50.0):
    """d/da of the UPPER incomplete gamma Gamma(a, z) = int_z^inf t^{a-1} e^{-t} dt.

    Valid for any real order ``a`` (incl. negative) at ``z > 0``. ``a`` and ``z``
    broadcast; the result has their broadcast shape."""
    a = np.asarray(a, float)
    z = np.asarray(z, float)
    a, z = np.broadcast_arrays(a, z)
    shp = a.shape
    a = a.ravel(); z = z.ravel()
    # support: e^{-z e^s} is < e^{-cutoff} once z e^s > cutoff  =>  s > ln(cutoff/z).
    Smax = np.maximum(np.log(cutoff / z), 2.0)                 # (N,)
    x, w = _gl(n)                                              # (n,)
    s = 0.5 * Smax[:, None] * (x[None, :] + 1.0)               # (N, n)
    jac = 0.5 * Smax[:, None]                                  # d s / d x
    g = np.exp(a[:, None] * s - z[:, None] * np.exp(s)) * (np.log(z)[:, None] + s)
    out = z ** a * np.sum(jac * w[None, :] * g, axis=1)
    return out.reshape(shp)


def dGammaL_da(a, z, n: int = 128, cutoff: float = 50.0):
    """d/da of the LOWER incomplete gamma gamma(a, z) = int_0^z t^{a-1} e^{-t} dt.

    Requires ``a > 0`` (the lower RW orders 1/2, 1/2 + alpha0*phi always are).
    Computed directly (no small-z cancellation). ``a``, ``z`` broadcast."""
    a = np.asarray(a, float)
    z = np.asarray(z, float)
    a, z = np.broadcast_arrays(a, z)
    shp = a.shape
    a = a.ravel(); z = z.ravel()
    if np.any(a <= 0):
        raise ValueError("dGammaL_da needs a > 0 (lower incomplete gamma diverges "
                         "at t=0 for a <= 0); use the RW lower orders 1/2, 1/2+a0*phi.")
    # support: e^{-a s} < e^{-cutoff} at s = cutoff/a; also keep ln(cutoff/z) room
    # near t=0 (s large) when z is small so the ln(z e^{-s}) tail is resolved.
    Smax = np.maximum(cutoff / a, np.where(z < 1.0, np.log(cutoff / z), 5.0))
    x, w = _gl(n)
    s = 0.5 * Smax[:, None] * (x[None, :] + 1.0)
    jac = 0.5 * Smax[:, None]
    g = np.exp(-a[:, None] * s - z[:, None] * np.exp(-s)) * (np.log(z)[:, None] - s)
    out = z ** a * np.sum(jac * w[None, :] * g, axis=1)
    return out.reshape(shp)


# ---------------------------------------------------------------------------
# mpmath references (exact to working precision) -- validation only, not for the hot path.
# ---------------------------------------------------------------------------
def dGammaU_da_ref(a, z, dps: int = 40) -> float:
    import mpmath as mp
    mp.mp.dps = dps
    return float(mp.diff(lambda aa: mp.gammainc(aa, mp.mpf(z), mp.inf), mp.mpf(a)))


def dGammaL_da_ref(a, z, dps: int = 40) -> float:
    import mpmath as mp
    mp.mp.dps = dps
    return float(mp.diff(lambda aa: mp.gammainc(aa, 0, mp.mpf(z)), mp.mpf(a)))
