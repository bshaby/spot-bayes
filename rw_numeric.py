"""
Guarded RW density transform ``dRW_safe``.

``dRW_safe`` wraps the closed-form analytic density ``dRW``
(cpp/rw_inte_cpp.cpp) with a strictly-positive floor, so the censored/exceedance
GPD log-likelihood stays finite.

History: the analytic density used to suffer catastrophic cancellation in the
UPPER TAIL -- exactly where the likelihood evaluates it (at exceedances) -- summing
huge opposite-sign incomplete gammas, biased by ~2-70% and GROWING with alpha0. So
this module used to difference ``pRW`` NUMERICALLY by default. The incomplete-gamma
accuracy fix (cpp/rw_inte_cpp.cpp: downward recurrence for negative-order upper
gammas + regularised lower gamma ``P(a,x)``, replacing the catastrophically-
cancelling ``Gamma(a) - Gamma(a,x)``) removed that bias: the analytic form is now
MACHINE-PRECISE -- validated to <=3e-15 relative error against a high-precision
mpmath reference across phi in [0.1, 10], alpha0 in [2, 5], gamma in [1, 5] and p up
to 1 - 1e-10 -- AND ~8x faster than the numeric difference (one closed-form call vs
two ``pRW`` evals). So ``dRW_safe`` now returns the analytic density directly and the
numeric-difference path has been removed.

See tests/test_rw.py: ``test_rw_derivatives_match_mpmath_in_tail`` pins the analytic
form to mpmath, and ``test_dRW_safe_is_strictly_positive_in_extreme_tail`` pins the
floor.
"""
from __future__ import annotations

import numpy as np

import rw_inte as _rw

# Smallest strictly-positive density returned (so log-likelihoods stay finite). The
# raw analytic dRW returns 0 for a tiny-negative roundoff, and underflows to 0 only
# deep in the tail where the TRUE density is < ~1e-308 (log < -708) -- so flooring
# there is ~exact, NOT the spurious moderate-x cliff the old numeric difference could
# hit (it underflowed while the true density was still ~1e-5).
_TINY = np.finfo(float).tiny


def dRW_safe(x, phi, gamma, alpha0):
    """Density of the multiplicative-nugget RW transform: the analytic closed form
    (``rw_inte.dRW``) with a strictly-positive ``_TINY`` floor.

    Accepts scalars or arrays (broadcast like the vectorised transforms). The floor
    engages only where the analytic value is non-finite or <= 0 -- a tiny-negative
    roundoff or a genuine far-tail underflow -- so the returned density is strictly
    positive and the GPD log-likelihood at exceedances never NaNs.
    """
    x, phi, gamma = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(phi, dtype=float),
        np.asarray(gamma, dtype=float))
    out = np.asarray(_rw.dRW(x, phi, gamma, float(alpha0)), dtype=float)
    scalar = out.ndim == 0
    out = np.atleast_1d(out).copy()
    bad = ~(np.isfinite(out) & (out > 0.0))
    if np.any(bad):
        out[bad] = _TINY
    return float(out[0]) if scalar else out.reshape(np.shape(x))
