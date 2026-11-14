"""Validate d/dnu K_nu(x) and d/dnu of the Matern correlation against mpmath.

mpmath gives an exact complex-step reference (besselk accepts complex order, so
Im[f(nu+ih)]/h has no subtractive cancellation). These gate the one new special
function needed to put the Matern smoothness nu under a gradient sampler.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bessel_nu as bn  # noqa: E402

mp = pytest.importorskip("mpmath")

NUS = [0.25, 0.5, 0.75, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5]
XS = np.array([0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])


def test_dkv_dnu_matches_mpmath():
    worst = 0.0
    for nu in NUS:
        fast = bn.dkv_dnu(nu, XS)
        ref = bn.dkv_dnu_ref(nu, XS)
        worst = max(worst, np.max(np.abs(fast - ref) / (np.abs(ref) + 1e-300)))
    assert worst < 1e-4, f"worst rel err {worst:.2e}"


def test_dmatern_dnu_matches_mpmath_complex_step():
    """d/dnu of M(x;nu) = (2^{1-nu}/Gamma(nu)) x^nu K_nu(x), independently via an
    mpmath complex-step on M itself (exercises the whole dmatern_dnu chain, not
    just dkv_dnu). Tight relative tolerance for x >= 0.3; below that the formula
    has catastrophic cancellation (large terms, result -> 0; the zone widens with
    nu, reaching ~0.1 at nu=3), so only an absolute bound is meaningful there -- and
    harmless for HMC (see dmatern_dnu docstring)."""
    mp.mp.dps = 50
    h = mp.mpf("1e-30")

    def dM_ref(nu, x):
        def M(v):
            return (mp.power(2, 1 - v) / mp.gamma(v)
                    * mp.power(x, v) * mp.besselk(v, x))
        return float(mp.im(M(mp.mpf(nu) + 1j * h)) / h)

    worst_rel, worst_abs = 0.0, 0.0
    for nu in NUS:
        fast = bn.dmatern_dnu(nu, XS)
        ref = np.array([dM_ref(nu, float(x)) for x in XS])
        big = XS >= 0.3
        worst_rel = max(worst_rel, np.max(np.abs(fast[big] - ref[big])
                                          / (np.abs(ref[big]) + 1e-12)))
        worst_abs = max(worst_abs, np.max(np.abs(fast - ref)))
    assert worst_rel < 1e-4, f"x>=0.1 worst rel err {worst_rel:.2e}"
    assert worst_abs < 1e-3, f"worst abs err {worst_abs:.2e}"


def test_dkv_dnu_half_integer_closed_form():
    """At nu=1/2, K_{1/2}(x)=sqrt(pi/2x) e^{-x}; d/dnu K_nu|_{1/2} has the known
    closed form sqrt(pi/2x) e^{-x} * [ ... ] -- just cross-check one point against
    mpmath to a tight tolerance (independent of the grid choice above)."""
    for x in (0.5, 1.0, 3.0):
        assert abs(bn.dkv_dnu(0.5, np.array([x]))[0]
                   - bn.dkv_dnu_ref(0.5, x)) / abs(bn.dkv_dnu_ref(0.5, x)) < 1e-4
