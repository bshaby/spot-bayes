"""Tests for the RW (Rosenblatt/scale-mixture) transforms.

Validates the closed-form multiplicative-nugget marginal, including the key
finding that it remains valid for phi >= 0.5 (the "positive" phi domain).
Run: pytest -q extremes_combined/tests/test_rw.py
"""
import os
import sys

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rw_inte as rw  # noqa: E402


PHIS = [0.3, 0.6, 0.9, 1.5]
GAMMA, ALPHA0 = 1.0, 2.5


@pytest.mark.parametrize("phi", PHIS)
def test_quantile_roundtrip(phi):
    for x in [1.5, 5.0, 50.0]:
        p = rw.pRW(x, phi, GAMMA, ALPHA0)
        xb = float(rw.qRW(p, phi, GAMMA, ALPHA0))
        assert abs(xb - x) / x < 1e-6


def test_qRW_warm_matches_cold_for_any_guess():
    """Warm-start qRW seeds the Brent bracket from a guess (the previous accepted
    qRW). It must return the SAME root as the cold solve regardless of guess quality
    -- a good guess only speeds up bracketing; a stale/garbage guess falls back to the
    exact cold path. This safety property is what makes it correct to pass a possibly-
    stale cached qv into the sampler's qd_RW_one."""
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.uniform(0.90, 0.9995, n)
    phi = rng.uniform(0.3, 0.9, n)
    gam = rng.uniform(1.0, 5.0, n)
    a0 = 3.0
    cold = rw.qRW(p, phi, gam, a0)
    # to Brent's ~1e-12 interval tolerance, both solves agree (bracket-dependent
    # last-digit jitter only).
    tol = 1e-9
    # 1) exact guess (the truth)
    assert np.max(np.abs(rw.qRW_warm(p, phi, gam, a0, cold) - cold) / cold) < tol
    # 2) realistic guess: root at slightly-perturbed inputs (a small MCMC step)
    p2 = np.clip(p + rng.normal(0, 2e-3, n), 0.5, 0.9999)
    phi2 = phi + rng.normal(0, 1e-2, n)
    cold2 = rw.qRW(p2, phi2, gam, a0)
    assert np.max(np.abs(rw.qRW_warm(p2, phi2, gam, a0, cold) - cold2) / cold2) < tol
    # 3) garbage guesses (zero, negative, absurdly large) -> cold fallback, still exact
    bad = np.where(rng.random(n) < 0.33, 0.0,
                   np.where(rng.random(n) < 0.5, -1.0, 1e40))
    assert np.max(np.abs(rw.qRW_warm(p, phi, gam, a0, bad) - cold) / cold) < tol


@pytest.mark.parametrize("phi", PHIS)
def test_drw_matches_finite_difference(phi):
    # test_rw_phi_gt_1 from the plan: dRW vs finite-diff of pRW, all phi.
    for x in [2.0, 5.0, 20.0]:
        h = 1e-4 * x
        fd = (rw.pRW(x + h, phi, GAMMA, ALPHA0)
              - rw.pRW(x - h, phi, GAMMA, ALPHA0)) / (2 * h)
        d = float(rw.dRW(x, phi, GAMMA, ALPHA0))
        assert abs(fd - d) / max(abs(d), 1e-12) < 1e-4


@pytest.mark.parametrize("phi", [0.5, 2.0, 5.0, 10.0])
def test_pRW_valid_up_to_phi_10(phi):
    """pRW (and qRW) remain correct up to phi=10 (Monte Carlo)."""
    import scipy.stats
    rng = np.random.default_rng(7)
    N = 3_000_000
    R = GAMMA / scipy.stats.norm.ppf(1 - rng.uniform(0, 1, N) / 2) ** 2
    W = 1.0 / (1.0 - rng.uniform(0, 1, N))
    Ue = rng.uniform(0, 1, N)
    eps = np.where(Ue < 0.5, np.power(Ue * 2.0, 1.0 / ALPHA0),
                   np.power(0.5 / (1.0 - Ue), 1.0 / ALPHA0))
    with np.errstate(over="ignore", invalid="ignore"):
        X = eps * R ** phi * W
    for p in [0.2, 0.5, 0.8]:
        x = float(qRW := rw.qRW(p, phi, GAMMA, ALPHA0))
        emp = np.mean(X <= x)
        cf = float(rw.pRW(x, phi, GAMMA, ALPHA0))
        se = np.sqrt(max(emp * (1 - emp), 1e-12) / N)
        assert abs(emp - cf) < 4 * se + 4e-3


@pytest.mark.parametrize("phi", [0.5, 2.0, 5.0, 10.0])
def test_dRW_safe_matches_high_order_derivative_up_to_phi_10(phi):
    """dRW_safe (now the analytic closed form) equals an independent 5-point-stencil
    derivative of pRW for all phi up to 10. (Pre incomplete-gamma fix the raw analytic
    dRW disagreed ~10-16% for phi>=5; post-fix it is machine-order -- see
    test_rw_derivatives_match_mpmath_in_tail.)"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rw_numeric import dRW_safe

    def deriv5(x):  # 4th-order central derivative of pRW in log-x
        lx = np.log(x); h = 1e-3
        xs = np.exp(lx + np.array([-2, -1, 1, 2]) * h)
        f = rw.pRW(xs, phi, GAMMA, ALPHA0)
        dpdlx = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h)
        return dpdlx / x  # chain rule d/dx = (d/dlx)/x

    for p in [0.3, 0.5, 0.7]:
        x = float(rw.qRW(p, phi, GAMMA, ALPHA0))
        ref = deriv5(x)
        got = float(dRW_safe(x, phi, GAMMA, ALPHA0))
        assert abs(got - ref) / max(abs(ref), 1e-300) < 1e-3


def _tail_integral(dens_fn, phi, g, a, a_lo=0.90, a_hi=0.999, N=2001):
    """Simpson integral (in log-x) of a density over an upper-tail interval and
    the exact pRW increment over the same interval. Robust ground truth for tail
    density accuracy -- insensitive to pointwise differentiation noise."""
    xlo = float(rw.qRW(a_lo, phi, g, a)); xhi = float(rw.qRW(a_hi, phi, g, a))
    lx = np.linspace(np.log(xlo), np.log(xhi), N); x = np.exp(lx)
    integrand = np.array([float(dens_fn(xi)) for xi in x]) * x       # dx = x dlx
    w = np.ones(N); w[1:-1:2] = 4; w[2:-1:2] = 2
    I = (lx[1] - lx[0]) / 3 * np.sum(w * integrand)
    return I, float(rw.pRW(xhi, phi, g, a) - rw.pRW(xlo, phi, g, a))


@pytest.mark.parametrize("alpha0", [2.5, 5.0])
@pytest.mark.parametrize("phi", [0.6, 1.0, 2.0])
def test_dRW_safe_tail_integral_identity(phi, alpha0):
    """The sampler's density must integrate to the pRW increment over the UPPER
    TAIL -- where the likelihood evaluates it -- for all phi and alpha0.

    After the incomplete-gamma accuracy fix (cpp/rw_inte_cpp.cpp: downward
    recurrence for negative-order upper gammas + regularised lower gamma P(a,x),
    replacing the catastrophically-cancelling Gamma(a)-Gamma(a,x)), dRW_safe returns
    the analytic closed form (this module formerly differenced pRW numerically). It is
    now machine-order across the tail, including the (alpha0=5, phi=2) corner that
    previously exposed a materially biased analytic and a ~1e-5 numeric error."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rw_numeric import dRW_safe
    g = 1.0
    I, exact = _tail_integral(lambda x: dRW_safe(x, phi, g, alpha0), phi, g, alpha0)
    assert abs(I - exact) / exact < 1e-8      # analytic density (machine-order)


@pytest.mark.parametrize("phi", [0.5, 0.9, 1.0, 1.5])
@pytest.mark.parametrize("alpha0", [2.5, 3.814, 5.0])
def test_rw_derivatives_match_mpmath_in_tail(phi, alpha0):
    """Regression for the incomplete-gamma accuracy fix (cpp/rw_inte_cpp.cpp).

    The multiplicative-nugget survival is a sum of four incomplete gammas; deep in
    the tail the negative-order UPPER gammas (Gamma(1/2 - alpha0*phi, lambda) with
    alpha0*phi > 1/2) and the lower gamma computed as Gamma(a)-Gamma(a,x) both lost
    most of their digits in raw GSL/double, biasing dRW ~5-70% and dlog_dRW_dx ~15%
    -- which manufactured the -log(dRW) likelihood CLIFF that trapped alpha0/beta.
    Pin pRW/dRW/dlog_dRW_dx to a high-precision mpmath evaluation of the SAME closed
    form across phi and deep into the tail; pre-fix this failed by up to 0.71."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    g = 1.5

    def S_mp(x):
        x = mp.mpf(x); ph = mp.mpf(phi); gg = mp.mpf(g); a0 = mp.mpf(alpha0)
        lam = gg / (2 * x ** (1 / ph)); g2 = gg / 2
        lg = lambda a, z: mp.gammainc(a, 0, z)
        ug = lambda a, z: mp.gammainc(a, z, mp.inf)
        return (lg(mp.mpf(1) / 2, lam)
                + a0 * a0 / (a0 * a0 - 1) * g2 ** ph * (1 / x) * ug(mp.mpf(1) / 2 - ph, lam)
                - 1 / (2 * (a0 + 1)) * x ** a0 * g2 ** (-a0 * ph) * lg(mp.mpf(1) / 2 + a0 * ph, lam)
                - 1 / (2 * (a0 - 1)) * x ** (-a0) * g2 ** (a0 * ph) * ug(mp.mpf(1) / 2 - a0 * ph, lam)
                ) / mp.sqrt(mp.pi)
    dRW_mp = lambda x: -mp.diff(S_mp, x)
    dlog_mp = lambda x: mp.diff(lambda xx: mp.log(dRW_mp(xx)), x)
    for x in [1e3, 1e6, 1e9]:
        assert abs(float(rw.pRW(x, phi, g, alpha0)) - float(1 - S_mp(x))) < 1e-9
        dm = float(dRW_mp(x)); lm = float(dlog_mp(x))
        assert abs(float(rw.dRW(x, phi, g, alpha0)) - dm) / abs(dm) < 1e-7
        assert abs(float(rw.dlog_dRW_dx(x, phi, g, alpha0)) - lm) / abs(lm) < 1e-7


def test_dRW_safe_is_strictly_positive_in_extreme_tail():
    """Regression: dRW_safe must never return a non-positive density, even deep in
    the upper tail where the numeric central difference underflows / hits pRW
    glitches. A <=0 density NaNs the GPD log-likelihood at exceedances (this broke
    a warm-started Ns~203 chain at an extreme time slice with R~5e5)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rw_numeric import dRW_safe
    for alpha0 in (2.5, 5.0):
        for phi in (0.6, 1.0, 2.0):
            xs = rw.qRW(np.array([0.90, 0.95, 0.99, 0.999, 0.99999]),
                        np.full(5, phi), np.ones(5), alpha0)
            d = dRW_safe(xs, np.full(5, phi), np.ones(5), alpha0)
            assert np.all(np.isfinite(d)) and np.all(d > 0.0), (phi, alpha0, d)


@pytest.mark.parametrize("phi", PHIS)
def test_closed_form_matches_monte_carlo(phi):
    """pRW equals the empirical CDF of a direct simulation, even for phi>0.5."""
    rng = np.random.default_rng(12345)
    N = 2_000_000
    U = rng.uniform(0, 1, N)
    R = GAMMA / scipy.stats.norm.ppf(1 - U / 2) ** 2     # Levy(scale=gamma)
    W = 1.0 / (1.0 - rng.uniform(0, 1, N))               # standard Pareto(b=1)
    Ue = rng.uniform(0, 1, N)                            # Log-Laplace(alpha0)
    eps = np.where(Ue < 0.5, np.power(Ue * 2.0, 1.0 / ALPHA0),
                   np.power(0.5 / (1.0 - Ue), 1.0 / ALPHA0))
    X = eps * R ** phi * W
    for x in [2.0, 5.0, 20.0]:
        emp = np.mean(X <= x)
        cf = rw.pRW(x, phi, GAMMA, ALPHA0)
        se = np.sqrt(emp * (1 - emp) / N)
        assert abs(emp - cf) < 4 * se + 2e-3
