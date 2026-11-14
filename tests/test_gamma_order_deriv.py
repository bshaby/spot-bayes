"""Validate the incomplete-gamma ORDER derivatives vs mpmath (see gamma_order_deriv).

Covers the (a, z) regime the RW density reaches: upper orders 1/2 - phi and
1/2 - alpha0*phi run negative (down to ~-6 at large alpha0*phi), lower orders
1/2 and 1/2 + alpha0*phi stay positive; z = lambda = (gamma/2) x^{-1/phi} spans
small-to-moderate. The small-z / negative-a corner is the one naive quadrature
gets catastrophically wrong, so it is exercised explicitly."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gamma_order_deriv import (dGammaU_da, dGammaL_da,          # noqa: E402
                               dGammaU_da_ref, dGammaL_da_ref)

mp = pytest.importorskip("mpmath")

A_UPPER = [-6.0, -3.3, -1.5, -0.7, -0.2, 0.1, 0.49, 0.5]      # can be negative
A_LOWER = [0.5, 0.9, 1.5, 3.0]                                # strictly positive
Z_GRID = [1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0, 8.0]


@pytest.mark.parametrize("a", A_UPPER)
@pytest.mark.parametrize("z", Z_GRID)
def test_upper_matches_mpmath(a, z):
    got = float(dGammaU_da(a, z))
    ref = dGammaU_da_ref(a, z)
    assert abs(got - ref) <= 1e-10 * max(abs(ref), 1e-12)


@pytest.mark.parametrize("a", A_LOWER)
@pytest.mark.parametrize("z", Z_GRID)
def test_lower_matches_mpmath(a, z):
    got = float(dGammaL_da(a, z))
    ref = dGammaL_da_ref(a, z)
    assert abs(got - ref) <= 1e-10 * max(abs(ref), 1e-12)


def test_hard_corner_small_z_negative_order():
    # a=-6, z=1e-3: dGammaU ~ -1.12e18; naive Gauss-Laguerre is off by ~10 orders.
    got = float(dGammaU_da(-6.0, 1e-3))
    ref = dGammaU_da_ref(-6.0, 1e-3)
    assert abs(got - ref) <= 1e-9 * abs(ref)


def test_sum_identity():
    # gamma + Gamma = Gamma(a)  =>  dGammaL + dGammaU = Gamma(a) psi(a)  (a > 0)
    from scipy.special import gamma, digamma
    for a in (0.5, 1.3, 2.7):
        for z in (0.05, 0.7, 4.0):
            lhs = float(dGammaL_da(a, z) + dGammaU_da(a, z))
            rhs = float(gamma(a) * digamma(a))
            assert abs(lhs - rhs) <= 1e-10 * abs(rhs)


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    a = rng.uniform(-6.0, 0.5, 500)
    z = rng.uniform(1e-3, 8.0, 500)
    vec = dGammaU_da(a, z)
    scal = np.array([float(dGammaU_da(ai, zi)) for ai, zi in zip(a, z)])
    assert np.allclose(vec, scal, rtol=1e-12, atol=0.0)
    assert vec.shape == (500,)
