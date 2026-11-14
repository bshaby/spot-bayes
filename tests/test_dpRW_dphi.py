"""The fused C d/dphi RW derivatives (rw_inte.dpRW_dphi + the incomplete-gamma
order-derivatives) vs references. See docs/phi_gradient_l2phi.md and
cpp/rw_inte_cpp.cpp:dpRW_dphi_standard_Pareto_multiplicative_nugget_C.

The C reuses the density's own upper/lower_gamma_C plus Gauss-Legendre order-
derivatives; these tests guard it against an mpmath/Python-kernel reference and
against a finite difference of the C pRW itself."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rw_inte import (dpRW_dphi, dlog_dRW_dphi, dupper_gamma_da,       # noqa: E402
                     dlower_gamma_da, pRW, dRW)
from gamma_order_deriv import dGammaU_da, dGammaL_da                  # noqa: E402

A_UPPER = [-6.0, -3.3, -1.5, -0.7, -0.2, 0.1, 0.49, 0.5]
A_LOWER = [0.5, 0.9, 1.5, 3.0]
Z_GRID = [1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0, 8.0]


@pytest.mark.parametrize("a", A_UPPER)
@pytest.mark.parametrize("z", Z_GRID)
def test_c_upper_order_deriv_matches_python(a, z):
    # C (n=64 GL) vs the validated Python kernel (n=128) -- both trace to mpmath.
    got = float(dupper_gamma_da(a, z))
    ref = float(dGammaU_da(a, z))
    assert abs(got - ref) <= 1e-9 * max(abs(ref), 1e-12)


@pytest.mark.parametrize("a", A_LOWER)
@pytest.mark.parametrize("z", Z_GRID)
def test_c_lower_order_deriv_matches_python(a, z):
    got = float(dlower_gamma_da(a, z))
    ref = float(dGammaL_da(a, z))
    assert abs(got - ref) <= 1e-7 * max(abs(ref), 1e-12)


@pytest.mark.parametrize("x", [1.5, 5.0, 20.0, 100.0, 1e3])
@pytest.mark.parametrize("phi", [0.3, 0.5, 0.8, 1.2])
@pytest.mark.parametrize("a0", [3.0, 5.0])
def test_c_dpRW_dphi_matches_central_diff(x, phi, a0):
    g = 1.0
    h = 1e-5
    ana = float(dpRW_dphi(x, phi, g, a0))
    fd = (float(pRW(x, phi + h, g, a0)) - float(pRW(x, phi - h, g, a0))) / (2 * h)
    # the tolerance is the FD reference's own floor (~5e-5), not the analytic's error
    assert abs(ana - fd) <= 1e-3 * max(abs(fd), 1e-8) + 1e-9


@pytest.mark.parametrize("x", [1.5, 5.0, 20.0, 100.0, 1e3, 1e5])
@pytest.mark.parametrize("phi", [0.3, 0.5, 0.8, 1.2])
@pytest.mark.parametrize("a0", [3.0, 5.0])
def test_c_dlog_dRW_dphi_matches_central_diff(x, phi, a0):
    g = 1.0
    h = 1e-6
    ana = float(dlog_dRW_dphi(x, phi, g, a0))
    fd = (np.log(float(dRW(x, phi + h, g, a0))) - np.log(float(dRW(x, phi - h, g, a0)))) / (2 * h)
    assert abs(ana - fd) <= 1e-3 * max(abs(fd), 1e-6) + 1e-8


def test_c_dpRW_dphi_vectorizes():
    x = np.array([5.0, 50.0, 500.0]); phi = np.full(3, 0.6); g = np.full(3, 1.5)
    out = dpRW_dphi(x, phi, g, 4.0)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
