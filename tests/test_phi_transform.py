"""Tests for the phi-domain transform abstraction (Phase 5)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_transform as pt  # noqa: E402
from config import PHI_UNIT, PHI_POSITIVE  # noqa: E402


def test_roundtrip_unit():
    c = np.array([0.01, 0.3, 0.5, 0.9, 0.99])
    u = pt.to_unconstrained(c, PHI_UNIT)
    cb = pt.to_constrained(u, PHI_UNIT)
    assert np.allclose(c, cb, atol=1e-10)
    assert np.all((cb > 0) & (cb < 1))


def test_roundtrip_positive():
    c = np.array([0.05, 0.5, 1.0, 2.5, 10.0])
    u = pt.to_unconstrained(c, PHI_POSITIVE)
    cb = pt.to_constrained(u, PHI_POSITIVE)
    assert np.allclose(c, cb, atol=1e-10)
    assert np.all(cb > 0)


def test_logprior_finite_both():
    u = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.isfinite(pt.logprior_unconstrained(u, PHI_UNIT))
    assert np.isfinite(pt.logprior_unconstrained(u, PHI_POSITIVE))


def test_field_from_knots_domains():
    rng = np.random.default_rng(0)
    W = rng.uniform(size=(20, 4))
    W = W / W.sum(axis=1, keepdims=True)
    eta = rng.standard_normal(4)
    f_unit = pt.field_from_knots(W, eta, PHI_UNIT)
    f_pos = pt.field_from_knots(W, eta, PHI_POSITIVE)
    assert np.all((f_unit > 0) & (f_unit < 1))
    assert np.all(f_pos > 0)
