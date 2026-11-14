"""Analytic gradient of the phi conditional (phi_gradient.phi_value_grad).

The gradient's whole point is that it never re-inverts qRW: dX/dphi comes from the
implicit-function identity dX/dphi = -(dpRW/dphi)/dRW at the root already in hand. So
the tests check it against central differences of the PRODUCTION value it claims to
differentiate, in BOTH phi domains -- the chain rule from the site field to the knots
differs between them, and getting only one right is the failure this file exists to
catch (the branch this came from exercised only "positive").
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi_gradient as pg  # noqa: E402


def _state(phi_domain, Ns=25, Nt=5, seed=4):
    import warnings
    import simulate as sm
    from config import ModelConfig
    from sampler import SamplerData, SamplerSettings, ModelState
    cfg = ModelConfig(model_type="shi", phi_domain=phi_domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = sm.simulate(cfg, sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = SamplerData(Y=o["Y_missing"], sites_xy=o["sites_xy"],
                       u_vec=o["u_mat"][:, 0], sigma_vec=o["sigma_mat"][:, 0],
                       xi_vec=o["xi_mat"][:, 0], p_cens=o["p_cens"],
                       knots_S=o["knots_S"], knots_phi=o["knots_phi"],
                       knots_rho=o["knots_rho"])
    return ModelState(cfg, data, SamplerSettings(nu=o["nu"], seed=1, n_iters=10))


@pytest.mark.parametrize("phi_domain", ["positive", "unit"])
def test_phi_gradient_matches_central_differences(phi_domain):
    import warnings
    state = _state(phi_domain)
    pk = state.phi_k.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, g = pg.phi_value_grad(state, pk)
        h = 1e-5
        fd = np.empty_like(g)
        for j in range(pk.size):
            a = pk.copy(); a[j] += h
            b = pk.copy(); b[j] -= h
            fd[j] = (pg.phi_value_grad(state, a)[0]
                     - pg.phi_value_grad(state, b)[0]) / (2 * h)
    assert np.max(np.abs(g - fd)) <= 1e-4 * max(1.0, np.max(np.abs(fd)))


@pytest.mark.parametrize("phi_domain", ["positive", "unit"])
def test_phi_value_matches_the_production_likelihood(phi_domain):
    """The value must be the sum the sampler itself would compute, not a variant --
    otherwise the gradient is of the wrong function even when it is self-consistent."""
    import warnings
    from mcmc_updates import phi_logprior
    state = _state(phi_domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val, _ = pg.phi_value_grad(state, state.phi_k.copy())
        qv, dv = state.qd_RW_all()
        ref = float(sum(state.loglik_t(t, qv=qv[:, t], dv=dv[:, t])
                        for t in range(state.Nt)))
        ref += float(phi_logprior(state, state.phi_k))
    assert abs(val - ref) <= 1e-9 * max(1.0, abs(ref))


def test_dX_dphi_identity_avoids_reinversion():
    """dX/dphi = -(dpRW/dphi)/dRW at the existing root, versus differencing qRW
    itself. This identity is why the gradient costs no extra inversions."""
    from rw_inte import qRW, dRW, dpRW_dphi
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.8, 0.999, n)
    phi = rng.uniform(0.2, 0.8, n)
    gam = np.full(n, 1.0)
    a0 = rng.uniform(1.5, 5.0, n)
    q = qRW(p, phi, gam, a0)
    analytic = -dpRW_dphi(q, phi, gam, a0) / dRW(q, phi, gam, a0)
    h = 1e-6
    fd = (qRW(p, phi + h, gam, a0) - qRW(p, phi - h, gam, a0)) / (2 * h)
    assert np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-8)) < 1e-5
