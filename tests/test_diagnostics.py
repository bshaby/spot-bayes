"""Tests for live diagnostic trace plots."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diagnostics  # noqa: E402
import simulate as sm  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, run_mcmc  # noqa: E402


def test_plot_traces_writes_png():
    n = 30
    TR = dict(ll=np.random.randn(n).cumsum(), alpha0=3 + np.random.randn(n) * 0.1,
              phi_k=0.5 + np.random.randn(n, 13) * 0.05,
              rho_k=1 + np.random.randn(n, 13) * 0.05,
              nu=1 + np.random.randn(n) * 0.05,
              beta_sigma=np.random.randn(n, 2))
    path = os.path.join(tempfile.mkdtemp(), "trace.png")
    diagnostics.plot_traces(TR, n, path, thin=5, iteration=145)
    assert os.path.exists(path) and os.path.getsize(path) > 1000


def test_rep_indices():
    assert diagnostics._rep_indices(2) == [0, 1]
    idx = diagnostics._rep_indices(13, m=4)
    assert idx[0] == 0 and idx[-1] == 12 and len(idx) == 4


def test_ess_iid_is_near_n():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    assert diagnostics.ess_1d(x) > 0.7 * 4000      # iid -> ESS ~ n


def test_ess_ar1_matches_theory():
    # AR(1) with rho=0.8: integrated autocorr time tau=(1+rho)/(1-rho)=9 -> ESS~n/9
    rng = np.random.default_rng(1)
    rho, n = 0.8, 20000
    x = np.empty(n); x[0] = rng.standard_normal()
    e = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    ess = diagnostics.ess_1d(x)
    assert 0.6 * n / 9 < ess < 1.6 * n / 9          # near the theoretical n/9


def test_ess_constant_returns_n():
    assert diagnostics.ess_1d(np.full(100, 3.0)) == 100.0


def test_ess_summary_structure():
    n = 200
    TR = dict(ll=np.random.randn(n).cumsum(), alpha0=np.random.randn(n),
              phi_k=np.random.randn(n, 5), rho_k=np.random.randn(n, 5))
    e = diagnostics.ess_summary(TR, n)
    assert isinstance(e["ll"], float)
    assert set(e["phi_k"]) == {"min", "median"}
    assert e["_n_used"] == n // 2


def test_sampler_writes_diagnostic_plot():
    out = sm.simulate(ModelConfig("shi"), sm.SimConfig(Ns=25, Nt=8, seed=11))
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"])
    png = os.path.join(tempfile.mkdtemp(), "live.png")
    st = SamplerSettings(n_iters=40, adapt_size=20, thin=5, nu=out["nu"], seed=1,
                         plot_every=20, plot_path=png)
    run_mcmc(ModelConfig("shi"), data, st)
    assert os.path.exists(png) and os.path.getsize(png) > 1000
