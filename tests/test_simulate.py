"""Tests for forward simulation (Phase 7)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
from config import ModelConfig  # noqa: E402


def test_shi_simulation_runs_and_matches_rng_path():
    sc = sm.SimConfig(Ns=40, Nt=8, seed=2345)
    out = sm.simulate(ModelConfig(model_type="shi"), sc)
    assert out["Y"].shape == (40, 8)
    assert np.allclose(out["W_wend_S"].sum(1), 1.0)
    # sites follow Shi's np.random path with the same seed
    np.random.seed(2345)
    minX, maxX, minY, maxY = sc.domain
    sites = np.random.random((40, 2)) * [maxX - minX, maxY - minY] + [minX, minY]
    assert np.allclose(sites, out["sites_xy"])


def test_hw_simulation_single_global_scale():
    out = sm.simulate(ModelConfig(model_type="huser-wadsworth",
                                  phi_domain="positive"),
                      sm.SimConfig(Ns=30, Nt=6, seed=1, delta=0.8))
    R = out["R_at_sites"]
    # single global Levy scale -> identical across sites within each time
    assert np.allclose(R[0], R[1])
    assert out["Y"].shape == (30, 6)


def test_wang_visgp_simulation():
    sc = sm.SimConfig(Ns=24, Nt=5, seed=7, rho_const=4.0)
    cfg = ModelConfig(model_type="wang", use_visgp=True, cliques_path="dummy.rds")
    # full-graph clique (no barriers) just to exercise the VisGP path
    cliques = [list(range(sc.Ns))]
    out = sm.simulate(cfg, sc, cliques=cliques, separators=[])
    assert out["Y"].shape == (24, 5)
    assert np.all(np.isfinite(out["Z"]))


def test_anisotropy_changes_simulation():
    sc = sm.SimConfig(Ns=40, Nt=4, seed=3)
    cfg = ModelConfig(model_type="shi", use_anisotropy=True)
    out_iso = sm.simulate(ModelConfig(model_type="shi"), sc)
    out_an = sm.simulate(cfg, sc, aniso_ratio=0.4, aniso_angle=0.6)
    assert not np.allclose(out_iso["W_wend_S"], out_an["W_wend_S"])
