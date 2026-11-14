"""Tests for Wendland kernel builders (Phase 2)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels  # noqa: E402
import utilities as u  # noqa: E402
from config import ModelConfig  # noqa: E402


def _grid(n=6):
    xs = np.linspace(0, 10, n)
    X, Y = np.meshgrid(xs, xs)
    return np.c_[X.ravel(), Y.ravel()]


def test_standard_matches_shi_build_weight_matrix():
    sites = _grid(6)
    knots = _grid(3)
    radius = 6.0
    W_new = kernels.wendland_standard(sites, knots, radius)
    theta = np.repeat(radius, knots.shape[0])
    W_shi = u.build_weight_matrix(sites, knots, kernel="wendland",
                                  wendland_theta=theta)
    assert np.allclose(W_new, W_shi, atol=0, rtol=0)


def test_domain_uncovered_mask_isotropic():
    # A radius large enough to reach the farthest site covers everything;
    # a tiny radius leaves sites between knots uncovered.
    sites = _grid(7)
    knots = _grid(3)
    assert not kernels.domain_uncovered_mask(sites, knots, 20.0).any()
    assert kernels.domain_uncovered_mask(sites, knots, 0.5).any()


def test_domain_uncovered_mask_anisotropy_shrinks_coverage():
    # An anisotropic (elliptical) support can uncover sites that the SAME radius
    # covers isotropically -- exactly the failure a static isotropic minimum
    # radius misses. With ratio<1 the support contracts along one axis.
    sites = _grid(7)
    knots = _grid(3)
    cfg = ModelConfig(model_type="shi", use_anisotropy=True)
    radius = 3.5
    iso_unc = kernels.domain_uncovered_mask(sites, knots, radius, config=cfg,
                                            aniso_ratio=1.0)
    aniso_unc = kernels.domain_uncovered_mask(sites, knots, radius, config=cfg,
                                              aniso_ratio=0.3, aniso_angle=0.0)
    assert aniso_unc.sum() > iso_unc.sum()


def test_anisotropy_isotropic():
    # ratio == 1 (with any angle) must equal the isotropic weights exactly.
    sites = _grid(6)
    knots = _grid(3)
    radius = 6.0
    W_iso = kernels.wendland_standard(sites, knots, radius)
    for angle in [0.0, 0.7, 2.0]:
        W_an = kernels.wendland_anisotropic(sites, knots, radius,
                                            aniso_ratio=1.0, aniso_angle=angle)
        assert np.allclose(W_an, W_iso)


def test_anisotropy_changes_weights_and_normalizes():
    sites = _grid(8)
    knots = _grid(3)
    W = kernels.wendland_anisotropic(sites, knots, 6.0,
                                     aniso_ratio=0.4, aniso_angle=np.pi / 6)
    assert np.allclose(W.sum(axis=1), 1.0)
    W_iso = kernels.wendland_standard(sites, knots, 6.0)
    assert not np.allclose(W, W_iso)


def test_gamma_bar_matches_formula():
    sites = _grid(6)
    knots = _grid(3)
    W = kernels.wendland_standard(sites, knots, 6.0)
    gamma_k = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 1.2, 0.9, 1.1])[:knots.shape[0]]
    gb = kernels.gamma_bar_from_weights(W, gamma_k)
    expected = np.sum((W * gamma_k) ** 0.5, axis=1) ** 2
    assert np.allclose(gb, expected)


def test_barrier_blocks_across_wall():
    # Rectangle [0,10]x[0,10] water with a thin land wall at x=5 (gap removed),
    # so points on opposite sides cannot see each other.
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    # water polygon = full rectangle minus a vertical slit wall around x=5.
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    wall = Polygon([(4.9, -1), (5.1, -1), (5.1, 11), (4.9, 11)])
    water = outer.difference(wall)

    left_site = np.array([[2.0, 5.0]])
    right_knot = np.array([[8.0, 5.0]])
    # large radius so they'd be in support if not for the wall
    W = kernels.wendland_barrier(left_site, right_knot, radius=20.0,
                                 water_poly=water)
    # only one knot, isolated -> fallback uniform (weight 1) but the *visibility*
    # is blocked; check the pre-normalisation visibility directly:
    assert not kernels._visible(left_site[0], right_knot[0], water)
    # a knot on the same side is visible
    left_knot = np.array([[3.0, 5.0]])
    assert kernels._visible(left_site[0], left_knot[0], water)


def _box_with_island():
    from shapely.geometry import box
    return box(0, 0, 10, 10).difference(box(4, 4, 6, 6))      # water, island in middle


def test_knot_outside_domain_is_visible_unless_island_blocks():
    """The outer-ring bug: a knot just outside the water polygon (as Faroe/NZ place
    their outer ring) was blocked from EVERY site because the sight line leaves the
    polygon at the knot end. Only the island should block it."""
    pytest.importorskip("shapely")
    water = _box_with_island()
    knot = np.array([-0.5, 5.0])                    # outside the box, west
    assert kernels._visible([2.0, 5.0], knot, water)          # clear line
    assert kernels._visible([8.0, 9.0], knot, water)          # passes north of island
    assert not kernels._visible([8.0, 5.0], knot, water)      # through the island
    # the original rule blocks all three; kept for runs fitted under it
    for site in ([2.0, 5.0], [8.0, 9.0], [8.0, 5.0]):
        assert not kernels._visible(site, knot, water, rule="domain")


def test_rules_agree_inside_the_domain():
    """For a knot INSIDE the bounding box the fix must change nothing."""
    pytest.importorskip("shapely")
    water = _box_with_island()
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(400, 2))
    pts = pts[[water.covers(__import__("shapely").Point(p)) for p in pts]]
    for a, b in zip(pts[:200], pts[200:400]):
        assert (kernels._visible(a, b, water)
                == kernels._visible(a, b, water, rule="domain"))
    # running along the coast is not blocking, under either rule (as before)
    for rule in ("obstacle", "domain"):
        assert kernels._visible([2.0, 4.0], [8.0, 4.0], water, rule)
        assert kernels._visible([4.0, 2.0], [4.0, 8.0], water, rule)


def test_barrier_rule_threads_through_config():
    pytest.importorskip("shapely")
    water = _box_with_island()
    sites = np.array([[2.0, 5.0], [1.0, 1.0], [9.0, 9.0]])
    knots = np.array([[-0.5, 5.0], [3.0, 2.0]])
    new = ModelConfig(model_type="wang", use_barrier_kernels=True, shapefile_path="x")
    old = ModelConfig(model_type="wang", use_barrier_kernels=True, shapefile_path="x",
                      barrier_visibility="domain")
    assert new.barrier_visibility == "obstacle"
    Wn = kernels.build_wendland_weights(sites, knots, 6.0, config=new, water_poly=water)
    Wo = kernels.build_wendland_weights(sites, knots, 6.0, config=old, water_poly=water)
    assert Wn[0, 0] > 0 and Wo[0, 0] == 0          # outside knot now contributes
    assert np.array_equal(kernels.visibility_mask(sites, knots, water, rule="domain"),
                          kernels.visibility_mask(sites, knots, water) & (knots[:, 0] > 0))


def test_saved_configs_without_the_field_load_as_the_old_rule():
    """Runs fitted before the fix saved no barrier_visibility; reloading must give
    the kernels they were fitted with, not the new default."""
    d = ModelConfig(model_type="wang", use_barrier_kernels=True,
                    shapefile_path="x").to_dict()
    d.pop("barrier_visibility")
    assert ModelConfig.from_dict(d).barrier_visibility == "domain"
    d["barrier_visibility"] = "obstacle"
    assert ModelConfig.from_dict(d).barrier_visibility == "obstacle"
    # irrelevant without barrier kernels: keep the default
    assert ModelConfig.from_dict({"model_type": "shi"}).barrier_visibility == "obstacle"
    with pytest.raises(ValueError):
        ModelConfig(barrier_visibility="nonsense")


def test_dispatch_via_config():
    sites = _grid(6)
    knots = _grid(3)
    cfg = ModelConfig(model_type="shi")  # no aniso, no barrier
    W = kernels.build_wendland_weights(sites, knots, 6.0, config=cfg)
    assert np.allclose(W, kernels.wendland_standard(sites, knots, 6.0))


def test_resume_keeps_the_checkpoint_barrier_rule(tmp_path):
    """Resuming a chain checkpointed under the old visibility rule must not switch
    the kernels mid-chain, even though the caller builds today's default config."""
    import warnings
    from sampler import SamplerSettings, keep_checkpoint_barrier_rule
    ck = str(tmp_path / "run.ckpt")
    open(ck + ".pkl", "wb").close()                          # a checkpoint exists
    old = ModelConfig(model_type="wang", use_barrier_kernels=True,
                      shapefile_path="x").to_dict()
    old.pop("barrier_visibility")                             # pre-fix config json
    import json
    with open(ck + ".traces.npz.config.json", "w") as fh:
        json.dump(old, fh)
    st = SamplerSettings(checkpoint_path=ck, resume=True)
    cfg = ModelConfig(model_type="wang", use_barrier_kernels=True, shapefile_path="x")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        keep_checkpoint_barrier_rule(cfg, st, ck + ".pkl")
    assert cfg.barrier_visibility == "domain" and w
    # a fresh run (no checkpoint) gets the new rule
    cfg2 = ModelConfig(model_type="wang", use_barrier_kernels=True, shapefile_path="x")
    keep_checkpoint_barrier_rule(cfg2, st, str(tmp_path / "absent.pkl"))
    assert cfg2.barrier_visibility == "obstacle"
