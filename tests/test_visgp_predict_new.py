"""Prediction at NEW sites not in the original VisGP graph (out-of-sample)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import covariance as cov  # noqa: E402
import domain  # noqa: E402


def test_fullgraph_matches_standard_kriging():
    """On a fully-visible domain, VisGP new-site prediction == ordinary GP
    kriging with the base Matern covariance."""
    rng = np.random.default_rng(0)
    train = rng.uniform(0, 10, (15, 2))
    new = rng.uniform(0, 10, (4, 2))
    rho, nu, s2 = 3.0, 1.0, 1.0

    K = cov.matern_wang(cov.dist_matrix(train), rho, s2, nu)
    Q = cov.visgp_precision([list(range(15))], [], train, rho, nu)  # full graph
    Z = rng.multivariate_normal(np.zeros(15), K)

    visible = np.ones((4, 15), dtype=bool)   # everything visible
    mean, var = cov.visgp_predict_new(new, train, Q, rho, nu, Z, visible, sigmasq=s2)

    # standard kriging reference
    for i in range(4):
        d = np.linalg.norm(train - new[i], axis=1)
        c = cov.matern_wang_cross(d, rho, s2, nu)
        m_ref = c @ np.linalg.solve(K, Z)
        v_ref = s2 - c @ np.linalg.solve(K, c)
        assert abs(mean[i] - m_ref) < 1e-8
        assert abs(var[i] - v_ref) < 1e-8


def test_predict_new_with_time_columns():
    rng = np.random.default_rng(1)
    train = rng.uniform(0, 10, (12, 2))
    new = rng.uniform(0, 10, (3, 2))
    rho, nu = 3.0, 1.0
    K = cov.matern_wang(cov.dist_matrix(train), rho, 1.0, nu)
    Q = cov.visgp_precision([list(range(12))], [], train, rho, nu)
    Z = rng.multivariate_normal(np.zeros(12), K, size=5).T  # (12, 5)
    visible = np.ones((3, 12), dtype=bool)
    mean, var = cov.visgp_predict_new(new, train, Q, rho, nu, Z, visible)
    assert mean.shape == (3, 5)
    assert var.shape == (3,)


def test_barrier_uses_only_visible_neighbors():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    wall = Polygon([(4.9, 2), (5.1, 2), (5.1, 11), (4.9, 11)])  # gap below y=2
    water = outer.difference(wall)

    # training sites: a cluster on the left and a cluster on the right
    left = np.array([[1.0, 5.0], [2.0, 6.0], [1.5, 4.0]])
    right = np.array([[8.0, 5.0], [9.0, 6.0], [8.5, 4.0]])
    train = np.vstack([left, right])
    rho, nu = 4.0, 1.0
    A = domain.build_visibility_graph(train, water)
    cl, seps = domain.build_junction_tree(domain.find_maximal_cliques(A))
    Q = cov.visgp_precision(cl, seps, train, rho, nu)
    Z = np.array([2.0, 2.0, 2.0, -2.0, -2.0, -2.0])   # left high, right low

    # a new site on the left side (behind the wall from the right cluster)
    new = np.array([[1.5, 5.5]])
    visible = domain.visible_from_new(new, train, water)
    # it should see left sites, not right sites
    assert visible[0, :3].all()
    assert not visible[0, 3:].any()
    mean, var = cov.visgp_predict_new(new, train, Q, rho, nu, Z, visible)
    # prediction driven by the (high) left cluster, not the (low) right cluster
    assert mean[0] > 0.5


def test_predict_new_interpolates_at_training_location():
    """A new site coinciding with a training site recovers its Z (var ~ 0)."""
    rng = np.random.default_rng(7)
    train = rng.uniform(0, 10, (10, 2))
    rho, nu, s2 = 3.0, 1.0, 1.0
    K = cov.matern_wang(cov.dist_matrix(train), rho, s2, nu)
    Q = cov.visgp_precision([list(range(10))], [], train, rho, nu)
    Z = rng.multivariate_normal(np.zeros(10), K)
    new = train[3:4].copy()                 # exactly at training site 3
    visible = np.ones((1, 10), bool)
    mean, var = cov.visgp_predict_new(new, train, Q, rho, nu, Z, visible, sigmasq=s2)
    assert abs(mean[0] - Z[3]) < 1e-6
    assert var[0] < 1e-8


def test_joint_matches_dense_kriging_full_visibility():
    rng = np.random.default_rng(3)
    train = rng.uniform(0, 10, (12, 2))
    new = rng.uniform(0, 10, (5, 2))
    rho, nu, s2 = 3.0, 1.0, 1.0
    K = cov.matern_wang(cov.dist_matrix(train), rho, s2, nu)
    Q = cov.visgp_precision([list(range(12))], [], train, rho, nu)
    Z = rng.multivariate_normal(np.zeros(12), K)
    vt = np.ones((5, 12), bool)
    vn = np.ones((5, 5), bool); np.fill_diagonal(vn, False)
    mean, covm = cov.visgp_predict_new_joint(new, train, Q, rho, nu, Z, vt, vn, sigmasq=s2)

    Knn = cov.matern_wang(cov.dist_matrix(new), rho, s2, nu)
    Knt = np.array([[cov.matern_wang_cross(np.array([np.linalg.norm(new[i] - train[j])]),
                                           rho, s2, nu)[0] for j in range(12)]
                    for i in range(5)])
    m_ref = Knt @ np.linalg.solve(K, Z)
    c_ref = Knn - Knt @ np.linalg.solve(K, Knt.T)
    assert np.allclose(mean, m_ref, atol=1e-9)
    assert np.allclose(covm, c_ref, atol=1e-9)
    # genuine joint dependence among new sites
    assert np.any(np.abs(covm[np.triu_indices(5, 1)]) > 1e-6)


def test_joint_barrier_decorrelates_opposite_sides():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    wall = Polygon([(4.9, -1), (5.1, -1), (5.1, 11), (4.9, 11)])  # full wall, no gap
    water = outer.difference(wall)
    train = np.array([[1.0, 5.0], [9.0, 5.0]])
    rho, nu = 4.0, 1.0
    A = domain.build_visibility_graph(train, water)
    cl, seps = domain.build_junction_tree(domain.find_maximal_cliques(A))
    Q = cov.visgp_precision(cl, seps, train, rho, nu)
    Z = np.array([1.0, -1.0])
    # two new sites: one on each side of the wall
    new = np.array([[2.0, 5.0], [8.0, 5.0]])
    vt = domain.visible_from_new(new, train, water)
    vn = domain.visible_among_new(new, water)
    assert not vn[0, 1]  # the two new sites cannot see each other across the wall
    mean, covm = cov.visgp_predict_new_joint(new, train, Q, rho, nu, Z, vt, vn)
    # conditional cross-covariance between opposite-side new sites is ~0
    assert abs(covm[0, 1]) < 1e-8


def test_isolated_new_site_falls_back_to_marginal():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon, Point
    # water is two disjoint disks; a new site in one, all training in the other
    left_disk = Point(2, 5).buffer(1.5)
    right_disk = Point(8, 5).buffer(1.5)
    water = left_disk.union(right_disk)
    train = np.array([[8.0, 5.0], [8.5, 5.5], [7.5, 4.5]])  # all in right disk
    rho, nu, s2 = 2.0, 1.0, 1.0
    Q = cov.visgp_precision([list(range(3))], [], train, rho, nu)
    Z = np.array([3.0, 3.0, 3.0])
    new = np.array([[2.0, 5.0]])  # in left disk, cannot see any training site
    visible = domain.visible_from_new(new, train, water)
    assert not visible.any()
    mean, var = cov.visgp_predict_new(new, train, Q, rho, nu, Z, visible, sigmasq=s2)
    assert mean[0] == 0.0          # marginal mean
    assert abs(var[0] - s2) < 1e-12  # marginal variance
