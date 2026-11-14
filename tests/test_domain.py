"""Tests for domain processing (Phase 3)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import domain  # noqa: E402
import covariance as cov  # noqa: E402

WANG_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Spatial-Extremes-on-Domains-with-Physical-Barriers-main", "Data")


def test_visibility_graph_with_barrier():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    wall = Polygon([(4.9, 2), (5.1, 2), (5.1, 11), (4.9, 11)])  # wall with a gap below y=2
    water = outer.difference(wall)
    sites = np.array([[2.0, 5.0], [8.0, 5.0], [3.0, 5.0], [2.0, 1.0], [8.0, 1.0]])
    A = domain.build_visibility_graph(sites, water)
    # across the wall (both at y=5) blocked:
    assert A[0, 1] == 0
    # same side visible:
    assert A[0, 2] == 1
    # around the gap (y=1, below wall) visible:
    assert A[3, 4] == 1
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)


def test_find_cliques_and_junction_tree_valid():
    # complete graph on 5 nodes -> single clique, no separators
    A = np.ones((5, 5), dtype=int)
    np.fill_diagonal(A, 0)
    cliques = domain.find_maximal_cliques(A)
    assert len(cliques) == 1 and sorted(cliques[0]) == [0, 1, 2, 3, 4]
    cl, seps = domain.build_junction_tree(cliques)
    assert domain.verify_running_intersection(cl, seps)


def test_junction_tree_chain_running_intersection():
    # two overlapping cliques sharing node 2 -> separator {2}
    cliques = [[0, 1, 2], [2, 3, 4]]
    cl, seps = domain.build_junction_tree(cliques)
    assert [2] in seps
    assert domain.verify_running_intersection(cl, seps)
    # and the resulting VisGP precision is PD on random coords
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (5, 2))
    Q = cov.visgp_precision(cl, seps, coords, 4.0, 1.0)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


@pytest.mark.skipif(not os.path.exists(os.path.join(WANG_DATA, "buffered_faroe_cliques.rds")),
                    reason="Wang clique .rds not present")
def test_load_wang_rds_cliques():
    import shutil
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    cliques = domain.load_cliques(os.path.join(WANG_DATA, "buffered_faroe_cliques.rds"))
    assert len(cliques) == 41           # known from the file
    # 0-indexed conversion: min index should be 0 (R min was 1)
    allidx = [i for c in cliques for i in c]
    assert min(allidx) == 0
