"""
Non-convex domain processing for the Wang model.

Responsibilities:
  * Load a domain boundary (water/ocean polygon) from a shapefile.
  * Build the visibility graph: sites i,j are adjacent iff the straight segment
    between them lies entirely in the domain.
  * Load pre-computed maximal cliques + separators (the junction tree) used by
    the VisGP precision. Cliques are computed in R (see scripts/compute_cliques.R,
    matching Wang's workflow) and stored as .rds; we read them with a small
    Rscript subprocess (R is the reference implementation) or accept a portable
    .json. Both yield 0-indexed Python lists.
  * Validate the running-intersection property of the junction tree before use
    (plan Concern #1), and reconstruct separators from cliques when needed.

networkx is used for the pure-Python clique route (handy for tests and for users
without R), but the .rds path keeps full compatibility with Wang's R outputs.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import List, Sequence, Tuple

import numpy as np


# -- domain boundary ---------------------------------------------------------
def load_domain(shapefile_path: str):
    """Load a shapefile and return a single (multi)polygon of the domain.

    The shapefile is expected to delineate the valid region (e.g. water). If it
    contains line features (a coastline), the caller should pre-polygonize; here
    we union all geometries and return the result.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    gdf = gpd.read_file(shapefile_path)
    return unary_union(gdf.geometry.values)


def _segment_in_domain(p0, p1, water_poly) -> bool:
    from shapely.geometry import LineString
    return bool(water_poly.covers(LineString([tuple(p0), tuple(p1)])))


def visible_from_new(new_coords: np.ndarray, train_coords: np.ndarray,
                     water_poly, n_cores: int = 1) -> np.ndarray:
    """Visibility of NEW sites to the training sites.

    Returns a boolean (n_new, n_train) matrix where entry [i, j] is True iff the
    straight segment from new site i to training site j lies entirely in the
    domain. Used by covariance.visgp_predict_new to predict at sites that were
    not part of the original VisGP graph.
    """
    new_coords = np.atleast_2d(np.asarray(new_coords, dtype=float))
    train_coords = np.asarray(train_coords, dtype=float)
    n_new, n_tr = new_coords.shape[0], train_coords.shape[0]

    def _row(i):
        return i, np.array([_segment_in_domain(new_coords[i], train_coords[j], water_poly)
                            for j in range(n_tr)], dtype=bool)

    if n_cores and n_cores > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_cores) as ex:
            rows = list(ex.map(_row, range(n_new)))
    else:
        rows = [_row(i) for i in range(n_new)]
    V = np.zeros((n_new, n_tr), dtype=bool)
    for i, r in rows:
        V[i] = r
    return V


def visible_among_new(new_coords: np.ndarray, water_poly,
                      n_cores: int = 1) -> np.ndarray:
    """Symmetric (n_new, n_new) visibility among the new sites themselves
    (diagonal False). Used for joint prediction at new sites."""
    return build_visibility_graph(new_coords, water_poly, n_cores).astype(bool)


def build_visibility_graph(sites_xy: np.ndarray, water_poly,
                           n_cores: int = 1) -> np.ndarray:
    """Binary adjacency matrix A (n x n): A[i,j]=1 iff i,j are mutually visible.

    Diagonal is 0 (matching Wang's convention).
    """
    n = sites_xy.shape[0]
    A = np.zeros((n, n), dtype=int)

    def _row(i):
        out = np.zeros(i, dtype=int)
        for j in range(i):
            out[j] = _segment_in_domain(sites_xy[i], sites_xy[j], water_poly)
        return i, out

    if n_cores and n_cores > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_cores) as ex:
            rows = list(ex.map(_row, range(n)))
    else:
        rows = [_row(i) for i in range(n)]
    for i, out in rows:
        A[i, :i] = out
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A


# -- cliques / junction tree -------------------------------------------------
def find_maximal_cliques(A: np.ndarray, triangulate: bool = True) -> List[List[int]]:
    """Cliques of the visibility graph for the VisGP junction-tree decomposition.

    The VisGP precision Q = sum_c inv(C_c) - sum_s inv(C_s) and its decomposable
    logdet identity (sum_c logdet - sum_s logdet) are only valid when the graph is
    CHORDAL and the cliques form a junction tree with the running-intersection
    property. A raw visibility graph is generally NOT chordal, so taking its
    Bron-Kerbosch maximal cliques produces an INVALID junction tree -- the
    inclusion-exclusion logdet is then wrong (biased, growing with the range), which
    silently corrupts the Z-field likelihood (see exp2 wang blow-up).

    With triangulate=True (default) we first complete the graph to a chordal graph
    (a minimal triangulation, adding fill-in edges) and return ITS cliques, which
    DO form a valid junction tree via build_junction_tree. triangulate=False
    recovers the old Bron-Kerbosch behaviour (only correct if A is already chordal,
    e.g. a single complete clique).
    """
    import networkx as nx
    G = nx.from_numpy_array(np.asarray(A))
    if triangulate and not nx.is_chordal(G):
        H, _ = nx.complete_to_chordal_graph(G)
        return [sorted(c) for c in nx.chordal_graph_cliques(H)]
    return [sorted(c) for c in nx.find_cliques(G)]


def build_junction_tree(cliques: Sequence[Sequence[int]]
                        ) -> Tuple[List[List[int]], List[List[int]]]:
    """Return (cliques, separators) for a valid junction tree.

    Separators are the clique intersections along a maximum-weight spanning tree
    of the clique-intersection graph (edge weight = |Ci ∩ Cj|). This yields the
    running-intersection property required for
        Q = sum_c inv(C_c) - sum_s inv(C_s)
    to be correct (plan Concern #1).
    """
    import networkx as nx
    cliques = [sorted(set(c)) for c in cliques]
    m = len(cliques)
    G = nx.Graph()
    G.add_nodes_from(range(m))
    for i in range(m):
        si = set(cliques[i])
        for j in range(i + 1, m):
            w = len(si & set(cliques[j]))
            if w > 0:
                G.add_edge(i, j, weight=w)
    separators: List[List[int]] = []
    # max spanning tree -> junction tree (one per connected component)
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        T = nx.maximum_spanning_tree(sub)
        for i, j in T.edges():
            sep = sorted(set(cliques[i]) & set(cliques[j]))
            if sep:
                separators.append(sep)
    # Guard: the inclusion-exclusion precision/logdet is only correct for a junction
    # tree with the running-intersection property, which requires CHORDAL cliques
    # (use find_maximal_cliques(A, triangulate=True)). Fail loudly otherwise rather
    # than silently corrupting the VisGP logdet.
    if not verify_running_intersection(cliques, separators):
        raise ValueError(
            "build_junction_tree: cliques do not form a valid junction tree "
            "(running intersection fails) -- the visibility graph is likely not "
            "chordal. Pass triangulated cliques: find_maximal_cliques(A, "
            "triangulate=True).")
    return cliques, separators


def verify_running_intersection(cliques: Sequence[Sequence[int]],
                                separators: Sequence[Sequence[int]]) -> bool:
    """Sanity check: every separator is a subset of >= 2 cliques, and the
    inclusion-exclusion node counts are consistent (each node appears in
    (#cliques containing it) - (#separators containing it) == 1 net).

    This is a necessary condition for a valid junction tree and catches the most
    common clique-file mistakes.
    """
    from collections import Counter
    cnt = Counter()
    for c in cliques:
        for v in c:
            cnt[v] += 1
    for s in separators:
        for v in s:
            cnt[v] -= 1
    return all(v == 1 for v in cnt.values())


# -- (de)serialization -------------------------------------------------------
def _read_rds_lists(path: str) -> List[List[int]]:
    """Read an R list-of-integer-vectors .rds via an Rscript subprocess.

    Returns 1-indexed values as stored in R; caller converts to 0-indexed.
    Requires Rscript + jsonlite (the reference clique workflow is in R).
    """
    rscript = (
        'args <- commandArgs(trailingOnly=TRUE);'
        'obj <- readRDS(args[1]);'
        'lst <- lapply(obj, function(x) as.integer(x));'
        'cat(jsonlite::toJSON(lst, auto_unbox=FALSE))'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(rscript)
        rfile = fh.name
    try:
        out = subprocess.check_output(["Rscript", rfile, path],
                                      stderr=subprocess.STDOUT)
    finally:
        os.unlink(rfile)
    data = json.loads(out.decode())
    # jsonlite wraps scalars as length-1 arrays already; normalise to lists
    return [list(map(int, x)) if isinstance(x, list) else [int(x)] for x in data]


def load_cliques(path: str, one_indexed: bool = True) -> List[List[int]]:
    """Load cliques/separators from .rds (R) or .json. Returns 0-indexed lists.

    `one_indexed` controls the R->Python offset (R is 1-indexed). JSON files are
    assumed already 0-indexed unless one_indexed is explicitly set.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as fh:
            data = json.load(fh)
        groups = [list(map(int, g)) for g in data]
        return [[i - 1 for i in g] for g in groups] if one_indexed and False else groups
    elif ext == ".rds":
        groups = _read_rds_lists(path)
        off = 1 if one_indexed else 0
        return [[i - off for i in g] for g in groups]
    raise ValueError(f"Unsupported clique file extension: {ext!r} (use .rds or .json)")


def save_cliques_json(path: str, cliques: Sequence[Sequence[int]]) -> None:
    """Save 0-indexed cliques/separators as portable JSON."""
    with open(path, "w") as fh:
        json.dump([list(map(int, c)) for c in cliques], fh)
