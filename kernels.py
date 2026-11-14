"""
Wendland kernel weight matrices for the random scale R(s) = W @ S.

Three flavours, selected by ModelConfig:
  * standard    : isotropic, no barriers  -- reproduces Shi's behaviour exactly.
  * anisotropic : elliptical support      -- Wang's geometric anisotropy.
  * barrier     : visibility-clipped      -- Wang's non-convex domain support,
                  optionally combined with anisotropy.

All builders return a per-row-normalised (Ns, n_knots) matrix W so that
  R(s,t) = (W @ S)[s]            and
  gamma_bar(s) = (sum_k sqrt(W[s,k] * gamma_k))**2     (Levy alpha=1/2)
match Shi's sampler (see sampler line `gamma_bar_vec = ...`).

The anisotropy transform follows Wang's functions.py exactly:
  B = diag(1, 1/ratio) @ [[cos a, sin a], [-sin a, cos a]]
  anisotropic squared distance: (s-b)^T B^T B (s-b)
With ratio == 1 the anisotropic distance reduces to Euclidean, so the
anisotropic builder is delegated to the standard one (guaranteeing identical
output -- see tests/test_kernels.py::test_anisotropy_isotropic).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.spatial

import utilities as _u


# -- raw Wendland basis (unnormalised), matching utilities.wendland_weights_fun
def _wendland_raw(d_over_theta: np.ndarray, k: int = 0, dimension: int = 2) -> np.ndarray:
    """Unnormalised Wendland basis value as a function of scaled distance d/theta.

    Mirrors the algebra in utilities.wendland_weights_fun but without the
    per-vector normalisation, so we can normalise per row afterwards.
    """
    d = np.asarray(d_over_theta, dtype=float)
    l = np.floor(dimension / 2) + k + 1
    if k == 0:
        res = np.where(d < 1, (1 - d) ** l, 0.0)
    elif k == 1:
        res = np.where(d < 1, (1 - d) ** (l + k) * ((l + 1) * d + 1), 0.0)
    elif k == 2:
        res = np.where(d < 1, (1 - d) ** (l + k) * ((l ** 2 + 4 * l + 3) * d ** 2
                                                    + (3 * l + 6) * d + 3), 0.0)
    elif k == 3:
        res = np.where(d < 1, (1 - d) ** (l + k) * ((l ** 3 + 9 * l ** 2 + 23 * l + 15) * d ** 3
                                                    + (6 * l ** 2 + 36 * l + 45) * d ** 2
                                                    + (15 * l + 45) * d + 15), 0.0)
    else:
        raise ValueError("k must be < 4")
    return res


def anisotropy_matrix(aniso_ratio: float, aniso_angle: float) -> np.ndarray:
    """Geometric anisotropy matrix B = diag(1, 1/ratio) @ R(angle).

    Matches Wang's functions.anisotropy_matrix(method='geometric').
    """
    c, s = np.cos(aniso_angle), np.sin(aniso_angle)
    R = np.array([[c, s], [-s, c]])
    S = np.array([[1.0, 0.0], [0.0, 1.0 / aniso_ratio]])
    return S @ R


def anisotropic_distances(sites_xy: np.ndarray, knot: np.ndarray,
                          aniso_ratio: float, aniso_angle: float) -> np.ndarray:
    """Anisotropic distance from every site to a single knot.

    h = sqrt( diff^T (B^T B) diff ),  B = anisotropy_matrix(ratio, angle).
    Reduces to Euclidean when ratio == 1.
    """
    diff = sites_xy - knot
    if aniso_ratio == 1.0:
        return np.linalg.norm(diff, axis=1)
    B = anisotropy_matrix(aniso_ratio, aniso_angle)
    A = B.T @ B
    h_sq = np.einsum("ij,jk,ik->i", diff, A, diff)
    return np.sqrt(np.maximum(h_sq, 0.0))


def domain_uncovered_mask(sites_xy: np.ndarray, knots_xy: np.ndarray, radius,
                          *, config=None, aniso_ratio: float = 1.0,
                          aniso_angle: float = 0.0, water_poly=None,
                          k: int = 0, tol: float = 0.0) -> np.ndarray:
    """Boolean (Ns,) mask of sites NOT covered by any Wendland kernel.

    A site is covered iff at least one knot gives it a strictly-positive RAW
    (un-normalised) Wendland weight -- and, for barrier kernels, that knot is
    visible. This is computed on the raw weights ON PURPOSE: the per-row uniform
    fallback in `_normalize_rows` makes every normalised row sum to 1, so a
    coverage gap is invisible after normalisation. The sampler uses this to
    REJECT any radius/anisotropy proposal that would leave part of the domain
    uncovered -- the correct, anisotropy-aware replacement for a single static
    minimum radius (which only holds under isotropy).
    """
    use_aniso = bool(getattr(config, "use_anisotropy", False)) and aniso_ratio != 1.0
    use_barrier = bool(getattr(config, "use_barrier_kernels", False))
    nk = knots_xy.shape[0]
    thetas = (np.repeat(float(radius), nk) if np.isscalar(radius)
              else np.asarray(radius, dtype=float))
    covered = np.zeros(sites_xy.shape[0], dtype=bool)
    for j in range(nk):
        knot = knots_xy[j]
        if use_aniso:
            d = anisotropic_distances(sites_xy, knot, aniso_ratio, aniso_angle)
        else:
            d = np.linalg.norm(sites_xy - knot, axis=1)
        contrib = _wendland_raw(d / thetas[j], k=k) > tol
        if use_barrier and water_poly is not None:
            for i in np.flatnonzero(contrib):
                if not _visible(sites_xy[i], knot, water_poly, _barrier_rule(config)):
                    contrib[i] = False
        covered |= contrib
    return ~covered


def visibility_mask(sites_xy: np.ndarray, knots_xy: np.ndarray,
                    water_poly=None, rule: str = "obstacle") -> np.ndarray:
    """(Ns, nk) boolean: True where the straight site->knot segment is visible
    (lies inside ``water_poly``). Visibility is a fixed geometric property of the
    site/knot/polygon -- it does NOT depend on radius or anisotropy (the metric
    warp cannot bend a blocked sightline), so this is computed ONCE and cached,
    keeping shapely out of the per-proposal path. All-True when there are no
    barriers (water_poly is None)."""
    Ns, nk = sites_xy.shape[0], knots_xy.shape[0]
    if water_poly is None:
        return np.ones((Ns, nk), dtype=bool)
    V = np.zeros((Ns, nk), dtype=bool)
    for j in range(nk):
        for i in range(Ns):
            V[i, j] = _visible(sites_xy[i], knots_xy[j], water_poly, rule)
    return V


def covering_radius(sites_xy: np.ndarray, knots_xy: np.ndarray,
                    aniso_ratio: float = 1.0, aniso_angle: float = 0.0,
                    vis_mask: np.ndarray = None) -> float:
    """Smallest radius threshold r_min such that ``r > r_min`` covers every site:

        r_min(ratio, angle) = max_s  min_{k visible to s}  h(s, k; ratio, angle)

    with ``h`` the (possibly anisotropic) site->knot distance. Coverage is exactly
    ``r > r_min`` (Wendland support is the strict ``h < r``), so this is the exact
    boundary the radius/anisotropy proposals must respect -- factoring it out lets
    a sampler reparametrise to the unconstrained slack ``r - r_min`` instead of
    rejecting at the boundary. ``vis_mask`` (Ns, nk bool) gates the inner min to
    visible knots (Wang barriers); None -> every knot visible (Shi / barrier-off).
    A site with no visible knot yields +inf (uncoverable at any radius)."""
    Ns, nk = sites_xy.shape[0], knots_xy.shape[0]
    H = np.empty((Ns, nk))
    for j in range(nk):
        H[:, j] = anisotropic_distances(sites_xy, knots_xy[j], aniso_ratio, aniso_angle)
    if vis_mask is not None:
        H = np.where(vis_mask, H, np.inf)
    return float(H.min(axis=1).max())


def _normalize_rows(W: np.ndarray, fallback_uniform: bool = True) -> np.ndarray:
    """Normalise each row to sum 1. Rows that sum to ~0 (a site that sees no
    knot through any barrier) get a uniform fallback + a warning, per the
    barrier-normalisation concern in the plan.

    NOTE: in the MCMC sampler this fallback should never fire, because the
    radius_S / anisotropy update blocks reject any proposal that leaves a site
    uncovered (see `domain_uncovered_mask` and `ModelState.domain_covered`). The
    fallback remains as a last-resort safety net for one-off forward builds; if
    you see this warning during sampling, a coverage check was bypassed.
    """
    row_sums = W.sum(axis=1)
    zero = row_sums <= 0
    if np.any(zero):
        if fallback_uniform:
            import warnings
            n = int(np.sum(zero))
            warnings.warn(
                f"{n} site(s) have zero total Wendland weight (isolated by "
                f"barriers); assigning uniform weights as a fallback.",
                RuntimeWarning,
            )
            W = W.copy()
            W[zero, :] = 1.0 / W.shape[1]
            row_sums = W.sum(axis=1)
        else:
            raise ValueError("Some sites have zero total Wendland weight.")
    return W / row_sums[:, None]


def wendland_standard(sites_xy: np.ndarray, knots_xy: np.ndarray,
                      radius, k: int = 0) -> np.ndarray:
    """Isotropic Wendland weights, identical to Shi's build_weight_matrix.

    `radius` may be a scalar or a per-knot vector (theta). Uses utilities so
    the `shi` path is bit-for-bit unchanged.
    """
    theta = radius
    if np.isscalar(radius):
        theta = np.repeat(float(radius), knots_xy.shape[0])
    return _u.build_weight_matrix(sites_xy, knots_xy, kernel="wendland",
                                  wendland_theta=theta)


def wendland_anisotropic(sites_xy: np.ndarray, knots_xy: np.ndarray, radius,
                         aniso_ratio: float, aniso_angle: float,
                         k: int = 0) -> np.ndarray:
    """Elliptical-support Wendland weights (geometric anisotropy)."""
    if aniso_ratio == 1.0:
        # identical to isotropic -> delegate for exact reproducibility
        return wendland_standard(sites_xy, knots_xy, radius, k=k)
    Ns = sites_xy.shape[0]
    nk = knots_xy.shape[0]
    thetas = (np.repeat(float(radius), nk) if np.isscalar(radius)
              else np.asarray(radius, dtype=float))
    W = np.empty((Ns, nk))
    for j in range(nk):
        h = anisotropic_distances(sites_xy, knots_xy[j], aniso_ratio, aniso_angle)
        W[:, j] = _wendland_raw(h / thetas[j], k=k)
    return _normalize_rows(W)


_OBSTACLE_CACHE: dict = {}


def _obstacles(water_poly):
    """The barriers inside water_poly's bounding box: envelope minus water (e.g.
    the island). Prepared and cached per polygon -- _visible runs ~10^5 times."""
    import shapely
    key = water_poly.wkb
    obs = _OBSTACLE_CACHE.get(key)
    if obs is None:
        obs = water_poly.envelope.difference(water_poly)
        shapely.prepare(obs)
        _OBSTACLE_CACHE.clear()          # one polygon per process in practice
        _OBSTACLE_CACHE[key] = obs
    return obs


def _barrier_rule(config) -> str:
    """config.barrier_visibility, defaulting to the fixed rule (see _visible)."""
    return getattr(config, "barrier_visibility", "obstacle")


def _visible(p0, p1, water_poly, rule: str = "obstacle") -> bool:
    """True unless the straight segment p0->p1 passes THROUGH a barrier.

    A barrier is any part of water_poly's bounding box that is not water (the
    island). Leaving the box is not a barrier: knots are routinely placed in a
    ring just outside the data domain (Faroe's outer ring sits at -9.2 against a
    water box ending at -9.1), and the old rule -- "the segment lies inside
    water_poly" -- blocked every sight line to them, so the whole ring got zero
    weight at every site. Within the box the two rules agree: a segment there
    leaves the water iff its interior meets a barrier's interior (touching the
    coast is still visible). rule="domain" reproduces the old rule, for runs
    fitted under it (ModelConfig.barrier_visibility).
    """
    from shapely.geometry import LineString
    seg = LineString([tuple(p0), tuple(p1)])
    if water_poly.covers(seg):           # fast path; exactly the old rule's True
        return True
    if rule == "domain":
        return False
    return not seg.relate_pattern(_obstacles(water_poly), "T********")


def wendland_barrier(sites_xy: np.ndarray, knots_xy: np.ndarray, radius,
                     water_poly, aniso_ratio: float = 1.0,
                     aniso_angle: float = 0.0, k: int = 0,
                     rule: str = "obstacle") -> np.ndarray:
    """Visibility-clipped Wendland weights for non-convex domains.

    A knot contributes to a site only if the straight line between them stays
    inside `water_poly` (the visibility criterion, consistent with the VisGP
    graph). Optionally combined with geometric anisotropy.
    """
    Ns = sites_xy.shape[0]
    nk = knots_xy.shape[0]
    thetas = (np.repeat(float(radius), nk) if np.isscalar(radius)
              else np.asarray(radius, dtype=float))
    W = np.zeros((Ns, nk))
    for j in range(nk):
        knot = knots_xy[j]
        if aniso_ratio == 1.0:
            d = np.linalg.norm(sites_xy - knot, axis=1)
        else:
            d = anisotropic_distances(sites_xy, knot, aniso_ratio, aniso_angle)
        raw = _wendland_raw(d / thetas[j], k=k)
        # zero out non-visible (barrier-blocked) site/knot pairs within support
        in_support = np.flatnonzero(raw > 0)
        for i in in_support:
            if _visible(sites_xy[i], knot, water_poly, rule):
                W[i, j] = raw[i]
    return _normalize_rows(W)


def build_wendland_weights(sites_xy: np.ndarray, knots_xy: np.ndarray, radius,
                           config=None, aniso_ratio: float = 1.0,
                           aniso_angle: float = 0.0,
                           water_poly=None, k: int = 0) -> np.ndarray:
    """Dispatch to the appropriate builder based on `config` flags.

    With config=None (or all flags off) this is exactly Shi's isotropic kernel.
    """
    use_aniso = bool(getattr(config, "use_anisotropy", False))
    use_barrier = bool(getattr(config, "use_barrier_kernels", False))

    if use_barrier:
        if water_poly is None:
            raise ValueError("Barrier kernels require a water_poly.")
        return wendland_barrier(sites_xy, knots_xy, radius, water_poly,
                                aniso_ratio=(aniso_ratio if use_aniso else 1.0),
                                aniso_angle=aniso_angle, k=k,
                                rule=_barrier_rule(config))
    if use_aniso:
        return wendland_anisotropic(sites_xy, knots_xy, radius,
                                    aniso_ratio, aniso_angle, k=k)
    return wendland_standard(sites_xy, knots_xy, radius, k=k)


def gamma_bar_from_weights(W: np.ndarray, gamma_k_vec: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """Effective per-site Levy scale gamma_bar(s).

    Matches Shi's sampler:  (sum_k (W[s,k]*gamma_k)**alpha)**(1/alpha).
    """
    return np.sum((W * gamma_k_vec) ** alpha, axis=1) ** (1.0 / alpha)
