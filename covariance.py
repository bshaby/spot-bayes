"""
Covariance models for the latent Gaussian field Z(s).

Two paths:
  * Shi (default): nonstationary Matern via utilities.ns_cov / ns_cov_cached_*.
    Re-exported here so callers have a single covariance module.
  * Wang (VisGP): stationary Matern combined with the visibility graph through
    Dempster covariance-selection on a junction tree (Gilbert & Datta 2024):
        Q = sum_{cliques}  inv(Matern_clique)  -  sum_{separators} inv(Matern_sep)
    Q is the precision of Z on the non-convex domain. Prediction uses the
    sparse-precision conditional formulas.

The Matern parameterisation for VisGP follows Wang's functions.matern:
    C(d) = sigmasq / Gamma(nu) * 2^{1-nu} * (d sqrt(8 nu)/rho)^nu * K_nu(d sqrt(8 nu)/rho)
with C(0) = sigmasq. Here rho is an effective range and nu the smoothness.
"""
from __future__ import annotations

from typing import Sequence, Optional, Tuple

import numpy as np
import scipy.spatial
from numpy.linalg import inv
from scipy.special import gamma as _gamma, kv as _kv

# Re-export Shi's covariance utilities for a single import surface.
from utilities import (ns_cov, ns_cov_precompute, ns_cov_cached,  # noqa: F401
                       ns_cov_cached_exp_nb, ns_cov_cached_matern32_nb,
                       ns_cov_cached_matern52_nb)


# --- canonical range parameterisation ---------------------------------------
# All three models expose rho on a single canonical scale: Wang's "effective
# range" convention, where the Matern correlation uses argument d*sqrt(8 nu)/rho
# (the standard practical-range parameterisation, also used by INLA). Wang and
# Huser-Wadsworth use matern_wang, which already takes canonical rho directly.
# Shi's nonstationary ns_cov uses a different internal range whose correlation is
# M(d / sqrt(range_internal)); the two agree iff
#     range_internal = rho_canonical**2 / (8 nu).
# Verified to machine precision in tests/test_rho_param.py.

def canonical_to_ns_range(rho_canonical, nu):
    """Convert canonical (effective-range) rho to Shi ns_cov internal range."""
    return np.asarray(rho_canonical, dtype=float) ** 2 / (8.0 * nu)


def ns_range_to_canonical(range_internal, nu):
    """Inverse of canonical_to_ns_range."""
    return np.sqrt(8.0 * nu * np.asarray(range_internal, dtype=float))


# --- Wang's stationary Matern -----------------------------------------------
def matern_wang(DistMtx: np.ndarray, rho: float, sigmasq: float,
                nu: float) -> np.ndarray:
    """Stationary Matern covariance matrix from a pairwise-distance matrix.

    Mirrors Wang's functions.matern (rho is an effective range with the
    sqrt(8 nu) scaling). Diagonal is set to sigmasq.
    """
    DistMtx = np.asarray(DistMtx, dtype=float)
    scaled = (DistMtx * np.sqrt(8.0 * nu)) / rho
    with np.errstate(divide="ignore", invalid="ignore"):
        term3 = scaled ** nu
        term4 = _kv(nu, scaled)
        cov = (sigmasq / _gamma(nu)) * (2.0 ** (1.0 - nu)) * term3 * term4
    np.fill_diagonal(cov, sigmasq)
    return cov


def dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean pairwise-distance matrix."""
    return scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(coords))


def matern_wang_cross(d, rho: float, sigmasq: float, nu: float) -> np.ndarray:
    """Stationary Matern covariance for a vector of distances (Wang scaling).

    Like matern_wang but for a 1-D distance vector (cross-covariances); returns
    sigmasq where d == 0.
    """
    d = np.asarray(d, dtype=float)
    out = np.full(d.shape, sigmasq, dtype=float)
    nz = d > 0
    scaled = (d[nz] * np.sqrt(8.0 * nu)) / rho
    out[nz] = (sigmasq / _gamma(nu)) * (2.0 ** (1.0 - nu)) * scaled ** nu * _kv(nu, scaled)
    return out


# --- VisGP precision via junction-tree inclusion-exclusion -------------------
def _clique_prec(clique: Sequence[int], d: int, DistMtx: np.ndarray,
                 rho: float, sigmasq: float, nu: float) -> Optional[np.ndarray]:
    """Precision of one clique embedded in a d x d zero matrix (or None)."""
    clique = list(clique)
    if len(clique) == 0:
        return None
    sub_cov = matern_wang(DistMtx[np.ix_(clique, clique)], rho, sigmasq, nu)
    return inv(sub_cov)


def _clique_prec_logdet(clique, DistMtx, rho, sigmasq, nu):
    """Clique precision sub-block and its log-determinant (None if empty)."""
    clique = list(clique)
    if len(clique) == 0:
        return None, 0.0
    sub_cov = matern_wang(DistMtx[np.ix_(clique, clique)], rho, sigmasq, nu)
    P = inv(sub_cov)
    _, ld = np.linalg.slogdet(P)
    return P, float(ld)


def visgp_precision(cliques: Sequence[Sequence[int]],
                    separators: Sequence[Sequence[int]],
                    coords: np.ndarray, rho: float, nu: float,
                    sigmasq: float = 1.0, DistMtx: Optional[np.ndarray] = None,
                    n_cores: int = 1, sparse: bool = False,
                    return_logdet: bool = False):
    """Assemble the VisGP precision matrix Q (Gilbert & Datta 2024).

    Q = sum_c inv(Matern[c]) - sum_s inv(Matern[s]) over the junction tree's
    cliques and separators. With a single clique == all nodes and no
    separators, Q == inv(full Matern) exactly (see tests).

    Options
    -------
    sparse : return Q as a scipy.sparse CSR matrix (assembled via COO, so memory
        is O(sum |clique|^2) rather than O(d^2)). The quadratic form Z^T Q Z is
        then a sparse matvec -- orders of magnitude faster for large localized
        graphs.
    return_logdet : also return logdet(Q), computed for free from the clique
        factorizations via the decomposable identity
            logdet(Q) = sum_c logdet(Q_c) - sum_s logdet(Q_s).
        This avoids a global O(d^3) slogdet in visgp_logprob.
    """
    from scipy.linalg import cho_factor, cho_solve
    coords = np.asarray(coords, dtype=float)
    d = coords.shape[0]
    if DistMtx is None:
        DistMtx = dist_matrix(coords)

    # Build the full Euclidean Matern ONCE and slice clique/separator submatrices
    # from it. The cliques overlap heavily, so the old per-block matern_wang calls
    # recomputed the same Bessel entries many times (the dominant cost); one pass
    # over all pairs computes each entry exactly once. Each block then needs a
    # single Cholesky, giving both the inverse (cho_solve) and the log-determinant
    # (sum log diag) -- replacing the old inv + separate slogdet.
    K = matern_wang(DistMtx, rho, sigmasq, nu)

    def _block(g):
        """(idx, P=inv(K_gg), logdet(P)=-logdet(K_gg)) for one clique/separator."""
        gi = np.asarray(g)
        if gi.size == 0:
            return gi, None, 0.0
        c, low = cho_factor(K[np.ix_(gi, gi)], lower=True, check_finite=False)
        ld = -2.0 * float(np.sum(np.log(np.diag(c))))          # logdet(inv(K_gg))
        P = cho_solve((c, low), np.eye(gi.size), check_finite=False)
        return gi, P, ld

    def _all(groups):
        if n_cores and n_cores > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_cores) as ex:
                return list(ex.map(_block, groups))
        return [_block(g) for g in groups]

    clique_blocks = _all(cliques)
    sep_blocks = _all(separators)

    logdet = (sum(ld for _, P, ld in clique_blocks if P is not None)
              - sum(ld for _, P, ld in sep_blocks if P is not None))

    # Assemble dense (fast: in-place block adds), then convert to CSR if requested.
    # The old per-block COO meshgrid was far slower than the dense scatter-add at the
    # Ns this model targets; a single dense->CSR conversion keeps the sparse API.
    Q = np.zeros((d, d))
    for blocks, sign in ((clique_blocks, 1.0), (sep_blocks, -1.0)):
        for gi, P, _ in blocks:
            if P is not None:
                Q[np.ix_(gi, gi)] += sign * P
    if sparse:
        import scipy.sparse as sp
        Q = sp.csr_matrix(Q)

    return (Q, logdet) if return_logdet else Q


def visgp_logprob(Z: np.ndarray, Q, logdet: Optional[float] = None) -> float:
    """log N(Z; 0, Q^{-1}) for a precision matrix Q (dense or sparse).

    = 0.5*logdet(Q) - 0.5 Z^T Q Z - (d/2) log(2 pi).
    Vectorised over columns if Z is (d, T): returns the summed log-density.

    Pass a precomputed `logdet` (e.g. from visgp_precision(return_logdet=True))
    to skip the global determinant; with a sparse Q the quadratic form is a
    sparse matvec.
    """
    import scipy.sparse as sp
    Z = np.asarray(Z, dtype=float)
    d = Q.shape[0]
    if logdet is None:
        if sp.issparse(Q):
            # sparse LU for the determinant when not precomputed
            lu = sp.linalg.splu(sp.csc_matrix(Q))
            diagU = lu.U.diagonal()
            if np.any(diagU <= 0) and np.any(lu.L.diagonal() <= 0):
                pass
            logdet = float(np.sum(np.log(np.abs(diagU))))
        else:
            sign, logdet = np.linalg.slogdet(Q)
            if sign <= 0:
                return -np.inf
    QZ = Q @ Z                      # dense or sparse matvec
    if Z.ndim == 1:
        quad = float(Z @ QZ); T = 1
    else:
        quad = float(np.einsum("it,it->", Z, QZ)); T = Z.shape[1]
    return float(0.5 * T * logdet - 0.5 * quad - 0.5 * T * d * np.log(2 * np.pi))


def visgp_conditional(Q, obs_idx: Sequence[int], pred_idx: Sequence[int],
                      Z_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Conditional N(mean, cov) of Z[pred] given Z[obs], from precision Q.

    For a joint Gaussian with precision Q partitioned into observed (o) and
    prediction (p) blocks:
        Z_p | Z_o ~ N( -Q_pp^{-1} Q_po Z_o,  Q_pp^{-1} ).
    Z_obs may be (n_obs,) or (n_obs, T). Handles dense or sparse Q (the small
    prediction block is densified for the solve).
    """
    import scipy.sparse as sp
    obs_idx = list(obs_idx)
    pred_idx = list(pred_idx)
    if sp.issparse(Q):
        Q_pp = np.asarray(Q[np.ix_(pred_idx, pred_idx)].todense())
        Q_po = Q[np.ix_(pred_idx, obs_idx)]            # keep sparse for matvec
    else:
        Q_pp = Q[np.ix_(pred_idx, pred_idx)]
        Q_po = Q[np.ix_(pred_idx, obs_idx)]
    cov = inv(Q_pp)
    mean = -cov @ (Q_po @ np.asarray(Z_obs))
    return mean, np.asarray(cov)


def _inverse_block(Q, idx):
    """Return (Q^{-1})[idx, idx] without forming the full inverse.

    Solves Q M = E (E selects the `idx` columns), then takes rows `idx`.
    Works for dense or sparse Q.
    """
    import scipy.sparse as sp
    idx = np.asarray(idx, dtype=int)
    d = Q.shape[0]
    E = np.zeros((d, idx.size))
    E[idx, np.arange(idx.size)] = 1.0
    if sp.issparse(Q):
        M = sp.linalg.spsolve(sp.csc_matrix(Q), E)
        M = np.asarray(M)
        if M.ndim == 1:
            M = M[:, None]
    else:
        M = np.linalg.solve(Q, E)
    return M[idx, :]


def visgp_predict_new(new_coords, train_coords, Q, rho: float, nu: float,
                      Z_train: np.ndarray, visible, sigmasq: float = 1.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Predict Z at NEW sites not present in the original VisGP graph/precision.

    For a fitted VisGP (precision Q on `train_coords`, with conditional means
    Z_train), predicts the latent field at `new_coords` -- sites the MCMC never
    saw -- using the VisGP Markov property: a new site s*'s Markov blanket is the
    set V* of training sites it can *see* (straight line in the domain), so

        Z(s*) | Z_train  =  Z(s*) | Z_{V*}
        mean(s*) = C_{*,V*} L_{V*,V*}^{-1} Z_{V*}
        var(s*)  = sigmasq - C_{*,V*} L_{V*,V*}^{-1} C_{V*,*}

    where C is the base Matern covariance (exact for visible pairs, Gilbert &
    Datta) and L = Q^{-1} is the fitted VisGP covariance among the neighbours
    (so barriers *among* the neighbours are respected). A new site that sees no
    training site (fully blocked by barriers) falls back to the marginal
    N(0, sigmasq) -- the zero-visibility case.

    Parameters
    ----------
    new_coords  : (n_new, 2)
    train_coords: (d, 2) -- the original MCMC sites
    Q           : (d, d) VisGP precision (dense or sparse)
    Z_train     : (d,) or (d, T) fitted latent values at the training sites
    visible     : (n_new, d) boolean -- visible[i, j] True iff new site i can see
                  training site j (see domain.visible_from_new)

    Returns
    -------
    mean : (n_new,) or (n_new, T) conditional mean(s)
    var  : (n_new,) conditional variance at each new site (marginal per site;
           dependence *among* new sites is not modelled -- adequate for per-site
           holdout scoring).
    """
    new_coords = np.atleast_2d(np.asarray(new_coords, dtype=float))
    train_coords = np.asarray(train_coords, dtype=float)
    Z_train = np.asarray(Z_train, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    n_new = new_coords.shape[0]
    T = Z_train.shape[1] if Z_train.ndim == 2 else None

    vis_lists = [np.flatnonzero(visible[i]) for i in range(n_new)]
    nonempty = [v for v in vis_lists if v.size]
    union = (np.unique(np.concatenate(nonempty)) if nonempty
             else np.array([], dtype=int))

    L_union = _inverse_block(Q, union) if union.size else None
    pos = {int(j): k for k, j in enumerate(union)}

    mean = np.zeros((n_new,) + ((T,) if T else ()))
    var = np.full(n_new, float(sigmasq))
    for i in range(n_new):
        V = vis_lists[i]
        if V.size == 0:
            continue  # isolated new site -> marginal N(0, sigmasq)
        loc = [pos[int(j)] for j in V]
        L_VV = L_union[np.ix_(loc, loc)]
        d_sv = np.linalg.norm(train_coords[V] - new_coords[i], axis=1)
        C_sV = matern_wang_cross(d_sv, rho, sigmasq, nu)          # (|V|,)
        sol = np.linalg.solve(L_VV, C_sV)                         # L^{-1} C
        ZV = Z_train[V] if T is None else Z_train[V, :]
        mean[i] = C_sV @ (np.linalg.solve(L_VV, ZV))
        var[i] = float(sigmasq - C_sV @ sol)
    return mean, var


def visgp_predict_new_joint(new_coords, train_coords, Q, rho: float, nu: float,
                            Z_train: np.ndarray, visible_train, visible_new,
                            sigmasq: float = 1.0, order=None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """JOINT conditional N(mean, cov) of Z at NEW sites given the training field.

    Unlike visgp_predict_new (which returns per-site marginals), this models the
    dependence *among* the new sites. It uses a Vecchia/NNGP-style sequential
    construction: ordering the new sites, each is regressed on its *visible*
    neighbours among the training sites and the already-ordered new sites,

        Z_new_i = sum_{j in N_i} a_{ij} Z_j + eps_i,   eps_i ~ N(0, f_i),

    which defines a valid joint Gaussian. Writing Z_new = A_n Z_new + mu0 + eps
    (A_n strictly lower-triangular in the order, mu0 the training contribution),

        mean = (I - A_n)^{-1} mu0,    cov = (I - A_n)^{-1} diag(f) (I - A_n)^{-T}.

    Properties:
    * Training marginal preserved exactly (training is the reference set; its
      neighbour blocks use the fitted VisGP covariance L = Q^{-1}).
    * New-site cross terms use the base Matern (exact for visible pairs).
    * Barriers respected: non-visible sites are not neighbours -> conditional
      independence. With full visibility and full neighbour sets the construction
      is the exact joint Gaussian conditional (equals ordinary kriging).

    Parameters
    ----------
    visible_train : (m, d) bool, new-to-training visibility.
    visible_new   : (m, m) bool, new-to-new visibility (symmetric, diag ignored).
    order         : optional ordering of the new sites (default 0..m-1).

    Returns (mean (m,) or (m, T), cov (m, m)).
    """
    new_coords = np.atleast_2d(np.asarray(new_coords, dtype=float))
    train_coords = np.asarray(train_coords, dtype=float)
    Z_train = np.asarray(Z_train, dtype=float)
    vt = np.asarray(visible_train, dtype=bool)
    vn = np.asarray(visible_new, dtype=bool)
    m = new_coords.shape[0]
    T = Z_train.shape[1] if Z_train.ndim == 2 else None
    if order is None:
        order = list(range(m))

    tn = [np.flatnonzero(vt[i]) for i in range(m)]
    nonempty = [v for v in tn if v.size]
    union = (np.unique(np.concatenate(nonempty)) if nonempty
             else np.array([], dtype=int))
    L_union = _inverse_block(Q, union) if union.size else None
    tpos = {int(j): k for k, j in enumerate(union)}

    A_n = np.zeros((m, m))
    F = np.full(m, float(sigmasq))
    mu0 = np.zeros((m,) + ((T,) if T else ()))
    processed = []
    for i in order:
        tnb = tn[i]                                   # visible training neighbours
        nnb = [j for j in processed if vn[i, j]]      # visible earlier-new neighbours
        kt, kn = tnb.size, len(nnb)
        if kt + kn == 0:
            continue                                  # isolated -> N(0, sigmasq)
        coords_blocks = []
        if kt:
            coords_blocks.append(train_coords[tnb])
        if kn:
            coords_blocks.append(new_coords[nnb])
        nb_coords = np.vstack(coords_blocks)
        # neighbour covariance: base Matern, with the train-train block replaced
        # by the fitted VisGP covariance L (barriers among training respected).
        S = matern_wang(dist_matrix(nb_coords), rho, sigmasq, nu)
        if kt:
            loc = [tpos[int(j)] for j in tnb]
            S[:kt, :kt] = L_union[np.ix_(loc, loc)]
        d_i = np.linalg.norm(nb_coords - new_coords[i], axis=1)
        c = matern_wang_cross(d_i, rho, sigmasq, nu)
        sol = np.linalg.solve(S, c)
        F[i] = float(sigmasq - c @ sol)
        if kt:
            ZT = Z_train[tnb] if T is None else Z_train[tnb, :]
            mu0[i] = sol[:kt] @ ZT
        for c_idx, j in enumerate(nnb):
            A_n[i, j] = sol[kt + c_idx]
        processed.append(i)

    M = np.eye(m) - A_n
    mean = np.linalg.solve(M, mu0)
    Minv = np.linalg.inv(M)
    F = np.maximum(F, 0.0)
    cov = (Minv * F) @ Minv.T
    return mean, 0.5 * (cov + cov.T)
