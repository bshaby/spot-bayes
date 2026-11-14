#!/usr/bin/env python3
"""
Shared setup for the Faroe Islands extreme-wave analysis (HW / Wang / Shi).

Picks up where the marginal pre-processing (Part I of
faroe_analysis/vignette/faroe_modeling_vignette.md) leaves off:
it consumes that stage's fitted 0.95-quantile threshold and GPD marginal coefficients, plus the
5-day-block-maxima wave field, and assembles everything the spot-bayes samplers
need to fit the three spatial-extremes models.

Imported by every faroe_task*.py so the data, knots, train/test split, model
configs, and (for Wang) the junction-tree cliques are identical across the fit,
comparison, and assessment stages. Deterministic given FAROE_SEED.

Design decisions (see the plan / conversation):
  * Modelling series  : 5-day block maxima (432 ocean sites x 1656 blocks); keeps
                        the provided threshold, p_cens=0.95 and beta inits consistent.
  * Coordinates       : lon/lat in DEGREES (matching the reference package).
  * Barrier           : the 14-vertex "buffered land" polygon from the reference
                        Buffered_Data_Analysis_Stage2.py. This is the MODELLING
                        barrier (Wang visibility graph + barrier Wendland kernels);
                        the coastline shapefile is only for plotting the real island.
  * Knots             : ported verbatim from the reference (Wendland centers
                        N_inner=16/N_outer=25 for S & phi; Gaussian centers
                        N_inner=4/N_outer=9 for rho).
  * Margins           : SAMPLED (update_GPD_sigma/xi) with the spatially-varying
                        covariates (intercept column dropped -- the sampler adds its
                        own), warm-started from beta_log_sigma / beta_logit_star_xi.

Env overrides: FAROE_OUT, FAROE_NITERS(50000), FAROE_THIN(10), FAROE_SEED,
FAROE_Z_SAMPLER(nuts), FAROE_NU_RHO_SAMPLER(nuts), FAROE_RHO_PRIOR_C(0.5),
FAROE_PROGRESS_EVERY(0=auto),
FAROE_NUTS_WARMUP(50), FAROE_NUTS_TARGET_ACCEPT(0.8), FAROE_EFF_RANGE_PHI(1.0),
FAROE_EFF_RANGE_RHO(1.5), FAROE_RADIUS_S(1.0), FAROE_WANG_BARRIER_KERNELS(1).
"""
from __future__ import annotations

import os
import sys
import pickle

# faroe_analysis/scripts/ -> package root is two levels up. Put BOTH on sys.path
# (the scripts dir for the faroe_* sibling modules, the repo root for the package).
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../faroe_analysis/scripts
ANALYSIS = os.path.dirname(_HERE)                           # .../faroe_analysis
REPO = os.path.dirname(ANALYSIS)                            # .../spot-bayes
for _p in (_HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                                     # noqa: E402
from config import ModelConfig                         # noqa: E402
from sampler import SamplerData, ModelState            # noqa: E402
import domain as dom                                   # noqa: E402

# -- paths (self-contained: everything lives under faroe_analysis/) -----------
DATA = os.path.join(ANALYSIS, "data")
NPZ = os.path.join(DATA, "faroe_workflow_outputs.npz")
# Block maxima pre-extracted to plain .npz (numpy) so the pipeline needs no netCDF4
# dependency -- see scripts/README note. The source .nc remains in data/ for provenance.
BM_NPZ = os.path.join(DATA, "faroe_block_maxima.npz")
SIGMA_COV_CSV = os.path.join(DATA, "sigma_covariates_std_2stage_MCMC.csv")
XI_COV_CSV = os.path.join(DATA, "xi_covariates_std_2stage_MCMC.csv")
CLIQUES_FILE = os.path.join(DATA, "buffered_faroe_cliques.json")
SEP_FILE = os.path.join(DATA, "buffered_faroe_separators.json")
COASTLINE = os.path.join(DATA, "ne_10m_coastline/ne_10m_coastline.shp")

OUT = os.environ.get("FAROE_OUT", os.path.join(ANALYSIS, "output"))
INPUTS_PKL = os.path.join(OUT, "inputs.pkl")
INPUTS_NPZ = os.path.join(OUT, "inputs.npz")

# -- domain (degrees) ---------------------------------------------------------
MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -9.0, -5.0, 60.0, 64.0
LON_OUTER = (MIN_LON - 0.2, MAX_LON + 0.2)
LAT_OUTER = (MIN_LAT - 0.2, MAX_LAT + 0.2)

# Buffered-land polygon (barrier) -- reference Buffered_Data_Analysis_Stage2.py:158-164
BUFFER_X = [-7.7, -7.2, -7, -6.91, -6.98, -6.63, -6.61, -6.69, -6.52, -6.27, -6.27,
            -6.53, -7.2, -7.24]
BUFFER_Y = [62.1, 62.01, 62.08, 61.9, 61.565, 61.35, 61.82, 62, 62.15, 62.24, 62.31,
            62.41, 62.33, 62.2]

# -- run parameters (env-overridable) -----------------------------------------
NITERS = int(os.environ.get("FAROE_NITERS", "50000"))
THIN = int(os.environ.get("FAROE_THIN", "10"))
SEED = int(os.environ.get("FAROE_SEED", "20260706"))
P_CENS = 0.95
Z_SAMPLER = os.environ.get("FAROE_Z_SAMPLER", "nuts").lower()
NU_RHO_SAMPLER = os.environ.get("FAROE_NU_RHO_SAMPLER", "nuts").lower()
RHO_PRIOR_C = float(os.environ.get("FAROE_RHO_PRIOR_C", "0.5"))
NUTS_WARMUP = int(os.environ.get("FAROE_NUTS_WARMUP", "50"))
NUTS_TARGET_ACCEPT = float(os.environ.get("FAROE_NUTS_TARGET_ACCEPT", "0.8"))
EFF_RANGE_PHI = float(os.environ.get("FAROE_EFF_RANGE_PHI", "1.0"))
EFF_RANGE_RHO = float(os.environ.get("FAROE_EFF_RANGE_RHO", "1.5"))
RADIUS_S = float(os.environ.get("FAROE_RADIUS_S", "1.0"))
# Progress-line cadence in ITERATIONS. 0 = auto (n_iters/50), which on a 75k-iteration
# fit is 1500 iters -- hours between updates at these speeds, and far too slow a feedback
# loop when you are benchmarking a rank count. Set e.g. FAROE_PROGRESS_EVERY=100.
PROGRESS_EVERY = int(os.environ.get("FAROE_PROGRESS_EVERY", "0"))
WANG_BARRIER_KERNELS = os.environ.get("FAROE_WANG_BARRIER_KERNELS", "1") == "1"

# -- train/test holdout-block boundaries (tunable; user reviews split.png) -----
# Three CONTIGUOUS blocks (~15-20% of the 432 sites). Positions per the brief:
#   1. East of the island, hugging the barrier (all sites east of the buffer in the
#      island latitude band -> no training points between barrier and block).
#   2. Open ocean, north-west.
#   3. South of the island, northern extent abutting the barrier.
BLOCK_E = dict(lon_min=-6.75, lon_max=-5.75, lat_min=61.55, lat_max=62.25)   # block 1
BLOCK_NW = dict(lon_min=-8.45, lon_max=-7.55, lat_min=62.75, lat_max=63.45)  # block 2
BLOCK_S = dict(lat_min=60.25, lat_max=61.25, lon_min=-7.45, lon_max=-6.55)   # block 3


# -- polygons -----------------------------------------------------------------
def buffer_polygon():
    from shapely.geometry import Polygon
    return Polygon(list(zip(BUFFER_X, BUFFER_Y)))


def _inner_box():
    from shapely.geometry import Polygon
    return Polygon([(MIN_LON - 0.1, MIN_LAT - 0.1), (MAX_LON + 0.1, MIN_LAT - 0.1),
                    (MAX_LON + 0.1, MAX_LAT + 0.1), (MIN_LON - 0.1, MAX_LAT + 0.1)])


def water_polygon():
    """Ocean modelling domain = inner boundary box MINUS the buffered land (barrier).
    Used for the Wang visibility graph and barrier Wendland kernels."""
    return _inner_box().difference(buffer_polygon())


# -- knots (ported from reference Buffered_Data_Analysis_Stage2.py:209-336) -----
def _reference_centers(n_inner_grid, n_outer_grid):
    from shapely.geometry import Point
    buf = buffer_polygon()
    box = _inner_box()
    # inner grid: two interleaved meshes, keep those strictly in-domain & off-land
    xd = (MAX_LON - MIN_LON) / (int(2 * np.sqrt(n_inner_grid)) - 1)
    yd = (MAX_LAT - MIN_LAT) / (int(2 * np.sqrt(n_inner_grid)) - 1)
    n = int(2 * np.sqrt(n_inner_grid))
    xc = np.linspace(MIN_LON + xd / 2, MAX_LON + xd / 2, num=n)
    yc = np.linspace(MIN_LAT + yd / 2, MAX_LAT + yd / 2, num=n)
    Xo, Yo = np.meshgrid(xc[0::2], yc[0::2])
    Xi, Yi = np.meshgrid(xc[1::2], yc[1::2])
    centers = np.vstack([np.c_[Xo.ravel(), Yo.ravel()], np.c_[Xi.ravel(), Yi.ravel()]])
    keep = [(MIN_LON < c[0] < MAX_LON and MIN_LAT < c[1] < MAX_LAT) for c in centers]
    centers = centers[np.array(keep)]
    inner = np.array([c for c in centers if not buf.contains(Point(c[0], c[1]))])
    # outer ring: linspace over the outer box, keep those OUTSIDE the inner box
    xo = np.linspace(LON_OUTER[0], LON_OUTER[1], int(np.sqrt(n_outer_grid)))
    yo = np.linspace(LAT_OUTER[0], LAT_OUTER[1], int(np.sqrt(n_outer_grid)))
    Xoo, Yoo = np.meshgrid(xo, yo)
    oc = np.c_[Xoo.ravel(), Yoo.ravel()]
    outer = np.array([c for c in oc if not box.contains(Point(c[0], c[1]))])
    return np.vstack([inner, outer])


def make_knots():
    """(knots_S, knots_phi, knots_rho). S & phi share the Wendland centers; rho uses
    the coarser Gaussian centers (collapsed to a scalar internally for Wang)."""
    wend = _reference_centers(16, 25)
    gauss = _reference_centers(4, 9)
    return wend, wend.copy(), gauss


# -- raw data -----------------------------------------------------------------
def load_raw():
    """Load the block-maxima field, threshold, marginal surfaces, covariates and
    beta inits over the 432 ocean cells (np.where(~land_mask) row-major order --
    the same order the covariate CSVs and the RDS clique node indices use)."""
    d = np.load(NPZ, allow_pickle=True)
    lat = d["latitude"].astype(float)
    lon = d["longitude"].astype(float)
    land_mask = d["land_mask"]
    water = ~land_mask
    LON, LAT = np.meshgrid(lon, lat)
    sites_xy = np.column_stack([LON[water], LAT[water]]).astype(float)

    bm = np.load(BM_NPZ)["block_max"].astype(float)             # (Nt, 21, 21)
    Y = bm[:, water].T.copy()                                    # (Ns, Nt), NaN=missing

    u_vec = d["threshold_q95"][water].astype(float)
    sigma_vec = d["sigma_regression"][water].astype(float)
    xi_vec = d["xi_regression"][water].astype(float)

    sig_cov = np.loadtxt(SIGMA_COV_CSV, delimiter=",")
    xi_cov = np.loadtxt(XI_COV_CSV, delimiter=",")
    assert np.allclose(sig_cov[:, 0], 1.0), "sigma covariate col 0 not all-ones"
    assert np.allclose(xi_cov[:, 0], 1.0), "xi covariate col 0 not all-ones"
    sigma_spatial = np.ascontiguousarray(sig_cov[:, 1:])         # (Ns, 10) drop intercept
    xi_spatial = np.ascontiguousarray(xi_cov[:, 1:])             # (Ns, 7)  drop intercept

    return dict(
        sites_xy=sites_xy, Y=Y, u_vec=u_vec, sigma_vec=sigma_vec, xi_vec=xi_vec,
        sigma_spatial=sigma_spatial, xi_spatial=xi_spatial,
        beta_sigma=d["beta_log_sigma"].astype(float),           # (11,) [b0, 10 spatial]
        beta_xi=d["beta_logit_star_xi"].astype(float),          # (8,)  [b0, 7 spatial]
        lat=lat, lon=lon, land_mask=land_mask, water=water)


# -- train/test split ---------------------------------------------------------
def split_sites_faroe(sites_xy):
    """Return an int block-id per site: 0=train, 1=East, 2=NW, 3=South."""
    lon, lat = sites_xy[:, 0], sites_xy[:, 1]
    block = np.zeros(sites_xy.shape[0], dtype=int)
    b1 = ((lon >= BLOCK_E["lon_min"]) & (lon <= BLOCK_E["lon_max"])
          & (lat >= BLOCK_E["lat_min"]) & (lat <= BLOCK_E["lat_max"]))
    b2 = ((lon >= BLOCK_NW["lon_min"]) & (lon <= BLOCK_NW["lon_max"])
          & (lat >= BLOCK_NW["lat_min"]) & (lat <= BLOCK_NW["lat_max"]))
    b3 = ((lat >= BLOCK_S["lat_min"]) & (lat <= BLOCK_S["lat_max"])
          & (lon >= BLOCK_S["lon_min"]) & (lon <= BLOCK_S["lon_max"]))
    block[b1] = 1
    block[b2] = 2
    block[b3] = 3
    return block


# -- model configs ------------------------------------------------------------
def _z_kw():
    if Z_SAMPLER == "nuts":
        return dict(z_sampler="nuts", nuts_warmup=NUTS_WARMUP,
                    nuts_target_accept=NUTS_TARGET_ACCEPT)
    if Z_SAMPLER == "rwmh":
        return dict(z_sampler="rwmh", z_update="ncp", z_block=5)
    raise SystemExit(f"FAROE_Z_SAMPLER={Z_SAMPLER!r} not in (rwmh, nuts)")


_MARGIN_KW = dict(update_GPD_sigma=True, update_GPD_xi=True, beta_sampler="component")


def models():
    """The three (ModelConfig, settings_kw) pairs. Built lazily so env overrides and
    the barrier-kernel toggle apply. All margins sampled; Shi/Wang anisotropic."""
    wang_cfg = ModelConfig(
        model_type="wang", phi_domain="positive", use_anisotropy=True,
        use_visgp=True, use_barrier_kernels=WANG_BARRIER_KERNELS,
        cliques_path=CLIQUES_FILE,
        shapefile_path=(COASTLINE if WANG_BARRIER_KERNELS else None), n_cores=1)
    return {
        "hw": (ModelConfig(model_type="huser-wadsworth", phi_domain="positive"),
               dict(update_S=True, update_Z=True, update_phi=True, update_rho=True,
                    update_alpha0=True, update_nu=True, **_z_kw(), **_MARGIN_KW)),
        "wang": (wang_cfg,
                 dict(update_S=True, update_Z=True, update_phi=True, update_rho=True,
                      update_alpha0=True, update_nu=True, update_radius_S=True,
                      update_anisotropy=True, **_z_kw(), **_MARGIN_KW)),
        "shi": (ModelConfig(model_type="shi", phi_domain="positive", use_anisotropy=True),
                dict(update_S=True, update_Z=True, update_phi=True, update_rho=True,
                     update_alpha0=True, update_nu=True, update_radius_S=True,
                     update_anisotropy=True, nu_rho_sampler=NU_RHO_SAMPLER,
                     **_z_kw(), **_MARGIN_KW)),
    }


def base_settings_kw():
    """Settings common to every fit (marginal priors, knot smoothing scales, rho
    prior, seed). Merged with the per-model kw from models()."""
    return dict(
        seed=SEED, thin=THIN, adapt_size=100, rho_prior_c=RHO_PRIOR_C,
        eff_range_phi=EFF_RANGE_PHI, eff_range_rho=EFF_RANGE_RHO, radius_S=RADIUS_S,
        progress_every=PROGRESS_EVERY, save_latent=True, stream_latent=True)


# -- SamplerData assembly -----------------------------------------------------
def build_data(cfg, inp, which):
    """SamplerData for one model on a site subset.

    which : "full" (all 432 sites; Wang uses the provided RDS junction tree),
            "train" (Wang recomputes the junction tree on the training subgraph).
    inp   : the inputs dict (from prepare_inputs / load_inputs).
    """
    idx = inp["full_idx"] if which == "full" else inp["train_idx"]
    kS, kphi, krho = inp["knots_S"], inp["knots_phi"], inp["knots_rho"]
    d = SamplerData(
        Y=inp["Y"][idx], sites_xy=inp["sites_xy"][idx],
        u_vec=inp["u_vec"][idx], sigma_vec=inp["sigma_vec"][idx],
        xi_vec=inp["xi_vec"][idx], p_cens=P_CENS,
        knots_S=kS, knots_phi=kphi, knots_rho=krho,
        sigma_spatial=inp["sigma_spatial"][idx], xi_spatial=inp["xi_spatial"][idx])
    if cfg.use_visgp:                       # Wang
        d.water_poly = water_polygon()
        if which == "full":
            d.cliques = inp["full_cliques"]
            d.separators = inp["full_separators"]
        else:
            d.cliques = inp["train_cliques"]
            d.separators = inp["train_separators"]
    return d


def init_snapshot_betas(cfg, data, settings, inp, idx, beta_sigma=None, beta_xi=None):
    """Full ModelState snapshot with the GPD betas overridden by fitted coefficients
    (warm-starting the margins). By default uses the reference workflow's 2-stage betas
    (inp["beta_sigma"]/["beta_xi"]); pass beta_sigma/beta_xi to override -- e.g. with the
    STAGE-1 marginal_fit posterior mean for a stage-2 fixed-margins run. Cold-init is
    deterministic (S=1, Z=0), so every MPI rank produces the identical snapshot."""
    st = ModelState(cfg, data, settings)
    snap = st.snapshot()
    bs = inp["beta_sigma"] if beta_sigma is None else beta_sigma
    bx = inp["beta_xi"] if beta_xi is None else beta_xi
    snap["beta_sigma"] = np.asarray(bs, float).copy()
    snap["beta_xi"] = np.asarray(bx, float).copy()
    return snap


# -- inputs (de)serialization -------------------------------------------------
def prepare_inputs(n_cores=1, verbose=True):
    """Build the full inputs dict from scratch: raw arrays, knots, split, and the
    Wang junction-tree cliques for BOTH the full graph (from the provided RDS) and
    the training subgraph (recomputed on the subset visibility graph)."""
    raw = load_raw()
    Ns = raw["Y"].shape[0]
    kS, kphi, krho = make_knots()
    block_id = split_sites_faroe(raw["sites_xy"])
    full_idx = np.arange(Ns)
    train_idx = np.flatnonzero(block_id == 0)
    test_idx = np.flatnonzero(block_id > 0)

    if verbose:
        print(f"[prep] {Ns} sites; knots S/phi={kS.shape[0]} rho={krho.shape[0]}; "
              f"train={train_idx.size} test={test_idx.size} "
              f"(E={int((block_id==1).sum())} NW={int((block_id==2).sum())} "
              f"S={int((block_id==3).sum())})", flush=True)

    # full-graph cliques: the provided buffered RDS (validated separable below)
    full_cliques = dom.load_cliques(CLIQUES_FILE)
    full_separators = dom.load_cliques(SEP_FILE)
    full_ok = dom.verify_running_intersection(full_cliques, full_separators)
    if not full_ok:
        raise ValueError("Provided RDS cliques/separators are NOT a valid junction "
                         "tree (running-intersection failed).")
    node_max = max(max(c) for c in full_cliques)
    if node_max != Ns - 1:
        raise ValueError(f"RDS clique node range 0..{node_max} != 0..{Ns-1} "
                         "(clique file does not match the 432-site graph).")

    # training-subgraph cliques: recompute the visibility graph + junction tree
    if verbose:
        print("[prep] recomputing training-subgraph visibility junction tree ...",
              flush=True)
    wpoly = water_polygon()
    A = dom.build_visibility_graph(raw["sites_xy"][train_idx], wpoly, n_cores=n_cores)
    tcl = dom.find_maximal_cliques(A, triangulate=True)
    train_cliques, train_separators = dom.build_junction_tree(tcl)

    inp = dict(**raw, knots_S=kS, knots_phi=kphi, knots_rho=krho,
               block_id=block_id, full_idx=full_idx, train_idx=train_idx,
               test_idx=test_idx, full_cliques=full_cliques,
               full_separators=full_separators, train_cliques=train_cliques,
               train_separators=train_separators)
    return inp


def save_inputs(inp, out=OUT):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "inputs.pkl"), "wb") as fh:
        pickle.dump(inp, fh)
    # readable rectangular-array copy (cliques omitted -- ragged)
    rect = {k: v for k, v in inp.items()
            if isinstance(v, np.ndarray)}
    np.savez(os.path.join(out, "inputs.npz"), **rect)


def load_inputs(out=OUT):
    with open(os.path.join(out, "inputs.pkl"), "rb") as fh:
        return pickle.load(fh)


# -- plotting helpers ---------------------------------------------------------
def add_coastline(ax, color="0.35", lw=0.8):
    """Overlay the real island coastline (from the Natural Earth shapefile).

    The shapefile is not tracked in git (large, redistributable); if it is absent
    we skip the overlay with a warning rather than failing the plot -- the barrier
    polygon (hardcoded) still shows."""
    if not os.path.exists(COASTLINE):
        import warnings
        warnings.warn(f"coastline shapefile not found at {COASTLINE}; skipping "
                      "island outline (see faroe_analysis/README.md).")
        return
    import geopandas as gpd
    gdf = gpd.read_file(COASTLINE)
    gdf = gdf.cx[MIN_LON - 0.5:MAX_LON + 0.5, MIN_LAT - 0.5:MAX_LAT + 0.5]
    gdf.plot(ax=ax, color=color, linewidth=lw, zorder=5)


def add_barrier(ax, color="black", lw=2.2):
    """Overlay the buffered-land modelling barrier (bold black outline)."""
    xs = BUFFER_X + [BUFFER_X[0]]
    ys = BUFFER_Y + [BUFFER_Y[0]]
    ax.plot(xs, ys, color=color, lw=lw, ls="-", zorder=6)


def set_domain(ax):
    ax.set_xlim(MIN_LON - 0.3, MAX_LON + 0.3)
    ax.set_ylim(MIN_LAT - 0.3, MAX_LAT + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def cell_edges(centers):
    """Cell-edge coordinates from a vector of regular cell centers."""
    c = np.asarray(centers, float)
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (c[1] - c[0]) / 2.0
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2.0
    return e


def grid_from_sites(inp, values, fill=np.nan):
    """Scatter per-site values back onto the full (lat, lon) grid (land = fill)."""
    g = np.full(inp["land_mask"].shape, fill, dtype=float)
    g[~inp["land_mask"]] = values
    return g


def plot_cells(ax, inp, values, cmap="viridis", vmin=None, vmax=None, bad="0.45"):
    """Filled cell map of a per-site quantity over the domain (land greyed)."""
    import matplotlib
    grid = np.ma.masked_invalid(grid_from_sites(inp, values))
    cm = matplotlib.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    cm.set_bad(bad)
    return ax.pcolormesh(cell_edges(inp["lon"]), cell_edges(inp["lat"]), grid,
                         cmap=cm, vmin=vmin, vmax=vmax, zorder=1)
