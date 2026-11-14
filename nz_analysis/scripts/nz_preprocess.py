#!/usr/bin/env python3
"""NZ marginal pre-processing: 3-day blocks, EDA, 0.95 quantile-regression threshold, GPD regression.

Part I of ``nz_analysis/vignette/nz_modeling_vignette.md``.  Starts at the upscaled
ERA5 daily precipitation for the January-April season and:

* runs the storm-duration EDA used to choose a storm-scale block length,
* forms non-overlapping 3-day precipitation sums at the 309 model sites,
* fits a pooled tau=0.95 quantile regression used as the high threshold,
* fits a generalized Pareto regression to the positive exceedances (maximum
  likelihood, with location-wise GPD MLEs for comparison),
* draws the coastline/topography/barrier maps.

Inputs (all in ``nz_analysis/data/``)::

    upscaled_era5_daily_precp_ARs.csv.gz   daily precipitation, 323 upscaled cells
    upscaled_NZ_SI_GEV_mle.csv             the 309 model sites (lon, lat, ...)
    mu_covariates_std_NZ_precip.csv        covariates (sites x 6), used for all three parts
    barrier_1500m/NZ_SI_barrier_1500m.shp  the 1500 m barrier (maps)
    ne_10m_coastline/ne_10m_coastline.shp  optional; map overlay only
    NZ_SI_SRTM.tif                         optional; SRTM topography for the domain map

Outputs go to ``nz_analysis/output/preprocess/`` (override with NZ_PREPROCESS_OUT):
``nz_workflow_outputs.npz`` (the same arrays as the shipped
``data/nz_workflow_outputs.npz``), the 3-day block CSV, the EDA tables,
``workflow_summary.csv``, ``model_coefficients.csv`` and the Part I figures in
``graphics/``.  The shipped data file is never overwritten.

Run from anywhere (serial, a few minutes)::

    python nz_analysis/scripts/nz_preprocess.py

Needs ``statsmodels`` (quantile regression) and ``pyshp`` (shapefiles); Pillow
(installed with matplotlib) reads the optional SRTM GeoTIFF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import pandas as pd
import shapefile
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import genpareto

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 19,
    }
)

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
DATA = ANALYSIS / "data"
OUTPUT = Path(os.environ.get("NZ_PREPROCESS_OUT", ANALYSIS / "output" / "preprocess"))
GRAPHICS = OUTPUT / "graphics"

DAILY_FILE = DATA / "upscaled_era5_daily_precp_ARs.csv.gz"
COVARIATES_FILE = DATA / "mu_covariates_std_NZ_precip.csv"
MODEL_SITES_FILE = DATA / "upscaled_NZ_SI_GEV_mle.csv"
SRTM_FILE = DATA / "NZ_SI_SRTM.tif"
BARRIER_HEIGHT_M = 1500
BARRIER_NAME = f"{BARRIER_HEIGHT_M}m"
BARRIER_SHP = DATA / f"barrier_{BARRIER_NAME}" / f"NZ_SI_barrier_{BARRIER_NAME}.shp"
COASTLINE = DATA / "ne_10m_coastline" / "ne_10m_coastline.shp"

TAU = 0.95
BLOCK_DAYS = 3
PLOT_EXTENT = (165.0, 174.8, -47.8, -40.0)
FIG_DPI = 260

COVARIATE_NAMES = [
    "Intercept",
    "Windward sites",
    "Distance to centroid",
    "cos(angle)",
    "sin(angle)",
    "Windward site x distance",
]


@dataclass
class BlockData:
    blocks: np.ndarray
    block_start: list[pd.Timestamp]
    block_end: list[pd.Timestamp]
    block_year: list[int]
    sites: pd.DataFrame
    site_columns: list[str]


def setup_dirs() -> None:
    for path in (OUTPUT, GRAPHICS):
        path.mkdir(parents=True, exist_ok=True)


def site_column(lon: float, lat: float, columns: list[str]) -> str:
    suffix = f"_lon{lon:.3f}_lat{lat:.3f}"
    matches = [column for column in columns if column.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column for lon={lon}, lat={lat}; found {matches}")
    return matches[0]


def load_daily_model_subset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    daily = pd.read_csv(DAILY_FILE)
    daily["Date"] = pd.to_datetime(daily["Date"])
    sites = pd.read_csv(MODEL_SITES_FILE)[["lon", "lat"]].copy()
    columns = [site_column(row.lon, row.lat, list(daily.columns)) for row in sites.itertuples(index=False)]
    return daily, sites, columns


def make_nonoverlap_blocks(daily: pd.DataFrame, site_columns: list[str], block_days: int = BLOCK_DAYS) -> BlockData:
    values = daily[site_columns].to_numpy(float)
    dates = list(daily["Date"])
    years = list(daily["Year"].astype(int))
    blocks = []
    starts = []
    ends = []
    block_years = []
    for year, idx in daily.groupby("Year").indices.items():
        idx = np.asarray(sorted(idx))
        k = 0
        while k + block_days <= len(idx):
            chunk = idx[k : k + block_days]
            if (dates[chunk[-1]] - dates[chunk[0]]).days == block_days - 1:
                blocks.append(values[chunk, :].sum(axis=0))
                starts.append(dates[chunk[0]])
                ends.append(dates[chunk[-1]])
                block_years.append(int(year))
            k += block_days
    sites = pd.read_csv(MODEL_SITES_FILE)[["lon", "lat"]].copy()
    return BlockData(np.vstack(blocks), starts, ends, block_years, sites, site_columns)


def storm_duration_summary(daily: pd.DataFrame, site_columns: list[str]) -> pd.DataFrame:
    total = daily[site_columns].sum(axis=1).to_numpy(float)
    dates = list(daily["Date"])
    rows = []
    for q in (0.90, 0.95, 0.98, 0.99):
        threshold = float(np.quantile(total, q))
        mask = total > threshold
        durations = []
        cur = 0
        for i, flag in enumerate(mask):
            contiguous = i == 0 or (dates[i] - dates[i - 1]).days == 1
            if flag and contiguous:
                cur += 1
            else:
                if cur:
                    durations.append(cur)
                cur = 1 if flag else 0
        if cur:
            durations.append(cur)
        arr = np.asarray(durations, dtype=float)
        rows.append(
            {
                "basis": "daily site-total precipitation",
                "threshold": f"{int(q * 100)}th",
                "events": int(len(arr)),
                "median_days": float(np.median(arr)),
                "p90_days": float(np.quantile(arr, 0.90)),
                "max_days": float(np.max(arr)),
            }
        )
    return pd.DataFrame(rows)


def block_length_diagnostics(daily: pd.DataFrame, site_columns: list[str]) -> pd.DataFrame:
    total = daily[site_columns].sum(axis=1).to_numpy(float)
    dates = list(daily["Date"])
    rows = []
    for lag in range(1, 8):
        xs = []
        ys = []
        for i in range(lag, len(dates)):
            if (dates[i] - dates[i - lag]).days == lag:
                xs.append(total[i - lag])
                ys.append(total[i])
        corr = float(np.corrcoef(xs, ys)[0, 1])
        rows.append({"lag_days": lag, "site_total_autocorrelation": corr, "pairs": len(xs)})
    return pd.DataFrame(rows)


def declustering_sensitivity(daily: pd.DataFrame, site_columns: list[str]) -> pd.DataFrame:
    total = daily[site_columns].sum(axis=1).to_numpy(float)
    dates = list(daily["Date"])
    threshold = float(np.quantile(total, TAU))
    exceed_idx = np.where(total > threshold)[0]
    rows = []
    for gap in range(1, 8):
        events = []
        current = []
        last_date = None
        for i in exceed_idx:
            if last_date is None or (dates[i] - last_date).days > gap:
                if current:
                    events.append(current)
                current = [i]
            else:
                current.append(i)
            last_date = dates[i]
        if current:
            events.append(current)
        durations = np.asarray([(dates[e[-1]] - dates[e[0]]).days + 1 for e in events], dtype=float)
        rows.append(
            {
                "declustering_gap_days": gap,
                "events": len(events),
                "median_duration_days": float(np.median(durations)),
                "p90_duration_days": float(np.quantile(durations, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def read_shape_parts(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    parts_out = []
    reader = shapefile.Reader(str(path))
    for shape in reader.shapes():
        points = np.asarray(shape.points, dtype=float)
        if points.size == 0:
            continue
        parts = list(shape.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            parts_out.append(points[start:end])
    return parts_out


def draw_coastline(ax: plt.Axes, color: str = "black", lw: float = 0.9) -> None:
    if not COASTLINE.exists():
        return
    xmin, xmax, ymin, ymax = PLOT_EXTENT
    reader = shapefile.Reader(str(COASTLINE))
    for shape in reader.shapes():
        sxmin, symin, sxmax, symax = shape.bbox
        if sxmax < xmin or sxmin > xmax or symax < ymin or symin > ymax:
            continue
        points = np.asarray(shape.points)
        parts = list(shape.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            seg = points[start:end]
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=6)


def draw_barrier(ax: plt.Axes) -> None:
    for part in read_shape_parts(BARRIER_SHP):
        if len(part) >= 3 and np.allclose(part[0], part[-1]):
            ax.add_patch(
                PolygonPatch(
                    part,
                    closed=True,
                    facecolor="none",
                    edgecolor="#b91c1c",
                    linewidth=1.6,
                    zorder=7,
                )
            )
        else:
            ax.plot(part[:, 0], part[:, 1], color="#b91c1c", lw=1.6, zorder=7)


def format_map(ax: plt.Axes) -> None:
    ax.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=12)


def geotiff_extent(path: Path, width: int, height: int) -> tuple[float, float, float, float]:
    from PIL import Image

    img = Image.open(path)
    tags = img.tag_v2
    scale = tags.get(33550)
    tie = tags.get(33922)
    if scale is None or tie is None:
        return PLOT_EXTENT
    x0 = float(tie[3])
    y0 = float(tie[4])
    dx = float(scale[0])
    dy = float(scale[1])
    return (x0, x0 + dx * width, y0 - dy * height, y0)


def load_topography(max_size: tuple[int, int] = (1400, 1200)) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(SRTM_FILE)
    full_size = img.size
    img.thumbnail(max_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=float)
    arr[arr <= -30000] = np.nan
    extent = geotiff_extent(SRTM_FILE, full_size[0], full_size[1])
    return arr, extent


def plot_domain_map(block_data: BlockData) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    if SRTM_FILE.exists():
        topo, extent = load_topography()
        vmax = float(np.nanquantile(topo, 0.995))
        ax.imshow(topo, extent=extent, origin="upper", cmap="gist_earth", vmin=0, vmax=vmax, alpha=0.92, zorder=1)
    draw_coastline(ax, lw=0.85)
    draw_barrier(ax)
    ax.scatter(block_data.sites["lon"], block_data.sites["lat"], s=12, color="#2563eb", alpha=0.7, label="Upscaled model sites", zorder=8)
    format_map(ax)
    ax.set_title(f"South Island precipitation domain: coastline, SRTM topography, and {BARRIER_HEIGHT_M} m barrier")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(GRAPHICS / "domain_map.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_eda(daily: pd.DataFrame, site_columns: list[str], duration: pd.DataFrame, acf: pd.DataFrame, declust: pd.DataFrame) -> None:
    total = daily[site_columns].sum(axis=1).to_numpy(float)
    regional_max = daily[site_columns].max(axis=1).to_numpy(float)
    dates = daily["Date"]
    recent = daily["Year"] >= daily["Year"].max() - 4
    q95 = float(np.quantile(total, TAU))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates[recent], total[recent], color="#2563eb", lw=1.0, label="Site-total daily precipitation")
    ax.axhline(q95, color="#b91c1c", lw=1.2, label="Overall q95")
    ax.set_title("Recent January-April precipitation episodes")
    ax.set_ylabel("Daily precipitation summed over model sites")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(GRAPHICS / "01_recent_time_series_events.png", dpi=FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(duration["threshold"], duration["events"], color="#2563eb", alpha=0.8)
    ax.set_title("Regional exceedance event counts by threshold")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(GRAPHICS / "02_event_duration_histograms.png", dpi=FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(duration))
    ax.plot(x, duration["median_days"], marker="o", label="Median")
    ax.plot(x, duration["p90_days"], marker="o", label="90th percentile")
    ax.plot(x, duration["max_days"], marker="o", label="Maximum")
    ax.set_xticks(x, duration["threshold"])
    ax.set_title("Exceedance event durations")
    ax.set_ylabel("Days")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(GRAPHICS / "03_duration_boxplots.png", dpi=FIG_DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].bar(acf["lag_days"], acf["site_total_autocorrelation"], color="#2563eb")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Site-total autocorrelation")
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Correlation")
    threshold = np.quantile(regional_max, TAU)
    peaks = np.where(regional_max > threshold)[0]
    sep = np.diff(np.asarray(dates.iloc[peaks], dtype="datetime64[D]")).astype("timedelta64[D]").astype(int)
    axes[1].hist(sep, bins=np.arange(1, 20) - 0.5, color="#238b45", alpha=0.8)
    axes[1].set_title("Separation between high regional maxima")
    axes[1].set_xlabel("Days")
    fig.savefig(GRAPHICS / "04_acf_peak_separation.png", dpi=FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(declust["declustering_gap_days"], declust["events"], marker="o", color="#2563eb")
    ax.set_title("Declustering sensitivity for daily site-total q95 exceedances")
    ax.set_xlabel("Run gap (days)")
    ax.set_ylabel("Event count")
    fig.tight_layout()
    fig.savefig(GRAPHICS / "05_declustering_sensitivity.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_covariates(covariates: np.ndarray, sites: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        vals = covariates[:, k]
        if k == 0:
            vals = np.zeros_like(vals)
        bound = float(np.nanmax(np.abs(vals))) or 1.0
        sc = ax.scatter(sites["lon"], sites["lat"], c=vals, s=20, cmap="coolwarm", vmin=-bound, vmax=bound, zorder=8)
        draw_coastline(ax, lw=0.65)
        draw_barrier(ax)
        format_map(ax)
        ax.set_title(COVARIATE_NAMES[k], fontsize=16)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=11)
    fig.suptitle("Covariates used for threshold and GPD regressions", fontsize=20)
    fig.savefig(GRAPHICS / "mu_covariate_maps.png", dpi=FIG_DPI)
    plt.close(fig)


def empirical_quantile(blocks: np.ndarray, tau: float = TAU) -> np.ndarray:
    return np.quantile(blocks, tau, axis=0, method="linear")


def fit_quantile_regression(blocks: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = blocks.reshape(-1)
    x = np.tile(covariates, (blocks.shape[0], 1))
    keep = np.isfinite(y)
    result = sm.QuantReg(y[keep], x[keep, :]).fit(q=TAU, max_iter=5000, p_tol=1e-6)
    fitted = covariates @ np.asarray(result.params)
    return np.asarray(result.params), fitted


def surface_scatter(ax: plt.Axes, sites: pd.DataFrame, values: np.ndarray, title: str, cmap, norm, label: str):
    sc = ax.scatter(sites["lon"], sites["lat"], c=values, s=26, cmap=cmap, norm=norm, zorder=8)
    draw_coastline(ax, lw=0.65)
    draw_barrier(ax)
    format_map(ax)
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=11)
    return sc


def plot_q95_results(empirical: np.ndarray, fitted: np.ndarray, sites: pd.DataFrame) -> None:
    q_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "#b91c1c"])
    q_norm = Normalize(vmin=float(np.nanmin(empirical)), vmax=float(np.nanmax(empirical)))
    residual = fitted - empirical
    bound = float(np.nanmax(np.abs(residual)))
    resid_norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    resid_cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#1d4ed8", "white", "#b91c1c"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), constrained_layout=True)
    surface_scatter(axes[0], sites, empirical, "Empirical location-wise q95", q_cmap, q_norm, "3-day precipitation sum")
    surface_scatter(axes[1], sites, fitted, "Quantile regression q95", q_cmap, q_norm, "3-day precipitation sum")
    surface_scatter(axes[2], sites, residual, "Fitted minus empirical", resid_cmap, resid_norm, "Difference")
    fig.savefig(GRAPHICS / "q95_fit_vs_empirical.png", dpi=FIG_DPI)
    plt.close(fig)


def xi_from_eta(eta: np.ndarray) -> np.ndarray:
    return -0.5 + 1.5 * expit(eta)


def eta_from_xi(xi: float) -> float:
    clipped = float(np.clip(xi, -0.499, 0.999))
    return float(logit((clipped + 0.5) / 1.5))


def locationwise_gpd_mles(blocks: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.full(blocks.shape[1], np.nan)
    xi = np.full(blocks.shape[1], np.nan)
    n_exceed = np.zeros(blocks.shape[1], dtype=int)
    for j in range(blocks.shape[1]):
        y = blocks[:, j] - threshold[j]
        y = y[np.isfinite(y) & (y > 0)]
        n_exceed[j] = len(y)
        if len(y) < 20:
            continue
        shape, _, scale = genpareto.fit(y, floc=0)
        if np.isfinite(shape) and np.isfinite(scale) and scale > 0:
            xi[j] = shape
            sigma[j] = scale
    return sigma, xi, n_exceed


def exceedance_table(blocks: np.ndarray, threshold: np.ndarray, x_sigma: np.ndarray, x_xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_parts = []
    sigma_parts = []
    xi_parts = []
    for j in range(blocks.shape[1]):
        y = blocks[:, j] - threshold[j]
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            y_parts.append(y)
            sigma_parts.append(np.repeat(x_sigma[j : j + 1, :], y.size, axis=0))
            xi_parts.append(np.repeat(x_xi[j : j + 1, :], y.size, axis=0))
    return np.concatenate(y_parts), np.vstack(sigma_parts), np.vstack(xi_parts)


def gpd_nll(params: np.ndarray, y: np.ndarray, x_sigma: np.ndarray, x_xi: np.ndarray) -> float:
    n_sigma = x_sigma.shape[1]
    beta_sigma = params[:n_sigma]
    beta_xi = params[n_sigma:]
    log_sigma = np.clip(x_sigma @ beta_sigma, -30, 30)
    sigma = np.exp(log_sigma)
    xi = xi_from_eta(x_xi @ beta_xi)
    z = y / sigma
    support = 1.0 + xi * z
    near_zero = np.abs(xi) < 1e-6
    invalid = (~near_zero) & (support <= 0)
    if np.any(invalid):
        return 1e100 + 1e20 * float(np.sum(-support[invalid] + 1.0))
    nll = np.empty_like(y)
    nll[near_zero] = log_sigma[near_zero] + z[near_zero]
    nz = ~near_zero
    nll[nz] = log_sigma[nz] + (1.0 / xi[nz] + 1.0) * np.log1p(xi[nz] * z[nz])
    if not np.all(np.isfinite(nll)):
        return 1e100
    return float(np.sum(nll))


def fit_gpd_regression(
    y: np.ndarray,
    x_sigma: np.ndarray,
    x_xi: np.ndarray,
    sigma_cov: np.ndarray,
    xi_cov: np.ndarray,
    sigma_mle: np.ndarray,
    xi_mle: np.ndarray,
) -> tuple[np.ndarray, float, bool, str]:
    n_sigma = x_sigma.shape[1]
    n_xi = x_xi.shape[1]
    start = np.zeros(n_sigma + n_xi)
    finite_sigma = np.isfinite(sigma_mle) & (sigma_mle > 0)
    finite_xi = np.isfinite(xi_mle)
    start[0] = np.log(float(np.nanmedian(sigma_mle[finite_sigma]))) if np.any(finite_sigma) else np.log(float(np.mean(y)))
    start[n_sigma] = eta_from_xi(float(np.nanmedian(xi_mle[finite_xi]))) if np.any(finite_xi) else eta_from_xi(0.0)
    ls_start = start.copy()
    if np.sum(finite_sigma) >= n_sigma:
        ls_start[:n_sigma] = np.linalg.lstsq(sigma_cov[finite_sigma, :], np.log(sigma_mle[finite_sigma]), rcond=None)[0]
    if np.sum(finite_xi) >= n_xi:
        ls_start[n_sigma:] = np.linalg.lstsq(xi_cov[finite_xi, :], np.array([eta_from_xi(v) for v in xi_mle[finite_xi]]), rcond=None)[0]
    starts = [start]
    for alpha in (0.25, 0.5, 1.0):
        candidate = start + alpha * (ls_start - start)
        if gpd_nll(candidate, y, x_sigma, x_xi) < 1e90:
            starts.append(candidate)
    best = None
    for candidate in starts:
        result = minimize(
            gpd_nll,
            candidate,
            args=(y, x_sigma, x_xi),
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-9, "gtol": 1e-5, "eps": 1e-4, "maxls": 50},
        )
        if np.isfinite(result.fun) and result.fun < 1e90 and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("No finite GPD regression optimization result.")
    return np.asarray(best.x), float(best.fun), bool(best.success), str(best.message)


def gpd_surfaces(params: np.ndarray, sigma_cov: np.ndarray, xi_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_sigma = sigma_cov.shape[1]
    sigma = np.exp(sigma_cov @ params[:n_sigma])
    xi = xi_from_eta(xi_cov @ params[n_sigma:])
    return sigma, xi


def plot_gpd_results(sites: pd.DataFrame, sigma_reg: np.ndarray, xi_reg: np.ndarray, sigma_mle: np.ndarray, xi_mle: np.ndarray) -> None:
    white_red = LinearSegmentedColormap.from_list("white_red", ["white", "#b91c1c"])
    blue_white_red = LinearSegmentedColormap.from_list("blue_white_red", ["#1d4ed8", "white", "#b91c1c"])
    sigma_values = np.r_[sigma_reg[np.isfinite(sigma_reg)], sigma_mle[np.isfinite(sigma_mle)]]
    sigma_norm = Normalize(vmin=float(np.nanmin(sigma_values)), vmax=float(np.nanmax(sigma_values)))
    xi_values = np.r_[xi_reg[np.isfinite(xi_reg)], xi_mle[np.isfinite(xi_mle)]]
    xi_bound = float(np.nanmax(np.abs(xi_values))) or 1.0
    xi_norm = TwoSlopeNorm(vmin=-xi_bound, vcenter=0.0, vmax=xi_bound)
    panels = [
        ("GPD regression sigma", sigma_reg, white_red, sigma_norm, "3-day precipitation sum"),
        ("Location-wise MLE sigma", sigma_mle, white_red, sigma_norm, "3-day precipitation sum"),
        ("GPD regression xi", xi_reg, blue_white_red, xi_norm, "shape"),
        ("Location-wise MLE xi", xi_mle, blue_white_red, xi_norm, "shape"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), constrained_layout=True)
    for ax, (title, surface, cmap, norm, label) in zip(axes.ravel(), panels):
        surface_scatter(ax, sites, surface, title, cmap, norm, label)
    fig.savefig(GRAPHICS / "gpd_threshold_and_parameter_maps.png", dpi=FIG_DPI)
    plt.close(fig)


def write_block_csv(block_data: BlockData) -> None:
    out = pd.DataFrame(block_data.blocks, columns=block_data.site_columns)
    out.insert(0, "block_end", block_data.block_end)
    out.insert(0, "block_start", block_data.block_start)
    out.insert(0, "Year", block_data.block_year)
    out.to_csv(OUTPUT / "upscaled_era5_3day_sum_precp_ARs_model_sites.csv", index=False)


def write_outputs(
    block_data: BlockData,
    covariates: np.ndarray,
    beta_mu: np.ndarray,
    q95_empirical: np.ndarray,
    q95_fit: np.ndarray,
    n_exceed: np.ndarray,
    gpd_params: np.ndarray,
    gpd_nll_value: float,
    success: bool,
    message: str,
    sigma_reg: np.ndarray,
    xi_reg: np.ndarray,
    sigma_mle: np.ndarray,
    xi_mle: np.ndarray,
) -> None:
    np.savez_compressed(
        OUTPUT / "nz_workflow_outputs.npz",
        lon=block_data.sites["lon"].to_numpy(),
        lat=block_data.sites["lat"].to_numpy(),
        blocks=block_data.blocks,
        q95_empirical=q95_empirical,
        q95_quantile_regression=q95_fit,
        threshold_q95=q95_fit,
        n_exceed=n_exceed,
        sigma_regression=sigma_reg,
        xi_regression=xi_reg,
        sigma_locationwise_mle=sigma_mle,
        xi_locationwise_mle=xi_mle,
        beta_mu=beta_mu,
        beta_log_sigma=gpd_params[: covariates.shape[1]],
        beta_logit_star_xi=gpd_params[covariates.shape[1] :],
        negative_log_likelihood=gpd_nll_value,
        optimizer_success=success,
        optimizer_message=message,
    )
    residual = q95_fit - q95_empirical
    rows = [
        {"metric": "q95_fit_rmse", "value": float(np.sqrt(np.mean(residual**2)))},
        {"metric": "q95_fit_mae", "value": float(np.mean(np.abs(residual)))},
        {"metric": "gpd_negative_log_likelihood", "value": float(gpd_nll_value)},
        {"metric": "gpd_optimizer_success", "value": int(success)},
        {"metric": "n_exceedances", "value": int(np.sum(n_exceed))},
        {"metric": "n_locations", "value": int(block_data.blocks.shape[1])},
        {"metric": "n_3day_blocks", "value": int(block_data.blocks.shape[0])},
        {"metric": "sigma_regression_median", "value": float(np.nanmedian(sigma_reg))},
        {"metric": "xi_regression_median", "value": float(np.nanmedian(xi_reg))},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "workflow_summary.csv", index=False)
    beta_rows = []
    for name, beta in zip(COVARIATE_NAMES, beta_mu):
        beta_rows.append({"model_part": "q95", "covariate": name, "beta": beta})
    for name, beta in zip(COVARIATE_NAMES, gpd_params[: len(COVARIATE_NAMES)]):
        beta_rows.append({"model_part": "log_sigma", "covariate": name, "beta": beta})
    for name, beta in zip(COVARIATE_NAMES, gpd_params[len(COVARIATE_NAMES) :]):
        beta_rows.append({"model_part": "logit_star_xi", "covariate": name, "beta": beta})
    pd.DataFrame(beta_rows).to_csv(OUTPUT / "model_coefficients.csv", index=False)


def main() -> None:
    setup_dirs()
    for optional in (COASTLINE, SRTM_FILE):
        if not optional.exists():
            print(f"({optional.name} not found in {optional.parent}; maps are drawn without it)")
    daily, sites, site_columns = load_daily_model_subset()
    block_data = make_nonoverlap_blocks(daily, site_columns, BLOCK_DAYS)
    write_block_csv(block_data)

    duration = storm_duration_summary(daily, site_columns)
    acf = block_length_diagnostics(daily, site_columns)
    declust = declustering_sensitivity(daily, site_columns)
    duration.to_csv(OUTPUT / "storm_duration_summary.csv", index=False)
    acf.to_csv(OUTPUT / "block_length_autocorrelation.csv", index=False)
    declust.to_csv(OUTPUT / "declustering_sensitivity.csv", index=False)

    covariates = np.loadtxt(COVARIATES_FILE, delimiter=",")
    if covariates.shape[0] != block_data.blocks.shape[1]:
        raise ValueError(f"Covariate rows {covariates.shape[0]} do not match sites {block_data.blocks.shape[1]}")

    plot_domain_map(block_data)
    plot_eda(daily, site_columns, duration, acf, declust)
    plot_covariates(covariates, block_data.sites)

    q95_emp = empirical_quantile(block_data.blocks, TAU)
    beta_mu, q95_fit = fit_quantile_regression(block_data.blocks, covariates)
    plot_q95_results(q95_emp, q95_fit, block_data.sites)

    sigma_mle, xi_mle, n_exceed = locationwise_gpd_mles(block_data.blocks, q95_fit)
    y, x_sigma, x_xi = exceedance_table(block_data.blocks, q95_fit, covariates, covariates)
    params, nll, success, message = fit_gpd_regression(y, x_sigma, x_xi, covariates, covariates, sigma_mle, xi_mle)
    sigma_reg, xi_reg = gpd_surfaces(params, covariates, covariates)
    plot_gpd_results(block_data.sites, sigma_reg, xi_reg, sigma_mle, xi_mle)

    write_outputs(block_data, covariates, beta_mu, q95_emp, q95_fit, n_exceed, params, nll, success, message, sigma_reg, xi_reg, sigma_mle, xi_mle)
    summary = pd.read_csv(OUTPUT / "workflow_summary.csv")

    print(f"3-day blocks: {block_data.blocks.shape}")
    print(f"q95 RMSE: {summary.loc[summary.metric == 'q95_fit_rmse', 'value'].iloc[0]:.3f}")
    print(f"GPD exceedances: {int(summary.loc[summary.metric == 'n_exceedances', 'value'].iloc[0])}")
    print(f"GPD optimizer success: {success}; {message}")
    print(f"Wrote outputs and figures to {OUTPUT}")


if __name__ == "__main__":
    main()
