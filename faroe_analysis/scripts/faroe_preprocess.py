#!/usr/bin/env python3
"""Faroe marginal pre-processing: 0.95 quantile-regression threshold + GPD regression.

Part I of ``faroe_analysis/vignette/faroe_modeling_vignette.md``.  Starts at the
5-day storm-season block maxima, fits the tau=0.95 spatial quantile regression used
as the high threshold, then fits a generalized Pareto regression to the positive
exceedances over that threshold (maximum likelihood, with location-wise GPD MLEs
for comparison).

Inputs (all in ``faroe_analysis/data/``)::

    faroe_block_maxima.npz                 block_max (1656, 21, 21), latitude, longitude
    mu_covariates_std_2stage_MCMC.csv      threshold covariates (water cells x 11)
    sigma_covariates_std_2stage_MCMC.csv   GPD scale covariates (water cells x 11)
    xi_covariates_std_2stage_MCMC.csv      GPD shape covariates (water cells x 8)
    ne_10m_coastline/ne_10m_coastline.shp  optional; map overlay only

Outputs go to ``faroe_analysis/output/preprocess/`` (override with
FAROE_PREPROCESS_OUT): ``faroe_workflow_outputs.npz`` (the same arrays as the shipped
``data/faroe_workflow_outputs.npz``), ``workflow_summary.csv``,
``model_coefficients.csv`` and the Part I figures in ``graphics/``.  The shipped
data file is never overwritten.

Run from anywhere (serial, about a minute)::

    python faroe_analysis/scripts/faroe_preprocess.py

Needs ``statsmodels`` (quantile regression); ``pyshp`` only for the coastline overlay.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
INPUT = ANALYSIS / "data"
OUTPUT = Path(os.environ.get("FAROE_PREPROCESS_OUT", ANALYSIS / "output" / "preprocess"))
GRAPHICS = OUTPUT / "graphics"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import genpareto


BLOCK_MAX_FILE = INPUT / "faroe_block_maxima.npz"
MU_COVARIATES = INPUT / "mu_covariates_std_2stage_MCMC.csv"
SIGMA_COVARIATES = INPUT / "sigma_covariates_std_2stage_MCMC.csv"
XI_COVARIATES = INPUT / "xi_covariates_std_2stage_MCMC.csv"
COASTLINE = INPUT / "ne_10m_coastline" / "ne_10m_coastline.shp"

TAU = 0.95
PLOT_EXTENT = (-9.4, -4.6, 59.7, 64.3)

# Buffered polygon used in the original barrier-aware analysis.  Grid-cell
# centers inside this outline are treated as land and excluded from fitting.
BUFFER_X = np.array(
    [-7.7, -7.2, -7, -6.91, -6.98, -6.63, -6.61, -6.69, -6.52, -6.27, -6.27, -6.53, -7.2, -7.24]
)
BUFFER_Y = np.array(
    [62.1, 62.01, 62.08, 61.9, 61.565, 61.35, 61.82, 62, 62.15, 62.24, 62.31, 62.41, 62.33, 62.2]
)
BUFFER_COORDS = np.column_stack([BUFFER_X, BUFFER_Y])

MU_NAMES = [
    "Intercept",
    "TPS linear 1",
    "TPS linear 2",
    "TPS smooth 1",
    "TPS smooth 2",
    "TPS smooth 3",
    "Depth",
    "Distance to centroid",
    "sin(angle)",
    "Depth x distance",
    "Distance x sin(angle)",
]
SIGMA_NAMES = MU_NAMES
XI_NAMES = [
    "Intercept",
    "TPS linear 1",
    "TPS linear 2",
    "TPS smooth 1",
    "Depth",
    "Bathymetric gradient",
    "Distance to centroid",
    "sin(angle)",
]


def grid_edges(values: np.ndarray) -> np.ndarray:
    """Convert regularly spaced grid centers to plotting cell edges."""
    step = float(np.diff(values).mean())
    return np.r_[values[0] - step / 2, values[:-1] + np.diff(values) / 2, values[-1] + step / 2]


def read_block_maxima() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the 5-day storm-season block maxima (VHM0, meters) and the grid."""
    with np.load(BLOCK_MAX_FILE) as d:
        block_max = np.asarray(d["block_max"], dtype=float)
        latitude = np.asarray(d["latitude"], dtype=float)
        longitude = np.asarray(d["longitude"], dtype=float)
    return block_max, latitude, longitude


def buffered_land_mask(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Return True for grid cells inside the buffered Faroe land polygon."""
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    return MplPath(BUFFER_COORDS).contains_points(points).reshape(lon_grid.shape)


def load_covariates(path: Path, land_mask: np.ndarray) -> np.ndarray:
    """Load covariates and check that their rows match the water-cell count."""
    covariates = np.loadtxt(path, delimiter=",")
    expected_rows = int((~land_mask).sum())
    if covariates.shape[0] != expected_rows:
        raise ValueError(f"{path.name} has {covariates.shape[0]} rows; expected {expected_rows} water cells.")
    return covariates


def draw_coastline(ax: plt.Axes) -> None:
    """Draw Natural Earth coastline segments intersecting the plotting domain."""
    if not COASTLINE.exists():
        return
    import shapefile  # pyshp; only needed for the optional coastline overlay

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
            ax.plot(seg[:, 0], seg[:, 1], color="black", lw=0.9, zorder=5)


def draw_buffer_polygon(ax: plt.Axes, facecolor: str = "white", alpha: float = 0.9) -> None:
    """Add the island/barrier outline used as the land mask."""
    ax.add_patch(
        PolygonPatch(
            BUFFER_COORDS,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.2,
            alpha=alpha,
            zorder=6,
        )
    )


def format_map_axis(ax: plt.Axes) -> None:
    ax.set_xlim(PLOT_EXTENT[0], PLOT_EXTENT[1])
    ax.set_ylim(PLOT_EXTENT[2], PLOT_EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)


def plot_domain_map(latitude: np.ndarray, longitude: np.ndarray, land_mask: np.ndarray) -> None:
    """Plot the analysis grid and the buffered island outline."""
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(lon_grid[~land_mask], lat_grid[~land_mask], s=20, color="#2563eb", alpha=0.65, label="Water grid cells")
    ax.scatter(lon_grid[land_mask], lat_grid[land_mask], s=20, color="#9ca3af", alpha=0.7, label="Masked land cells")
    draw_coastline(ax)
    draw_buffer_polygon(ax)
    format_map_axis(ax)
    ax.set_title("Faroe Islands North Atlantic analysis domain")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(GRAPHICS / "domain_map.png", dpi=180)
    plt.close(fig)


def empirical_quantile(block_max: np.ndarray, land_mask: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Compute location-wise empirical quantiles for non-land cells."""
    q = np.full(block_max.shape[1:], np.nan, dtype=float)
    for i in range(block_max.shape[1]):
        for j in range(block_max.shape[2]):
            if land_mask[i, j]:
                continue
            values = block_max[:, i, j]
            values = values[np.isfinite(values)]
            if values.size:
                q[i, j] = np.quantile(values, tau, method="linear")
    return q


def fit_quantile_regression(
    block_max: np.ndarray, covariates: np.ndarray, land_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a block-level tau=0.95 quantile regression over all water cells."""
    y_by_location = block_max[:, ~land_mask]
    y = y_by_location.reshape(-1)
    x = np.tile(covariates, (block_max.shape[0], 1))
    keep = np.isfinite(y)

    model = sm.QuantReg(y[keep], x[keep, :])
    result = model.fit(q=TAU, max_iter=5000, p_tol=1e-6)

    fitted_water = covariates @ np.asarray(result.params)
    surface = np.full(block_max.shape[1:], np.nan, dtype=float)
    surface[~land_mask] = fitted_water
    return np.asarray(result.params), surface


def plot_covariates(covariates: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, land_mask: np.ndarray) -> None:
    """Plot the 11 mu covariates used in the q95 regression."""
    lon_edges = grid_edges(longitude)
    lat_edges = grid_edges(latitude)
    water = ~land_mask
    surfaces = []
    for k in range(covariates.shape[1]):
        surface = np.full(land_mask.shape, np.nan, dtype=float)
        surface[water] = covariates[:, k]
        surfaces.append(surface)

    finite = np.concatenate([surface[np.isfinite(surface) & water] for surface in surfaces])
    bound = float(np.max(np.abs(finite)))
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#1d4ed8", "white", "#b91c1c"])
    norm = Normalize(vmin=-bound, vmax=bound)

    fig, axes = plt.subplots(3, 4, figsize=(15.0, 10.6), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(surfaces):
            ax.axis("off")
            continue
        masked = np.ma.array(surfaces[k], mask=land_mask | ~np.isfinite(surfaces[k]))
        mesh = ax.pcolormesh(lon_edges, lat_edges, masked, cmap=cmap, norm=norm, shading="auto")
        draw_coastline(ax)
        draw_buffer_polygon(ax)
        format_map_axis(ax)
        ax.set_title(MU_NAMES[k], fontsize=10)
        ax.tick_params(labelsize=8)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Standardized covariate value")
    fig.suptitle("Covariates for the 0.95 quantile regression", fontsize=15)
    fig.savefig(GRAPHICS / "mu_covariate_maps.png", dpi=180)
    plt.close(fig)


def plot_q95_results(
    empirical: np.ndarray,
    fitted: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    land_mask: np.ndarray,
) -> None:
    """Compare empirical and fitted q95 surfaces on common color scales."""
    lon_edges = grid_edges(longitude)
    lat_edges = grid_edges(latitude)
    water = ~land_mask
    q_values = empirical[np.isfinite(empirical) & water]
    q_norm = Normalize(vmin=float(q_values.min()), vmax=float(q_values.max()))
    q_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "#b91c1c"])

    residual = fitted - empirical
    resid_values = residual[np.isfinite(residual) & water]
    bound = float(np.max(np.abs(resid_values)))
    resid_norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    resid_cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#1d4ed8", "white", "#b91c1c"])

    panels = [
        ("Empirical location-wise q95", empirical, q_cmap, q_norm, "5-day max VHM0 (m)"),
        ("Quantile regression q95", fitted, q_cmap, q_norm, "5-day max VHM0 (m)"),
        ("Fitted minus empirical", residual, resid_cmap, resid_norm, "Difference (m)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), constrained_layout=True)
    for ax, (title, surface, cmap, norm, label) in zip(axes, panels):
        masked = np.ma.array(surface, mask=land_mask | ~np.isfinite(surface))
        mesh = ax.pcolormesh(lon_edges, lat_edges, masked, cmap=cmap, norm=norm, shading="auto")
        draw_coastline(ax)
        draw_buffer_polygon(ax)
        format_map_axis(ax)
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82)
        cbar.set_label(label)
    fig.savefig(GRAPHICS / "q95_fit_vs_empirical.png", dpi=180)
    plt.close(fig)


def xi_from_eta(eta: np.ndarray) -> np.ndarray:
    """Map an unconstrained linear predictor to xi in (-0.5, 1)."""
    return -0.5 + 1.5 * expit(eta)


def eta_from_xi(xi: float) -> float:
    """Inverse of xi_from_eta, with clipping for stable initialization."""
    clipped = float(np.clip(xi, -0.499, 0.999))
    return float(logit((clipped + 0.5) / 1.5))


def locationwise_gpd_mles(
    block_max: np.ndarray, threshold: np.ndarray, land_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit independent GPD MLEs at each water cell for comparison."""
    sigma = np.full(land_mask.shape, np.nan, dtype=float)
    xi = np.full(land_mask.shape, np.nan, dtype=float)
    n_exceed = np.zeros(land_mask.shape, dtype=np.int16)
    for i in range(block_max.shape[1]):
        for j in range(block_max.shape[2]):
            if land_mask[i, j] or not np.isfinite(threshold[i, j]):
                continue
            exceedances = block_max[:, i, j] - threshold[i, j]
            exceedances = exceedances[np.isfinite(exceedances) & (exceedances > 0)]
            n_exceed[i, j] = len(exceedances)
            if len(exceedances) < 20:
                continue
            shape, _, scale = genpareto.fit(exceedances, floc=0)
            if np.isfinite(shape) and np.isfinite(scale) and scale > 0:
                sigma[i, j] = scale
                xi[i, j] = shape
    return sigma, xi, n_exceed


def exceedance_table(
    block_max: np.ndarray,
    threshold: np.ndarray,
    land_mask: np.ndarray,
    sigma_covariates: np.ndarray,
    xi_covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack all positive exceedances with the covariates for their grid cell."""
    y_parts = []
    sigma_parts = []
    xi_parts = []
    water_values = block_max[:, ~land_mask]
    water_threshold = threshold[~land_mask]
    for loc_idx in range(water_values.shape[1]):
        y = water_values[:, loc_idx] - water_threshold[loc_idx]
        y = y[np.isfinite(y) & (y > 0)]
        if y.size == 0:
            continue
        y_parts.append(y)
        sigma_parts.append(np.repeat(sigma_covariates[loc_idx : loc_idx + 1, :], y.size, axis=0))
        xi_parts.append(np.repeat(xi_covariates[loc_idx : loc_idx + 1, :], y.size, axis=0))
    return np.concatenate(y_parts), np.vstack(sigma_parts), np.vstack(xi_parts)


def gpd_negative_log_likelihood(params: np.ndarray, y: np.ndarray, x_sigma: np.ndarray, x_xi: np.ndarray) -> float:
    """Negative log likelihood for a covariate-dependent GPD exceedance model."""
    n_sigma = x_sigma.shape[1]
    beta_sigma = params[:n_sigma]
    beta_xi = params[n_sigma:]
    log_sigma = x_sigma @ beta_sigma
    sigma = np.exp(np.clip(log_sigma, -30.0, 30.0))
    xi = xi_from_eta(x_xi @ beta_xi)
    z = y / sigma

    near_zero = np.abs(xi) < 1e-6
    support = 1.0 + xi * z
    invalid = (~near_zero) & (support <= 0.0)
    if np.any(invalid):
        return 1e100 + 1e20 * float(np.sum(-support[invalid] + 1.0))

    nll = np.empty_like(y, dtype=float)
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
    sigma_covariates: np.ndarray,
    xi_covariates: np.ndarray,
    sigma_mle: np.ndarray,
    xi_mle: np.ndarray,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, float, bool, str]:
    """Fit the GPD regression, initialized from location-wise MLEs when possible."""
    n_sigma = x_sigma.shape[1]
    n_xi = x_xi.shape[1]
    water = ~land_mask
    start = np.zeros(n_sigma + n_xi, dtype=float)

    finite_sigma = np.isfinite(sigma_mle[water]) & (sigma_mle[water] > 0)
    finite_xi = np.isfinite(xi_mle[water])
    start[0] = np.log(float(np.median(sigma_mle[water][finite_sigma]))) if np.any(finite_sigma) else np.log(float(np.mean(y)))
    start[n_sigma] = eta_from_xi(float(np.median(xi_mle[water][finite_xi]))) if np.any(finite_xi) else eta_from_xi(0.0)

    ls_start = start.copy()
    if np.sum(finite_sigma) >= n_sigma:
        ls_start[:n_sigma] = np.linalg.lstsq(
            sigma_covariates[finite_sigma, :], np.log(sigma_mle[water][finite_sigma]), rcond=None
        )[0]
    if np.sum(finite_xi) >= n_xi:
        ls_start[n_sigma:] = np.linalg.lstsq(
            xi_covariates[finite_xi, :],
            np.array([eta_from_xi(value) for value in xi_mle[water][finite_xi]]),
            rcond=None,
        )[0]

    # Several starts reduce sensitivity to noisy location-wise tail estimates.
    starts = [start]
    for alpha in (0.25, 0.5, 1.0):
        candidate = start + alpha * (ls_start - start)
        if gpd_negative_log_likelihood(candidate, y, x_sigma, x_xi) < 1e90:
            starts.append(candidate)

    best = None
    for candidate in starts:
        result = minimize(
            gpd_negative_log_likelihood,
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


def gpd_regression_surfaces(
    params: np.ndarray, sigma_covariates: np.ndarray, xi_covariates: np.ndarray, land_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fitted GPD sigma and xi at each water cell."""
    n_sigma = sigma_covariates.shape[1]
    sigma_water = np.exp(sigma_covariates @ params[:n_sigma])
    xi_water = xi_from_eta(xi_covariates @ params[n_sigma:])
    sigma = np.full(land_mask.shape, np.nan, dtype=float)
    xi = np.full(land_mask.shape, np.nan, dtype=float)
    sigma[~land_mask] = sigma_water
    xi[~land_mask] = xi_water
    return sigma, xi


def plot_gpd_results(
    threshold: np.ndarray,
    n_exceed: np.ndarray,
    sigma_regression: np.ndarray,
    xi_regression: np.ndarray,
    sigma_mle: np.ndarray,
    xi_mle: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    land_mask: np.ndarray,
) -> None:
    """Plot GPD parameter surfaces with regression and location-wise fits adjacent."""
    lon_edges = grid_edges(longitude)
    lat_edges = grid_edges(latitude)
    white_red = LinearSegmentedColormap.from_list("white_red", ["white", "#b91c1c"])
    blue_white_red = LinearSegmentedColormap.from_list("blue_white_red", ["#1d4ed8", "white", "#b91c1c"])

    sigma_values = np.r_[sigma_regression[np.isfinite(sigma_regression)], sigma_mle[np.isfinite(sigma_mle)]]
    sigma_norm = Normalize(vmin=float(np.nanmin(sigma_values)), vmax=float(np.nanmax(sigma_values)))
    xi_values = np.r_[xi_regression[np.isfinite(xi_regression)], xi_mle[np.isfinite(xi_mle)]]
    xi_bound = float(np.nanmax(np.abs(xi_values)))
    xi_norm = TwoSlopeNorm(vmin=-xi_bound, vcenter=0.0, vmax=xi_bound)

    panels = [
        ("GPD regression sigma", sigma_regression, white_red, sigma_norm, "m"),
        ("Location-wise MLE sigma", sigma_mle, white_red, sigma_norm, "m"),
        ("GPD regression xi", xi_regression, blue_white_red, xi_norm, "shape"),
        ("Location-wise MLE xi", xi_mle, blue_white_red, xi_norm, "shape"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.5), constrained_layout=True)
    for ax, (title, surface, cmap, norm, label) in zip(axes.ravel(), panels):
        masked = np.ma.array(surface, mask=land_mask | ~np.isfinite(surface))
        mesh = ax.pcolormesh(lon_edges, lat_edges, masked, cmap=cmap, norm=norm, shading="auto")
        draw_coastline(ax)
        draw_buffer_polygon(ax)
        format_map_axis(ax)
        ax.set_title(title)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82)
        cbar.set_label(label)
    fig.savefig(GRAPHICS / "gpd_threshold_and_parameter_maps.png", dpi=180)
    plt.close(fig)


def write_outputs(
    beta_mu: np.ndarray,
    q95_empirical: np.ndarray,
    q95_fit: np.ndarray,
    threshold: np.ndarray,
    n_exceed: np.ndarray,
    gpd_params: np.ndarray,
    gpd_nll: float,
    optimizer_success: bool,
    optimizer_message: str,
    sigma_regression: np.ndarray,
    xi_regression: np.ndarray,
    sigma_mle: np.ndarray,
    xi_mle: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    land_mask: np.ndarray,
) -> None:
    """Persist compact arrays and human-readable summaries."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "faroe_workflow_outputs.npz",
        latitude=latitude,
        longitude=longitude,
        land_mask=land_mask,
        q95_empirical=q95_empirical,
        q95_quantile_regression=q95_fit,
        threshold_q95=threshold,
        n_exceed=n_exceed,
        sigma_regression=sigma_regression,
        xi_regression=xi_regression,
        sigma_locationwise_mle=sigma_mle,
        xi_locationwise_mle=xi_mle,
        beta_mu=beta_mu,
        beta_log_sigma=gpd_params[: len(SIGMA_NAMES)],
        beta_logit_star_xi=gpd_params[len(SIGMA_NAMES) :],
        negative_log_likelihood=gpd_nll,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
    )

    water = ~land_mask
    resid = q95_fit - q95_empirical
    resid_values = resid[np.isfinite(resid) & water]
    rows = [
        {"metric": "q95_fit_rmse_m", "value": float(np.sqrt(np.mean(resid_values**2)))},
        {"metric": "q95_fit_mae_m", "value": float(np.mean(np.abs(resid_values)))},
        {"metric": "gpd_negative_log_likelihood", "value": float(gpd_nll)},
        {"metric": "gpd_optimizer_success", "value": int(optimizer_success)},
        {"metric": "n_exceedances", "value": int(np.sum(n_exceed[water]))},
        {"metric": "n_water_locations", "value": int(water.sum())},
        {"metric": "sigma_regression_median", "value": float(np.nanmedian(sigma_regression[water]))},
        {"metric": "xi_regression_median", "value": float(np.nanmedian(xi_regression[water]))},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "workflow_summary.csv", index=False)

    beta_rows = []
    for name, beta in zip(MU_NAMES, beta_mu):
        beta_rows.append({"model_part": "q95", "covariate": name, "beta": beta})
    for name, beta in zip(SIGMA_NAMES, gpd_params[: len(SIGMA_NAMES)]):
        beta_rows.append({"model_part": "log_sigma", "covariate": name, "beta": beta})
    for name, beta in zip(XI_NAMES, gpd_params[len(SIGMA_NAMES) :]):
        beta_rows.append({"model_part": "logit_star_xi", "covariate": name, "beta": beta})
    pd.DataFrame(beta_rows).to_csv(OUTPUT / "model_coefficients.csv", index=False)


def main() -> None:
    GRAPHICS.mkdir(parents=True, exist_ok=True)
    if not COASTLINE.exists():
        print(f"(coastline not found at {COASTLINE}; maps are drawn without it)")

    block_max, latitude, longitude = read_block_maxima()
    land_mask = buffered_land_mask(longitude, latitude)
    mu_covariates = load_covariates(MU_COVARIATES, land_mask)
    sigma_covariates = load_covariates(SIGMA_COVARIATES, land_mask)
    xi_covariates = load_covariates(XI_COVARIATES, land_mask)

    print(f"block maxima shape: {block_max.shape}")
    print(f"water cells: {(~land_mask).sum()}, masked land cells: {land_mask.sum()}")

    plot_domain_map(latitude, longitude, land_mask)
    plot_covariates(mu_covariates, latitude, longitude, land_mask)

    q95_empirical = empirical_quantile(block_max, land_mask, TAU)
    beta_mu, q95_fit = fit_quantile_regression(block_max, mu_covariates, land_mask)
    plot_q95_results(q95_empirical, q95_fit, latitude, longitude, land_mask)

    threshold = q95_fit
    sigma_mle, xi_mle, n_exceed = locationwise_gpd_mles(block_max, threshold, land_mask)
    y, x_sigma, x_xi = exceedance_table(block_max, threshold, land_mask, sigma_covariates, xi_covariates)
    gpd_params, gpd_nll, success, message = fit_gpd_regression(
        y, x_sigma, x_xi, sigma_covariates, xi_covariates, sigma_mle, xi_mle, land_mask
    )
    sigma_regression, xi_regression = gpd_regression_surfaces(gpd_params, sigma_covariates, xi_covariates, land_mask)
    plot_gpd_results(
        threshold,
        n_exceed,
        sigma_regression,
        xi_regression,
        sigma_mle,
        xi_mle,
        latitude,
        longitude,
        land_mask,
    )

    write_outputs(
        beta_mu,
        q95_empirical,
        q95_fit,
        threshold,
        n_exceed,
        gpd_params,
        gpd_nll,
        success,
        message,
        sigma_regression,
        xi_regression,
        sigma_mle,
        xi_mle,
        latitude,
        longitude,
        land_mask,
    )

    water = ~land_mask
    residual = q95_fit - q95_empirical
    keep = np.isfinite(residual) & water
    print(f"q95 RMSE against empirical location-wise q95: {np.sqrt(np.mean(residual[keep] ** 2)):.3f} m")
    print(f"GPD exceedances: {len(y)}")
    print(f"GPD optimizer success: {success}; {message}")
    print(f"Wrote graphics to {GRAPHICS}")
    print(f"Wrote model outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
