#!/usr/bin/env python3
"""
Task 1 -- data preparation for the Faroe Islands wave-height analysis.

Gathers and saves everything the HW / Wang / Shi samplers need:
  * the 5-day-block-maxima wave matrix Y (432 ocean sites x 1656 blocks),
  * spatially-varying threshold (u_vec) and GPD marginal covariates + beta inits,
  * the knot grids (S / phi / rho),
  * the train/test split (three contiguous holdout blocks),
  * the Wang junction-tree cliques for the full graph (validated separable) and the
    training subgraph (recomputed).

Outputs (under FAROE_OUT, default faroe_analysis/output):
  * inputs.pkl / inputs.npz   -- consumed by every downstream task
  * figures/split.png         -- train/test blocks over island + barrier (REVIEW ME)
  * figures/knots.png         -- knot layout + sites + barrier
  * figures/threshold.png     -- the 0.95-quantile threshold surface

Serial (no MPI needed):  python faroe_analysis/scripts/faroe_task1_prep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # faroe_analysis/scripts
import numpy as np                                     # noqa: E402
import faroe_setup as F                                # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


def _edges(centers):
    """Cell-edge coordinates from a vector of regular cell centers."""
    c = np.asarray(centers, float)
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (c[1] - c[0]) / 2.0
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2.0
    return e


def _to_grid(inp, values):
    """Scatter per-site values back onto the full (lat, lon) grid (land = NaN)."""
    g = np.full(inp["land_mask"].shape, np.nan)
    g[~inp["land_mask"]] = values
    return g


def plot_split(inp, path):
    sites = inp["sites_xy"]
    block = inp["block_id"]
    colors = {0: "0.6", 1: "tab:red", 2: "tab:blue", 3: "tab:orange"}
    fig, ax = plt.subplots(figsize=(8, 8))
    for b, col in colors.items():
        m = block == b
        ax.scatter(sites[m, 0], sites[m, 1], s=70, c=col, zorder=3,
                   edgecolors="k", linewidths=0.3)
    F.add_barrier(ax)
    F.add_coastline(ax)
    F.set_domain(ax)
    ax.set_title("Train/Test Split")
    _savefig(fig, path)


def plot_knots(inp, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(inp["sites_xy"][:, 0], inp["sites_xy"][:, 1], s=8, c="0.7", zorder=2)
    ax.scatter(inp["knots_S"][:, 0], inp["knots_S"][:, 1], marker="*", s=180,
               c="tab:blue", zorder=4, edgecolors="k", linewidths=0.3)
    ax.scatter(inp["knots_rho"][:, 0], inp["knots_rho"][:, 1], marker="P", s=90,
               c="tab:red", zorder=5, edgecolors="k", linewidths=0.3)
    F.add_barrier(ax)
    F.add_coastline(ax)
    F.set_domain(ax)
    ax.set_title("Knot Layout")
    _savefig(fig, path)


def plot_threshold(inp, path):
    grid = np.ma.masked_invalid(_to_grid(inp, inp["u_vec"]))
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = matplotlib.colormaps["Reds"].copy()
    cmap.set_bad("white")                                  # land cells white
    pcm = ax.pcolormesh(_edges(inp["lon"]), _edges(inp["lat"]), grid, cmap=cmap,
                        zorder=1)                          # no cell gridlines
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04,
                 label="0.95-quantile threshold (m)")
    F.add_barrier(ax)
    F.add_coastline(ax)
    F.set_domain(ax)
    ax.set_title("Threshold Surface (Spatial 0.95 Quantile Regression)")
    _savefig(fig, path)


def main():
    os.makedirs(F.OUT, exist_ok=True)
    figdir = os.path.join(F.OUT, "figures")

    inp = F.prepare_inputs(n_cores=int(os.environ.get("FAROE_PREP_CORES", "4")))

    # --- validations --------------------------------------------------------
    Y = inp["Y"]
    Ns, Nt = Y.shape
    assert (Ns, Nt) == (432, 1656), f"unexpected Y shape {Y.shape}"
    allnan = np.isnan(Y).all(axis=1)
    assert not allnan.any(), f"{int(allnan.sum())} sites are all-NaN"
    assert np.isfinite(inp["u_vec"]).all(), "non-finite threshold"
    assert inp["sigma_spatial"].shape == (Ns, 10)
    assert inp["xi_spatial"].shape == (Ns, 7)
    assert inp["beta_sigma"].shape == (11,) and inp["beta_xi"].shape == (8,)
    frac_missing = float(np.isnan(Y).mean())
    print(f"[prep] Y {Y.shape}; missing frac {frac_missing:.4f}; "
          f"threshold [{inp['u_vec'].min():.2f}, {inp['u_vec'].max():.2f}] m")
    print(f"[prep] full junction tree: {len(inp['full_cliques'])} cliques / "
          f"{len(inp['full_separators'])} separators (separable: OK)")
    print(f"[prep] train junction tree: {len(inp['train_cliques'])} cliques / "
          f"{len(inp['train_separators'])} separators")

    # --- save + figures -----------------------------------------------------
    F.save_inputs(inp)
    print(f"[prep] saved -> {F.INPUTS_PKL} and {F.INPUTS_NPZ}")
    plot_split(inp, os.path.join(figdir, "split.png"))
    plot_knots(inp, os.path.join(figdir, "knots.png"))
    plot_threshold(inp, os.path.join(figdir, "threshold.png"))
    print(f"[prep] figures -> {figdir}/  (REVIEW split.png before fitting)")


if __name__ == "__main__":
    main()
