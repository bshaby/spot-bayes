#!/usr/bin/env python3
"""
Task 1 -- data preparation for the NZ South Island extreme-precipitation analysis.

Gathers and saves everything the HW / Wang / Shi samplers need:
  * the 3-day-sum precipitation matrix Y (309 sites x 3080 blocks),
  * spatially-varying threshold (u_vec) and GPD marginal covariates + beta inits,
  * the knot grids (S / phi / rho),
  * the train/test split (NW open-ocean block + SE leeward barrier-hugging strip),
  * the Wang junction-tree cliques for the full 1500 m visibility graph and the
    training subgraph.

Outputs (under NZ_OUT, default nz_analysis/output):
  * inputs.pkl / inputs.npz   -- consumed by every downstream task
  * figures/split.png         -- train/test blocks over island + barrier (REVIEW ME)
  * figures/knots.png         -- knot layout + sites + barrier
  * figures/threshold.png     -- the 0.95-quantile threshold surface

Serial (no MPI needed):  python nz_analysis/scripts/nz_task1_prep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nz_analysis/scripts
import numpy as np                                     # noqa: E402
import nz_setup as N                                   # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

# Larger fonts so the maps stay legible when displayed small (the vignette renders
# them at a reduced width) while the high save DPI below keeps the PNGs crisp when
# blown back up on a projector.
plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 23,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
})


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=200, bbox_inches="tight")   # large px/filesize -> projector-crisp
    plt.close(fig)
    os.replace(tmp, path)


def plot_split(inp, path):
    sites = inp["sites_xy"]
    block = inp["block_id"]
    NW_C, SE_C = "red", "dodgerblue"                # bright, saturated -> test pops
    colors = {0: "0.72", 1: NW_C, 2: SE_C}          # train muted grey
    sizes = {0: 34, 1: 110, 2: 150}                 # test dots much larger; SE largest
    labels = {0: "train", 1: "NW ocean / windward", 2: "SE leeward"}
    fig, ax = plt.subplots(figsize=(9, 8))
    N.add_coastline(ax)
    N.add_barrier(ax)
    for b in (0, 1, 2):                              # draw train first, test on top
        m = block == b
        ax.scatter(sites[m, 0], sites[m, 1], s=sizes[b], c=colors[b],
                   zorder=3 if b == 0 else 4,
                   edgecolors="k", linewidths=0.4 if b == 0 else 0.9,
                   label=f"{labels[b]} ({int(m.sum())})")
    # NW holdout rectangle (bold dashed) + SE convex-hull polygon (bold dashed)
    b = N.BLOCK_NW
    ax.add_patch(Rectangle((b["lon_min"], b["lat_min"]),
                 b["lon_max"] - b["lon_min"], b["lat_max"] - b["lat_min"],
                 fill=False, edgecolor=NW_C, lw=2.8, ls="--", zorder=5))
    N.add_se_hull(ax, sites, block, color=SE_C, lw=2.8, ls="--")
    N.set_domain(ax)
    frac = (block > 0).mean()
    ax.set_title(f"NZ train/test split (holdout {int((block>0).sum())}/{block.size} "
                 f"= {frac:.0%})")
    ax.legend(loc="lower right", framealpha=0.9)
    _savefig(fig, path)


def plot_knots(inp, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    N.add_coastline(ax)
    N.add_barrier(ax)
    ax.scatter(inp["sites_xy"][:, 0], inp["sites_xy"][:, 1], s=8, c="0.7", zorder=2,
               label=f"sites ({inp['sites_xy'].shape[0]})")
    ax.scatter(inp["knots_S"][:, 0], inp["knots_S"][:, 1], marker="*", s=170,
               c="tab:red", zorder=4, edgecolors="k", linewidths=0.3,
               label=f"Wendland S/phi ({inp['knots_S'].shape[0]})")
    ax.scatter(inp["knots_rho"][:, 0], inp["knots_rho"][:, 1], marker="X", s=80,
               c="tab:blue", zorder=5, edgecolors="k", linewidths=0.3,
               label=f"Gaussian rho ({inp['knots_rho'].shape[0]})")
    N.set_domain(ax)
    ax.set_title("Knot Layout (barrier = 1500 m; no knots inside barrier)")
    ax.legend(loc="lower right", framealpha=0.9)
    _savefig(fig, path)


def plot_threshold(inp, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    pcm = N.plot_cells(ax, inp, inp["u_vec"], cmap="Reds")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("0.95-quantile threshold (mm, 3-day sum)", fontsize=16)
    cb.ax.tick_params(labelsize=13)
    N.add_coastline(ax)
    N.add_barrier(ax)
    N.set_domain(ax)
    ax.set_title("0.95 Quantile Threshold Surface")
    _savefig(fig, path)


def main():
    os.makedirs(N.OUT, exist_ok=True)
    figdir = os.path.join(N.OUT, "figures")

    inp = N.prepare_inputs(n_cores=int(os.environ.get("NZ_PREP_CORES", "1")))

    # --- validations --------------------------------------------------------
    Y = inp["Y"]
    Ns, Nt = Y.shape
    assert (Ns, Nt) == (309, 3080), f"unexpected Y shape {Y.shape}"
    assert np.isfinite(Y).all(), "non-finite entries in Y"
    assert np.isfinite(inp["u_vec"]).all(), "non-finite threshold"
    assert inp["sigma_spatial"].shape == (Ns, 5)
    assert inp["xi_spatial"].shape == (Ns, 5)
    assert inp["beta_sigma"].shape == (6,) and inp["beta_xi"].shape == (6,)
    n_exc = int((Y > inp["u_vec"][:, None]).sum())
    print(f"[prep] Y {Y.shape}; exceedances {n_exc}; "
          f"threshold [{inp['u_vec'].min():.1f}, {inp['u_vec'].max():.1f}] mm")
    print(f"[prep] full junction tree: {len(inp['full_cliques'])} cliques / "
          f"{len(inp['full_separators'])} separators (separable: OK)")
    print(f"[prep] train junction tree: {len(inp['train_cliques'])} cliques / "
          f"{len(inp['train_separators'])} separators")

    # --- save + figures -----------------------------------------------------
    N.save_inputs(inp)
    print(f"[prep] saved -> {N.INPUTS_PKL} and {N.INPUTS_NPZ}")
    plot_split(inp, os.path.join(figdir, "split.png"))
    plot_knots(inp, os.path.join(figdir, "knots.png"))
    plot_threshold(inp, os.path.join(figdir, "threshold.png"))
    print(f"[prep] figures -> {figdir}/  (REVIEW split.png before fitting)")


if __name__ == "__main__":
    main()
