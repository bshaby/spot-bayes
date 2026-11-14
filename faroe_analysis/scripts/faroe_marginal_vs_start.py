#!/usr/bin/env python3
"""
Compare the STAGE-1 marginal fit's posterior-mean GPD parameter surfaces against the
STARTING values that kick off Task 2 (the wang/full dependence fit warm-starts its margins
from the reference workflow betas = marginal_fit.npz's beta_sigma_ref / beta_xi_ref).

Produces a 2x3 figure -- rows = (sigma, xi), columns = (Task-2 start | stage-1 posterior
mean | difference = stage1 - start) -- plus printed summary stats. Everything margin-related
(covariates, the ref betas, the draws) is read from the stage-1 npz so it is exactly what
the run used; only the plotting grid (land mask / lon / lat) comes from faroe_setup.load_raw.

Usage (defaults to <analysis>/output/marginal):
    MARG_OUT=/path/to/output/marginal python faroe_analysis/scripts/faroe_marginal_vs_start.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(_HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (_HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
import marginal as mg                                         # noqa: E402
import faroe_setup as F                                       # noqa: E402

OUT = os.environ.get("MARG_OUT", os.path.join(ANALYSIS, "output", "marginal"))
NPZ = os.path.join(OUT, "marginal_fit.npz")
CFG = os.path.join(OUT, "marginal_fit.config.json")


def _burn_draws(n_draws):
    """warmup // thin recorded draws (clamped to <= half), matching faroe_fit._stage1_betas."""
    warm = 0
    if os.path.exists(CFG):
        c = json.load(open(CFG))
        warm = int(c.get("warmup", 0)) // max(1, int(c.get("thin", 1)))
    return min(warm, n_draws // 2)


def _sigma_surface(beta, Xsp):
    return mg.sigma_from_eta(beta[0] + Xsp @ beta[1:])


def _xi_surface(beta, Xsp):
    return mg.xi_from_eta(beta[0] + Xsp @ beta[1:])


def main():
    if not os.path.exists(NPZ):
        sys.exit(f"[compare] {NPZ} not found; run stage 1 first.")
    d = np.load(NPZ, allow_pickle=True)
    burn = _burn_draws(d["beta_sigma"].shape[0])

    b_sig_post = d["beta_sigma"][burn:].mean(0)
    b_xi_post = d["beta_xi"][burn:].mean(0)
    b_sig_start = d["beta_sigma_ref"]          # = Task-2 warm-start betas
    b_xi_start = d["beta_xi_ref"]
    Xs, Xx = d["sigma_spatial"], d["xi_spatial"]

    sig_start = _sigma_surface(b_sig_start, Xs)
    sig_post = _sigma_surface(b_sig_post, Xs)
    xi_start = _xi_surface(b_xi_start, Xx)
    xi_post = _xi_surface(b_xi_post, Xx)

    inp = F.load_raw()                          # grid geometry for the cell maps
    nwater = int((~inp["land_mask"]).sum())
    assert sig_start.size == nwater, f"site count {sig_start.size} != water cells {nwater}"

    # ---- summary stats to stdout -----------------------------------------
    def _s(name, a, b):
        print(f"[compare] {name:>10}: start med={np.median(a):.3f} "
              f"[{np.percentile(a,5):.3f},{np.percentile(a,95):.3f}]  "
              f"stage1 med={np.median(b):.3f} [{np.percentile(b,5):.3f},{np.percentile(b,95):.3f}]  "
              f"Δ med={np.median(b-a):+.3f} max|Δ|={np.max(np.abs(b-a)):.3f}")
    print(f"[compare] burn {burn} draws dropped; averaging {d['beta_sigma'].shape[0]-burn} draws")
    _s("sigma", sig_start, sig_post)
    _s("xi", xi_start, xi_post)

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    rows = [("sigma", sig_start, sig_post, "viridis"),
            ("xi", xi_start, xi_post, "cividis")]
    col_titles = ["Task-2 start (reference betas)", "Stage-1 posterior mean",
                  "Difference (stage1 - start)"]
    for r, (pname, s_start, s_post, cmap) in enumerate(rows):
        vmin = float(min(s_start.min(), s_post.min()))
        vmax = float(max(s_start.max(), s_post.max()))
        diff = s_post - s_start
        dabs = float(np.max(np.abs(diff))) or 1e-9
        panels = [(s_start, cmap, vmin, vmax), (s_post, cmap, vmin, vmax),
                  (diff, "RdBu_r", -dabs, dabs)]
        for c, (vals, cm, vlo, vhi) in enumerate(panels):
            a = ax[r, c]
            pm = F.plot_cells(a, inp, vals, cmap=cm, vmin=vlo, vmax=vhi)
            F.add_barrier(a)
            F.add_coastline(a)
            F.set_domain(a)
            fig.colorbar(pm, ax=a, fraction=0.046, pad=0.04)
            if r == 0:
                a.set_title(col_titles[c], fontsize=11)
            if c == 0:
                a.set_ylabel(f"{pname}\nLatitude", fontsize=11)
    fig.suptitle("Stage-1 marginal posterior mean vs Task-2 starting margins", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = os.path.join(OUT, "marginal_vs_task2start.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[compare] saved {png}", flush=True)


if __name__ == "__main__":
    main()
