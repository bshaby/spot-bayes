#!/usr/bin/env python3
"""
Compare two ways of forming the stage-1 posterior-mean GPD parameter surfaces:

  A. plug-in    : average the beta draws first, THEN form one surface  -> surface(mean beta)
  B. mean-surface: form a surface at EVERY MCMC draw, THEN average     -> mean_k surface(beta_k)

Because sigma = exp(eta) and xi = scaled-logit(eta) are NONLINEAR, E[surface] != surface(E[beta])
(Jensen): for sigma (convex exp) the mean-of-surfaces is >= the plug-in surface; for xi the sign
varies. This makes a 2x3 figure -- rows (sigma, xi), columns (A: mean beta -> surface | B: mean of
per-draw surfaces | difference B - A) -- plus printed stats. The gap is tiny when the marginal
posterior is tight.

Usage (defaults to <analysis>/output/marginal):
    MARG_OUT=/path/to/output/marginal python faroe_analysis/scripts/faroe_marginal_meansurface.py
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
    warm = 0
    if os.path.exists(CFG):
        c = json.load(open(CFG))
        warm = int(c.get("warmup", 0)) // max(1, int(c.get("thin", 1)))
    return min(warm, n_draws // 2)


def _eta_all(draws, Xsp):
    """Linear predictor for every draw: (K, n_params) x (Ns, p) -> (K, Ns)."""
    return draws[:, 0][:, None] + draws[:, 1:] @ Xsp.T


def main():
    if not os.path.exists(NPZ):
        sys.exit(f"[meansurf] {NPZ} not found; run stage 1 first.")
    d = np.load(NPZ, allow_pickle=True)
    burn = _burn_draws(d["beta_sigma"].shape[0])
    ds = d["beta_sigma"][burn:]                 # (K, 11)
    dx = d["beta_xi"][burn:]                     # (K, 8)
    Xs, Xx = d["sigma_spatial"], d["xi_spatial"]
    K = ds.shape[0]

    # B: per-draw surfaces then average
    sig_meansurf = mg.sigma_from_eta(_eta_all(ds, Xs)).mean(0)     # (Ns,)
    xi_meansurf = mg.xi_from_eta(_eta_all(dx, Xx)).mean(0)
    # A: mean beta then surface
    sig_plugin = mg.sigma_from_eta(ds.mean(0)[0] + Xs @ ds.mean(0)[1:])
    xi_plugin = mg.xi_from_eta(dx.mean(0)[0] + Xx @ dx.mean(0)[1:])

    inp = F.load_raw()
    nwater = int((~inp["land_mask"]).sum())
    assert sig_plugin.size == nwater, f"site count {sig_plugin.size} != water {nwater}"

    def _s(name, a, b):
        dif = b - a
        print(f"[meansurf] {name:>6}: plug-in med={np.median(a):.4f}  mean-surface med={np.median(b):.4f}"
              f"  Δ(B-A) med={np.median(dif):+.4f} max|Δ|={np.max(np.abs(dif)):.4f} "
              f"min={dif.min():+.4f} max={dif.max():+.4f}")
    print(f"[meansurf] averaging {K} post-burn draws")
    _s("sigma", sig_plugin, sig_meansurf)       # expect Δ >= 0 everywhere (Jensen, convex exp)
    _s("xi", xi_plugin, xi_meansurf)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    rows = [("sigma", sig_plugin, sig_meansurf, "viridis"),
            ("xi", xi_plugin, xi_meansurf, "cividis")]
    col_titles = ["A: mean(beta) -> surface", "B: mean of per-draw surfaces",
                  "Difference (B - A)"]
    for r, (pname, a_surf, b_surf, cmap) in enumerate(rows):
        vmin = float(min(a_surf.min(), b_surf.min()))
        vmax = float(max(a_surf.max(), b_surf.max()))
        diff = b_surf - a_surf
        dabs = float(np.max(np.abs(diff))) or 1e-12
        panels = [(a_surf, cmap, vmin, vmax), (b_surf, cmap, vmin, vmax),
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
    fig.suptitle("Posterior-mean margins: plug-in surface(mean beta) vs mean of per-draw surfaces",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = os.path.join(OUT, "marginal_meansurface_vs_plugin.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[meansurf] saved {png}", flush=True)


if __name__ == "__main__":
    main()
