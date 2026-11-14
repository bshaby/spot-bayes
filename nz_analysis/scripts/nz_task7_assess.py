#!/usr/bin/env python3
"""
Task 7 -- model assessment for the Wang model fitted to the FULL dataset (Task 2).

Produces two diagnostics (see nz_diagnostics.py for the methodology):

  1. Site-wise QQ plots against a standard Gumbel margin, at 4 chosen locations.
     Observed exceedances are transformed through the posterior-averaged censored-GPD
     CDF; binomial 95% bands come from the uniform order-statistic (Beta) law.

  2. Extremal-dependence chi(u) of every site RELATIVE TO a few reference locations
     (empirical vs model-based, for u in {0.90, 0.95, 0.98}). Model-based chi is the
     posterior mean over draws, each simulating fresh replicates from the fitted
     process and rank-transforming to uniform margins.

The QQ sites and chi reference locations contrast windward (W-coast, high-exposure)
vs leeward (SE, sheltered-by-the-Alps) exposure; adjust QQ_SITES / CHI_REFS below.

MPI parallelises the (heavier) model-based chi over posterior draws:
    OMP_NUM_THREADS=1 ... mpiexec -n <ranks> python nz_analysis/scripts/nz_task7_assess.py
Serial also works. Outputs -> nz_analysis/output/assess/.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nz_analysis/scripts
import numpy as np                                     # noqa: E402
from sampler import load_run, SamplerSettings          # noqa: E402
import nz_setup as N                                   # noqa: E402
import nz_diagnostics as D                             # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

# (lon, lat) targets -- nearest data site is used. Contrast windward vs leeward.
QQ_SITES = [(167.0, -43.0), (172.5, -43.5), (172.5, -41.0), (169.0, -46.0)]
QQ_LABELS = ["windward (W coast)", "leeward (SE)", "north", "south"]
CHI_REFS = [(167.0, -43.0), (172.5, -43.5), (172.5, -41.0)]
CHI_REF_LABELS = ["windward (W coast)", "leeward (SE)", "north"]
CHI_US = [0.90, 0.95, 0.98]


def _progress(tag, every=10):
    t0 = time.time()

    def cb(b, B):
        if b == 1 or b % every == 0 or b == B:
            print(f"  [{tag}] draw {b}/{B}  {time.time()-t0:5.1f}s", flush=True)
    return cb


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)


def plot_qq(run, inp, sites, labels, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, s, lab in zip(axes.ravel(), sites, labels):
        q = D.gumbel_qq_site(run, s, inp["Y"], inp["u_vec"],
                             inp["sigma_spatial"], inp["xi_spatial"], p_cens=N.P_CENS)
        if q is None:
            ax.set_title(f"{lab}: too few exceedances"); continue
        ax.fill_between(q["z_theory"], q["z_lo"], q["z_hi"], color="0.85",
                        label="95% band", zorder=1)
        lim = [min(q["z_theory"][0], q["z_data"][0]) - 0.3,
               max(q["z_theory"][-1], q["z_data"][-1]) + 0.3]
        ax.plot(lim, lim, "k--", lw=1, zorder=2)
        ax.scatter(q["z_theory"], q["z_data"], s=16, c="tab:green", zorder=3)
        lon, lat = inp["sites_xy"][s]
        ax.set_title(f"{lab}  ({lon:.1f}, {lat:.1f})  n={q['n_exceed']}", fontsize=10)
        ax.set_xlabel("theoretical Gumbel quantile")
        ax.set_ylabel("fitted (observed) Gumbel quantile")
        ax.set_aspect("equal")
    fig.suptitle("Wang model -- site-wise Gumbel QQ plots (95% binomial bands)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, path)


def plot_chi_reference(inp, ref_idx, ref_label, rows, us, path):
    """Rows = ordered [(name, chi_array), ...] (empirical + each model mode);
    cols = thresholds u. Cell maps of chi vs the reference site (marked with a star)."""
    nrow, ncol = len(rows), len(us)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    rlon, rlat = inp["sites_xy"][ref_idx]
    for row, (name, arr) in enumerate(rows):
        for col, uu in enumerate(us):
            ax = axes[row][col]
            pcm = N.plot_cells(ax, inp, arr[:, col], cmap="magma", vmin=0.0, vmax=1.0)
            N.add_barrier(ax)
            N.add_coastline(ax)
            N.set_domain(ax)
            ax.scatter([rlon], [rlat], marker="*", s=260, c="cyan",
                       edgecolors="k", linewidths=0.6, zorder=8)
            ax.set_title(f"{name}  chi(u={uu:.2f})", fontsize=10)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Extremal dependence chi relative to {ref_label} "
                 f"({rlon:.1f}, {rlat:.1f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=150, help="posterior draws for chi")
    ap.add_argument("--n-rep", type=int, default=2000, help="replicates/draw for chi")
    ap.add_argument("--burn", type=float, default=0.5)
    ap.add_argument("--run", default="wang_full.npz",
                    help="run to assess: a basename under the output dir, or a path")
    ap.add_argument("--outdir", default=None,
                    help="output subdir under the output dir (default: 'assess' for "
                         "wang_full.npz, else 'assess_<run-stem>' so runs don't clobber)")
    args = ap.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception:
        comm, rank = None, 0

    rp = args.run if os.path.isabs(args.run) else os.path.join(N.OUT, args.run)
    if not os.path.exists(rp):
        sys.exit(f"Missing Wang full-data fit at {rp}. Run Task 2 first.")
    stem = os.path.basename(rp)[:-4] if rp.endswith(".npz") else os.path.basename(rp)
    if args.outdir is not None:
        sub = args.outdir
    else:
        sub = "assess" if stem == "wang_full" else f"assess_{stem}"
    outdir = os.path.join(N.OUT, sub)
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)

    inp = N.load_inputs()
    run = load_run(rp)

    # reconstruct the Wang config/settings/data used at fit time (for the geometry)
    _, kw = N.models()["wang"]
    # the geometry must be the one the run was FITTED with, so take its saved
    # config (its barrier-kernel visibility rule in particular) rather than
    # today's in-code default
    cfg = run["config"]
    settings = SamplerSettings(**{**N.base_settings_kw(), **kw, "n_iters": N.NITERS})
    data = N.build_data(cfg, inp, "full")

    ref_idx = [D.nearest_site(inp["sites_xy"], ll) for ll in CHI_REFS]
    us = np.array(CHI_US)

    # --- model-based chi (MPI over draws) -----------------------------------
    if rank == 0:
        print(f"[assess] model-based chi (marginal + conditional): {args.n_draws} "
              f"draws x {args.n_rep} reps, {len(ref_idx)} references ...", flush=True)
    mod = D.model_chi(run, cfg, data, settings, ref_idx, us,
                      n_draws=args.n_draws, n_rep=args.n_rep, burn=args.burn,
                      mode="both", comm=comm,
                      progress=_progress("chi") if rank == 0 else None)

    if rank == 0:
        # --- empirical chi (cheap) ------------------------------------------
        emp = np.stack([D.empirical_chi(inp["Y"], ri, us) for ri in ref_idx])
        np.savez(os.path.join(outdir, "chi.npz"), empirical=emp,
                 model_marginal=mod["marginal"], model_conditional=mod["conditional"],
                 ref_idx=np.array(ref_idx), us=us,
                 ref_coords=inp["sites_xy"][ref_idx])
        for r, (ri, lab) in enumerate(zip(ref_idx, CHI_REF_LABELS)):
            rows = [("empirical", emp[r]),
                    ("model (marginal)", mod["marginal"][r]),
                    ("model (conditional)", mod["conditional"][r])]
            plot_chi_reference(inp, ri, lab, rows, us,
                               os.path.join(outdir, f"chi_ref{r}_{lab.split()[0]}.png"))
        # --- QQ plots -------------------------------------------------------
        qq_idx = [D.nearest_site(inp["sites_xy"], ll) for ll in QQ_SITES]
        plot_qq(run, inp, qq_idx, QQ_LABELS, os.path.join(outdir, "qq.png"))
        print(f"[assess] wrote qq.png + chi_ref*.png + chi.npz to {outdir}/", flush=True)
    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":
    main()
