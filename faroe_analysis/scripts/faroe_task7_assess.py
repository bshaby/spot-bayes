#!/usr/bin/env python3
"""
Task 7 -- model assessment for the Wang model fitted to the FULL dataset (Task 2).

Produces two diagnostics (see faroe_diagnostics.py for the methodology):

  1. Site-wise QQ plots against a standard Gumbel margin, at 4 chosen locations.
     Observed exceedances are transformed through the posterior-averaged censored-GPD
     CDF; binomial 95% bands come from the uniform order-statistic (Beta) law.

  2. Extremal-dependence chi(u) of every site RELATIVE TO a few reference locations
     (empirical vs model-based, for u in CHI_US). BOTH sides use the fitted marginal
     CDF -- the model field is thresholded at the draw's qRW_i(u), the observations
     are pushed through the draw's censored-GPD CDF -- and both are recomputed at
     every posterior draw, so chi.npz carries 2.5/97.5% bands for each. Levels must
     exceed the censoring probability p_cens; see faroe_diagnostics.chi_assess.

The QQ sites and chi reference locations are chosen to contrast open-ocean vs
sheltered (lee-of-island) exposure; adjust QQ_SITES / CHI_REFS below as desired.

MPI parallelises the (heavier) model-based chi over posterior draws:
    OMP_NUM_THREADS=1 ... mpiexec -n <ranks> python faroe_analysis/scripts/faroe_task7_assess.py
Serial also works. Outputs -> faroe_analysis/output/assess/.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # faroe_analysis/scripts
import numpy as np                                     # noqa: E402
from sampler import load_run, SamplerSettings          # noqa: E402
import faroe_setup as F                                # noqa: E402
import faroe_diagnostics as D                          # noqa: E402

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

# (lon, lat) targets -- nearest data site is used. Contrast exposures.
QQ_SITES = [(-8.5, 60.6), (-6.0, 62.0), (-7.0, 63.5), (-6.8, 60.8)]
QQ_LABELS = ["open ocean (SW)", "lee of island (E)", "north", "south"]
CHI_REFS = [(-8.5, 60.6), (-6.0, 62.0), (-7.2, 63.4), (-8.35, 62.0)]
CHI_REF_LABELS = ["open ocean (SW)", "lee of island (E)", "north",
                  "open ocean (W)"]
# The 4th ref mirrors Wang (2026a) Fig. 2 top-left panel: open ocean due-west of
# the island, roughly midway between the island's western edge (barrier min lon
# -7.7) and the western domain edge (MIN_LON -9.0), at the island's latitude band.
# Must all exceed F.P_CENS (=0.95): the fitted censored-GPD margin is unidentified
# below the censoring threshold. At Nt=1656 these leave ~66/33/17 empirical
# exceedances at a reference site; 0.999 would leave ~2 and is not usable without
# the moving-window pair pooling that GPD_ScaleMixture does (we have none).
CHI_US = [0.96, 0.98, 0.99]


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
                             inp["sigma_spatial"], inp["xi_spatial"], p_cens=F.P_CENS,
                             sigma_fixed=inp["sigma_vec"], xi_fixed=inp["xi_vec"])
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
    """Rows = ordered [(name, chi_array), ...] (empirical + each model construction);
    cols = thresholds u. Cell maps of chi vs the reference site (marked w/ a star)."""
    nrow, ncol = len(rows), len(us)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    rlon, rlat = inp["sites_xy"][ref_idx]
    for row, (name, arr) in enumerate(rows):
        for col, uu in enumerate(us):
            ax = axes[row][col]
            pcm = F.plot_cells(ax, inp, arr[:, col], cmap="Reds", vmin=0.0, vmax=1.0)
            F.add_barrier(ax)
            F.add_coastline(ax)
            F.set_domain(ax)
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
    ap.add_argument("--run", default="wang_full_fixmargins_stage1",
                    help="fit to assess (basename under FAROE_OUT, no .npz). Default is "
                         "the Task-2 fixed-margin fit; pass wang_full for a sampled-margin "
                         "fit. Results go to assess_<run> so runs never overwrite each other.")
    args = ap.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception:
        comm, rank = None, 0

    outdir = os.path.join(F.OUT, "assess" if args.run == "wang_full"
                          else f"assess_{args.run}")
    rp = os.path.join(F.OUT, f"{args.run}.npz")
    if not os.path.exists(rp):
        sys.exit(f"Missing fit at {rp}. Run Task 2 first"
                 f"{' (with --fix-margins)' if 'fixmargins' in args.run else ''}.")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)

    inp = F.load_inputs()
    run = load_run(rp)

    # reconstruct the Wang config/settings/data used at fit time (for the geometry)
    _, kw = F.models()["wang"]
    # the geometry must be the one the run was FITTED with, so take its saved
    # config (its barrier-kernel visibility rule in particular) rather than
    # today's in-code default
    cfg = run["config"]
    if "fixmargins" in args.run:                       # mirror faroe_fit --fix-margins
        kw = {**kw, "update_GPD_sigma": False, "update_GPD_xi": False}
    settings = SamplerSettings(**{**F.base_settings_kw(), **kw, "n_iters": F.NITERS})
    data = F.build_data(cfg, inp, "full")

    ref_idx = [D.nearest_site(inp["sites_xy"], ll) for ll in CHI_REFS]
    us = np.array(CHI_US)

    # --- chi: model AND empirical, both per-draw on the fitted-CDF scale ------
    if rank == 0:
        print(f"[assess] chi (fitted-CDF margins): {args.n_draws} draws x "
              f"{args.n_rep} reps, {len(ref_idx)} references, u={CHI_US} ...",
              flush=True)
    chi = D.chi_assess(run, cfg, data, settings, inp["Y"], inp["u_vec"],
                       inp["sigma_spatial"], inp["xi_spatial"], F.P_CENS,
                       ref_idx, us,
                       # a --fix-margins fit records no beta traces; it used exactly
                       # these (calibrated) 2-stage surfaces, so the empirical chi must
                       # too. Ignored when the run has sampled betas.
                       sigma_fixed=inp["sigma_vec"], xi_fixed=inp["xi_vec"],
                       n_draws=args.n_draws, n_rep=args.n_rep,
                       burn=args.burn, comm=comm,
                       progress=_progress("chi") if rank == 0 else None)
    emp, mod = chi["emp_mean"], chi["model_mean"]
    mod_pz, mod_c = chi["model_priorz_mean"], chi["model_cond_mean"]

    if rank == 0:
        np.savez(os.path.join(outdir, "chi.npz"),
                 empirical=emp, model=mod, model_priorz=mod_pz, model_cond=mod_c,
                 empirical_lo=chi["emp_lo"], empirical_hi=chi["emp_hi"],
                 model_lo=chi["model_lo"], model_hi=chi["model_hi"],
                 model_priorz_lo=chi["model_priorz_lo"],
                 model_priorz_hi=chi["model_priorz_hi"],
                 model_cond_lo=chi["model_cond_lo"], model_cond_hi=chi["model_cond_hi"],
                 n_draws_used=chi["n_draws_used"], p_cens=F.P_CENS,
                 ref_idx=np.array(ref_idx), us=us,
                 ref_coords=inp["sites_xy"][ref_idx])
        for r, (ri, lab) in enumerate(zip(ref_idx, CHI_REF_LABELS)):
            rows = [("empirical", emp[r]),
                    ("model (marginal)", mod[r]),
                    ("model (prior Z, post. S)", mod_pz[r]),
                    ("model (conditional)", mod_c[r])]
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
