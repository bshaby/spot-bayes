"""
Shared helpers for the simulation-study experiment scripts (exp1_* and exp2_*:
setup/fit/compare). Knot setup (Shi's config), train/test site split, MPI
checkpoint-interval auto-calibration to a wall-time target, holdout scoring
(predictive log-lik + vanilla/TW-CRPS) for all 3 models, and surface plots.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # faroe_analysis/scripts
import simulate as sm                     # noqa: E402
import utilities as u                     # noqa: E402
import predictive as pr                   # noqa: E402
from config import PHI_UNIT               # noqa: E402


# -- geometry -----------------------------------------------------------------
def shi_knots(domain=(0.0, 10.0, 0.0, 10.0), grid=9):
    """Shi's isometric outer+inner knot grids (same grid count for S/phi/rho)."""
    k = sm.make_isometric_knots(*domain, grid)
    return k, k.copy(), k.copy()           # knots_S, knots_phi, knots_rho


def split_sites(n, frac=0.10, seed=0):
    """Deterministic train/test site split (same on every MPI rank)."""
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(frac * n)))
    test = np.sort(rng.choice(n, size=n_test, replace=False))
    train = np.setdiff1d(np.arange(n), test)
    return train, test


# -- MPI checkpoint-interval calibration --------------------------------------
def calibrate_checkpoint_every(run_fn, comm, target_hours=2.0, n_cal=10):
    """Time a short throwaway fit and return checkpoint_every for ~target_hours.

    run_fn(n_iters) -> runs the (warm) MPI sampler for n_iters with no
    checkpointing. Returns an int interval (>= 50), broadcast to all ranks.
    """
    run_fn(2)                              # warm numba / caches
    t0 = time.time()
    run_fn(n_cal)
    per_iter = (time.time() - t0) / n_cal
    per_iter = comm.bcast(per_iter, root=0)
    return max(50, int(round(target_hours * 3600.0 / max(per_iter, 1e-6))))


# -- holdout scoring (all 3 models, new-site prediction) ----------------------
def _thin_to(nrec, burn, max_draws, last_n=None):
    """Post-burn indices thinned to at most `max_draws` (None -> keep all).

    If `last_n` is given it OVERRIDES the `burn` fraction: keep the final `last_n`
    recorded draws. For runs of different length this starts at a different
    absolute iteration but keeps the same-length tail (a convergence knob -- use
    the well-mixed end of each chain regardless of total run length).
    """
    lo = max(0, nrec - int(last_n)) if last_n else int(burn * nrec)
    avail = nrec - lo
    step = 1 if (not max_draws or avail <= max_draws) else int(np.ceil(avail / max_draws))
    return lo, step


def score_holdout(run_path, geom, holdout, tail_threshold, M=50, burn=0.5,
                  max_draws=None, seed=0, progress=None, comm=None, last_n=None,
                  regions=None):
    """Predictive log-lik + vanilla/TW-CRPS at the held-out sites for one run.

    Uses predictive.score_holdout_all: the (expensive) per-draw new-site kriging
    is computed ONCE and reused for log-lik AND both CRPS weightings -- ~3x fewer
    kriging solves than the old three-pass version. `max_draws` thins the
    post-burn posterior draws to at most that many (speed/accuracy knob).

    geom : predictive.NewSiteGeometry for the held-out coords.
    holdout : predictive.HoldoutData (Y/u/sigma/xi at the held-out sites).
    regions : optional {name: LOCAL site indices into `holdout`}. Scores every named
        region from ONE pass and returns {name: <score dict>} instead of one dict --
        the pooled set and its blocks for the price of the pooled set alone.
    """
    import sampler as smp
    run = smp.load_run(run_path)
    nrec = run["phi_k"].shape[0]
    lo, step = _thin_to(nrec, burn, max_draws, last_n=last_n)
    draws = smp.build_draws(run, burn=lo, thin=step)
    bw = 0.25 * np.std(holdout.Y)
    res = pr.score_holdout_all(geom, draws, holdout, M=M, seed=seed,
                               tail_threshold=tail_threshold, tail_smooths=(bw,),
                               progress=progress, comm=comm, regions=regions)

    def _flat(r):
        # `_pt` carries the per-TIME contributions. They cost nothing to keep and are
        # the only way to attach a standard error afterwards: the scores are sums/means
        # over T exchangeable time replicates, and without the per-replicate values a
        # score is a single number with no measurable uncertainty. Recovering them
        # otherwise means re-running the whole pass.
        return dict(loglik=r["loglik"]["total"], crps=r["crps_vanilla_mean"],
                    twcrps_hard=r["twcrps_hard_mean"],
                    twcrps_smooth=r["twcrps_smooth_mean"][bw],
                    n_draws=r["n_draws"],
                    _pt=dict(loglik=np.asarray(r["loglik"]["per_time"], float),
                             crps=np.asarray(r["crps_per_time"]["vanilla"], float),
                             twcrps_smooth=np.asarray(
                                 r["crps_per_time"][f"twcrps_smooth_{bw:g}"], float)))

    if regions:
        return {nm: _flat(r) for nm, r in res["by_region"].items()}
    return _flat(res)


def waic(run_path, geom, train, burn=0.5, max_draws=None, seed=0,
         progress=None, comm=None, last_n=None):
    """Per-(site,time) CONDITIONAL WAIC on the TRAINING data for one run.

    Post-processing only (no effect on fitting): reuses the saved per-draw latent
    (Z, S) via build_draws. `geom` is the same predictive.NewSiteGeometry used for
    holdout scoring (its train_coords/cliques define the training-site geometry);
    `train` is a predictive.HoldoutData holding the TRAINING Y/u/sigma/xi. Returns
    dict with waic/lppd/p_waic.
    """
    import sampler as smp
    run = smp.load_run(run_path)
    nrec = run["phi_k"].shape[0]
    lo, step = _thin_to(nrec, burn, max_draws, last_n=last_n)
    draws = smp.build_draws(run, burn=lo, thin=step)
    res = pr.waic_conditional(geom, draws, train, seed=seed,
                              progress=progress, comm=comm)
    return dict(waic=res["waic"], lppd=res["lppd"], p_waic=res["p_waic"],
                p_waic_max=res["p_waic_max"], n_pwaic_bad=res["n_pwaic_bad"],
                _pt_waic=np.asarray(res["elpd_per_time"], float))


def write_table(rows, path, title=""):
    """Write a model-comparison table to CSV and return a printable string.

    Columns are the union of a fixed base set and any optional keys present in the
    rows (e.g. WAIC), so callers can attach extra scores without changing this."""
    lead = ["model"] + (["region"] if any("region" in r for r in rows) else [])
    base = ["loglik", "crps", "twcrps_hard", "twcrps_smooth", "n_draws"]
    extra = [k for k in ("waic", "lppd", "p_waic", "p_waic_max", "n_pwaic_bad")
             if any(k in r for r in rows)]
    keys = lead + extra + base
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    lines = [title, "  ".join(f"{k:>13s}" for k in keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None:
                cells.append(f"{'':>13s}")               # missing (e.g. waic on a block row)
            elif isinstance(v, str):
                cells.append(f"{v:>13s}")
            else:
                cells.append(f"{v:13.3f}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


# -- surfaces (posterior-mean phi / rho over a grid) --------------------------
def _grid(domain=(0.0, 10.0, 0.0, 10.0), n=40):
    xs = np.linspace(domain[0], domain[1], n)
    ys = np.linspace(domain[2], domain[3], n)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel()], X, Y


def posterior_mean_surface(run_path, which, eff_range, phi_domain=None,
                           domain=(0.0, 10.0, 0.0, 10.0), n=40):
    """Posterior-mean phi(s) or rho(s) surface on a grid from a saved run.

    which : 'phi' (uses phi_k trace + Gaussian smoother) or 'rho' (rho_k trace).
    phi_domain : defaults to the run's saved config (so positive-domain runs are
    exp-transformed correctly); pass explicitly only to override.
    Returns (X, Y, surface) for a heatmap; None if the run lacks that block
    (e.g. rho surface for a model with a scalar range).
    """
    import sampler as smp
    run = smp.load_run(run_path)
    if phi_domain is None:
        phi_domain = getattr(run["config"], "phi_domain", PHI_UNIT)
    key = "phi_k" if which == "phi" else "rho_k"
    knots = run["knots_phi"] if which == "phi" else run["knots_rho"]
    trace = run[key]
    nrec = trace.shape[0]
    mean_k = trace[nrec // 2:].mean(axis=0)           # post-burn posterior mean
    grid, X, Y = _grid(domain, n)
    bw = eff_range ** 2 / 6.0
    W = u.build_weight_matrix(grid, knots, kernel="gaussian",
                              radius=eff_range, bandwidth=bw, cutoff=False)
    field = W @ mean_k
    if which == "phi" and phi_domain != PHI_UNIT:
        field = np.exp(field)
    return X, Y, field.reshape(X.shape)


def plot_surface_panel(panels, path, suptitle="", truth=None):
    """panels: list of (title, X, Y, surface). truth: optional scalar truth line
    shown in the colorbar label. Writes a PNG (Agg, headless-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4), squeeze=False)
    for ax, (title, X, Y, S) in zip(axes[0], panels):
        im = ax.pcolormesh(X, Y, S, shading="auto", cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if truth is not None and truth.get(title.split()[0].lower()) is not None:
            cb.ax.axhline(truth[title.split()[0].lower()], color="r", lw=2)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110); plt.close(fig); os.replace(tmp, path)
