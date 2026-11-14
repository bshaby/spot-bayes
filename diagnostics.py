"""
Live MCMC diagnostics: multi-panel trace plots written to a PNG at regular
intervals so a run can be monitored (including headless on a cluster).

`plot_traces` renders the log-likelihood plus the global scalars (alpha0, nu,
radius_S) and a few representative knot/coefficient traces (phi_k, rho_k,
beta_sigma, beta_xi). It overwrites the same file each call, so the user just
re-opens it to see the latest progress. A light recent-acceptance summary can be
added when adaptive blocks are passed.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np


def format_hms(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS' (or 'MM:SS' under an hour)."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ProgressMeter:
    """Lightweight wall-clock progress/ETA reporter for the MCMC loops.

    Tracks elapsed time and an exponentially-weighted iteration rate, and emits a
    one-line summary (iter k/N, ll, elapsed, ETA, it/s) at most every
    `every` iterations. Shared by the serial and MPI samplers so both give the
    same feedback. `tag` distinguishes ranks/runs in the printout.
    """

    def __init__(self, total: int, every: int, tag: str = "", start_iter: int = 0):
        self.total = int(total)
        self.every = max(1, int(every))
        self.tag = tag
        self.t0 = time.time()
        self.t_last = self.t0
        self.it_last = start_iter
        self.rate = None              # EWMA of iters/sec

    def update(self, it: int, ll: float, *, force: bool = False) -> None:
        """Call once per iteration; prints only on the `every` cadence (or force)."""
        done = it + 1
        if not (force or done % self.every == 0 or done == self.total):
            return
        now = time.time()
        d_it = max(1, done - self.it_last)
        inst = d_it / max(now - self.t_last, 1e-9)
        self.rate = inst if self.rate is None else 0.7 * self.rate + 0.3 * inst
        self.it_last, self.t_last = done, now
        remaining = (self.total - done) / self.rate if self.rate else float("inf")
        pfx = f"[{self.tag}] " if self.tag else ""
        print(f"{pfx}iter {done:>6d}/{self.total}  ll={ll:.1f}  "
              f"elapsed {format_hms(now - self.t0)}  ETA {format_hms(remaining)}  "
              f"{self.rate:.2f} it/s", flush=True)


def _rep_indices(k, m=4):
    """Up to m evenly-spaced column indices (first/.../last) of a knot block."""
    if k <= m:
        return list(range(k))
    return sorted(set(np.linspace(0, k - 1, m).astype(int).tolist()))


def ess_1d(x) -> float:
    """Effective sample size of a 1-D MCMC trace.

    Uses the normalized autocorrelation (via FFT) and Geyer's initial
    positive-sequence estimator of the integrated autocorrelation time:
        tau = -1 + 2 * sum_k (rho_{2k} + rho_{2k+1})   (truncated at the first
              non-positive pair), and ESS = n / tau.
    Returns n for a (near-)constant or too-short trace.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return float(n)
    x = x - x.mean()
    v = np.dot(x, x) / n
    if not np.isfinite(v) or v <= 0:
        return float(n)
    m = 1 << int(np.ceil(np.log2(2 * n)))      # zero-pad for linear autocorr
    f = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(f * np.conj(f), n=m)[:n].real
    rho = ac / ac[0]
    s, k = 0.0, 0
    while 2 * k + 1 < n:
        pair = rho[2 * k] + rho[2 * k + 1]
        if pair <= 0:
            break
        s += pair
        k += 1
    tau = max(-1.0 + 2.0 * s, 1e-12)
    return float(min(n / tau, float(n)))


def ess_summary(TR: dict, n: int, burn_frac: float = 0.5) -> dict:
    """Per-parameter ESS over the post-burn portion of the traces.

    Scalars give a single ESS; vector blocks (phi_k, rho_k, beta_*) report the
    min and median ESS across their components (the min flags the worst-mixing
    component). `n` is the number of recorded samples so far.
    """
    lo = int(burn_frac * n)
    if n - lo < 8:
        lo = 0
    out = {}
    for key in ("ll", "alpha0", "nu", "radius_S"):
        if key in TR:
            out[key] = ess_1d(np.asarray(TR[key][:n])[lo:])
    for key in ("phi_k", "rho_k", "beta_sigma", "beta_xi"):
        if key in TR:
            arr = np.asarray(TR[key][:n])[lo:]
            if arr.ndim == 2 and arr.shape[1] > 0:
                e = np.array([ess_1d(arr[:, c]) for c in range(arr.shape[1])])
                out[key] = {"min": float(e.min()), "median": float(np.median(e))}
    out["_n_used"] = n - lo
    return out


def print_ess(ess: dict) -> None:
    """Print a compact ESS summary table (e.g. at the end of a run)."""
    print(f"--- Effective sample size (last {ess.get('_n_used', '?')} draws) ---")
    for key in ("ll", "alpha0", "nu", "radius_S"):
        if key in ess:
            print(f"  {key:<12s} {ess[key]:8.1f}")
    for key in ("phi_k", "rho_k", "beta_sigma", "beta_xi"):
        if key in ess:
            print(f"  {key:<12s} min {ess[key]['min']:7.1f}  median {ess[key]['median']:7.1f}")


def plot_traces(TR: dict, n: int, path: str, thin: int = 1,
                iteration: Optional[int] = None, burn_frac: float = 0.0,
                dpi: int = 150, phi_domain: Optional[str] = None) -> None:
    """Write a multi-panel trace plot of the first `n` recorded samples to `path`.

    TR        : the trace dict from run_mcmc / run_mcmc_mpi.
    n         : number of recorded samples so far (rows to plot).
    thin      : recording interval (x-axis is labelled in iterations).
    iteration : current iteration, shown in the title.
    burn_frac : fraction of the shown trace to grey out as burn-in (visual only).
    phi_domain: if given ('unit'/'positive'), the phi_k panel is shown on phi's
                NATURAL scale (inverse logit/exp) instead of the sampled
                unconstrained scale; None keeps the raw transformed values.
    """
    if n < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(n) * thin
    panels = [("log-likelihood", np.asarray(TR["ll"][:n]), None),
              ("alpha0", np.asarray(TR["alpha0"][:n]), None)]
    if "nu" in TR:
        panels.append(("nu", np.asarray(TR["nu"][:n]), None))
    if "radius_S" in TR:
        panels.append(("radius_S", np.asarray(TR["radius_S"][:n]), None))
    for name in ("phi_k", "rho_k"):
        arr = np.asarray(TR[name][:n])
        if arr.ndim == 2 and arr.shape[1] > 0:
            idx = _rep_indices(arr.shape[1])
            label = name
            if name == "phi_k" and phi_domain is not None:  # show phi natural-scale
                import phi_transform as pt
                arr = pt.to_constrained(arr, phi_domain)
                label = "phi"
            panels.append((f"{label} (knots {idx})", arr[:, idx], idx))
    for name in ("beta_sigma", "beta_xi"):
        if name in TR:
            arr = np.asarray(TR[name][:n])
            panels.append((name, arr, list(range(arr.shape[1]))))

    # ESS summary (second half of the shown trace) as a text panel
    ess = ess_summary(TR, n, burn_frac=0.5)

    npanel = len(panels) + 1                    # +1 for the ESS text panel
    ncol = 2
    nrow = (npanel + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow), squeeze=False)
    burn = int(burn_frac * n)
    for ax, (title, data, labels) in zip(axes.ravel(), panels):
        data = np.asarray(data)
        if data.ndim == 1:
            ax.plot(x, data, lw=0.8, color="C0")
        else:
            for c in range(data.shape[1]):
                lab = str(labels[c]) if labels is not None else None
                ax.plot(x, data[:, c], lw=0.7, label=lab)
            if labels is not None and data.shape[1] <= 8:
                ax.legend(fontsize=6, ncol=2, loc="best")
        if burn > 0:
            ax.axvspan(x[0], x[min(burn, n - 1)], color="0.9", zorder=0)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("iteration", fontsize=6)

    # ESS text panel (immediately after the trace panels)
    ess_ax = axes.ravel()[len(panels)]
    ess_ax.axis("off")
    lines = [f"Effective sample size  (last {ess.get('_n_used', n)} draws)", ""]
    for key in ("ll", "alpha0", "nu", "radius_S"):
        if key in ess:
            lines.append(f"  {key:<10s} {ess[key]:8.1f}")
    for key in ("phi_k", "rho_k", "beta_sigma", "beta_xi"):
        if key in ess:
            lines.append(f"  {key:<10s} min {ess[key]['min']:7.1f}  med {ess[key]['median']:7.1f}")
    ess_ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=8, transform=ess_ax.transAxes)

    for ax in axes.ravel()[npanel:]:
        ax.axis("off")
    ttl = "MCMC diagnostics" + (f"  (iter {iteration})" if iteration is not None else "")
    fig.suptitle(ttl, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    # atomic-ish: write to tmp then replace so a reader never sees a half-written file
    import os
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    os.replace(tmp, path)


# --- end-of-run summary ------------------------------------------------------
def _run_TR(run: dict) -> tuple:
    """Map a saved run (sampler.load_run) to a plot_traces-style TR dict + n.
    The saved run stores nu/radius_S as scalars and their traces under *_trace."""
    TR = {"ll": run["ll"], "alpha0": run["alpha0"],
          "phi_k": run["phi_k"], "rho_k": run["rho_k"]}
    for key, trace in (("nu", "nu_trace"), ("radius_S", "radius_S_trace")):
        if trace in run:
            TR[key] = run[trace]
    for key in ("beta_sigma", "beta_xi"):
        if key in run:
            TR[key] = run[key]
    return TR, int(run["phi_k"].shape[0])


def _mean_field(run, which, burn_frac, domain, grid_n):
    """Post-burn posterior-mean phi(s) or rho(s) surface (honours phi_domain)."""
    import utilities as u
    key = "phi_k" if which == "phi" else "rho_k"
    knots = run["knots_phi"] if which == "phi" else run["knots_rho"]
    eff = float(run["eff_range_phi"] if which == "phi" else run["eff_range_rho"])
    trace = run[key]
    mean_k = trace[int(burn_frac * trace.shape[0]):].mean(axis=0)
    xs = np.linspace(domain[0], domain[1], grid_n)
    ys = np.linspace(domain[2], domain[3], grid_n)
    X, Y = np.meshgrid(xs, ys)
    grid = np.c_[X.ravel(), Y.ravel()]
    W = u.build_weight_matrix(grid, knots, kernel="gaussian", radius=eff,
                              bandwidth=eff ** 2 / 6.0, cutoff=False)
    field = W @ mean_k
    phi_dom = getattr(run["config"], "phi_domain", "unit")
    if which == "phi" and phi_dom != "unit":
        field = np.exp(field)
    return X, Y, field.reshape(X.shape)


def plot_run_summary(run, out_prefix: str, truth: Optional[dict] = None,
                     burn_frac: float = 0.5, dpi: int = 150,
                     domain=(0.0, 10.0, 0.0, 10.0), grid_n: int = 40) -> list:
    """Write a comprehensive end-of-run summary for a fitted run.

    Produces two PNGs: `<out_prefix>.traces.png` (multi-panel parameter traces +
    ESS) and `<out_prefix>.summary.png` (post-burn posterior-mean phi/rho
    surfaces + marginal posterior densities of the key scalars). `run` may be a
    loaded-run dict (sampler.load_run) or a path to its .npz. `truth` optionally
    maps parameter name -> scalar truth, drawn as a reference line.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(run, str):
        import sampler as smp
        run = smp.load_run(run)
    truth = truth or {}
    written = []

    # (a) trace panel (reuse plot_traces)
    TR, n = _run_TR(run)
    tp = out_prefix + ".traces.png"
    plot_traces(TR, n, tp, thin=1, burn_frac=burn_frac, dpi=dpi,
                phi_domain=getattr(run.get("config"), "phi_domain", None))
    written.append(tp)

    # (b) surfaces + scalar marginals
    lo = int(burn_frac * n)
    surfaces = [("phi", "phi(s) posterior mean")]
    # rho surface only meaningful with >1 rho knot (skip for HW single-knot)
    if run["knots_rho"].shape[0] > 1:
        surfaces.append(("rho", "rho(s) posterior mean"))
    scalars = []
    for name, src in (("alpha0", "alpha0"), ("nu", "nu_trace"),
                      ("radius_S", "radius_S_trace"),
                      ("aniso_ratio", "aniso_ratio_trace")):
        if src in run:
            scalars.append((name, np.asarray(run[src])[lo:]))

    ncol = max(len(surfaces) + 1, len(scalars))
    fig, axes = plt.subplots(2, ncol, figsize=(4.3 * ncol, 8.0), squeeze=False)
    # row 0: surfaces
    for ax, (which, title) in zip(axes[0], surfaces):
        X, Y, S = _mean_field(run, which, burn_frac, domain, grid_n)
        im = ax.pcolormesh(X, Y, S, shading="auto", cmap="viridis")
        ax.set_title(title, fontsize=10); ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if truth.get(which) is not None:
            cb.ax.axhline(truth[which], color="r", lw=2)
    # ll trace as the last cell of row 0
    ax_ll = axes[0][len(surfaces)]
    ax_ll.plot(np.asarray(run["ll"]), lw=0.7, color="C0")
    ax_ll.axvspan(0, lo, color="0.9", zorder=0)
    ax_ll.set_title("log-likelihood", fontsize=10); ax_ll.tick_params(labelsize=7)
    for ax in axes[0][len(surfaces) + 1:]:
        ax.axis("off")
    # row 1: scalar marginal posteriors (post-burn)
    for ax, (name, vals) in zip(axes[1], scalars):
        ax.hist(vals, bins=40, color="C0", alpha=0.85)
        q = np.percentile(vals, [2.5, 50, 97.5])
        for v in q:
            ax.axvline(v, color="0.3", ls="--", lw=0.8)
        if truth.get(name) is not None:
            ax.axvline(truth[name], color="r", lw=2)
        ax.set_title(f"{name}  (med {q[1]:.3g})", fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in axes[1][len(scalars):]:
        ax.axis("off")
    fig.suptitle("Posterior summary (post-burn)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    sp = out_prefix + ".summary.png"
    tmp = sp + ".tmp.png"
    fig.savefig(tmp, dpi=dpi); plt.close(fig); os.replace(tmp, sp)
    written.append(sp)
    return written


def plot_live_surfaces(state, trace_path, phi_domain=None):
    """Companion to the live trace plot: scatter the CURRENT spatial fields
    phi(s), sigma(s) and xi(s) (the latter two from the sampled GPD-margin betas)
    over the site coordinates, at the present iteration. Written to a sibling of
    `trace_path` (``<base>.surfaces.png``) so it refreshes at the same cadence.

    Generic (site scatter, no basemap), so it is safe for any domain. sigma/xi use
    the first time column (spatially-varying, temporally-constant margins here)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = state.data.sites_xy
    fields = [("phi(s)", np.asarray(state.phi_vec, float))]
    if getattr(state, "sigma_mat", None) is not None:
        fields.append(("sigma(s)", np.asarray(state.sigma_mat[:, 0], float)))
    if getattr(state, "xi_mat", None) is not None:
        fields.append(("xi(s)", np.asarray(state.xi_mat[:, 0], float)))
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), squeeze=False)
    for ax, (title, val) in zip(axes[0], fields):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=18, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.splitext(trace_path)[0] + ".surfaces.png"
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110); plt.close(fig); os.replace(tmp, path)
    return path
