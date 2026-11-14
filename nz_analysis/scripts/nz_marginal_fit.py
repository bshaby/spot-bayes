#!/usr/bin/env python3
"""
Stage-1 marginal fit for the NZ analysis: a standalone Bayesian GPD regression on the
threshold exceedances, conditionally independent given the GPD parameters (NO spatial-
dependence field). This is the first stage of the two-stage scheme and the source of the
fixed margins that Tasks 2-5 hold frozen.

It reuses nz_setup.load_raw() for the 3-day-sum field Y (309 x 3080), the per-site 0.95
threshold u_vec, the standardized sigma/xi covariates (intercept dropped -- the design
adds its own), and the reference workflow's fitted coefficients as the NUTS warm start.
The model, sampler and gradient live in the top-level marginal_model.py; this script is
just the NZ wiring + output. Mirrors faroe_analysis/scripts/faroe_marginal_fit.py.

Serial, no MPI (the likelihood is cheap: 5% of cells are exceedances -> 47,589 terms).
Deterministic given MARG_SEED.

Env overrides (production defaults in parens):
  MARG_OUT ($NZ_OUT/marginal, else <analysis>/output/marginal)   MARG_NITERS (20000)
  MARG_WARMUP (2000)   MARG_THIN (5)   MARG_SEED (20260813)   MARG_TARGET_ACCEPT (0.8)
  MARG_SIGMA_BETA_SD (5.0)   MARG_XI_BETA_SD (5.0)   MARG_CKPT_EVERY (1000)
  MARG_FORCE (0 -> refuse to clobber)
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))            # nz_analysis/scripts
ANALYSIS = os.path.dirname(_HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (_HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                            # noqa: E402
import marginal as mg                                         # noqa: E402
import marginal_model as mm                                   # noqa: E402
import nz_setup as N                                          # noqa: E402


def _envf(k, d):
    return float(os.environ.get(k, d))


def _envi(k, d):
    return int(os.environ.get(k, d))


# Default under NZ_OUT so the marginal co-locates with the fits on Alpine scratch
# (all output on one knob); falls back to the repo-local output/ when NZ_OUT is unset.
OUT = os.environ.get(
    "MARG_OUT",
    os.path.join(os.environ.get("NZ_OUT", os.path.join(ANALYSIS, "output")), "marginal"))
NITERS = _envi("MARG_NITERS", 20000)
WARMUP = _envi("MARG_WARMUP", 2000)
THIN = _envi("MARG_THIN", 5)
SEED = _envi("MARG_SEED", 20260813)
TARGET_ACCEPT = _envf("MARG_TARGET_ACCEPT", 0.8)
SIGMA_BETA_SD = _envf("MARG_SIGMA_BETA_SD", 5.0)
XI_BETA_SD = _envf("MARG_XI_BETA_SD", 5.0)
# Snapshot/checkpoint cadence: every CKPT_EVERY iters write the npz + png and a resume
# checkpoint (0 disables). Makes long runs inspectable AND crash-safe: re-running the same
# command CONTINUES from the last checkpoint (extends if MARG_NITERS is larger).
CKPT_EVERY = _envi("MARG_CKPT_EVERY", 1000)
FORCE = os.environ.get("MARG_FORCE", "0") == "1"

NPZ = os.path.join(OUT, "marginal_fit.npz")
CFG = os.path.join(OUT, "marginal_fit.config.json")
CKPT = os.path.join(OUT, "marginal_fit.ckpt.pkl")


def build_data():
    """MarginalData over all 309 sites with the NZ covariates (spatial-only)."""
    raw = N.load_raw()
    Ns, Nt = raw["Y"].shape
    sd = mg.MarginalDesign(Ns, Nt, spatial=raw["sigma_spatial"])   # (Ns, 5) -> 6 params
    xd = mg.MarginalDesign(Ns, Nt, spatial=raw["xi_spatial"])      # (Ns, 5) -> 6 params
    data = mm.MarginalData(Y=raw["Y"], u_vec=raw["u_vec"], sigma_design=sd, xi_design=xd,
                           sigma_beta_sd=SIGMA_BETA_SD, xi_beta_sd=XI_BETA_SD)
    return data, raw


def _emit(res, raw, b_sig0, b_xi0):
    """Write the inspectable artifacts (npz + config json + trace png) from a (possibly
    partial) fit result. Used as the fit's on_checkpoint callback, so a long run leaves an
    up-to-date npz/png every MARG_CKPT_EVERY iters."""
    np.savez(NPZ, beta_sigma=res["beta_sigma"], beta_xi=res["beta_xi"],
             accept=res["accept"], depth=res["depth"],
             beta_sigma_ref=b_sig0, beta_xi_ref=b_xi0,
             sigma_spatial=raw["sigma_spatial"], xi_spatial=raw["xi_spatial"],
             u_vec=raw["u_vec"], sites_xy=raw["sites_xy"])
    with open(CFG, "w") as fh:
        json.dump(dict(niters=NITERS, warmup=WARMUP, thin=THIN, seed=SEED,
                       target_accept=TARGET_ACCEPT, sigma_beta_sd=SIGMA_BETA_SD,
                       xi_beta_sd=XI_BETA_SD, n_exc=int(res["n_exc"]),
                       n_sigma=int(res["n_sigma"]), n_xi=int(res["n_xi"])), fh, indent=2)
    _diagnostic_plot(res, WARMUP, NITERS)


def main():
    if FORCE:
        for p in (NPZ, CFG, CKPT):
            if os.path.exists(p):
                os.remove(p)
    elif os.path.exists(NPZ) and not os.path.exists(CKPT):
        # a finished (pre-checkpoint) run with no resume state -> don't silently clobber
        sys.exit(f"[marginal] {NPZ} exists with no checkpoint; set MARG_FORCE=1 to refit.")
    os.makedirs(OUT, exist_ok=True)

    data, raw = build_data()
    b_sig0 = np.asarray(raw["beta_sigma"], float)      # reference warm start (len 6)
    b_xi0 = np.asarray(raw["beta_xi"], float)          # (len 6)
    assert b_sig0.size == data.n_sigma and b_xi0.size == data.n_xi, \
        "reference beta length != design n_params"

    print(f"[marginal] Ns={data.Y.shape[0]} Nt={data.Y.shape[1]} exceedances={data.n_exc} "
          f"(params sigma={data.n_sigma} xi={data.n_xi}); niters={NITERS} warmup={WARMUP} "
          f"thin={THIN} seed={SEED} ckpt_every={CKPT_EVERY}", flush=True)

    res = mm.fit(data, b_sig0, b_xi0, n_iters=NITERS, warmup=WARMUP, thin=THIN,
                 seed=SEED, target_accept=TARGET_ACCEPT, progress=True,
                 checkpoint_path=CKPT, checkpoint_every=CKPT_EVERY, resume=True,
                 on_checkpoint=lambda r: _emit(r, raw, b_sig0, b_xi0))

    # post-warmup summary (draws are pre-thinned; drop the warmup portion by count)
    keep = min(res["beta_sigma"].shape[0] * WARMUP // max(1, NITERS),
               res["beta_sigma"].shape[0] - 1)
    bs, bx = res["beta_sigma"][keep:], res["beta_xi"][keep:]
    print("[marginal] posterior mean beta_sigma:", np.round(bs.mean(0), 4))
    print("[marginal]     reference beta_sigma:", np.round(b_sig0, 4))
    print("[marginal] posterior mean beta_xi:   ", np.round(bx.mean(0), 4))
    print("[marginal]     reference beta_xi:   ", np.round(b_xi0, 4))
    print(f"[marginal] accept(last)~{res['accept'][-500:].mean():.3f} "
          f"depth(last)~{res['depth'][-500:].mean():.2f}", flush=True)
    print(f"[marginal] saved {NPZ}", flush=True)


def _diagnostic_plot(res, warmup, niters):
    """Trace panels for beta_sigma / beta_xi + accept/depth. Best-effort (skips if no
    matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # noqa: BLE001
        print(f"[marginal] skipping plot ({e})", flush=True)
        return
    bs, bx = res["beta_sigma"], res["beta_xi"]
    it = np.arange(bs.shape[0]) * res["thin"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for k in range(bs.shape[1]):
        ax[0, 0].plot(it, bs[:, k], lw=0.6)
    ax[0, 0].set_title("beta_sigma traces"); ax[0, 0].axvline(warmup, color="k", ls=":")
    for k in range(bx.shape[1]):
        ax[0, 1].plot(it, bx[:, k], lw=0.6)
    ax[0, 1].set_title("beta_xi traces"); ax[0, 1].axvline(warmup, color="k", ls=":")
    ax[1, 0].plot(res["accept"], lw=0.4); ax[1, 0].set_title("NUTS accept_stat")
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].plot(res["depth"], lw=0.4); ax[1, 1].set_title("NUTS tree depth")
    fig.tight_layout()
    png = os.path.join(OUT, "marginal_fit.trace.png")
    fig.savefig(png, dpi=110); plt.close(fig)
    print(f"[marginal] saved {png}", flush=True)


if __name__ == "__main__":
    main()
