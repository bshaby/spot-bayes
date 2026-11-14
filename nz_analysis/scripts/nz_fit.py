#!/usr/bin/env python3
"""
Shared MPI fit driver for the NZ analysis (Tasks 2-5).

Fits ONE model (hw / wang / shi) on either the full 309-site dataset or the training
subset. Auto-calibrated checkpointing + automatic resume (relaunch the same command to
continue/extend). On completion writes the run (.npz + streamed latent S/Z) and a
diagnostic summary (trace + ESS + posterior-mean surfaces).

    OMP_NUM_THREADS=1 ... mpiexec -n <ranks> python nz_analysis/scripts/nz_fit.py \
        --model wang --which full --fix-margins

TWO-STAGE (the production path). The task wrappers run_nz_task2-5.sh fix the margins at
the stage-1 marginal run (run_nz_marginal.sh) via --fix-margins + NZ_STAGE1_NPZ, so their
outputs carry the _fixmargins_stage1 suffix:
    Task 2:  --model wang --which full  --fix-margins -> output/wang_full_fixmargins_stage1.*
    Task 3:  --model hw   --which train --fix-margins -> output/train/hw_fixmargins_stage1.*
    Task 4:  --model wang --which train --fix-margins -> output/train/wang_fixmargins_stage1.*
    Task 5:  --model shi  --which train --fix-margins -> output/train/shi_fixmargins_stage1.*
Without --fix-margins the driver instead SAMPLES the margins (warm-started from the
stage-1 posterior mean when NZ_STAGE1_NPZ is set, else the workflow betas).

STARTING VALUES. --plugin-init (ON by default) replaces the cold init (S=1, Z=0, phi=1,
alpha0=5, rho=1) with init_field.plugin_init_snapshot: the transformed data
X = qRW(pCGP(Y)) is inverted into a starting (S, Z) field, and nu/rho are read off the
implied correlogram. On NZ this starts ~2.1M nats higher and puts Z in the Gaussian
typical set instead of at the prior mode -- which is what the long transient was.
Runs using it carry a _plug suffix so they never resume an old cold-start chain.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nz_analysis/scripts
import numpy as np                                                 # noqa: E402
import marginal as mg                                              # noqa: E402
from sampler import SamplerSettings, save_run, load_run            # noqa: E402
from sampler_mpi import run_mcmc_mpi                               # noqa: E402
import diagnostics as dg                                          # noqa: E402
import init_field as ini                                          # noqa: E402
import _common as C                                               # noqa: E402
import nz_setup as N                                              # noqa: E402
from mpi4py import MPI                                            # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
CKPT_HOURS = float(os.environ.get("NZ_CKPT_HOURS", "0.5"))


def _stage1_load():
    """(npz, burn) for the NZ_STAGE1_NPZ stage-1 fit, or None if unset. `burn` is the
    warmup-equivalent recorded-draw count to drop (warmup // thin from the sibling config
    json, clamped to <= half the draws)."""
    path = os.environ.get("NZ_STAGE1_NPZ")
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"NZ_STAGE1_NPZ={path} not found (run stage 1 first: "
                         f"bash nz_analysis/scripts/run_nz_marginal.sh).")
    d = np.load(path, allow_pickle=True)
    warm = 0
    cfgp = path[:-4] + ".config.json" if path.endswith(".npz") else path + ".config.json"
    if os.path.exists(cfgp):
        import json
        c = json.load(open(cfgp))
        warm = int(c.get("warmup", 0)) // max(1, int(c.get("thin", 1)))
    warm = min(warm, int(d["beta_sigma"].shape[0]) // 2)
    return d, warm


def _stage1_betas():
    """Posterior-mean (beta_sigma, beta_xi) from the stage-1 fit -- the warm start for a
    SAMPLED-margin run, and (for a fixed-margin run) the coefficients whose
    surface(mean beta) fixes the margins. (None, None) if NZ_STAGE1_NPZ is unset."""
    loaded = _stage1_load()
    if loaded is None:
        return None, None
    d, warm = loaded
    return d["beta_sigma"][warm:].mean(0), d["beta_xi"][warm:].mean(0)


def _paths(model, which, fix_margins=False, fix_radius=None, init_radius=None,
           stage1=False, plugin=False):
    if which == "full":
        outdir = N.OUT
        name = f"{model}_full"
    else:
        outdir = os.path.join(N.OUT, "train")
        name = model
    if fix_margins:
        name += "_fixmargins"          # never clobber the sampled-margin fit
    if stage1:
        name += "_stage1"              # stage-1-seeded margins: distinct from ref-beta run
    if plugin:
        name += "_plug"                # plug-in starting field: distinct from a cold start
    if fix_radius is not None:
        name += f"_fixr{fix_radius:g}".replace(".", "p")
    elif init_radius is not None:
        name += f"_r0{init_radius:g}".replace(".", "p")
    return outdir, name, os.path.join(outdir, f"{name}.npz")


def fit(model, which, fix_margins=False, fix_radius=None, init_radius=None,
        plugin_init=True, phi0=None, alpha0=None, nu0=None, rho0=None):
    cfg, model_kw = N.models()[model]
    b_s1, b_x1 = _stage1_betas()                # stage-1 posterior mean, or (None, None)
    stage1 = b_s1 is not None
    if fix_margins:
        # Hold the GPD margins at the stage-1 posterior-mean surface instead of sampling
        # them. This is the two-stage scheme's "defensible fixed margins" path: the
        # margins come from a clean standalone marginal analysis, and only the dependence
        # field (Z/S/phi/rho/nu/alpha0/geometry) adapts to them. In a SAMPLED-margin fit
        # the betas and the latent field weld to each other through the nugget X ~= X*,
        # and the pair can settle far from a calibrated margin with no single-block move
        # able to come back (see the Faroe analysis).
        model_kw = {**model_kw, "update_GPD_sigma": False, "update_GPD_xi": False}
    if fix_radius is not None or init_radius is not None:
        # The Wendland support radius CANNOT be sampled across basins. R = W(r) S with
        # S >= 0 (Levy), so R lives in the CONE of the Wendland basis with nonnegative
        # coefficients, and that cone changes with r -- a solo radius move costs tens of
        # thousands of nats and is always rejected, so the chain simply keeps whatever
        # basin it was INITIALISED in. Hence: set the radius, don't ask the sampler to
        # find it. NB ModelState raises the requested radius to its covering value if
        # some site would otherwise be isolated (NZ: 1.0 -> ~2.6).
        model_kw = {**model_kw,
                    "radius_S": float(fix_radius if fix_radius is not None
                                      else init_radius)}
        if fix_radius is not None:
            model_kw["update_radius_S"] = False
    outdir, name, rp = _paths(model, which, fix_margins, fix_radius, init_radius,
                              stage1, plugin_init)
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        if stage1:
            print(f"[{name}] margins {'FIXED at' if fix_margins else 'seeded from'} the "
                  f"stage-1 posterior mean ({os.environ['NZ_STAGE1_NPZ']})", flush=True)
    comm.Barrier()

    inp = N.load_inputs()
    data = N.build_data(cfg, inp, which)
    idx = inp["full_idx"] if which == "full" else inp["train_idx"]

    fixed_betas = None
    if fix_margins and not stage1:
        # No stage 1 to pin to, so the margins stay at the workflow's own fitted surface
        # (data.sigma_vec/xi_vec, = surface(inp["beta_*"]) -- verified identical). Record
        # those coefficients anyway: update_GPD is off, so the sampler writes no beta
        # trace, and without one Tasks 6/7 cannot form margins at held-out sites.
        fixed_betas = (np.asarray(inp["beta_sigma"], float),
                       np.asarray(inp["beta_xi"], float))
        if rank == 0:
            print(f"[{name}] no NZ_STAGE1_NPZ: margins fixed at the WORKFLOW surface "
                  f"(run run_nz_marginal.sh for the two-stage path)", flush=True)
    elif fix_margins and stage1:
        # Fix the margins at the stage-1 posterior-mean surface, surface(mean beta). With
        # update_GPD off the sampler reads data.sigma_vec/xi_vec (ModelState.__init__), so
        # overwrite them here. Also stash the (constant) betas to RECORD in the run: a
        # per-site fixed surface has no coefficient representation, and new-site prediction
        # (predictive._holdout_marginal) needs beta_sigma/beta_xi to form margins at
        # held-out sites. NZ margins are spatial-only, so sigma/xi are constant in time.
        Xs, Xx = inp["sigma_spatial"][idx], inp["xi_spatial"][idx]
        data.sigma_vec = mg.sigma_from_eta(b_s1[0] + Xs @ b_s1[1:])
        data.xi_vec = mg.xi_from_eta(b_x1[0] + Xx @ b_x1[1:])
        fixed_betas = (np.asarray(b_s1, float), np.asarray(b_x1, float))
        if rank == 0:
            print(f"[{name}] fixed margins = stage-1 surface(mean beta); betas recorded "
                  f"for new-site prediction", flush=True)

    settings_kw = {**N.base_settings_kw(), **model_kw,
                   "latent_out": rp,
                   "plot_every": max(N.THIN, N.NITERS // 100),
                   "plot_path": os.path.join(outdir, f"{name}.trace.png"),
                   "checkpoint_path": os.path.join(outdir, f"{name}.ckpt"),
                   "resume": True}

    # idempotent auto-extend: skip if already fitted to NITERS.
    if os.path.exists(rp):
        with np.load(rp) as d:
            done = int(np.asarray(d["ll"]).size) * N.THIN
        ckpt_global = os.path.join(outdir, f"{name}.ckpt.global.pkl")
        if done >= N.NITERS:
            if rank == 0:
                print(f"[{name}] already fitted ~{done} >= {N.NITERS}; skipping.",
                      flush=True)
            return rp
        if not os.path.exists(ckpt_global):
            if rank == 0:
                print(f"[{name}] final run exists (~{done}) but no checkpoint to "
                      f"extend; delete {os.path.basename(rp)} to refit.", flush=True)
            return rp
        if rank == 0:
            print(f"[{name}] extending ~{done} -> {N.NITERS} iters.", flush=True)

    # Starting values (ignored on checkpoint resume). Every rank builds its own snapshot;
    # both builders are deterministic, so the ranks agree bit-for-bit.
    st_full = SamplerSettings(**{**settings_kw, "n_iters": N.NITERS})
    if plugin_init:
        snap, info = ini.plugin_init_snapshot(
            cfg, data, st_full,
            phi0=N.PHI0 if phi0 is None else phi0,
            alpha0=N.ALPHA0 if alpha0 is None else alpha0,
            nu=nu0, rho=rho0, beta_sigma=b_s1, beta_xi=b_x1,
            return_info=True, verbose=(rank == 0))
        if rank == 0:
            print(f"[{name}] plug-in start: phi={info['phi0']} alpha0={info['alpha0']} "
                  f"nu={info['nu']:.3f} rho={info['rho']:.3f} "
                  f"radius_S={info['radius_S']:.3f}", flush=True)
    else:
        snap = N.init_snapshot_betas(cfg, data, st_full, inp, idx,
                                     beta_sigma=b_s1, beta_xi=b_x1)

    # auto-calibrate the checkpoint interval to ~CKPT_HOURS
    def _cal(nit):
        run_mcmc_mpi(cfg, data, SamplerSettings(
            **{**settings_kw, "n_iters": nit, "save_latent": False,
               "stream_latent": False, "latent_out": None, "plot_every": 0,
               "checkpoint_every": 0, "resume": False, "checkpoint_path": None}),
            init_snapshot=snap)
    ck_every = C.calibrate_checkpoint_every(_cal, comm, target_hours=CKPT_HOURS)
    if rank == 0:
        print(f"[{name}] checkpoint_every={ck_every} iters (~{CKPT_HOURS} h)", flush=True)

    TR = run_mcmc_mpi(cfg, data, SamplerSettings(
        **{**settings_kw, "n_iters": N.NITERS, "checkpoint_every": ck_every}),
        verbose=True, init_snapshot=snap)

    if rank == 0:
        if fixed_betas is not None:
            # record the constant stage-1 betas per draw so build_draws / predictive.
            # _holdout_marginal can form margins at held-out sites (update_GPD was off, so
            # the sampler recorded no beta trace of its own).
            n_save = TR["ll"].shape[0]
            TR["beta_sigma"] = np.tile(fixed_betas[0], (n_save, 1))
            TR["beta_xi"] = np.tile(fixed_betas[1], (n_save, 1))
        save_run(rp, TR, cfg, st_full, data)
        try:
            paths = dg.plot_run_summary(rp, os.path.join(outdir, f"{name}.summary"))
            print(f"[{name}] done -> {rp}; summary: "
                  f"{', '.join(os.path.basename(p) for p in paths)}", flush=True)
        except Exception as e:
            print(f"[{name}] done -> {rp}; summary plot failed: {e}", flush=True)
    comm.Barrier()
    return rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hw", "wang", "shi"], required=True)
    ap.add_argument("--which", choices=["full", "train"], required=True)
    ap.add_argument("--fix-margins", action="store_true",
                    help="hold the GPD margins at the STAGE-1 posterior-mean surface "
                         "(NZ_STAGE1_NPZ) instead of sampling them; writes "
                         "<name>_fixmargins_stage1.*")
    ap.add_argument("--no-plugin-init", dest="plugin_init", action="store_false",
                    help="cold-start (S=1, Z=0, phi=1, alpha0=5, rho=1) instead of the "
                         "plug-in starting field. Drops the _plug name suffix.")
    ap.set_defaults(plugin_init=os.environ.get("NZ_PLUGIN_INIT", "1") == "1")
    ap.add_argument("--phi0", type=float, default=None,
                    help=f"starting tail exponent for the plug-in init "
                         f"(default NZ_PHI0={N.PHI0}). 0.5 is the asymptotic "
                         f"independence/dependence boundary.")
    ap.add_argument("--alpha0", type=float, default=None,
                    help=f"starting nugget tail parameter (default NZ_ALPHA0={N.ALPHA0}); "
                         f"must exceed alpha0_min=2.")
    ap.add_argument("--nu0", type=float, default=None,
                    help="pin the starting Matern smoothness instead of estimating it "
                         "from the plug-in correlogram.")
    ap.add_argument("--rho0", type=float, default=None,
                    help="pin the starting Matern range instead of estimating it from "
                         "the plug-in correlogram.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fix-radius", type=float, metavar="R",
                   help="PIN radius_S at R (update_radius_S=False). The radius cannot "
                        "be sampled across basins, so profile over it: writes "
                        "<name>_fixr<R>.*")
    g.add_argument("--init-radius", type=float, metavar="R",
                   help="START radius_S at R but keep sampling it -- tests whether a "
                        "large-radius basin is self-sustaining. Writes <name>_r0<R>.*")
    args = ap.parse_args()
    if rank == 0:
        bits = []
        if args.fix_margins:
            # which surface it is depends on NZ_STAGE1_NPZ; fit() prints that once it knows
            bits.append("margins FIXED (not sampled)")
        if args.plugin_init:
            bits.append("plug-in starting field")
        if args.fix_radius is not None:
            bits.append(f"radius_S PINNED at {args.fix_radius:g}")
        elif args.init_radius is not None:
            bits.append(f"radius_S STARTED at {args.init_radius:g} (still sampled)")
        print(f"Fitting {args.model} on {args.which} data: "
              f"{N.NITERS} iters, thin {N.THIN}, z={N.Z_SAMPLER}"
              f"{('; ' + ', '.join(bits)) if bits else ''}.", flush=True)
    fit(args.model, args.which, fix_margins=args.fix_margins,
        fix_radius=args.fix_radius, init_radius=args.init_radius,
        plugin_init=args.plugin_init, phi0=args.phi0, alpha0=args.alpha0,
        nu0=args.nu0, rho0=args.rho0)


if __name__ == "__main__":
    main()
