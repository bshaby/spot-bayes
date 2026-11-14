#!/usr/bin/env python3
"""
Shared MPI fit driver for the Faroe analysis (Tasks 2-5).

Fits ONE model (hw / wang / shi) on either the full 432-site dataset or the
training subset, warm-starting the GPD margins from the workflow's fitted betas.
Auto-calibrated checkpointing + automatic resume (relaunch the same command to
continue/extend). On completion writes the run (.npz + streamed latent S/Z) and a
diagnostic summary (trace + ESS + posterior-mean surfaces).

    OMP_NUM_THREADS=1 ... mpiexec -n <ranks> python faroe_analysis/scripts/faroe_fit.py \
        --model wang --which full

Task wrappers (run_faroe_task2-5.sh) fix the margins at the marginal run via
--fix-margins + FAROE_STAGE1_NPZ, so their outputs carry the _fixmargins_stage1 suffix:
    Task 2:  --model wang --which full  --fix-margins -> output/wang_full_fixmargins_stage1.*
    Task 3:  --model hw   --which train --fix-margins -> output/train/hw_fixmargins_stage1.*
    Task 4:  --model wang --which train --fix-margins -> output/train/wang_fixmargins_stage1.*
    Task 5:  --model shi  --which train --fix-margins -> output/train/shi_fixmargins_stage1.*
Without --fix-margins the driver instead SAMPLES the margins (warm-started from the
workflow betas) and drops the suffix (e.g. wang_full.*).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # faroe_analysis/scripts
import numpy as np                                                 # noqa: E402
import marginal as mg                                              # noqa: E402
from sampler import SamplerSettings, save_run, load_run            # noqa: E402
from sampler_mpi import run_mcmc_mpi                               # noqa: E402
import diagnostics as dg                                          # noqa: E402
import init_field as ini                                          # noqa: E402
import _common as C                                               # noqa: E402
import faroe_setup as F                                           # noqa: E402
from mpi4py import MPI                                            # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
CKPT_HOURS = float(os.environ.get("FAROE_CKPT_HOURS", "0.5"))


def _stage1_load():
    """(npz, burn) for the FAROE_STAGE1_NPZ stage-1 fit, or None if unset. `burn` is the
    warmup-equivalent recorded-draw count to drop (warmup // thin from the sibling config
    json, clamped to <= half the draws)."""
    path = os.environ.get("FAROE_STAGE1_NPZ")
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"FAROE_STAGE1_NPZ={path} not found (run stage 1 first).")
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
    SAMPLED-margin / recursive run, and (for a fixed-margin run) the coefficients whose
    surface(mean beta) fixes the margins. (None, None) if FAROE_STAGE1_NPZ is unset."""
    loaded = _stage1_load()
    if loaded is None:
        return None, None
    d, warm = loaded
    return d["beta_sigma"][warm:].mean(0), d["beta_xi"][warm:].mean(0)


def _paths(model, which, fix_margins=False, fix_radius=None, init_radius=None,
           stage1=False, beta_sampler=None, warm=False, plugin=False):
    if which == "full":
        outdir = F.OUT
        name = f"{model}_full"
    else:
        outdir = os.path.join(F.OUT, "train")
        name = model
    if plugin:
        name += "_plug"                # plug-in starting field (init_field.py)
    if warm:
        name += "_warm"                # goodinit-warm-started full-model (sampled margins)
    if beta_sampler == "recursive":
        name += "_recursive"           # two-stage recursive-Bayes run: distinct output
    if fix_margins:
        name += "_fixmargins"          # never clobber the sampled-margin fit
    if stage1:
        name += "_stage1"              # stage-1-seeded margins: distinct from ref-beta run
    if fix_radius is not None:
        name += f"_fixr{fix_radius:g}".replace(".", "p")
    elif init_radius is not None:
        name += f"_r0{init_radius:g}".replace(".", "p")
    return outdir, name, os.path.join(outdir, f"{name}.npz")


def fit(model, which, fix_margins=False, fix_radius=None, init_radius=None,
        beta_sampler=None, warm_field=None, shi_rho_npz=None, plugin_init=False,
        phi0=0.5, alpha0=None):
    cfg, model_kw = F.models()[model]
    if warm_field is not None and fix_margins:
        raise SystemExit("--warm-field warm-starts the FULL (sampled-margin) model; "
                         "do not combine with --fix-margins.")
    if warm_field is not None and plugin_init:
        raise SystemExit("--plugin-init and --warm-field are two different starting "
                         "fields; pick one.")
    b_s1, b_x1 = _stage1_betas()                # stage-1 posterior mean, or (None, None)
    stage1 = b_s1 is not None and warm_field is None   # warm start seeds betas from goodinit
    if beta_sampler is not None:
        model_kw = {**model_kw, "beta_sampler": beta_sampler}
    if beta_sampler == "recursive":
        npz = os.environ.get("FAROE_STAGE1_NPZ")
        if not npz:
            raise SystemExit("--beta-sampler recursive requires FAROE_STAGE1_NPZ "
                             "(the stage-1 marginal_fit.npz) -- run stage 1 first.")
        # recursive proposes beta from the stage-1 posterior; margins stay SAMPLED, so
        # do not combine with --fix-margins.
        model_kw = {**model_kw, "recursive_prior_npz": npz}
    if fix_margins:
        # Hold the GPD margins at the workflow's 2-stage betas (which are calibrated:
        # pooled exceedance rates 1.00/1.00/0.94 at u=0.96/0.98/0.99) instead of
        # sampling them. In the sampled-margin fit the betas drift far off that
        # calibration (median sigma 1.27 -> 0.43, median xi -0.17 -> +0.05, exceedance
        # ratio at u=0.99 -> 2.67) and the latent field then locks in around them, so
        # no single-block move can come back. Pinning the margins changes the TARGET,
        # forcing Z/S/phi/radius_S to adapt to a calibrated margin rather than the
        # reverse -- which is the only way to test whether the short-range dependence
        # is caused by the margin drift.
        model_kw = {**model_kw, "update_GPD_sigma": False, "update_GPD_xi": False}
    if fix_radius is not None or init_radius is not None:
        # The Wendland support radius CANNOT be sampled across basins. R = W(r) S with
        # S >= 0 (Levy), so R lives in the CONE of the Wendland basis with nonnegative
        # coefficients, and that cone changes with r: the fitted (spatially rough) R is
        # 38% / 62% unreachable inside the r=3.8 / r=7.6 cones (nonneg least squares).
        # No reparameterisation of S can carry the field across, so a solo radius move
        # costs ~51k nats and is always rejected -- and the chain simply keeps whatever
        # basin it was INITIALISED in (Faroe started at 1.0 and never exceeded 1.91 in
        # 75k iters; exp3-shi started at 3.0 and roamed 2.97-5.72). Hence: set the
        # radius, don't ask the sampler to find it.
        model_kw = {**model_kw,
                    "radius_S": float(fix_radius if fix_radius is not None
                                      else init_radius)}
        if fix_radius is not None:
            model_kw["update_radius_S"] = False
    outdir, name, rp = _paths(model, which, fix_margins, fix_radius, init_radius, stage1,
                              beta_sampler, warm=warm_field is not None,
                              plugin=plugin_init)
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        if stage1:
            print(f"[{name}] margins seeded from stage-1 posterior mean "
                  f"({os.environ['FAROE_STAGE1_NPZ']})", flush=True)
        if warm_field is not None:
            print(f"[{name}] warm-started from goodinit field {warm_field}"
                  + (f" (+ shi rho from {shi_rho_npz})" if model == "shi" else ""), flush=True)
    comm.Barrier()

    inp = F.load_inputs()
    data = F.build_data(cfg, inp, which)
    idx = inp["full_idx"] if which == "full" else inp["train_idx"]

    fixed_betas = None
    if fix_margins and stage1:
        # Fix the margins at the stage-1 posterior-mean surface, surface(mean beta). With
        # update_GPD off the sampler reads data.sigma_vec/xi_vec (ModelState.__init__), so
        # overwrite them here. Also stash the (constant) betas to RECORD in the run: a
        # per-site fixed surface has no coefficient representation, and new-site prediction
        # (predictive._holdout_marginal) needs beta_sigma/beta_xi to form margins at
        # held-out sites. Faroe margins are spatial-only, so sigma/xi are constant in time.
        Xs, Xx = inp["sigma_spatial"][idx], inp["xi_spatial"][idx]
        data.sigma_vec = mg.sigma_from_eta(b_s1[0] + Xs @ b_s1[1:])
        data.xi_vec = mg.xi_from_eta(b_x1[0] + Xx @ b_x1[1:])
        fixed_betas = (np.asarray(b_s1, float), np.asarray(b_x1, float))
        if rank == 0:
            print(f"[{name}] fixed margins = stage-1 surface(mean beta); betas recorded "
                  f"for new-site prediction", flush=True)

    settings_kw = {**F.base_settings_kw(), **model_kw,
                   "latent_out": rp,
                   "plot_every": max(F.THIN, F.NITERS // 100),
                   "plot_path": os.path.join(outdir, f"{name}.trace.png"),
                   "checkpoint_path": os.path.join(outdir, f"{name}.ckpt"),
                   "resume": True}

    # idempotent auto-extend: skip if already fitted to NITERS.
    if os.path.exists(rp):
        with np.load(rp) as d:
            done = int(np.asarray(d["ll"]).size) * F.THIN
        ckpt_global = os.path.join(outdir, f"{name}.ckpt.global.pkl")
        if done >= F.NITERS:
            if rank == 0:
                print(f"[{name}] already fitted ~{done} >= {F.NITERS}; skipping.",
                      flush=True)
            return rp
        if not os.path.exists(ckpt_global):
            if rank == 0:
                print(f"[{name}] final run exists (~{done}) but no checkpoint to "
                      f"extend; delete {os.path.basename(rp)} to refit.", flush=True)
            return rp
        if rank == 0:
            print(f"[{name}] extending ~{done} -> {F.NITERS} iters.", flush=True)

    # warm start (ignored on checkpoint resume). --warm-field: seed the FULL state (Z field
    # + all params) from the goodinit run; else just the margins (stage-1 or workflow betas).
    st_full = SamplerSettings(**{**settings_kw, "n_iters": F.NITERS})
    if warm_field is not None:
        from faroe_warmstart import build_warm_snapshot
        snap = build_warm_snapshot(model, cfg, data, st_full, inp, warm_field, idx=idx,
                                   shi_rho_npz=shi_rho_npz)
    elif plugin_init:
        # Dataset-generic plug-in ("X-inversion") start: invert X = qRW(pCGP(Y)) into a
        # starting (S, Z) field and read nu/rho off its correlogram, instead of the cold
        # S=1/Z=0 prior mode. See init_field.py; measured +2.1M nats on NZ.
        snap = ini.plugin_init_snapshot(
            cfg, data, st_full, phi0=phi0, alpha0=alpha0,
            beta_sigma=b_s1 if b_s1 is not None else inp["beta_sigma"],
            beta_xi=b_x1 if b_x1 is not None else inp["beta_xi"],
            verbose=(rank == 0))
    else:
        snap = F.init_snapshot_betas(cfg, data, st_full, inp, idx,
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
        **{**settings_kw, "n_iters": F.NITERS, "checkpoint_every": ck_every}),
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
                    help="hold the GPD betas at the workflow's calibrated 2-stage "
                         "values instead of sampling them; writes <name>_fixmargins.*")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fix-radius", type=float, metavar="R",
                   help="PIN radius_S at R (update_radius_S=False). The radius cannot "
                        "be sampled across basins, so profile over it: writes "
                        "<name>_fixr<R>.*")
    g.add_argument("--init-radius", type=float, metavar="R",
                   help="START radius_S at R but keep sampling it -- tests whether a "
                        "large-radius basin is self-sustaining. Writes <name>_r0<R>.*")
    ap.add_argument("--beta-sampler", default=None,
                    choices=["component", "nuts", "barker", "recursive"],
                    help="override the GPD-margin coefficient sampler. 'recursive' is the "
                         "two-stage recursive-Bayes STUB (raises NotImplementedError until "
                         "stage 2 is implemented; see docs/two_stage_recursive_bayes.md).")
    ap.add_argument("--plugin-init", action="store_true",
                    help="start from the dataset-generic plug-in ('X-inversion') field "
                         "(init_field.py) instead of the cold S=1/Z=0 prior mode: the "
                         "starting (S, Z) are read off X = qRW(pCGP(Y)) and nu/rho off "
                         "the implied correlogram. Writes <name>_plug.*. Not with "
                         "--warm-field.")
    ap.add_argument("--phi0", type=float, default=0.5,
                    help="starting tail exponent for --plugin-init (default 0.5, the "
                         "asymptotic independence/dependence boundary).")
    ap.add_argument("--alpha0", type=float, default=None,
                    help="starting nugget tail parameter for --plugin-init (default "
                         "alpha0_min + 0.05 = 2.05).")
    ap.add_argument("--warm-field", default=None, metavar="NPZ",
                    help="warm-start the FULL (sampled-margin) model from the goodinit "
                         "field (goodinit_field.npz from reassemble_goodinit.py): seeds the "
                         "Z field + all params, writes <name>_warm.*. Not with --fix-margins.")
    ap.add_argument("--shi-rho-npz", default=os.path.join(F.OUT, "shi_fixmargins_stage1.npz"),
                    metavar="NPZ", help="for --model shi --warm-field: the run whose rho_k "
                         "tail-mean seeds the 13-knot rho field (default: shi_fixmargins_stage1).")
    args = ap.parse_args()
    if rank == 0:
        bits = []
        if args.fix_margins:
            bits.append("margins FIXED at the 2-stage betas")
        if args.plugin_init:
            bits.append("plug-in starting field")
        if args.fix_radius is not None:
            bits.append(f"radius_S PINNED at {args.fix_radius:g}")
        elif args.init_radius is not None:
            bits.append(f"radius_S STARTED at {args.init_radius:g} (still sampled)")
        print(f"Fitting {args.model} on {args.which} data: "
              f"{F.NITERS} iters, thin {F.THIN}, z={F.Z_SAMPLER}"
              f"{('; ' + ', '.join(bits)) if bits else ''}.", flush=True)
    fit(args.model, args.which, fix_margins=args.fix_margins,
        fix_radius=args.fix_radius, init_radius=args.init_radius,
        beta_sampler=args.beta_sampler, warm_field=args.warm_field,
        shi_rho_npz=args.shi_rho_npz, plugin_init=args.plugin_init,
        phi0=args.phi0, alpha0=args.alpha0)


if __name__ == "__main__":
    main()
