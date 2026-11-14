#!/usr/bin/env python3
"""
Entry point: fit one of the three models to a dataset.

Examples
--------
  # Shi scale-aware model on a simulated dataset directory
  python scripts/run_sampler.py --model shi --data-dir ./data --n-iters 5000

  # Wang non-convex model with VisGP + barrier kernels
  python scripts/run_sampler.py --model wang --use-visgp --use-barrier-kernels \
      --shapefile domain.shp --cliques cliques.rds --data-dir ./data

  # Huser-Wadsworth with phi in (0, inf)
  python scripts/run_sampler.py --model huser-wadsworth --phi-domain positive --data-dir ./data

A data directory must contain: Y.npy (Ns,Nt; NaN = missing), sites_xy.npy (Ns,2),
u.npy (Ns,), sigma.npy (Ns,), xi.npy (Ns,). Optional: knots_S.npy etc.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import add_config_args, config_from_args  # noqa: E402
from sampler import SamplerData, SamplerSettings, run_mcmc, save_run  # noqa: E402
import domain  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Fit a spatial-extremes model.")
    add_config_args(p)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="./mcmc_out.npz")
    p.add_argument("--n-iters", type=int, default=5000)
    p.add_argument("--adapt-size", type=int, default=50)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--mpi", action="store_true",
                   help="Use the MPI sampler (launch with mpirun -n P).")
    p.add_argument("--save-latent", dest="save_latent",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Record the latent S/Z traces (needed for holdout "
                        "scoring). Use --no-save-latent to save disk + memory "
                        "when you only need parameter posteriors.")
    p.add_argument("--latent-float32", dest="latent_float32", action="store_true",
                   help="Store latent traces as float32 (halves their size).")
    p.add_argument("--stream-latent", dest="stream_latent", action="store_true",
                   help="Stream latent traces to disk (memmap) instead of holding "
                        "them in RAM -- decouples runtime memory from chain length.")
    p.add_argument("--latent-chunk", type=int, default=50,
                   help="Flush the streamed latent trace to disk every N saves.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write a resumable checkpoint every N iters (0 = off).")
    p.add_argument("--checkpoint-path", default=None,
                   help="Checkpoint file path (default: <out>.ckpt).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint at --checkpoint-path.")
    p.add_argument("--plot-every", type=int, default=0,
                   help="Write a diagnostic trace plot every N iters (0 = off).")
    p.add_argument("--plot-path", default=None,
                   help="Diagnostic plot PNG path (default: <out>.trace.png).")
    # --- GPD-marginal regression (optional) ---------------------------------
    g = p.add_argument_group("GPD marginal regression")
    g.add_argument("--update-gpd-sigma", action="store_true")
    g.add_argument("--update-gpd-xi", action="store_true")
    p.add_argument("--update-nu", action="store_true",
                   help="Update the Matern smoothness nu.")
    p.add_argument("--update-radius-s", dest="update_radius_s", action="store_true",
                   help="Update the Wendland support radius for R's kernel.")
    for par in ("sigma", "xi"):
        for grp in ("spatial", "temporal", "spatiotemporal"):
            g.add_argument(f"--{par}-{grp}", dest=f"{par}_{grp}", default=None,
                           help=f".npy covariates for {par} ({grp}). spatial=(Ns,p), "
                                f"temporal=(Nt,p), spatiotemporal=(Ns,Nt,p).")
    args = p.parse_args()

    cfg = config_from_args(args)
    cfg.n_cores = args.n_cores

    d = args.data_dir
    def load(name, required=True):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return np.load(path)
        if required:
            raise FileNotFoundError(path)
        return None

    def load_cov(p):
        return np.load(p) if p else None

    Y = load("Y.npy")
    data = SamplerData(
        Y=Y, sites_xy=load("sites_xy.npy"),
        u_vec=load("u.npy"), sigma_vec=load("sigma.npy"), xi_vec=load("xi.npy"),
        knots_S=load("knots_S.npy", required=False),
        knots_phi=load("knots_phi.npy", required=False),
        knots_rho=load("knots_rho.npy", required=False),
        sigma_spatial=load_cov(args.sigma_spatial),
        sigma_temporal=load_cov(args.sigma_temporal),
        sigma_spatiotemporal=load_cov(args.sigma_spatiotemporal),
        xi_spatial=load_cov(args.xi_spatial),
        xi_temporal=load_cov(args.xi_temporal),
        xi_spatiotemporal=load_cov(args.xi_spatiotemporal))

    if cfg.use_visgp:
        data.cliques = domain.load_cliques(cfg.cliques_path)
        sep_path = cfg.cliques_path.replace("cliques", "separators")
        data.separators = domain.load_cliques(sep_path) if os.path.exists(sep_path) else []
    if cfg.use_barrier_kernels:
        data.water_poly = domain.load_domain(cfg.shapefile_path)

    st = SamplerSettings(n_iters=args.n_iters, adapt_size=args.adapt_size,
                         thin=args.thin, nu=args.nu, seed=args.seed,
                         save_latent=args.save_latent,
                         latent_dtype=("float32" if args.latent_float32 else "float64"),
                         stream_latent=args.stream_latent, latent_out=args.out,
                         latent_chunk=args.latent_chunk,
                         update_GPD_sigma=args.update_gpd_sigma,
                         update_GPD_xi=args.update_gpd_xi,
                         update_nu=args.update_nu,
                         update_radius_S=args.update_radius_s,
                         checkpoint_every=args.checkpoint_every,
                         checkpoint_path=(args.checkpoint_path or (args.out + ".ckpt")),
                         resume=args.resume,
                         plot_every=args.plot_every,
                         plot_path=(args.plot_path or (args.out + ".trace.png")))
    if args.save_latent:
        n_save = len(range(0, args.n_iters, args.thin))
        Ns, T = Y.shape
        bytes_ = n_save * Ns * T * (4 if args.latent_float32 else 8)
        print(f"Saving latent Z trace (~{bytes_/1e6:.0f} MB; needed for holdout "
              f"scoring). Use --no-save-latent to skip.")
    else:
        print("Not saving latent traces (--no-save-latent): holdout scoring "
              "will require a refit with --save-latent.")
    if args.mpi:
        from sampler_mpi import run_mcmc_mpi
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        TR = run_mcmc_mpi(cfg, data, st, verbose=True)
        if rank != 0:
            return                       # only rank 0 holds the traces
    else:
        TR = run_mcmc(cfg, data, st, verbose=True)

    # full run artifact (traces + geometry) for prediction / scoring
    save_run(args.out, TR, cfg, st, data)
    print(f"Saved run to {args.out} (+ .config.json"
          + (", .cliques.pkl" if data.cliques is not None else "") + ")")


if __name__ == "__main__":
    main()
