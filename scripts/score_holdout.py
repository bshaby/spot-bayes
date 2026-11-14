#!/usr/bin/env python3
"""
Post-processing: predictive log-likelihood and CRPS on a holdout set, from a
fitted run saved with sampler.save_run.

Two holdout regimes:
  * --new-sites : the holdout locations were NOT in the fitting graph (genuinely
    new coordinates). Uses the joint VisGP new-site predictor (or nonstationary /
    stationary kriging for shi / huser-wadsworth).
  * --in-graph  : the holdout sites were part of the fitting set but masked
    (a miss_mat); uses the in-graph conditional.

Holdout directory must contain:
  Y.npy        (n_holdout, T)  observed responses
  coords.npy   (n_holdout, 2)  locations            [new-sites mode]
  u.npy, sigma.npy, xi.npy     marginal params at holdout sites
  obs_idx.npy, holdout_idx.npy (indices into the fitted set) [in-graph mode]

Example:
  python scripts/score_holdout.py --run run.npz --holdout-dir ./holdout \
      --new-sites --shapefile domain.shp --M 50 --burn 2000 --thin 10 \
      --tail-threshold 60 --tail-smooth 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sampler as smp  # noqa: E402
import predictive as pr  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Score a holdout set from a saved run.")
    p.add_argument("--run", required=True, help="Path to the .npz saved by save_run.")
    p.add_argument("--holdout-dir", required=True)
    p.add_argument("--out", default="./holdout_scores.npz")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-sites", action="store_true")
    mode.add_argument("--in-graph", action="store_true")
    p.add_argument("--shapefile", default=None, help="Domain polygon (wang new-sites).")
    p.add_argument("--M", type=int, default=50, help="Conditional latent draws per posterior draw.")
    p.add_argument("--burn", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--tail-threshold", type=float, default=None)
    p.add_argument("--tail-smooth", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run = smp.load_run(args.run)
    cfg = run["config"]
    draws = smp.build_draws(run, burn=args.burn, thin=args.thin)

    d = args.holdout_dir
    Y = np.load(os.path.join(d, "Y.npy"))
    u = np.load(os.path.join(d, "u.npy"))
    sigma = np.load(os.path.join(d, "sigma.npy"))
    xi = np.load(os.path.join(d, "xi.npy"))
    p_cens = float(run["p_cens"])
    holdout = pr.HoldoutData(Y=Y, u=u, sigma=sigma, xi=xi, p_cens=p_cens)

    if args.new_sites:
        new_coords = np.load(os.path.join(d, "coords.npy"))
        water_poly = None
        if args.shapefile is not None:
            import domain
            water_poly = domain.load_domain(args.shapefile)

        # optional GPD-marginal covariates at the holdout sites: if present, the
        # holdout sigma/xi are derived from the fitted beta per draw.
        def hcov(name):
            pth = os.path.join(d, name)
            return np.load(pth) if os.path.exists(pth) else None
        geom = pr.NewSiteGeometry(
            new_coords=new_coords, train_coords=run["sites_xy"],
            knots_S=run["knots_S"], knots_phi=run["knots_phi"],
            knots_rho=run["knots_rho"], radius_S=float(run["radius_S"]),
            eff_range_phi=float(run["eff_range_phi"]),
            eff_range_rho=float(run["eff_range_rho"]), config=cfg,
            cliques=run.get("cliques"), separators=run.get("separators"),
            water_poly=water_poly, sigsq_Z=1.0,
            sigma_spatial=hcov("sigma_spatial.npy"),
            sigma_temporal=hcov("sigma_temporal.npy"),
            sigma_spatiotemporal=hcov("sigma_spatiotemporal.npy"),
            xi_spatial=hcov("xi_spatial.npy"),
            xi_temporal=hcov("xi_temporal.npy"),
            xi_spatiotemporal=hcov("xi_spatiotemporal.npy"))
        ll = pr.predictive_loglik_new_sites(geom, draws, holdout, M=args.M, seed=args.seed)
        cr = pr.crps_holdout_new_sites(geom, draws, holdout, M=args.M, seed=args.seed,
                                       tail_threshold=args.tail_threshold,
                                       tail_smooth=args.tail_smooth)
    else:  # in-graph
        obs_idx = np.load(os.path.join(d, "obs_idx.npy")).tolist()
        holdout_idx = np.load(os.path.join(d, "holdout_idx.npy")).tolist()
        # covariance per draw: rebuild from rho/nu (VisGP) or dense Matern
        import covariance as cov

        def cov_for_draw(draw):
            if cfg.use_visgp:
                return cov.visgp_precision(run["cliques"], run.get("separators", []),
                                           run["sites_xy"], draw["rho"], draw["nu"],
                                           sparse=True)
            return cov.matern_wang(cov.dist_matrix(run["sites_xy"]),
                                   draw["rho"], 1.0, draw["nu"])
        # in-graph draws need Z_obs at obs_idx
        ig_draws = [dict(Z_obs=dr["Z_train"][obs_idx],
                         phi_holdout=np.full(len(holdout_idx), 0.5),  # placeholder
                         gamma_bar_holdout=np.ones(len(holdout_idx)),
                         R_holdout=np.ones((len(holdout_idx), Y.shape[1])),
                         alpha0=dr["alpha0"]) for dr in draws]
        ll = pr.predictive_loglik(ig_draws, holdout, obs_idx=obs_idx,
                                  holdout_idx=holdout_idx, cov_for_draw=cov_for_draw,
                                  use_visgp=cfg.use_visgp, M=args.M, seed=args.seed)
        cr = pr.crps_holdout(ig_draws, holdout, obs_idx=obs_idx,
                             holdout_idx=holdout_idx, cov_for_draw=cov_for_draw,
                             use_visgp=cfg.use_visgp, M=args.M, seed=args.seed,
                             tail_threshold=args.tail_threshold, tail_smooth=args.tail_smooth)

    out = {"predictive_loglik_total": ll["total"],
           "predictive_loglik_per_draw": ll["per_draw"],
           "crps_vanilla_mean": cr["vanilla_mean"]}
    if "twcrps_mean" in cr:
        out["twcrps_mean"] = cr["twcrps_mean"]
    np.savez(args.out, **out)
    print(f"predictive log-lik (total): {ll['total']:.2f}")
    print(f"CRPS (vanilla mean):        {cr['vanilla_mean']:.4f}")
    if "twcrps_mean" in cr:
        print(f"TW-CRPS (mean):             {cr['twcrps_mean']:.4f}")
    print(f"Saved scores to {args.out}")


if __name__ == "__main__":
    main()
