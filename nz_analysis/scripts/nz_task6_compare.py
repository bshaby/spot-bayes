#!/usr/bin/env python3
"""
Task 6 -- model comparison on the held-out sites (run AFTER Tasks 3/4/5).

Loads the three training-set fits (hw / wang / shi) and scores each at the held-out
sites (NW open-ocean windward + SE leeward blocks) with:
  * predictive log-score (higher = better),
  * CRPS and tail-weighted CRPS (hard + smooth; lower = better),
  * conditional WAIC on the training data (lower = better).

Holdout marginal parameters are derived PER POSTERIOR DRAW from each model's
sampled GPD betas applied to the held-out sites' covariates (predictive.py handles
this when the geometry carries sigma_spatial/xi_spatial and the draw carries
beta_sigma/beta_xi). WAIC uses each model's posterior-mean marginal surface on the
training sites (the WAIC path conditions on fixed margins).

ONE PASS PER MODEL. The pooled test set and the per-block breakdown all come from a
single scoring pass. The block log-scores are not ADDITIVE (the score is a joint
density; the Monte-Carlo logsumexp over latent-field draws does not distribute over the
site-sum) but they ARE EXTRACTABLE, because conditional on the latent field the sites
are independent -- so the per-site log-densities the pass already computes can be summed
over any index set. That halves the job.

CRASH SAFETY. Each (model, region) score is appended to model_comparison.partial.jsonl
and the CSV rewritten the moment it is computed, so a failure loses only the block in
flight; re-running RESUMES. This matters more here than at Faroe: NZ runs under a Slurm
wall-time limit, and task 6 is one-shot in alpine_nz.sbatch, so a job that overruns its
window used to lose everything. Now a resubmission continues. Use --fresh to ignore the
journal.

MPI: the per-draw kriging is parallelised across ranks (rank-independent scores).
    OMP_NUM_THREADS=1 ... mpiexec -n <ranks> python nz_analysis/scripts/nz_task6_compare.py
Options: -B/--draws (cap posterior draws), --M (inner draws), --burn, --last-n,
    --suffix / --suffix-<model> (per-model fit selection; 'skip' drops a model),
    --no-blocks, --qq-per-block, --no-waic.
Writes nz_analysis/output/compare/model_comparison.csv (+ bar-chart + oos_qq figures)
on rank 0.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # nz_analysis/scripts
import numpy as np                                                 # noqa: E402
from mpi4py import MPI                                             # noqa: E402
from sampler import load_run                                       # noqa: E402
import predictive as pr                                            # noqa: E402
import marginal as mg                                              # noqa: E402
import _common as C                                                # noqa: E402
import nz_setup as N                                               # noqa: E402
import nz_diagnostics as D                                         # noqa: E402

MODELS = ("hw", "wang", "shi")
MODEL_COLORS = {"hw": "tab:gray", "wang": "tab:green", "shi": "tab:blue"}
# test-set holdout blocks (nz_setup.split_sites_nz ids) -> per-block scores
BLOCKS = {"NW": 1, "SE": 2}


def _progress(tag, every=25):
    t0 = time.time()

    def cb(b, B):
        if b == 1 or b % every == 0 or b == B:
            el = time.time() - t0
            print(f"  [{tag}] draw {b}/{B}  elapsed {el:5.1f}s  "
                  f"ETA {el/b*(B-b):5.1f}s", flush=True)
    return cb


def _jsonable(o):
    """numpy scalars/arrays -> JSON, preserving int-ness so n_draws stays an int."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _row_key(model, region):
    return f"{model}|{region}"


def _load_partial(path):
    """{key: row} from the append-only journal; a later record for a key WINS.

    That is what lets WAIC be attached by appending rather than rewriting. A torn
    final line (the process died mid-write) is skipped rather than fatal -- losing the
    last block is the situation we are already in, and refusing to start would throw
    away every block before it."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "model" in r and "region" in r:
                out[_row_key(r["model"], r["region"])] = r
    return out


def _append_partial(path, row):
    """Append one result and force it to disk NOW -- buffered output is exactly what
    goes missing when Slurm kills a job at its wall-time limit."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=_jsonable) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _postmean_margin(run, inp, idx):
    """Per-site (sigma, xi) for the GLOBAL site indices `idx`.

    Sampled-margin runs: posterior-mean betas applied to those sites' covariates.

    FIXED-margin runs never sample the GPD betas, so save_run stores no beta traces at
    all and the beta path dies on KeyError: 'beta_sigma'. That is EVERY fit in the NZ
    two-stage pipeline (<model>_fixmargins_stage1_plug). Those runs held the margins at
    inp's per-site stage-1 values -- ModelState sets sigma_mat from data.sigma_vec when
    update_sigma is off -- so that is the surface they used and the one the holdout must
    be scored against."""
    idx = np.asarray(idx)
    if "beta_sigma" not in run or "beta_xi" not in run:
        return np.asarray(inp["sigma_vec"])[idx], np.asarray(inp["xi_vec"])[idx]
    bsig = np.asarray(run["beta_sigma"]).mean(axis=0)
    bxi = np.asarray(run["beta_xi"]).mean(axis=0)
    dsig = mg.MarginalDesign(idx.size, 1, inp["sigma_spatial"][idx], None, None)
    dxi = mg.MarginalDesign(idx.size, 1, inp["xi_spatial"][idx], None, None)
    sigma = mg.sigma_from_eta(dsig.eta(bsig))[:, 0]
    xi = mg.xi_from_eta(dxi.eta(bxi))[:, 0]
    return sigma, xi


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ap = argparse.ArgumentParser(description="NZ held-out model comparison")
    ap.add_argument("-B", "--draws", type=int, default=None,
                    help="cap on post-burn posterior draws (thinned); default all")
    ap.add_argument("--M", type=int, default=50, help="inner conditional draws")
    ap.add_argument("--burn", type=float, default=0.5, help="post-burn discard frac")
    ap.add_argument("--last-n", type=int, default=None,
                    help="score using only the FINAL N recorded draws of each run "
                         "(overrides --burn; a convergence knob -- take the well-mixed "
                         "tail of each chain even when runs differ in total length).")
    ap.add_argument("--no-waic", action="store_true", help="skip conditional WAIC")
    ap.add_argument("--suffix", default="",
                    help="training-fit name suffix to load (default '': the plain "
                         "train/<model>.npz fits from Tasks 3-5). Override per model with "
                         "--suffix-<model>.")
    for _m in MODELS:
        ap.add_argument(f"--suffix-{_m}", dest=f"suffix_{_m}", default=None,
                        help=f"per-model suffix override for {_m.upper()} (defaults to "
                             f"--suffix). Pass 'skip' to DROP {_m.upper()} from the "
                             f"comparison -- e.g. compare only hw/wang: --suffix-shi skip.")
    ap.add_argument("--qq-per-block", type=int, default=2,
                    help="test sites per holdout block for the out-of-sample Gumbel QQ (0=off).")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing model_comparison.partial.jsonl and "
                         "recompute every block from scratch (the default RESUMES).")
    ap.add_argument("--no-blocks", action="store_true",
                    help="score only the pooled test set (skip the per-block breakdown).")
    args = ap.parse_args()

    inp = N.load_inputs()
    train_idx, test_idx = inp["train_idx"], inp["test_idx"]
    outdir = os.path.join(N.OUT, "compare")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)

    rundir = os.path.join(N.OUT, "train")
    # per-model suffix (falls back to the global --suffix); 'skip'/'none' drops a model
    active, paths, sfx_used = [], {}, {}
    for m in MODELS:
        sfx = getattr(args, f"suffix_{m}")
        if sfx is None:
            sfx = args.suffix
        if str(sfx).lower() in ("skip", "none"):
            continue
        active.append(m)
        paths[m], sfx_used[m] = os.path.join(rundir, f"{m}{sfx}.npz"), sfx
    if not active:
        sys.exit("No models selected -- every model was set to 'skip'.")
    missing = [m for m in active if not os.path.exists(paths[m])]
    if missing:
        sys.exit(f"Missing training fits {[m + sfx_used[m] for m in missing]} in "
                 f"{rundir}. Run Tasks 3-5 first (or set --suffix/--suffix-<model> to "
                 f"match your fits, or --suffix-<model> skip to drop one).")

    # ---- PREFLIGHT: a VisGP (wang) fit needs the TRAIN-subgraph junction tree ----
    # save_run writes <fit>.cliques.pkl only when data.cliques was set, and load_run
    # populates run["cliques"] only when that sidecar exists -- so run.get("cliques")
    # can be None and goes straight into covariance.visgp_precision as
    # "TypeError: 'NoneType' object is not iterable". At Faroe that surfaced two days
    # into a run, after the finished models had been scored and discarded. Resolve it
    # here, at startup, and fail in seconds if it cannot be.
    def _uses_visgp(fit_path):
        """use_visgp from the config sidecar -- cheap, no npz or latent memmap."""
        try:
            with open(fit_path + ".config.json") as fh:
                return bool(json.load(fh).get("use_visgp"))
        except (OSError, json.JSONDecodeError):
            return False

    _vis_models = [m for m in active if _uses_visgp(paths[m])]
    _fallback = [m for m in _vis_models
                 if not os.path.exists(paths[m] + ".cliques.pkl")]
    if _fallback and not inp.get("train_cliques"):
        sys.exit(f"VisGP fits {_fallback} carry no <fit>.cliques.pkl sidecar and "
                 f"inputs.pkl has no 'train_cliques' to fall back on. The junction "
                 f"tree is required to score a wang model; re-run Task 1 to rebuild "
                 f"inputs.pkl, or drop these fits from the comparison.")
    if _fallback and rank == 0:
        print(f"note: {_fallback} have no .cliques.pkl sidecar -- using inputs.pkl "
              f"train_cliques ({len(inp['train_cliques'])} cliques / "
              f"{len(inp.get('train_separators') or [])} separators), the same "
              f"junction tree the training fits were built with.", flush=True)

    def _vis_parts(run):
        """(cliques, separators) over the TRAINING sites for a VisGP run, else (None, None)."""
        if not run["config"].use_visgp:
            return None, None
        cl, sp = run.get("cliques"), run.get("separators")
        if not cl:
            cl, sp = inp.get("train_cliques"), inp.get("train_separators")
        # Indices address the TRAIN coordinate array (visgp_precision is called with
        # geom.train_coords). A FULL-graph set would index past it and score the wrong
        # geometry silently rather than erroring, so check.
        top = max(max(c) for c in cl)
        if top >= train_idx.size:
            sys.exit(f"clique index {top} exceeds the {train_idx.size} training "
                     f"sites -- these look like FULL-graph cliques, which would score "
                     f"the wrong geometry. Expected the train subgraph.")
        return cl, list(sp or [])

    Y = inp["Y"]
    tail_thr = float(np.median(inp["u_vec"][test_idx]))
    # per-block test partition (NW/SE) -> region -> GLOBAL site indices
    tbid = N.split_sites_nz(inp["sites_xy"])[test_idx]           # block id per test site
    regions = {"all": test_idx}
    if not args.no_blocks:
        for name, bid in BLOCKS.items():
            reg = test_idx[tbid == bid]
            if reg.size:
                regions[name] = reg
    if rank == 0:
        window = f"last {args.last_n}" if args.last_n else f"burn {args.burn:g}"
        print(f"=== NZ model comparison ({comm.Get_size()} ranks) ===\n"
              f"models={{ {', '.join(f'{m}:{sfx_used[m]!r}' for m in active)} }}; "
              f"train={train_idx.size} test={test_idx.size} "
              f"blocks={{ {', '.join(f'{k}:{v.size}' for k, v in regions.items() if k!='all')} }}; "
              f"tail_threshold={tail_thr:.1f} mm; draws<={args.draws or 'all'} ({window}) "
              f"M={args.M}", flush=True)

    def _geom_hold(run, reg):
        """(NewSiteGeometry, HoldoutData) for the GLOBAL test-site indices `reg`."""
        sig_r, xi_r = inp["sigma_spatial"][reg], inp["xi_spatial"][reg]
        _cl, _sep = _vis_parts(run)
        geom = pr.NewSiteGeometry(
            new_coords=inp["sites_xy"][reg], train_coords=inp["sites_xy"][train_idx],
            knots_S=run["knots_S"], knots_phi=run["knots_phi"], knots_rho=run["knots_rho"],
            radius_S=float(run["radius_S"]), eff_range_phi=float(run["eff_range_phi"]),
            eff_range_rho=float(run["eff_range_rho"]), config=run["config"],
            cliques=_cl, separators=_sep,
            water_poly=(N.modeling_polygon() if run["config"].use_visgp else None),
            sigma_spatial=sig_r, xi_spatial=xi_r)
        s_r, x_r = _postmean_margin(run, inp, reg)        # fallback margins
        hold = pr.HoldoutData(Y=Y[reg], u=inp["u_vec"][reg], sigma=s_r, xi=x_r,
                              p_cens=N.P_CENS)
        return geom, hold

    # Regions as LOCAL indices into the pooled test set: one scoring pass over all test
    # sites yields every region, because the per-site conditional log-densities it
    # computes can be summed over any index set (see score_holdout_all's docstring).
    _pos = {int(g): i for i, g in enumerate(test_idx)}
    local_regions = {nm: np.array([_pos[int(g)] for g in reg], dtype=int)
                     for nm, reg in regions.items()}
    WAIC_KEYS = ("waic", "lppd", "p_waic", "p_waic_max", "n_pwaic_bad")

    # ---- resume from whatever a previous (killed) run already finished --------
    partial_path = os.path.join(outdir, "model_comparison.partial.jsonl")
    csv_path = os.path.join(outdir, "model_comparison.csv")
    if rank == 0:
        if args.fresh and os.path.exists(partial_path):
            os.replace(partial_path, partial_path + ".bak")
            print(f"--fresh: moved existing partial results to "
                  f"{os.path.basename(partial_path)}.bak", flush=True)
        done = _load_partial(partial_path)
    else:
        done = None
    # Every rank must agree on what to skip: score_holdout/waic are COLLECTIVE, so a
    # rank that skipped a block while others entered it would deadlock the job.
    done = comm.bcast(done, root=0)
    if done and rank == 0:
        print(f"\nresuming: {len(done)} block(s) already on disk in "
              f"{os.path.basename(partial_path)} ({', '.join(sorted(done))}) "
              f"-- pass --fresh to recompute", flush=True)

    ptdir = os.path.join(outdir, "pointwise")

    def _save_pointwise(row):
        """Split the per-TIME arrays out of `row` into a sidecar npz.

        They are far too big for the JSON journal but are the only route to a standard
        error after the fact -- and, more importantly, to a PAIRED difference between
        two models, which is the comparison that actually has power. Merged rather than
        overwritten, because the holdout scores and the WAIC are persisted at different
        moments and land in the same file."""
        pt = {k: row.pop(k) for k in [k for k in row if k.startswith("_pt")]}
        if not pt:
            return
        arrs = {}
        for k, v in pt.items():
            if isinstance(v, dict):
                arrs.update(v)
            else:
                arrs[k.replace("_pt_", "")] = v
        os.makedirs(ptdir, exist_ok=True)
        fp = os.path.join(ptdir, f"{row['model']}.{row['region']}.npz")
        if os.path.exists(fp):
            with np.load(fp) as d:
                merged = {k: d[k] for k in d.files}
            merged.update(arrs)
            arrs = merged
        np.savez(fp, **arrs)

    def _persist(row, all_rows):
        """Journal the row and refresh the CSV, so results are readable mid-run."""
        if rank != 0:
            return
        _save_pointwise(row)
        _append_partial(partial_path, row)
        C.write_table(all_rows, csv_path)

    rows, runs = [], {}
    for m in active:
        rp = paths[m]
        run = load_run(rp); runs[m] = run
        cached = {nm: dict(done.get(_row_key(m, nm)) or {}) for nm in regions}
        missing = [nm for nm in regions if not cached[nm]]
        geom_all = None

        if missing:
            # ONE pass over the pooled test set for every region. Scoring each region
            # separately re-ran the whole kriging and cost as much again as the pooled
            # run: the blocks are not ADDITIVE, but they are EXTRACTABLE from this pass.
            geom_all, hold_all = _geom_hold(run, test_idx)
            if rank == 0:
                print(f"\n[{m}] scoring {test_idx.size} held-out sites -> regions "
                      f"{list(regions)} in ONE pass (missing: {missing})", flush=True)
            scored = C.score_holdout(rp, geom_all, hold_all, tail_threshold=tail_thr,
                                     M=args.M, burn=args.burn, max_draws=args.draws,
                                     last_n=args.last_n,
                                     progress=_progress(m) if rank == 0 else None,
                                     comm=comm, regions=local_regions)
            # Every region is refreshed, not just the missing ones: they all come from
            # this single pass at no extra cost, which also keeps a model's rows on
            # COMMON RANDOM NUMBERS. Any WAIC already attached is carried over -- it is a
            # training-data score and does not depend on this pass.
            for nm in regions:
                sc = dict(scored[nm])
                sc["model"], sc["region"] = m, nm
                for k in WAIC_KEYS:
                    if k in cached[nm]:
                        sc[k] = cached[nm][k]
                cached[nm] = sc
        elif rank == 0:
            print(f"\n[{m}] all {len(regions)} region score(s) already on disk "
                  f"-- skipping", flush=True)

        model_rows = [cached[nm] for nm in regions]
        rows.extend(model_rows)
        if missing:
            # PERSIST BEFORE WAIC: the scoring pass is hours and its WAIC minutes, so a
            # WAIC failure must never discard the scores.
            for sc in model_rows:
                _persist(sc, rows)

        # conditional WAIC is a TRAINING-data score -> attach once, to the pooled row
        sc_all = cached["all"]
        if not args.no_waic and "waic" not in sc_all:
            s_tr, x_tr = _postmean_margin(run, inp, train_idx)
            train = pr.HoldoutData(Y=Y[train_idx], u=inp["u_vec"][train_idx],
                                   sigma=s_tr, xi=x_tr, p_cens=N.P_CENS)
            if rank == 0:
                print(f"[{m}] WAIC...", flush=True)
            if geom_all is None:
                geom_all, _ = _geom_hold(run, test_idx)
            sc_all.update(C.waic(rp, geom_all, train, burn=args.burn,
                                 max_draws=args.draws, last_n=args.last_n,
                                 progress=_progress(m + ":waic") if rank == 0 else None,
                                 comm=comm))
            _persist(sc_all, rows)  # journal wins on re-read: last record per key

    if rank == 0:
        # full per-region table (pooled 'all' + per block NW/SE)
        rows.sort(key=lambda r: (active.index(r["model"]),
                                 list(regions).index(r["region"])))
        tbl = C.write_table(rows, csv_path,
                            title="\nNZ held-out predictive scores by region "
                                  "(higher loglik / lower CRPS,TWCRPS,WAIC = better):")
        print(tbl, flush=True)
        # The figures are presentation only and come after hours of scoring, so a
        # plotting failure must not be what ends the job -- the numbers are already in
        # the CSV and the journal by this point.
        try:
            # pooled bar chart (one bar per model) uses the 'all' rows only
            _plot_comparison([r for r in rows if r["region"] == "all"],
                             os.path.join(outdir, "model_comparison.png"))
            # out-of-sample Gumbel QQ (one row per model, columns = a few test sites)
            if args.qq_per_block > 0:
                qq_sites, qq_labels = _pick_qq_sites(inp, test_idx, tbid,
                                                     args.qq_per_block)
                if qq_sites:
                    plot_oos_qq(runs, inp, qq_sites, qq_labels,
                                os.path.join(outdir, "oos_qq.png"), burn=args.burn)
                    print(f"Wrote oos_qq.png ({len(qq_sites)} test sites x "
                          f"{len(active)} models)", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"WARNING: figures failed ({type(exc).__name__}: {exc}); the scores "
                  f"are safe in {os.path.basename(csv_path)} and "
                  f"{os.path.basename(partial_path)}.", flush=True)
        print(f"\nWrote model_comparison.csv to {outdir}/", flush=True)
    comm.Barrier()


def _pick_qq_sites(inp, test_idx, tbid, per_block):
    """A few GLOBAL test-site indices per holdout block (most exceedances first)."""
    Y, uu = inp["Y"], inp["u_vec"]
    sites, labels = [], []
    for name, bid in BLOCKS.items():
        cand = test_idx[tbid == bid]
        nexc = [(int(np.sum(np.isfinite(Y[s]) & (Y[s] > uu[s]))), int(s)) for s in cand]
        for _, s in sorted(nexc, reverse=True)[:per_block]:
            sites.append(s)
            labels.append(f"{name} ({inp['sites_xy'][s, 0]:.2f},{inp['sites_xy'][s, 1]:.2f})")
    return sites, labels


def plot_oos_qq(runs, inp, sites, labels, path, burn=0.5):
    """Out-of-sample Gumbel QQ: rows = models, columns = held-out test sites. Each panel
    pushes the site's observed exceedances through that model's posterior-averaged
    conditional-GPD CDF (margins = the model's sampled betas at the test-site covariates)
    and maps to the standard-Gumbel scale, with the binomial 95% band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(runs)
    nS = len(sites)
    fig, axs = plt.subplots(len(models), nS, figsize=(3.0 * nS, 2.9 * len(models)),
                            squeeze=False)
    for i, m in enumerate(models):
        for j, (site, lab) in enumerate(zip(sites, labels)):
            a = axs[i][j]
            q = D.gumbel_qq_site(runs[m], site, inp["Y"], inp["u_vec"],
                                 inp["sigma_spatial"], inp["xi_spatial"], burn=burn)
            if q is None:
                a.set_axis_off(); continue
            a.fill_between(q["z_theory"], q["z_lo"], q["z_hi"], color="0.85")
            lim = [min(q["z_theory"][0], q["z_data"].min()) - 0.2,
                   max(q["z_theory"][-1], q["z_data"].max()) + 0.2]
            a.plot(lim, lim, "k--", lw=1)
            a.plot(q["z_theory"], q["z_data"], "o", ms=3.5, color="tab:blue")
            if i == 0:
                a.set_title(lab, fontsize=8)
            if j == 0:
                a.set_ylabel(f"{m}\nobs (Gumbel)", fontsize=9)
            if i == len(models) - 1:
                a.set_xlabel("theoretical Gumbel", fontsize=8)
    fig.suptitle("Out-of-sample Gumbel QQ at held-out test sites "
                 "(off diagonal / outside band = marginal misfit)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=125, bbox_inches="tight"); plt.close(fig); os.replace(tmp, path)


def _plot_comparison(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("loglik", "pred. log-score", True),
               ("crps", "CRPS", False), ("twcrps_hard", "TW-CRPS (hard)", False),
               ("twcrps_smooth", "TW-CRPS (smooth)", False), ("waic", "WAIC", False)]
    metrics = [mtr for mtr in metrics if all(mtr[0] in r for r in rows)]
    names = [r["model"] for r in rows]
    colors = [MODEL_COLORS.get(n, "tab:orange") for n in names]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4),
                             squeeze=False)
    for ax, (k, lab, higher) in zip(axes[0], metrics):
        vals = [r[k] for r in rows]
        ax.bar(names, vals, color=colors)
        ax.set_title(lab + ("  (higher=better)" if higher else "  (lower=better)"),
                     fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("NZ held-out model comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=120); plt.close(fig); os.replace(tmp, path)


if __name__ == "__main__":
    main()
