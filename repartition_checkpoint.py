#!/usr/bin/env python3
"""Change the RANK COUNT of an MPI checkpoint in place -- continue a run on a
different number of ranks without restarting it.

`run_mcmc_mpi` refuses a resume whose rank count differs from the one that wrote the
checkpoint (the per-rank files are a round-robin partition, `local_t = range(rank, Nt,
size)`). The usual escape is a warm start, which is rank-count independent -- but a
warm start deliberately RESTARTS the chain: cold adaptives, fresh RNG, trace history
back to iteration 0.

That is unnecessary here, because everything per-rank is either a partition of a
global object or keyed by TIME:

    S_local / Z_local   columns of the full (k_S, Nt) / (Ns, Nt) field  -> re-split
    loc_ad["S"/"Z"]     dicts KEYED BY t, one adaptive per time column  -> regroup
    lrng                per-rank RNG stream                             -> reseeded
    nuts_z              per-rank Z step adapter (scalar eps)            -> carried over

So the field, the learned per-time proposal covariances (the 41x41 S matrices that
took the whole run to adapt), the global parameters, the global RNG, the iteration
counter and the recorded traces ALL survive exactly. Only the local RNG streams are
re-derived, which carries no posterior information -- and an MPI resume is already
not bit-reproducible anyway (float reduction ordering depends on rank count).

    python repartition_checkpoint.py --ckpt <base> --ranks 32 [--out <base>] [--dry-run]

By default it rewrites <base> IN PLACE, first moving every existing per-rank file into
<base>.size<old>.bak/ -- which also removes the stale rank32..63 files that would
otherwise sit next to a 32-rank set and silently corrupt a later 64-rank resume.
"""
import argparse
import glob
import os
import pickle
import re
import shutil
import sys

# Lives at the REPO ROOT: nothing in here is analysis-specific (it only knows the
# sampler's own checkpoint layout), so Faroe and NZ share one copy via their
# run_*_repartition.sh wrappers.
_HERE = os.path.dirname(os.path.abspath(__file__))            # repo root
sys.path.insert(0, _HERE)                                     # mcmc_updates, numpy deps
import numpy as np                                            # noqa: E402


def load_set(base):
    gp = base + ".global.pkl"
    if not os.path.exists(gp):
        sys.exit(f"missing {gp}")
    with open(gp, "rb") as fh:
        G = pickle.load(fh)
    old = int(G["size"])
    ranks = {}
    for p in glob.glob(base + ".rank*.pkl"):
        m = re.search(r"\.rank(\d+)\.pkl$", p)
        if m:
            ranks[int(m.group(1))] = p
    missing = [r for r in range(old) if r not in ranks]
    if missing:
        sys.exit(f"missing per-rank files for ranks {missing} (need all {old})")
    R = {}
    for r in range(old):
        with open(ranks[r], "rb") as fh:
            R[r] = pickle.load(fh)
    return G, R, old, {r: ranks[r] for r in ranks}


def check_not_torn(base, old, paths, max_skew, force):
    """A checkpoint is written as 1 + `size` SEPARATE files: each is replaced
    atomically, but the SET is not. Killing a job mid-write can therefore leave a
    mix of iterations -- new global.pkl beside stale rank files -- and nothing in the
    per-rank payload stamps its iteration, so a torn set would reassemble into a
    silently wrong field. A clean checkpoint writes all files within seconds; a wide
    mtime spread means the job was killed during the write."""
    files = [base + ".global.pkl"] + [paths[r] for r in range(old)]
    mt = {p: os.path.getmtime(p) for p in files}
    skew = max(mt.values()) - min(mt.values())
    print(f"  file mtime spread   {skew:.1f} s across {len(files)} files", end="")
    if skew <= max_skew:
        print("  (consistent)")
        return
    print("  <-- SUSPICIOUS")
    oldest = min(mt, key=mt.get); newest = max(mt, key=mt.get)
    msg = (f"\ncheckpoint may be TORN: {os.path.basename(oldest)} is {skew:.0f}s older "
           f"than {os.path.basename(newest)}.\nIf the job was killed mid-checkpoint, the "
           f"per-rank files span two different iterations and the reassembled field "
           f"would be wrong.\nPrefer the previous good checkpoint, or re-run with "
           f"--force if you are sure (e.g. a slow shared filesystem).")
    if not force:
        sys.exit(msg)
    print(msg + "\n--force given: continuing anyway.")


def reassemble(G, R, old):
    """Full-field S/Z plus time-keyed adaptive dicts, from the old partition."""
    Nt = sum(R[r]["Z_local"].shape[1] for r in range(old))
    kS = R[0]["S_local"].shape[0]
    Ns = R[0]["Z_local"].shape[0]
    S = np.empty((kS, Nt)); Z = np.empty((Ns, Nt))
    S[:] = np.nan; Z[:] = np.nan
    ad_S, ad_Z = {}, {}
    for r in range(old):
        cols = list(range(r, Nt, old))
        if len(cols) != R[r]["Z_local"].shape[1]:
            sys.exit(f"rank {r}: {R[r]['Z_local'].shape[1]} columns but the "
                     f"round-robin partition of Nt={Nt} over {old} ranks wants "
                     f"{len(cols)} -- is this really a round-robin checkpoint?")
        S[:, cols] = R[r]["S_local"]
        Z[:, cols] = R[r]["Z_local"]
        for key, dst in (("S", ad_S), ("Z", ad_Z)):
            src = R[r]["loc_ad"][key]
            if isinstance(src, dict):
                dst.update(src)
    if np.isnan(S).any() or np.isnan(Z).any():
        sys.exit("reassembled field has gaps -- the per-rank columns do not cover 0..Nt-1")
    for nm, d in (("S", ad_S), ("Z", ad_Z)):
        if d and set(d) != set(range(Nt)):
            sys.exit(f"loc_ad[{nm!r}] covers {len(d)} times, expected {Nt}")
    return S, Z, ad_S, ad_Z, Nt


def write_set(out, G, S, Z, ad_S, ad_Z, nuts_z, new, Nt, seed):
    payload = dict(G)
    payload["size"] = int(new)
    tmp = out + ".global.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out + ".global.pkl")
    for r in range(new):
        cols = list(range(r, Nt, new))
        d = dict(S_local=S[:, cols].copy(), Z_local=Z[:, cols].copy(),
                 loc_ad={"S": {t: ad_S[t] for t in cols} if ad_S else {},
                         "Z": {t: ad_Z[t] for t in cols} if ad_Z else {}},
                 # same formula run_mcmc_mpi uses to seed a fresh local stream
                 lrng=np.random.default_rng(seed * 100003 + r + 1).bit_generator.state)
        if nuts_z is not None and r in nuts_z:
            d["nuts_z"] = nuts_z[r]
        tmp = f"{out}.rank{r}.pkl.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, f"{out}.rank{r}.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint base path (.../name.ckpt)")
    ap.add_argument("--ranks", type=int, required=True, help="NEW rank count")
    ap.add_argument("--out", default=None,
                    help="write the new set here (default: in place, old files backed up)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the re-derived local RNG streams (default: the "
                         "run's own seed if recorded, else 0)")
    ap.add_argument("--dry-run", action="store_true", help="inspect and report only")
    ap.add_argument("--max-skew", type=float, default=300.0, metavar="SEC",
                    help="refuse if the checkpoint files' mtimes span more than this "
                         "(default 300 s) -- catches a set torn by a mid-write kill")
    ap.add_argument("--force", action="store_true", help="proceed despite a wide skew")
    args = ap.parse_args()

    G, R, old, paths = load_set(args.ckpt)
    new = args.ranks
    print(f"checkpoint {args.ckpt}")
    print(f"  iteration      {G['it']}  (resumes at {G['it'] + 1})")
    check_not_torn(args.ckpt, old, paths, args.max_skew, args.force)
    S, Z, ad_S, ad_Z, Nt = reassemble(G, R, old)
    print(f"  ranks          {old} -> {new}")
    print(f"  field          Z {Z.shape}  S {S.shape}   (finite: "
          f"{np.isfinite(Z).all() and np.isfinite(S).all()}, S>0: {(S > 0).all()})")
    print(f"  per-time adaptives  S {len(ad_S)}  Z {len(ad_Z)}  of Nt={Nt} "
          f"-> regrouped, NOT rebuilt")
    print(f"  recorded traces     {len(G.get('small_traces', {}))} arrays, kept")
    print(f"  gsnap NUTS adapters {sorted((G['gsnap'].get('nuts') or {}).keys()) or 'none'}")
    nz = {r: R[r]["nuts_z"] for r in range(old) if R[r].get("nuts_z") is not None}
    print(f"  per-rank nuts_z     {len(nz)} present -> "
          f"{min(len(nz), new) if nz else 0} carried over")
    if Nt % new:
        print(f"  NB Nt={Nt} is not divisible by {new}: columns split "
              f"{Nt // new}/{Nt // new + 1} per rank (fine, the sampler handles it)")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = args.out or args.ckpt
    if out == args.ckpt:
        bak = f"{args.ckpt}.size{old}.bak"
        os.makedirs(bak, exist_ok=True)
        for p in list(paths.values()) + [args.ckpt + ".global.pkl"]:
            shutil.move(p, os.path.join(bak, os.path.basename(p)))
        print(f"\nbacked up the {old}-rank set -> {bak}/")
    seed = args.seed if args.seed is not None else int(G.get("seed", 0) or 0)
    write_set(out, G, S, Z, ad_S, ad_Z, nz, new, Nt, seed)
    print(f"wrote {new}-rank checkpoint at {out}  (local RNG streams reseeded "
          f"from seed={seed})")
    print(f"\nnow relaunch with {new} ranks; the run continues from iteration "
          f"{G['it'] + 1}.")


if __name__ == "__main__":
    main()
