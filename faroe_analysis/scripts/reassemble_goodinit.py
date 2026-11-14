#!/usr/bin/env python3
"""Reassemble the goodinit run's FINAL latent field (Z, S) from its MPI checkpoint.

The goodinit run was saved with save_latent=False, so the field lives only in the
per-rank checkpoint pkls (round-robin time partition: rank r owns t = r, r+size, ...).
This stitches them back into full (Ns, Nt) Z and (k_S, Nt) S, and bundles the global
snapshot (phi_k, rho_k, alpha0, nu, radius_S, anisotropy, betas, W_wend_S, gamma_bar)
into a single self-contained goodinit_field.npz -- the exact final state, ready to
seed the task 3-5 warm starts. No dependence on the (stale) goodinit.npz.

    python reassemble_goodinit.py [--ckpt output/goodinit/goodinit.ckpt] [--out ...]
"""
import argparse, glob, os, pickle, re, sys
# the checkpoint pkls hold adaptive-block objects defined in mcmc_updates / at the repo
# root, so those modules must be importable to unpickle. Put repo root + scripts on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root (mcmc_updates, sampler)
sys.path.insert(0, _HERE)                                     # faroe_analysis/scripts
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="checkpoint base path (…/goodinit.ckpt)")
ap.add_argument("--out", default=None, help="output npz (default: <ckptdir>/goodinit_field.npz)")
ap.add_argument("--float32", action="store_true", help="store Z/S as float32 (~half size)")
args = ap.parse_args()

if args.ckpt is None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    args.ckpt = os.path.join(os.environ.get("FAROE_OUT", os.path.join(root, "faroe_analysis/output")),
                             "goodinit", "goodinit.ckpt")
gpath = args.ckpt + ".global.pkl"
if not os.path.exists(gpath):
    sys.exit(f"missing global checkpoint {gpath}")
G = pickle.load(open(gpath, "rb"))
size = int(G["size"]); it = int(G["it"]); gsnap = G["gsnap"]
print(f"checkpoint: iter {it}, size {size} ranks")

ranks = {}
for p in glob.glob(args.ckpt + ".rank*.pkl"):
    m = re.search(r"\.rank(\d+)\.pkl$", p)
    if m:
        ranks[int(m.group(1))] = p
missing = [r for r in range(size) if r not in ranks]
if missing:
    sys.exit(f"missing rank pkls: {missing} (need all {size} from the run machine)")

# infer Nt = total columns = sum over ranks (round-robin partitions 0..Nt-1 exactly)
loc = {}
ncols = 0
for r in range(size):
    d = pickle.load(open(ranks[r], "rb"))
    loc[r] = (np.asarray(d["S_local"]), np.asarray(d["Z_local"]))
    ncols += loc[r][1].shape[1]
Nt = ncols
kS = loc[0][0].shape[0]; Ns = loc[0][1].shape[0]
S = np.empty((kS, Nt)); Z = np.empty((Ns, Nt))
for r in range(size):
    cols = range(r, Nt, size)
    Sl, Zl = loc[r]
    assert Zl.shape[1] == len(list(cols)), f"rank {r} col mismatch"
    S[:, list(cols)] = Sl
    Z[:, list(cols)] = Zl
print(f"reassembled  Z {Z.shape}  S {S.shape}  "
      f"(finite: Z {np.isfinite(Z).all()}, S {np.isfinite(S).all()}; S>0: {(S>0).all()})")

dt = np.float32 if args.float32 else np.float64
out = args.out or os.path.join(os.path.dirname(gpath), "goodinit_field.npz")
save = {"Z": Z.astype(dt), "S": S.astype(dt), "it": it}
for k, v in gsnap.items():          # phi_k, rho_k, alpha0, nu, radius_S, aniso_*, betas, W_wend_S, gamma_bar
    save[f"g_{k}"] = np.asarray(v)
np.savez(out, **save)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
print("globals:", {k: (np.asarray(v).shape if hasattr(v, '__len__') else v) for k, v in gsnap.items()})
