#!/usr/bin/env bash
# Repartition an MPI checkpoint to a different rank count, in the `extremes` env.
#
# Wrapper for repartition_checkpoint.py. Use this rather than calling python directly:
# the script needs Python 3 AND an interpreter that can import numpy + mcmc_updates
# (the per-rank pickles contain Adaptive objects), and a bare `python` on the login
# node may well be the system Python 2 -- which fails with a confusing SyntaxError
# rather than anything informative.
#
# USAGE (paths are absolute; _faroe_env.sh cd's to the repo root)
#   bash faroe_analysis/scripts/run_faroe_repartition.sh \
#        --ckpt "$FAROE_OUT/train/shi_fixmargins_stage1.ckpt" --ranks 32 --dry-run
#   bash faroe_analysis/scripts/run_faroe_repartition.sh \
#        --ckpt "$FAROE_OUT/train/shi_fixmargins_stage1.ckpt" --ranks 32
#
# STOP THE JOB FIRST, just after a checkpoint. A checkpoint is 1 + size separate
# files; killing mid-write can tear the set. The script checks the mtime spread and
# refuses a torn one, but the clean sequence is: wait for the run's .npz mtime to
# change, wait ~30 s for the other ranks to clear the Barrier, then kill.
#
# --seed defaults to FAROE_SEED (the run's own seed) so the re-derived local RNG
# streams match what a fresh run at the new rank count would have used.
set -eo pipefail
source "$(dirname "$0")/_faroe_env.sh"

args=("$@")
case " ${args[*]} " in
  *" --seed "*) : ;;
  *) args+=(--seed "${FAROE_SEED:-20260706}") ;;
esac

# repartition_checkpoint.py lives at the REPO ROOT (nothing in it is analysis-specific;
# NZ shares the same copy). _faroe_env.sh has already cd'd there.
exec python "$(pwd)/repartition_checkpoint.py" "${args[@]}"
