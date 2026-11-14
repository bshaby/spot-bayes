#!/usr/bin/env bash
# Repartition an MPI checkpoint to a different rank count, in the `extremes` env.
#
# Wrapper for the repo-root repartition_checkpoint.py (shared with Faroe -- nothing in
# that tool is analysis-specific; it only knows the sampler's checkpoint layout). Use
# this rather than calling python directly: the script needs Python 3 AND an interpreter
# that can import numpy + mcmc_updates (the per-rank pickles contain Adaptive objects),
# and a bare `python` on the login node may well be the system Python 2 -- which fails
# with a confusing SyntaxError rather than anything informative.
#
# WHY: run_mcmc_mpi REFUSES a resume whose rank count differs from the one that wrote the
# checkpoint (the per-rank files are a round-robin partition, local_t = range(rank, Nt,
# size)). So if you want to restart a 16-rank NZ fit on 32 ranks, repartition FIRST. The
# alternative -- a warm start -- deliberately restarts the chain: cold adaptives, fresh
# RNG, trace history back to iteration 0. Repartitioning keeps all of it.
#
# USAGE (paths are absolute; _nz_env.sh cd's to the repo root)
#   bash nz_analysis/scripts/run_nz_repartition.sh \
#        --ckpt "$NZ_OUT/train/shi_fixmargins_stage1_plug.ckpt" --ranks 32 --dry-run
#   bash nz_analysis/scripts/run_nz_repartition.sh \
#        --ckpt "$NZ_OUT/train/shi_fixmargins_stage1_plug.ckpt" --ranks 32
#
# Checkpoint bases for the standard NZ runs (under $NZ_OUT):
#   task 2   wang_full_fixmargins_stage1_plug.ckpt
#   task 3   train/hw_fixmargins_stage1_plug.ckpt
#   task 4   train/wang_fixmargins_stage1_plug.ckpt
#   task 5   train/shi_fixmargins_stage1_plug.ckpt
#
# THEN resubmit at the new rank count:
#   sbatch --cpus-per-task=32 --export=ALL,NZ_TASK=5 nz_analysis/scripts/alpine_nz.sbatch
# (and make sure no stale NRANKS is exported in your shell -- it would beat the flag).
#
# STOP THE JOB FIRST, just after a checkpoint. A checkpoint is 1 + size separate files;
# killing mid-write can tear the set. The script checks the mtime spread and refuses a
# torn one, but the clean sequence is: wait for the run's .npz mtime to change, wait
# ~30 s for the other ranks to clear the Barrier, then kill.
#
# --seed defaults to NZ_SEED (the run's own seed) so the re-derived local RNG streams
# match what a fresh run at the new rank count would have used.
set -eo pipefail
source "$(dirname "$0")/_nz_env.sh"

args=("$@")
case " ${args[*]} " in
  *" --seed "*) : ;;
  *) args+=(--seed "${NZ_SEED:-20260720}") ;;
esac

exec python "$(pwd)/repartition_checkpoint.py" "${args[@]}"
