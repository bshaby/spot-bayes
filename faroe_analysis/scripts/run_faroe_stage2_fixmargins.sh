#!/usr/bin/env bash
# STAGE 2 (runnable path) -- fixed-margins dependence fit seeded from stage 1.
#
# Runs the production MPI dependence model (wang, full 432 sites) with the GPD margins
# FROZEN at the STAGE-1 posterior-mean surface, surface(mean beta), instead of sampling
# them. This is the "defensible fixed margins" path from docs/two_stage_recursive_bayes.md:
# the margins come from the standalone marginal analysis, and only the dependence field
# (Z/S/phi/rho/nu/alpha0/geometry) adapts to them.
#
# PREREQUISITE: stage 1 must be finished --
#     bash faroe_analysis/scripts/run_faroe_marginal.sh
# produces the marginal_fit.npz this script consumes (default location below).
#
# How the seeding works: FAROE_STAGE1_NPZ points faroe_fit.py at the stage-1 draws; with
# --fix-margins it fixes sigma/xi at surface(mean beta) (= the plug-in posterior-mean
# surface; the mean-of-per-draw-surfaces differs by only ~0.1% but has no coefficient form)
# and RECORDS those constant betas in the run so held-out prediction can form margins at new
# sites (predictive._holdout_marginal needs beta_sigma/beta_xi). Output goes to a DISTINCT
# name so it never clobbers the reference-beta fixmargins run:
#     faroe_analysis/output/wang_full_fixmargins_stage1.*
# (Without FAROE_STAGE1_NPZ, --fix-margins instead pins the REFERENCE workflow surface.)
#
# Checkpointed + auto-resuming: re-run the same command to extend/continue.
#
# Env overrides: NRANKS(64), FAROE_STAGE1_NPZ(<output>/marginal/marginal_fit.npz),
# FAROE_NITERS, FAROE_THIN, FAROE_CKPT_HOURS, plus the standard faroe_setup knobs.
source "$(dirname "$0")/_faroe_env.sh"     # activates `extremes`, GSL, MPI env, cd repo root

export FAROE_STAGE1_NPZ="${FAROE_STAGE1_NPZ:-${FAROE_OUT:-$FAROE_SCRIPTS/../output}/marginal/marginal_fit.npz}"
if [ ! -f "$FAROE_STAGE1_NPZ" ]; then
  echo "ERROR: stage-1 output not found at $FAROE_STAGE1_NPZ" >&2
  echo "       run  bash faroe_analysis/scripts/run_faroe_marginal.sh  first." >&2
  exit 1
fi
echo "[stage2] seeding fixed margins from $FAROE_STAGE1_NPZ"

# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} \
    python "$FAROE_SCRIPTS/faroe_fit.py" --model wang --which full --fix-margins "$@"
