#!/usr/bin/env bash
# Task 2 -- fit the wang model on the FULL data with FIXED margins from the marginal run.
#
# The GPD margins are FROZEN at the stage-1 marginal fit's posterior-mean surface,
# surface(mean beta), instead of being sampled: only the dependence field
# (Z/S/phi/rho/nu/alpha0/geometry) adapts to them. Output goes to the distinct name
#     faroe_analysis/output/wang_full_fixmargins_stage1.*
# so it never clobbers a sampled-margin fit. (This folds run_faroe_stage2_fixmargins.sh
# into the standard Task 2.)
#
# PREREQUISITE: the marginal run must be finished --
#     bash faroe_analysis/scripts/run_faroe_marginal.sh
# produces the marginal_fit.npz consumed here (default location below).
#
# Env overrides: NRANKS(64), FAROE_STAGE1_NPZ(<output>/marginal/marginal_fit.npz),
# plus the standard faroe_setup knobs.
source "$(dirname "$0")/_faroe_env.sh"

export FAROE_STAGE1_NPZ="${FAROE_STAGE1_NPZ:-${FAROE_OUT:-$FAROE_SCRIPTS/../output}/marginal/marginal_fit.npz}"
if [ ! -f "$FAROE_STAGE1_NPZ" ]; then
  echo "ERROR: marginal-run output not found at $FAROE_STAGE1_NPZ" >&2
  echo "       run  bash faroe_analysis/scripts/run_faroe_marginal.sh  first." >&2
  exit 1
fi
echo "[task2] fixing margins from $FAROE_STAGE1_NPZ"

# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} \
    python "$FAROE_SCRIPTS/faroe_fit.py" --model wang --which full --fix-margins "$@"
