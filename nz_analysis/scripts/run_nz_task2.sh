#!/usr/bin/env bash
# Task 2 -- fit the wang model on the FULL data with FIXED margins from stage 1.
#
# The GPD margins are FROZEN at the stage-1 marginal fit's posterior-mean surface,
# surface(mean beta), instead of being sampled: only the dependence field
# (Z/S/phi/rho/nu/alpha0/geometry) adapts to them. The latent field starts from the
# plug-in ("X-inversion") decomposition rather than S=1/Z=0, so the run begins ~2.1M
# nats up with Z already in its Gaussian typical set. Output:
#     nz_analysis/output/wang_full_fixmargins_stage1_plug.*
#
# PREREQUISITE: the marginal run must be finished --
#     bash nz_analysis/scripts/run_nz_marginal.sh
#
# Env overrides: NRANKS(16), NZ_STAGE1_NPZ(<output>/marginal/marginal_fit.npz),
# NZ_PHI0(0.5), NZ_ALPHA0(2.05), NZ_PLUGIN_INIT(1), plus the standard nz_setup knobs.
# Extra args pass through to nz_fit.py (e.g. --no-plugin-init, --fix-radius 2.6).
source "$(dirname "$0")/_nz_env.sh"
nz_require_stage1 task2

# shellcheck disable=SC2086
mpiexec ${NZ_MPIEXEC_ARGS:-} -n "$NRANKS" ${NZ_RANKWRAP:-} \
    python "$NZ_SCRIPTS/nz_fit.py" --model wang --which full --fix-margins "$@"
