#!/usr/bin/env bash
# Task 6 -- held-out model comparison of the tasks 3-5 TRAIN fits: predictive log-score,
# CRPS, TW-CRPS, WAIC, PER TEST BLOCK (NW/SE) + pooled, plus an out-of-sample Gumbel QQ
# (oos_qq.png, one row per model). Writes output/compare/model_comparison.csv (+ .png).
#
# Defaults to the two-stage, plug-in-initialised fits that tasks 3-5 now produce
# (train/<model>_fixmargins_stage1_plug.npz). Override the whole default with NZ_SUFFIX,
# or per-model / per-arg by appending your own flags (later args win):
#     --suffix ''                   # the plain cold-start train/<model>.npz
#     --suffix-shi skip             # drop shi (e.g. compare only hw/wang)
#     --suffix-shi _fixmargins_stage1  # per-model suffix override
#     --last-n 10000                # score on the FINAL 10k draws of each run (convergence
#                                   #   knob; overrides --burn, per-run tail even if lengths differ)
#     --no-blocks                   # pooled scores only (skip the per-block breakdown)
#     --qq-per-block 3              # more QQ sites per block (0 = skip the QQ)
#
# One-shot (not resumable): under the Alpine sbatch submit with --array=1.
source "$(dirname "$0")/_nz_env.sh"
# shellcheck disable=SC2086
mpiexec ${NZ_MPIEXEC_ARGS:-} -n "$NRANKS" ${NZ_RANKWRAP:-} \
    python "$NZ_SCRIPTS/nz_task6_compare.py" \
    --suffix "${NZ_SUFFIX:-_fixmargins_stage1_plug}" "$@"
