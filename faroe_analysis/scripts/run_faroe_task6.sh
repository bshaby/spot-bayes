#!/usr/bin/env bash
# Task 6 -- held-out model comparison of the tasks 3-5 TRAIN fits: predictive log-score,
# CRPS, TW-CRPS, WAIC, now PER TEST BLOCK (E/NW/S) + pooled, plus an out-of-sample Gumbel
# QQ (oos_qq.png, one row per model). Writes output/compare/model_comparison.csv (+ .png).
#
# Compares the goodinit-WARM-started sampled-margin fits by default (train/<model>_warm.npz
# from run_faroe_task{3,4,5}_warm.sh). Override the target with args / FAROE_ARGS:
#     --suffix _fixmargins_stage1   # the fixed-margin fits instead
#     --suffix ''                   # plain train/<model>.npz
#     --suffix-shi skip             # drop shi (e.g. compare only hw/wang warm fits)
#     --suffix-shi _fixmargins_stage1  # per-model suffix: mix hw/wang warm w/ shi fixmargins
#     --fit hw_warm --fit hw_fixmargins_stage1 --fit shi_fixmargins_stage1
#                                   # name the fits EXPLICITLY by train/<stem>.npz, one
#                                   #   --fit each. The only way to score two variants of
#                                   #   the SAME model together (--suffix allows one entry
#                                   #   per model); labels are the stems.
#     --last-n 10000                # score on the FINAL 10k draws of each run (convergence
#                                   #   knob; overrides --burn, per-run tail even if lengths differ)
#     --no-blocks                   # pooled scores only (skip the per-block breakdown)
#     --qq-per-block 3              # more QQ sites per block (0 = skip the QQ)
#
# One-shot (not resumable): under the Alpine sbatch submit with --array=1.
source "$(dirname "$0")/_faroe_env.sh"
# shellcheck disable=SC2086
mpiexec -n "$NRANKS" ${FAROE_RANKWRAP:-} python "$FAROE_SCRIPTS/faroe_task6_compare.py" "$@"
