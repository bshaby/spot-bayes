#!/usr/bin/env bash
# Task 7 -- assessment of the full Wang fit (Gumbel QQ + chi maps).
#
# Defaults to the Task 2 two-stage, plug-in-initialised run. Point it elsewhere with
# --run <name.npz> (or NZ_RUN); output lands in output/assess_<run-stem>/.
#
# One-shot (not resumable): under the Alpine sbatch submit with --array=1.
source "$(dirname "$0")/_nz_env.sh"
# shellcheck disable=SC2086
mpiexec ${NZ_MPIEXEC_ARGS:-} -n "$NRANKS" ${NZ_RANKWRAP:-} \
    python "$NZ_SCRIPTS/nz_task7_assess.py" \
    --run "${NZ_RUN:-wang_full_fixmargins_stage1_plug.npz}" "$@"
