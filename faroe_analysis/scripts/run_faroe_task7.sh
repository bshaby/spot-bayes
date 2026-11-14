#!/usr/bin/env bash
# Task 7 -- assessment of the Wang full-data fit (Gumbel QQ + chi maps). Run after 2.
source "$(dirname "$0")/_faroe_env.sh"
mpiexec -n "$NRANKS" ${FAROE_RANKWRAP:-} python "$FAROE_SCRIPTS/faroe_task7_assess.py" "$@"
