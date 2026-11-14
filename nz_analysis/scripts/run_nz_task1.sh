#!/usr/bin/env bash
# Task 1 -- data prep (serial): build output/inputs.* + review figures.
source "$(dirname "$0")/_nz_env.sh"
NZ_PREP_CORES="${NZ_PREP_CORES:-1}" python "$NZ_SCRIPTS/nz_task1_prep.py"
