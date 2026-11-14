#!/usr/bin/env bash
# Task 1 -- data prep (serial). Builds faroe_analysis/output/inputs.{pkl,npz} + figures.
source "$(dirname "$0")/_faroe_env.sh"
FAROE_PREP_CORES="${FAROE_PREP_CORES:-8}" python "$FAROE_SCRIPTS/faroe_task1_prep.py"
