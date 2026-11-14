#!/usr/bin/env bash
# Task 2 (WARM) -- fit the wang model on the FULL 432-site data with SAMPLED margins,
# warm-started from the goodinit run's converged state (Z field + all params). This is
# the full-domain sampled-margin fit for Task 7 assessment (~ a pipeline-named
# continuation of goodinit). Output:
#     faroe_analysis/output/wang_full_warm.*
#
# PREREQUISITE: goodinit_field.npz (reassemble_goodinit.py) in output/goodinit/.
# Env overrides: NRANKS(64), FAROE_WARMFIELD, plus faroe_setup knobs.
source "$(dirname "$0")/_faroe_env.sh"
WARM="${FAROE_WARMFIELD:-${FAROE_OUT:-$FAROE_SCRIPTS/../output}/goodinit/goodinit_field.npz}"
if [ ! -f "$WARM" ]; then
  echo "ERROR: goodinit field not found at $WARM" >&2
  echo "       run  python $FAROE_SCRIPTS/reassemble_goodinit.py --float32  first." >&2
  exit 1
fi
echo "[task2-warm] wang full, sampled margins, warm-started from $WARM"
# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} \
    python "$FAROE_SCRIPTS/faroe_fit.py" --model wang --which full --warm-field "$WARM" "$@"
