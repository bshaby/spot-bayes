#!/usr/bin/env bash
# Task 5 (WARM) -- fit the shi model on the TRAIN data with SAMPLED margins, warm-started
# from the goodinit run's converged full-model state (Z field + all params). shi inherits goodinit's phi and S fields; its 13-knot rho field comes from the
# shi_fixmargins_stage1 rho tail-mean. Output:
#     faroe_analysis/output/train/shi_warm.*
# Consumed by Task 6 (the full-model, sampled-margin comparison).
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
echo "[task5-warm] shi, sampled margins, warm-started from $WARM"
# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} \
    python "$FAROE_SCRIPTS/faroe_fit.py" --model shi --which train --warm-field "$WARM" "$@"
