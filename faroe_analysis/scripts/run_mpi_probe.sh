#!/usr/bin/env bash
# Run mpi_probe.py the way the real fits run: source the shared env (which derives
# FAROE_MPIEXEC_ARGS / FAROE_RANKWRAP from FAROE_MPI_BIND / FAROE_NUMACTL) then launch,
# exactly like run_faroe_task*.sh. Kept as its own script so the sweep sbatch can invoke
# it once per config in a FRESH shell -- re-sourcing _faroe_env.sh in one shell would
# accumulate duplicate -bind-to flags in FAROE_MPIEXEC_ARGS.
source "$(dirname "$0")/_faroe_env.sh"
# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} python "$FAROE_SCRIPTS/mpi_probe.py" "$@"
