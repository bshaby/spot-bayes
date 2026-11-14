#!/bin/bash -l
# ===========================================================================
# Alpine MPI diagnostic -- run mpi_probe.py exactly the way alpine_faroe.sbatch runs
# the real fits (same login shell, same CONDA_BASE, same fork-bootstrap MPI, same 1
# task x 64 cpus), so its verdict actually reflects the fits' environment.
#
# Use this INSTEAD of `sbatch --wrap=...`: --wrap runs under /bin/sh, and _faroe_env.sh
# is bash (BASH_SOURCE, arrays), so sourcing it under sh dies at init -- the giveaway is
# a bash-completion "syntax error near unexpected token '<'", not any probe output.
#
#     sbatch faroe_analysis/scripts/alpine_mpi_probe.sbatch
#     # optional: pin the probe's sampler-timing radius
#     sbatch --export=ALL,PROBE_RADIUS=7.5 faroe_analysis/scripts/alpine_mpi_probe.sbatch
#
# It finishes in ~1-2 min and backfills easily. Read mpiprobe_<jobid>.log:
#   WORLD SIZE == 1        -> MPI not wired (N serial copies); mpiexec/mpi4py mismatch
#   Allreduce in ms not us -> slow fabric; try I_MPI_FABRICS=shm (this env is Intel MPI)
#   real s/iter high anyway-> genuinely compute-bound
# ===========================================================================
#SBATCH --job-name=mpiprobe
#SBATCH --partition=acpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # ONE task owning the node (we FORK ranks, not srun)
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00               # [3] at ~100 s/iter needs ~8-10 min; 15 was too tight
#SBATCH --qos=cpu-normal
#SBATCH --account=YOUR_ALLOCATION     # EDIT: your Slurm allocation, or pass --account=... to sbatch
#SBATCH --output=mpiprobe_%j.log

set -eo pipefail

# --- conda (identical to alpine_faroe.sbatch) ------------------------------
export CONDA_BASE="${CONDA_BASE:-/curc/sw/anaconda3/2023.09}"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "WARNING: no conda.sh under CONDA_BASE=$CONDA_BASE -- falling back to Lmod." >&2
  unset CONDA_BASE
  module reset 2>/dev/null || module purge 2>/dev/null || true
  module load "${FAROE_CONDA_MODULE:-anaconda}"
fi
export FAROE_ENV="${FAROE_ENV:-extremes}"
export NRANKS="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-64}}"

_ncpu="${SLURM_CPUS_ON_NODE:-$NRANKS}"
if command -v taskset >/dev/null 2>&1; then
  taskset -pc "0-$((_ncpu - 1))" $$ >/dev/null 2>&1 || true
fi
echo "CPU check: nproc=$(nproc)  cpus_on_node=${_ncpu}  ranks=$NRANKS  mask=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')"

cd "${FAROE_REPO:-$SLURM_SUBMIT_DIR}"

# _faroe_env.sh sets the fork-bootstrap MPI vars (HYDRA_BOOTSTRAP=fork etc.), activates
# the env, and cd's to the repo root -- source it so the probe launches like the fits.
# shellcheck source=/dev/null
source faroe_analysis/scripts/_faroe_env.sh

echo "mpiprobe: ranks=$NRANKS  node=$(hostname)  conda_base=${CONDA_BASE:-<module>}" \
     "  rankwrap='${FAROE_RANKWRAP:-}'  probe_radius=${PROBE_RADIUS:-<none>}"
# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} python faroe_analysis/scripts/mpi_probe.py
