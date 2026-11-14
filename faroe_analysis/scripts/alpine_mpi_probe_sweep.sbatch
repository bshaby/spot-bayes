#!/bin/bash -l
# ===========================================================================
# Alpine MPI diagnostic SWEEP -- run mpi_probe.py under several rank-placement configs
# BACK-TO-BACK in ONE allocation, so every config executes on the SAME physical node.
#
# WHY: single-config submissions each land on a different node (1-NUMA vs 2-NUMA, and
# different noisy neighbours), so their [3] s/iter numbers are not comparable -- the
# 24x has been chasing a moving target. Running the configs in one job fixes the node,
# turning the comparison into a controlled A/B/C/D, and pays the multi-hour queue wait
# ONCE instead of per config.
#
#     sbatch --export=ALL,FAROE_OUT=/scratch/alpine/$USER/faroe_out \
#            faroe_analysis/scripts/alpine_mpi_probe_sweep.sbatch
#
# Read the log config-by-config (each fenced with a ##### CONFIG banner). Per config:
#   [2b] concurrent cholesky ~1x + [2] allreduce in us  -> 1 rank/core, placement good
#   [2b] ~27x + allreduce in ms                          -> ranks STACKED (bad placement)
#   NUMA: box has N node(s)                              -> is there anything to interleave?
#   [3] s/iter                                           -> the payoff; compare ACROSS configs
#
# Configs run in graceful-degradation order: the two with NO binding risk go first, so a
# wall-time kill still leaves the unpinned baseline + interleave comparison intact.
#   unpinned         OS schedules ranks (mask=64) -- current production behaviour
#   interleave       unpinned, but each rank's pages round-robin across NUMA nodes
#   bind_core        1 rank per distinct core (-bind-to core -map-by core)
#   bind_core_local  1 rank/core AND memory first-touched local (numactl --localalloc)
#
# Overrides: FAROE_SWEEP="name=VARS|name=VARS|..." to change the list; PROBE_ITERS to set
# timed [3] iterations (default 1); PROBE_SKIP_SAMPLER=1 for a fast placement-only sweep
# (skips the slow [3] entirely -- finishes in ~1 min/config).
# ===========================================================================
#SBATCH --job-name=mpiprobesweep
#SBATCH --partition=acpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # ONE task owning the node (we FORK ranks, not srun)
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00               # 4 configs x (~2 min probe + [3]); 1h leaves margin
#SBATCH --qos=cpu-normal
#SBATCH --account=YOUR_ALLOCATION     # EDIT: your Slurm allocation, or pass --account=... to sbatch
#SBATCH --output=mpiprobesweep_%j.log

set -eo pipefail

# --- conda (identical to alpine_mpi_probe.sbatch) --------------------------
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
echo "CPU check: nproc=$(nproc)  cpus_on_node=${_ncpu}  ranks=$NRANKS  node=$(hostname)" \
     "  mask=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')"

cd "${FAROE_REPO:-$SLURM_SUBMIT_DIR}"

# Config list: "name=VARS" entries separated by '|'; VARS is space-separated KEY=VAL
# passed to `env` for that run. Empty VARS (the "unpinned=" entry) = production default.
_default_sweep="unpinned=|interleave=FAROE_NUMACTL=interleave|bind_core=FAROE_MPI_BIND=core|bind_core_local=FAROE_MPI_BIND=core FAROE_NUMACTL=local"
IFS='|' read -r -a _cfgs <<< "${FAROE_SWEEP:-$_default_sweep}"

for _spec in "${_cfgs[@]}"; do
  _name="${_spec%%=*}"        # text before the first '=' is the label
  _cfgenv="${_spec#*=}"       # everything after it is the KEY=VAL env for this config
  echo
  echo "############################################################################"
  echo "##  CONFIG: ${_name}    [env: ${_cfgenv:-<none>}]    node=$(hostname)  $(date '+%H:%M:%S')"
  echo "############################################################################"
  # Fresh child shell per config: _faroe_env.sh APPENDS to FAROE_MPIEXEC_ARGS, so
  # re-sourcing it in this same shell would stack duplicate -bind-to flags. A child
  # `env ... bash` starts with FAROE_MPIEXEC_ARGS unset and derives it cleanly. `env`
  # (no -i) preserves the inherited environment (CONDA_BASE, NRANKS, FAROE_OUT, ...).
  # shellcheck disable=SC2086
  env ${_cfgenv} PROBE_ITERS="${PROBE_ITERS:-1}" \
      bash faroe_analysis/scripts/run_mpi_probe.sh \
    || echo "##  CONFIG ${_name} exited nonzero (rc=$?) -- continuing sweep"
done
echo
echo "##  SWEEP COMPLETE  $(date '+%H:%M:%S')"
