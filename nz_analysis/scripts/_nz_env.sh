#!/usr/bin/env bash
# Shared environment for the NZ run scripts. `source` me from a run_nz_task*.sh.
# Activates the `extremes` conda env, pins one thread per rank (thermal/oversubscription
# safety), sets NRANKS (default 16), exports NZ_SCRIPTS, and cd's to the repo root.
# NB: no `set -u` -- Intel-MPI's mpivars activate script references SETVARS_CALL
# unbound, which would abort activation under `set -u`.
# This file is SOURCED, so `set -e` here applies to the SOURCING shell. When that
# shell is interactive that is actively dangerous: under errexit any command that
# returns non-zero exits the shell, and bash completion functions return non-zero
# routinely -- so `source _nz_env.sh` followed by a TAB logs you out. The run_*.sh
# wrappers do want the strict options, so enable them only for non-interactive shells,
# and remember the caller's errexit so the conda bracket below RESTORES it instead of
# switching it on unconditionally (which would re-arm the same trap).
case $- in
  *i*) : ;;                       # interactive: leave the user's shell alone
  *)   set -eo pipefail ;;
esac
case $- in *e*) _nz_had_e=1 ;; *) _nz_had_e= ;; esac

# Locate conda robustly (override with CONDA_BASE=/path/to/conda). Falls back to
# common install dirs, then to an already-initialized conda on PATH.
_conda_base="${CONDA_BASE:-}"
if [ -z "$_conda_base" ]; then
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
           "/opt/conda" "/opt/anaconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then _conda_base="$c"; break; fi
  done
fi
if [ -z "$_conda_base" ] && command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
fi
if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$_conda_base/etc/profile.d/conda.sh"
fi
# Bracket the activation with `set +e`. conda runs the env's activate.d hooks here,
# and a hook that merely PROBES for something absent can return non-zero -- the
# libfabric one does `FIP=$(which fi_info)`, whose assignment inherits `which`'s exit
# status when fi_info is not on PATH. Under `set -e` that aborts the sourcing shell
# MID-ACTIVATION, and because it dies inside conda's own code the wrapper exits 1 with
# no message at all. Same family as the `set -u` note above: activate.d scripts are
# not written to survive a strict shell. Verify the activation ourselves afterwards.
set +e
conda activate "${NZ_ENV:-extremes}"
if [ -n "$_nz_had_e" ]; then set -e; fi
if [ "${CONDA_DEFAULT_ENV:-}" != "${NZ_ENV:-extremes}" ]; then
  echo "ERROR: could not activate conda env '${NZ_ENV:-extremes}'" "(got '${CONDA_DEFAULT_ENV:-none}')." >&2
  echo "       set NZ_ENV, or CONDA_BASE=/path/to/conda if conda was not found." >&2
  # `exit` would close an interactive shell that sourced this file. `return` aborts
  # the sourced file and, because every wrapper runs under errexit, still stops them.
  if [ "${BASH_SOURCE[0]}" != "${0}" ]; then return 1; else exit 1; fi
fi
export PATH="${CONDA_PREFIX}/bin:$PATH"
# geopandas/shapely need PROJ's data dir; point at the env's copy if present.
if [ -z "${PROJ_LIB:-}" ] && [ -d "${CONDA_PREFIX}/share/proj" ]; then
  export PROJ_LIB="${CONDA_PREFIX}/share/proj"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# GSL for the RW C-extension: rw_inte_cpp.so dlopens libgsl.so.28 at import. On Alpine
# that comes from Lmod (gsl/2.8), NOT conda -- conda's gsl kept getting evicted whenever a
# channel-mixed `conda install` (e.g. numba from defaults) pulled MKL and dropped the
# OpenBLAS libblas gsl links against. Loading it here makes fits find libgsl.so.28 without
# depending on a fragile conda solve. Skipped cleanly where no module system exists (the
# workstation gets gsl from conda); a failed load surfaces loudly as an ImportError at
# runtime, so keeping it quiet here is safe. Override with NZ_GSL_MODULE.
if type module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load ${NZ_GSL_MODULE:-gcc/14.2.0 gsl/2.8} 2>/dev/null || true
fi

# Single-node MPI: launch all ranks LOCALLY on this node, ignoring any Slurm resource
# manager (we run <=64 ranks on one 64-core node, so no cross-node wire-up / PMI is
# needed). Each MPI reads only its OWN variables; the others are harmless no-ops.
export I_MPI_HYDRA_BOOTSTRAP=fork              # Intel MPI: fork ranks locally
export HYDRA_BOOTSTRAP=fork                    # MPICH Hydra: fork ranks locally
export OMPI_MCA_plm=isolated                   # OpenMPI: launch locally, don't call srun
export OMPI_MCA_rmaps_base_oversubscribe=1     # OpenMPI: don't refuse if slots underdetected

# OpenMPI single-node tuning (harmless no-ops for Intel MPI / MPICH):
export OMPI_MCA_btl="^openib"
export OMPI_MCA_orte_tmpdir_base="${NZ_OMPI_TMPDIR:-/dev/shm}"
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0

# Quiet MPICH/UCX startup noise on CPU nodes. UCX probes InfiniBand and CUDA and
# spams WARN/ERROR lines (e.g. "ibv_query_port_speed ... Protocol not supported",
# "cuDeviceGetCount ... initialization error") -- harmless here since we run
# single-node over shared memory, but they bury real errors. These are logging /
# transport-selection knobs ONLY (they don't change what MPI does on one node);
# unset them if MPI ever misbehaves. Both honor a pre-set value so you can override.
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-error}"   # drop the InfiniBand WARN wall
export UCX_TLS="${UCX_TLS:-^cuda}"               # skip CUDA probing (no GPUs on amilan)

# MPI parallelism is OVER TIME BLOCKS (3080 of them), so any rank count divides the work
# cleanly. 16 is the production default because four 16-core jobs SCHEDULE sooner than four
# 64-core ones (a small request backfills into gaps a whole-node request waits for), and
# because this sampler does ~150 allreduces per iteration -- each a straggler barrier -- so
# parallel efficiency is already falling off well before 64 ranks. It is a claim about
# allocation size, NOT about placement: the jobs are independent submissions and Slurm may
# well scatter them across nodes, which is fine (see alpine_nz.sbatch on bandwidth).
# The trade is per-iteration wall: 3080/16 = 193 blocks per rank against Faroe's 1656/64 =
# 26. Raise it for a single fit you want back fast:
#     NRANKS=64 bash nz_analysis/scripts/run_nz_task2.sh
export NRANKS="${NRANKS:-16}"
NZ_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NZ_SCRIPTS
cd "$(cd "$NZ_SCRIPTS/../.." && pwd)"   # spot-bayes repo root

# Launch-line hooks, both empty by default so the launch is byte-for-byte unchanged when
# they are unset. NZ_MPIEXEC_ARGS goes right after `mpiexec` (e.g. "-bind-to core -map-by
# core" to pin ranks on a multi-NUMA cluster node); NZ_RANKWRAP is prepended to `python`
# INSIDE mpiexec, so it wraps each RANK (e.g. "numactl --interleave=all" for a per-rank
# memory policy). See _faroe_env.sh for what these are for and how to verify they took.
export NZ_MPIEXEC_ARGS="${NZ_MPIEXEC_ARGS:-}"
export NZ_RANKWRAP="${NZ_RANKWRAP:-}"

# Stage-1 guard used by the Task 2-5 wrappers: resolve NZ_STAGE1_NPZ (honouring a pre-set
# value), abort with instructions if the marginal fit has not been run, and announce it.
nz_require_stage1() {
  export NZ_STAGE1_NPZ="${NZ_STAGE1_NPZ:-${NZ_OUT:-$NZ_SCRIPTS/../output}/marginal/marginal_fit.npz}"
  if [ ! -f "$NZ_STAGE1_NPZ" ]; then
    echo "ERROR: stage-1 marginal output not found at $NZ_STAGE1_NPZ" >&2
    echo "       run  bash nz_analysis/scripts/run_nz_marginal.sh  first." >&2
    return 1        # not exit: the wrappers run under errexit, and a hand-sourced
                    # shell should not be closed by calling a helper
  fi
  echo "[${1:-task}] fixing margins from $NZ_STAGE1_NPZ"
}
