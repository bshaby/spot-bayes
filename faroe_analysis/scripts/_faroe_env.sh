#!/usr/bin/env bash
# Shared environment for the Faroe run scripts. `source` me from a run_faroe_task*.sh.
# Activates the `extremes` conda env, pins one thread per rank (thermal/oversubscription
# safety), sets NRANKS (default 64), exports FAROE_SCRIPTS, and cd's to the repo root.
# NB: no `set -u` -- Intel-MPI's mpivars activate script references SETVARS_CALL
# unbound, which would abort activation under `set -u`.
# This file is SOURCED, so `set -e` here applies to the SOURCING shell. When that
# shell is interactive that is actively dangerous: under errexit any command that
# returns non-zero exits the shell, and bash completion functions return non-zero
# routinely -- so `source _faroe_env.sh` followed by a TAB logs you out. The run_*.sh
# wrappers do want the strict options, so enable them only for non-interactive shells,
# and remember the caller's errexit so the conda bracket below RESTORES it instead of
# switching it on unconditionally (which would re-arm the same trap).
case $- in
  *i*) : ;;                       # interactive: leave the user's shell alone
  *)   set -eo pipefail ;;
esac
case $- in *e*) _faroe_had_e=1 ;; *) _faroe_had_e= ;; esac

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
conda activate "${FAROE_ENV:-extremes}"
if [ -n "$_faroe_had_e" ]; then set -e; fi
if [ "${CONDA_DEFAULT_ENV:-}" != "${FAROE_ENV:-extremes}" ]; then
  echo "ERROR: could not activate conda env '${FAROE_ENV:-extremes}'" "(got '${CONDA_DEFAULT_ENV:-none}')." >&2
  echo "       set FAROE_ENV, or CONDA_BASE=/path/to/conda if conda was not found." >&2
  # `exit` would close an interactive shell that sourced this file. `return` aborts
  # the sourced file and, because every wrapper runs under errexit, still stops them.
  if [ "${BASH_SOURCE[0]}" != "${0}" ]; then return 1; else exit 1; fi
fi
export PATH="${CONDA_PREFIX}/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# GSL for the RW C-extension: rw_inte_cpp.so dlopens libgsl.so.28 at import. On Alpine
# that comes from Lmod (gsl/2.8), NOT conda -- conda's gsl kept getting evicted whenever a
# channel-mixed `conda install` (e.g. numba from defaults) pulled MKL and dropped the
# OpenBLAS libblas gsl links against. Loading it here makes fits/probes find libgsl.so.28
# without depending on a fragile conda solve. Skipped cleanly where no module system
# exists (the workstation gets gsl from conda); a failed load surfaces loudly as an
# ImportError at runtime, so keeping it quiet here is safe. Override with FAROE_GSL_MODULE.
if type module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load ${FAROE_GSL_MODULE:-gcc/14.2.0 gsl/2.8} 2>/dev/null || true
fi

# Single-node MPI: launch all ranks LOCALLY on this node, ignoring any Slurm resource
# manager (we run <=64 ranks on one 64-core node, so no cross-node wire-up / PMI is
# needed). This is what makes a conda-provided MPI (OpenMPI / MPICH / Intel MPI) run
# cleanly inside a Slurm allocation without the cluster's PMI integration. Each MPI
# reads only its OWN variables; the others are harmless no-ops.
export I_MPI_HYDRA_BOOTSTRAP=fork              # Intel MPI: fork ranks locally
export HYDRA_BOOTSTRAP=fork                    # MPICH Hydra: fork ranks locally
export OMPI_MCA_plm=isolated                   # OpenMPI: launch locally, don't call srun
export OMPI_MCA_rmaps_base_oversubscribe=1     # OpenMPI: don't refuse if slots underdetected

# conda-forge MPICH 5.x links UCX, which on startup probes hardware it can't use here --
# CUDA (cuDeviceGetCount: no GPU on a CPU node) and the InfiniBand HCA (ib_iface: Protocol
# not supported for our single-node fork launch) -- logging a WARN per probe PER RANK, so at
# 64 ranks it floods the log and can bury mpi_probe.py's [3] result. We only ever run
# intra-node, so restrict UCX to shared-memory + self transports: kills the probes (and the
# spam) and keeps the fast on-node path. Overridable; a no-op for non-UCX MPIs.
export UCX_TLS="${UCX_TLS:-sm,self}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-error}"
# UCX_LOG_LEVEL=error silences the repeating ib_iface WARN flood but NOT the error-level
# `cuDeviceGetCount ... initialization error` UCX emits probing for an absent GPU. Tell
# MPICH not to do any GPU init, which stops that probe at the source (a no-op on GPU-less
# CPU nodes anyway). If the cuda errors somehow persist, set UCX_LOG_LEVEL=fatal.
export MPIR_CVAR_ENABLE_GPU="${MPIR_CVAR_ENABLE_GPU:-0}"

# OpenMPI single-node tuning (harmless no-ops for Intel MPI / MPICH):
#  * skip the InfiniBand (openib) BTL -- on one node ranks talk via shared memory, so
#    probing the IB card only produces "no device params" noise;
#  * keep the shared-memory backing file on NODE-LOCAL RAM (/dev/shm) instead of Lustre
#    (/scratch), which OpenMPI warns is slow -- important for the per-sweep allreduces.
#    Override the location with FAROE_OMPI_TMPDIR if /dev/shm is unavailable.
export OMPI_MCA_btl="^openib"
export OMPI_MCA_orte_tmpdir_base="${FAROE_OMPI_TMPDIR:-/dev/shm}"
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_shmem_mmap_enable_nfs_warning=0

# Yield the CPU when a rank blocks in a collective instead of busy-spinning. The fit
# does ~150 allreduces PER ITERATION, each a straggler barrier; on unpinned ranks the
# fast ones spin and steal cores from the straggler still computing -- a per-iter cost
# that no per-primitive micro-bench shows (see mpi_probe.py [2d]). OpenMPI-only var;
# MPICH/Intel ignore it (MPICH already blocks/yields rather than hard-spins by default,
# which is one reason the mpi4py->MPICH rebuild may help this exact path).
export OMPI_MCA_mpi_yield_when_idle=1

# --- NUMA-local rank binding (opt-in: export FAROE_MPI_BIND=core) -------------
# On a multi-NUMA cluster node (EPYC Milan is often NPS2/NPS4), UNBOUND ranks float
# across NUMA domains and their memory lands non-local, so every access to the
# streamed Z/S/R field pays a 2-5x remote penalty. Observed: identical Wang fit,
# 64 ranks, runs ~24x slower on an Alpine EPYC node (ranks unpinned, mask=64) than
# on a Threadripper workstation (flat/UMA, so placement is moot). Binding each rank
# to a core makes Linux first-touch put its pages on the local NUMA node.
#
# Opt-in, because it must NOT change the workstation path that already works and
# because OpenMPI's mapper can misbehave under the isolated/fork bootstrap. Verify it
# took with mpi_probe.py: the per-rank affinity mask should drop from 64 to 1.
#   sbatch --export=ALL,FAROE_MPI_BIND=core ...
# FAROE_MPIEXEC_ARGS is injected right after `mpiexec` in every launch line (the run
# scripts and the probe sbatch). Default empty -> launch is byte-for-byte unchanged when
# the lever is off, so the workstation path is untouched.
export FAROE_MPIEXEC_ARGS="${FAROE_MPIEXEC_ARGS:-}"
if [ -n "${FAROE_MPI_BIND:-}" ]; then
  export OMPI_MCA_hwloc_base_binding_policy="$FAROE_MPI_BIND"        # core (or numa)
  export OMPI_MCA_rmaps_base_mapping_policy="${FAROE_MPI_MAP:-core}" # one rank per core
  export I_MPI_PIN=1                                                 # Intel MPI: pin ranks
  export I_MPI_PIN_DOMAIN="${FAROE_MPI_MAP:-core}"
  # CRITICAL after the mpi4py->MPICH rebuild: MPICH's Hydra launcher ignores every
  # OMPI_MCA_* and I_MPI_* var above, so those alone leave ranks UNBOUND under MPICH.
  # Hydra binds only via the `-bind-to` CLI flag (single dash -- accepted by MPICH,
  # Intel, and OpenMPI Hydra launchers alike), so inject it too.
  #
  # `-map-by` is NOT optional here: `-bind-to core` ALONE tells Hydra to bind each rank
  # to one core but does NOT say WHICH -- under the fork bootstrap it stacked all 64
  # ranks onto a handful of cores (mpi_probe.py measured 27x cholesky contention and
  # 224 ms allreduces, vs 1x/9us unbound). `-map-by core` round-robins one rank per
  # DISTINCT core, which is the placement we actually want. Verify with mpi_probe.py:
  # per-rank mask -> 1 AND [2b] concurrent cholesky back to ~1x (NOT 27x) AND [2]
  # allreduce back to microseconds. If [2b] is still >>1x, the map-by didn't spread --
  # fall back to a per-rank wrapper: FAROE_NUMACTL pins+localallocs each rank instead.
  export FAROE_MPIEXEC_ARGS="${FAROE_MPIEXEC_ARGS} -bind-to $FAROE_MPI_BIND -map-by ${FAROE_MPI_MAP:-core}"
fi

# --- NUMA MEMORY policy (opt-in: export FAROE_NUMACTL=interleave) -------------
# The lever CPU-binding does NOT pull: memory PLACEMENT. On a partitioned (NPS2/NPS4)
# HPC node, an unbound job's pages concentrate on ONE NUMA domain, so all ranks share a
# single memory controller's bandwidth (~55 GB/s = one domain) while the other domains
# sit idle -- vs a workstation whose BIOS interleaves memory across all channels, giving
# every core the full aggregate (~200 GB/s). `numactl --interleave=all` reproduces the
# workstation's flat memory: every page round-robins across all controllers.
#
# FAROE_RANKWRAP is prepended to `python` INSIDE mpiexec, so it wraps each RANK (per-rank
# memory policy), not the launcher. The run scripts and probe use it.
#   sbatch --export=ALL,FAROE_NUMACTL=interleave ...
FAROE_RANKWRAP=""
case "${FAROE_NUMACTL:-}" in
  interleave) FAROE_RANKWRAP="numactl --interleave=all" ;;   # spread pages across domains
  local)      FAROE_RANKWRAP="numactl --localalloc" ;;       # pages on the rank's domain
  "")         : ;;
  *)          FAROE_RANKWRAP="numactl ${FAROE_NUMACTL}" ;;    # pass-through
esac
if [ -n "$FAROE_RANKWRAP" ] && ! command -v numactl >/dev/null 2>&1; then
  echo "WARNING: FAROE_NUMACTL set but numactl not found; ignoring." >&2
  FAROE_RANKWRAP=""
fi
export FAROE_RANKWRAP

export NRANKS="${NRANKS:-64}"
FAROE_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FAROE_SCRIPTS
cd "$(cd "$FAROE_SCRIPTS/../.." && pwd)"   # spot-bayes repo root
