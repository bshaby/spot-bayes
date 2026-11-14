#!/bin/bash -l
# ===========================================================================
# Alpine (CU Research Computing) Slurm batch script for the Faroe fits.
#
# NB `-l` above: a LOGIN shell, so /etc/profile.d/*.sh runs and the module system is
# initialised the same way it is in an interactive session. Without it the module
# function may be missing or MODULEPATH incomplete on the compute node.
#
# Runs ONE task on a SINGLE acpu node (64 cores). Because it is a single node,
# we launch the ranks with the conda env's own mpiexec using a LOCAL (fork)
# bootstrap -- no Slurm/PMI/srun integration required, which is the most robust
# way to run a conda-provided MPI (Intel MPI or MPICH) under Slurm.
#
# EDIT the three marked lines (account, time/qos, and which task), then:
#     sbatch faroe_analysis/scripts/alpine_faroe.sbatch
# Submit FROM the repo root (spot-bayes/), or set FAROE_REPO below.
#
# The fits checkpoint every ~30 min and AUTO-RESUME: if a job hits its wall-time
# limit, just `sbatch` this script again and it continues from the checkpoint.
#
# LONGER-THAN-24h RUNS: the --array line below chains consecutive wall-time windows
# automatically. `--array=1-4%1` submits 4 elements that run ONE AT A TIME (%1); each
# element resumes from the checkpoint the previous one left, so 4 x 24h gives up to
# ~96h of compute under the cpu-normal qos without babysitting. Elements run in sequence
# regardless of how the previous one exited (wall-time kill included), and once the
# fit reaches its target iteration the remaining elements detect it and no-op in
# seconds -- so over-provisioning the array count is harmless.
# ===========================================================================
#SBATCH --job-name=faroe
#SBATCH --partition=acpu            # standard AMD Milan CPU partition (64-core nodes)
#SBATCH --nodes=1
# ONE Slurm task owning all 64 cores -- NOT --ntasks=64. We do not launch with srun: the
# conda mpiexec FORKS its 64 ranks from this batch shell. With --ntasks=64 the batch step
# is task 0 and Slurm binds it to a single CPU, so every forked rank inherits that
# one-core affinity mask and all 64 ranks timeshare one core (~36 s/iter instead of ~1).
# Asking for 1 task x 64 cpus gives the batch shell the whole node's mask to fork into.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00               # EDIT: cpu-normal qos <=24h; for a longer single run use --qos=cpu-long --time=7-00:00:00
#SBATCH --qos=cpu-normal                  # EDIT: 'cpu-normal' (<=24h) or 'cpu-long' (<=7d, lower priority)
#SBATCH --account=YOUR_ALLOCATION     # EDIT: your Slurm allocation, or pass --account=... to sbatch
#SBATCH --array=1-4%1                  # EDIT: consecutive resume windows for >24h fits (%1 = one at a
                                       #   time). Long fits: bump the 4. Short/one-shot (tasks 6,7 or a
                                       #   run that fits in one window): override at submit with --array=1.
#SBATCH --output=faroe_%x_%A_%a.log    # -> faroe_<jobname>_<arrayjobid>_<arrayidx>.log

set -eo pipefail

# --- which task to run (2=Wang full, 3=hw train, 4=wang train, 5=shi train, ---
# --- 6=compare, 7=assess). Override at submit: sbatch --export=ALL,FAROE_TASK=4 ... ---
FAROE_TASK="${FAROE_TASK:-2}"          # EDIT or pass via --export

# The array chains RESUME windows, which only makes sense for the checkpointing fits
# (tasks 2-5). The post-processing tasks (6=compare, 7=assess) run once and are not
# resumable, so skip any array element past the first to avoid recomputation.
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-1}"
if { [ "$FAROE_TASK" = "6" ] || [ "$FAROE_TASK" = "7" ]; } && [ "$ARRAY_IDX" -gt 1 ]; then
  echo "Task $FAROE_TASK is one-shot (not resumable); array element $ARRAY_IDX -> skip."
  exit 0
fi

# --- conda -----------------------------------------------------------------
# Point STRAIGHT at the CURC anaconda install and skip Lmod entirely. `module load
# anaconda` succeeds in an interactive job on a compute node but fails from the batch
# script on the same node type ("unknown module"), so the module system is not a
# dependable way in from here. _faroe_env.sh sources $CONDA_BASE/etc/profile.d/conda.sh
# directly, which needs no modules at all.
#
# Override the prefix at submit time if CURC moves it:
#     sbatch --export=ALL,CONDA_BASE=/curc/sw/anaconda3/<version> ...
export CONDA_BASE="${CONDA_BASE:-/curc/sw/anaconda3/2023.09}"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "WARNING: no conda.sh under CONDA_BASE=$CONDA_BASE -- falling back to Lmod." >&2
  unset CONDA_BASE
  module reset 2>/dev/null || module purge 2>/dev/null || true
  module load "${FAROE_CONDA_MODULE:-anaconda}"
fi
export FAROE_ENV="${FAROE_ENV:-extremes}"   # the env you created on Alpine (see README)

# NB: the single-node local-launch settings (OMPI_MCA_plm=isolated for OpenMPI,
# HYDRA_BOOTSTRAP=fork for MPICH/Intel) are set by _faroe_env.sh, which every
# run_faroe_task*.sh sources -- so `mpiexec` launches all ranks on this node without
# needing Slurm's MPI/PMI integration.
# Ranks default to the cores Slurm gave us, but an EXPLICIT NRANKS wins:
#   sbatch --export=ALL,NRANKS=32 ...
# Keep --cpus-per-task=64 even when running fewer ranks. Asking Slurm for only the
# cores you will use invites a co-tenant onto the other half of the node, and this fit
# is memory-bandwidth bound (update_nu_rho_nuts is recomputed identically on EVERY rank
# and streams ~70 MB per gradient), so a neighbour competing for bandwidth costs more
# than the idle cores do. Holding the whole node also leaves room to run a second
# independent chain alongside the first.
# Record WHERE the value came from: --export=ALL faithfully carries a stale NRANKS from
# your login shell, which then silently beats an explicit --cpus-per-task on the sbatch
# line. That surfaces much later as an opaque rank-count mismatch against the
# checkpoint, so name the source in the CPU check line below.
if [ -n "${NRANKS:-}" ]; then _nranks_src="explicit NRANKS env"; else
  _nranks_src="SLURM_CPUS_PER_TASK"; fi
export NRANKS="${NRANKS:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-64}}}"

# Belt and braces: widen this shell's CPU mask before the ranks are forked, in case the
# site's task/affinity plugin narrowed the batch step to one core.
#
# ONLY widen when the mask is genuinely too small. On a PARTIAL-node allocation the
# cores Slurm gives us need not be 0..n-1: with a co-tenant holding 0-15 we might get
# 16-47, and `taskset -pc 0-31` would then be intersected by the kernel down to 16-31 --
# HALF the cores, for the full rank count, silently. `nproc` already reports the current
# affinity mask, so if it covers the ranks we need, leave Slurm's placement alone.
_ncpu="${SLURM_CPUS_ON_NODE:-$NRANKS}"
if [ "$(nproc)" -lt "$NRANKS" ] && command -v taskset >/dev/null 2>&1; then
  echo "CPU check: mask has only $(nproc) cpu(s) for $NRANKS ranks -- widening to 0-$((_ncpu - 1))"
  taskset -pc "0-$((_ncpu - 1))" $$ >/dev/null 2>&1 || true
fi
echo "CPU check: nproc=$(nproc)  cpus_on_node=${_ncpu}  ranks=$NRANKS (from ${_nranks_src})  mask=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')"
# ^ If mask is a single CPU (e.g. "0") the ranks will all land on one core: STOP and fix
#   the affinity before burning a 24h window -- expect roughly "0-63".

# --- output location: SCRATCH (default) ------------------------------------------
# /home has a small quota and the streamed latent Z/S traces are large, so all task
# output (inputs.pkl, fits, latent .npy, figures, and the marginal run) goes to scratch.
# This defaults to /scratch/alpine/$USER/faroe_out and is CREATED here -- no manual mkdir
# needed, nothing to pass at submit time. Override with --export=ALL,FAROE_OUT=/some/path.
#
# IMPORTANT: the interactively-run steps (Task 1 prep and run_faroe_marginal.sh) MUST write
# to the SAME FAROE_OUT so the fits find inputs.pkl and marginal_fit.npz. Either export the
# identical value in your shell (`export FAROE_OUT=/scratch/alpine/$USER/faroe_out`) before
# running them, or run Task 1 through this script: `sbatch --array=1 --export=ALL,FAROE_TASK=1 ...`.
export FAROE_OUT="${FAROE_OUT:-/scratch/alpine/$USER/faroe_out}"
mkdir -p "$FAROE_OUT"

# --- go to the repo and run the task via its normal run script -----------------
cd "${FAROE_REPO:-$SLURM_SUBMIT_DIR}"
echo "Alpine job: task=$FAROE_TASK  ranks=$NRANKS  node=$(hostname)  env=$FAROE_ENV" \
     "  FAROE_OUT=$FAROE_OUT  conda_base=${CONDA_BASE:-<module>}  args='${FAROE_ARGS:-}'" \
     "  array=${SLURM_ARRAY_JOB_ID:-none}:${ARRAY_IDX}/${SLURM_ARRAY_TASK_COUNT:-1}"
# FAROE_ARGS is forwarded to the task script, so the flags work under Slurm:
#   sbatch --export=ALL,FAROE_TASK=2,FAROE_ARGS="--fix-radius 7.5" ...
# (unquoted on purpose -- it must word-split into separate arguments)
# shellcheck disable=SC2086
bash "faroe_analysis/scripts/run_faroe_task${FAROE_TASK}.sh" ${FAROE_ARGS:-}
