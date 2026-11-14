#!/bin/bash -l
# ===========================================================================
# Alpine (CU Research Computing) Slurm batch script for the NZ fits.
#
# NB `-l` above: a LOGIN shell, so /etc/profile.d/*.sh runs and the module system is
# initialised the same way it is in an interactive session. Without it the module
# function may be missing or MODULEPATH incomplete on the compute node.
#
# Runs ONE task with 16 MPI ranks. Because it is a single node, we launch the
# ranks with the conda env's own mpiexec using a LOCAL (fork) bootstrap -- no
# Slurm/PMI/srun integration required, which is the most robust way to run a
# conda-provided MPI (OpenMPI / MPICH / Intel MPI) under Slurm.
#
# WHY 16 AND NOT 64: MPI parallelism here is over TIME BLOCKS (3080 of them), so
# any rank count divides the work. Four 16-core jobs SCHEDULE sooner than four
# 64-core ones -- a 16-core request backfills into gaps a whole-node request has to
# wait for -- and in ideal scaling 4x16 running concurrently finishes in about the
# same wall time as 4x64 running one after another. In practice 16 does better than
# that, because the sampler does ~150 allreduces PER ITERATION and each is a
# straggler barrier, so parallel efficiency falls off well before 64 ranks.
#
# This is about ALLOCATION SIZE, not placement. The four jobs are independent
# submissions; Slurm puts each wherever it finds 16 free cores, which may be four
# different nodes. That is fine, and better than the alternative -- see the
# --cpus-per-task note below.
#
# The trade is per-iteration wall (3080/16 = 193 blocks per rank). For a single fit
# you want back fast, submit with --cpus-per-task=64.
#
# EDIT the marked lines (account, time/qos, and which task), then:
#     sbatch nz_analysis/scripts/alpine_nz.sbatch
# Submit FROM the repo root (spot-bayes/), or set NZ_REPO below.
#
# The fits checkpoint every ~30 min and AUTO-RESUME: if a job hits its wall-time
# limit, just `sbatch` this script again and it continues from the checkpoint.
#
# LONGER-THAN-24h RUNS: the --array line chains consecutive wall-time windows
# automatically. `--array=1-4%1` submits 4 elements that run ONE AT A TIME (%1); each
# resumes from the checkpoint the previous one left. Once the fit reaches its target
# iteration the remaining elements detect it and no-op in seconds -- so over-
# provisioning the array count is harmless.
# ===========================================================================
#SBATCH --job-name=nz
#SBATCH --partition=acpu            # standard AMD Milan CPU partition (64-core nodes)
#SBATCH --nodes=1
# ONE Slurm task owning all 16 cores -- NOT --ntasks=16. We do not launch with srun: the
# conda mpiexec FORKS its ranks from this batch shell. With --ntasks=16 the batch step is
# task 0 and Slurm binds it to a SINGLE CPU, so every forked rank inherits that one-core
# affinity mask and all 16 ranks timeshare one core (Faroe measured ~36 s/iter instead of
# ~1). Asking for 1 task x 16 cpus gives the batch shell a 16-core mask to fork into.
#
# Faroe holds the WHOLE node (--cpus-per-task=64) even when running fewer ranks, to keep a
# bandwidth-competing co-tenant off it: that fit is memory-bandwidth bound
# (update_nu_rho_nuts is recomputed identically on EVERY rank and streams ~70 MB per
# gradient), so a neighbour costs more than the idle cores do. That reasoning applies here
# too, and it cuts AGAINST packing our own fits together -- four NZ fits on one node would
# compete for its memory controllers exactly as a stranger's job would. Slurm scattering
# the four submissions across nodes is the good outcome, not a missed optimisation.
#
# If a fit looks bandwidth-starved, buy it the whole node rather than the whole rank count:
#     sbatch --cpus-per-task=64 --export=ALL,NRANKS=16 ...
# which allocates 64 cores but still runs 16 ranks, leaving the rest idle on purpose.
#
# CHANGING THE RANK COUNT: NRANKS follows SLURM_CPUS_PER_TASK, and a --cpus-per-task on
# the sbatch COMMAND LINE overrides the directive below, so a later resume window can run
# wider with no edit to this file:
#     sbatch --cpus-per-task=32 --export=ALL,NZ_TASK=5 nz_analysis/scripts/alpine_nz.sbatch
# Two conditions. (a) REPARTITION THE CHECKPOINT FIRST -- run_mcmc_mpi refuses a resume
# whose rank count differs from the one that wrote it:
#     bash nz_analysis/scripts/run_nz_repartition.sh \
#          --ckpt "$NZ_OUT/train/shi_fixmargins_stage1_plug.ckpt" --ranks 32
# (b) make sure no stale NRANKS is exported in your login shell: --export=ALL carries it
# through and it BEATS the flag. The "CPU check:" line below names which source won.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00               # EDIT: cpu-normal qos <=24h; for a longer single run use --qos=cpu-long --time=7-00:00:00
#SBATCH --qos=cpu-normal                  # EDIT: 'cpu-normal' (<=24h) or 'cpu-long' (<=7d, lower priority)
#SBATCH --account=YOUR_ALLOCATION     # EDIT: your Slurm allocation, or pass --account=... to sbatch
#SBATCH --array=1-4%1                  # EDIT: consecutive resume windows for >24h fits (%1 = one at a
                                       #   time). Short/one-shot (tasks 6,7): override at submit with --array=1.
#SBATCH --output=nz_%x_%A_%a.log       # -> nz_<jobname>_<arrayjobid>_<arrayidx>.log

set -eo pipefail

# --- which task to run: 1=prep, marginal=STAGE 1 (run this before 2-5), -------
# --- 2=Wang full, 3=hw train, 4=wang train, 5=shi train, 6=compare, 7=assess. -
# --- Override at submit: sbatch --export=ALL,NZ_TASK=4 ... --------------------
NZ_TASK="${NZ_TASK:-2}"                # EDIT or pass via --export

# The array chains RESUME windows, which only makes sense for the checkpointing runs
# (the marginal fit and tasks 2-5). Prep and the post-processing tasks run once and are
# not resumable, so skip any array element past the first to avoid recomputation.
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-1}"
case "$NZ_TASK" in
  1|6|7) ONE_SHOT=1 ;;
  *)     ONE_SHOT=0 ;;
esac
if [ "$ONE_SHOT" = "1" ] && [ "$ARRAY_IDX" -gt 1 ]; then
  echo "Task $NZ_TASK is one-shot (not resumable); array element $ARRAY_IDX -> skip."
  exit 0
fi

# --- conda -----------------------------------------------------------------
# Point STRAIGHT at the CURC anaconda install and skip Lmod entirely. `module load
# anaconda` succeeds in an interactive job on a compute node but fails from the batch
# script on the same node type ("Lmod has detected the following error: The following
# module(s) are unknown: anaconda"), so the module system is not a dependable way in from
# here. _nz_env.sh sources $CONDA_BASE/etc/profile.d/conda.sh directly, needing no modules.
#
# Override the prefix at submit time if CURC moves it:
#     sbatch --export=ALL,CONDA_BASE=/curc/sw/anaconda3/<version> ...
export CONDA_BASE="${CONDA_BASE:-/curc/sw/anaconda3/2023.09}"

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "WARNING: no conda.sh under CONDA_BASE=$CONDA_BASE -- falling back to Lmod." >&2
  unset CONDA_BASE
  module reset 2>/dev/null || module purge 2>/dev/null || true
  module load "${NZ_CONDA_MODULE:-anaconda}"
fi
export NZ_ENV="${NZ_ENV:-extremes}"    # the env you created on Alpine (see README)

# NB: the single-node local-launch settings (OMPI_MCA_plm=isolated for OpenMPI,
# HYDRA_BOOTSTRAP=fork for MPICH/Intel) are set by _nz_env.sh, which every
# run_nz_task*.sh sources -- so `mpiexec` launches all ranks on this node without
# needing Slurm's MPI/PMI integration.
# Ranks default to the cores Slurm gave us, but an EXPLICIT NRANKS wins:
#   sbatch --export=ALL,NRANKS=32 ...
# Record WHERE the value came from: --export=ALL faithfully carries a stale NRANKS from
# your login shell, which then silently beats an explicit --cpus-per-task on the sbatch
# line. That surfaces much later as an opaque rank-count mismatch against the checkpoint,
# so name the source in the CPU check line below.
if [ -n "${NRANKS:-}" ]; then _nranks_src="explicit NRANKS env"; else
  _nranks_src="SLURM_CPUS_PER_TASK"; fi
export NRANKS="${NRANKS:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-16}}}"

# Belt and braces: widen this shell's CPU mask before the ranks are forked, in case the
# site's task/affinity plugin narrowed the batch step to one core.
#
# ONLY widen when the mask is genuinely too small. On a PARTIAL-node allocation (which is
# the normal case here at 16 of 64 cores) the cores Slurm gives us need not be 0..n-1:
# with a co-tenant holding 0-15 we might get 16-31, and `taskset -pc 0-15` would then be
# intersected by the kernel down to nothing usable. `nproc` already reports the current
# affinity mask, so if it covers the ranks we need, leave Slurm's placement alone.
_ncpu="${SLURM_CPUS_ON_NODE:-$NRANKS}"
if [ "$(nproc)" -lt "$NRANKS" ] && command -v taskset >/dev/null 2>&1; then
  echo "CPU check: mask has only $(nproc) cpu(s) for $NRANKS ranks -- widening to 0-$((_ncpu - 1))"
  taskset -pc "0-$((_ncpu - 1))" $$ >/dev/null 2>&1 || true
fi
echo "CPU check: nproc=$(nproc)  cpus_on_node=${_ncpu}  ranks=$NRANKS (from ${_nranks_src})  mask=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')"
# ^ If mask is a single CPU (e.g. "0") the ranks will all land on one core: STOP and fix
#   the affinity before burning a 24h window -- expect a 16-wide range.

# --- output location: SCRATCH (default) ------------------------------------------
# /home has a small quota and the streamed latent Z/S traces are large, so all task
# output (inputs.pkl, fits, latent .npy, figures) goes to scratch. This defaults to
# /scratch/alpine/$USER/nz_out and is CREATED here -- no manual mkdir needed, and
# nothing to pass at submit time. Override with --export=ALL,NZ_OUT=/some/other/path.
#
# IMPORTANT: Task 1 (run interactively before the fits) MUST write to the SAME NZ_OUT
# so the fits find inputs.pkl. Either export the identical value in your interactive
# shell (`export NZ_OUT=/scratch/alpine/$USER/nz_out`) before running Task 1, or run
# Task 1 through this script too: `sbatch --array=1 --export=ALL,NZ_TASK=1 ...`.
export NZ_OUT="${NZ_OUT:-/scratch/alpine/$USER/nz_out}"
mkdir -p "$NZ_OUT"

# --- go to the repo and run the task via its normal run script -----------------
# The stage-1 marginal fit has no task number (it precedes them all), so it gets its own
# name here rather than being squeezed into the numbering.
case "$NZ_TASK" in
  marginal|stage1) RUN_SCRIPT="nz_analysis/scripts/run_nz_marginal.sh" ;;
  *)               RUN_SCRIPT="nz_analysis/scripts/run_nz_task${NZ_TASK}.sh" ;;
esac
cd "${NZ_REPO:-$SLURM_SUBMIT_DIR}"
echo "Alpine job: task=$NZ_TASK  ranks=$NRANKS  node=$(hostname)  env=$NZ_ENV" \
     "  NZ_OUT=$NZ_OUT  conda_base=${CONDA_BASE:-<module>}  args='${NZ_ARGS:-}'" \
     "  array=${SLURM_ARRAY_JOB_ID:-none}:${ARRAY_IDX}/${SLURM_ARRAY_TASK_COUNT:-1}"
# NZ_ARGS is forwarded to the task script, so the flags work under Slurm:
#   sbatch --export=ALL,NZ_TASK=2,NZ_ARGS="--fix-radius 2.6" ...
# (unquoted on purpose -- it must word-split into separate arguments)
# shellcheck disable=SC2086
bash "$RUN_SCRIPT" ${NZ_ARGS:-}
