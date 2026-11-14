#!/usr/bin/env bash
# Preflight sanity check -- run BEFORE launching the MPI fits (Tasks 2-5).
# Verifies, in the same env the fits use:
#   1. the conda env activates and the core Python deps import;
#   2. the mpiexec launcher and mpi4py library actually agree (a real 2-rank
#      allreduce -- catches the "size-1 singleton" wrong-MPI failure);
#   3. the input data files are present, and whether Task 1 has been run.
# Exits non-zero if deps are missing, data is missing, or the MPI pairing is broken.
#
#   bash faroe_analysis/scripts/run_faroe_preflight.sh          # 2-rank test
#   PREFLIGHT_RANKS=4 bash faroe_analysis/scripts/run_faroe_preflight.sh
source "$(dirname "$0")/_faroe_env.sh"     # activates env, cd repo root, sets FAROE_SCRIPTS
set +e                                     # report problems, don't abort on the first one
DATA_DIR="$FAROE_SCRIPTS/../data"
fail=0

echo "=== conda env ==="
echo "env: ${CONDA_DEFAULT_ENV:-?}    python: $(command -v python)"

echo "=== core Python deps ==="
python - <<'PY'
mods = ["numpy", "scipy", "matplotlib", "shapely", "networkx", "mpi4py"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m, str(e)))
for m, e in missing:
    print(f"  MISSING {m}: {e}")
if missing:
    raise SystemExit(1)
print("  core deps import OK:", ", ".join(mods))
try:
    import geopandas  # noqa: F401
    print("  geopandas OK (coastline overlay available)")
except Exception:
    print("  geopandas absent (coastline overlay skipped -- harmless)")
PY
[ $? -ne 0 ] && fail=1

echo "=== compiled RW extension (cpp/rw_inte_cpp.so) ==="
REPO_ROOT="$(cd "$FAROE_SCRIPTS/../.." && pwd)"
if PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -c "import rw_inte" 2>/dev/null; then
    echo "  rw_inte_cpp.so builds/loads OK"
else
    echo "  NOT BUILT: the native RW extension did not load. Build it (C++ + GSL):"
    echo "     conda install -c conda-forge gsl patchelf   # if the env lacks them"
    echo "     module load gcc                             # or: conda install -c conda-forge gxx"
    echo "     make -C \"$REPO_ROOT/cpp\""
    fail=1
fi

echo "=== MPI launcher / library ==="
echo "mpiexec: $(command -v mpiexec || echo 'NOT FOUND')"
mpiexec --version 2>&1 | head -1
echo "cores available here: $(nproc 2>/dev/null || echo '?')   SLURM_NTASKS=${SLURM_NTASKS:-unset}"
# A DIRECT singleton MPI_Init can fail under OpenMPI-in-Slurm even when launched jobs
# work fine, so this is informational only -- the N-rank test below is authoritative.
python -c "from mpi4py import MPI; print('mpi4py built against:', MPI.Get_library_version().splitlines()[0])" 2>/dev/null \
    || echo "(singleton MPI query failed -- not fatal under OpenMPI+Slurm; see the launch test below)"

echo "=== ${PREFLIGHT_RANKS:-2}-rank wire-up test ==="
R="${PREFLIGHT_RANKS:-2}"
out="$(mpiexec -n "$R" python -c "from mpi4py import MPI
c = MPI.COMM_WORLD
peers = c.allreduce(1, op=MPI.SUM)
if c.Get_rank() == 0:
    print(f'PREFLIGHT_MPI_PEERS={peers}')" 2>&1)"
echo "$out" | sed 's/^/  /'
peers="$(printf '%s\n' "$out" | sed -n 's/^PREFLIGHT_MPI_PEERS=//p' | head -1)"
nlines="$(printf '%s\n' "$out" | grep -c '^PREFLIGHT_MPI_PEERS=')"
if [ "$peers" = "$R" ] && [ "$nlines" = "1" ]; then
    echo "  MPI OK: launcher and mpi4py agree ($R ranks in one world)."
else
    echo "  MPI WIRE-UP FAILED: expected exactly one 'PREFLIGHT_MPI_PEERS=$R'."
    echo "     ($nlines lines / peers=${peers:-?}.) Common causes:"
    echo "     * 'not enough slots' -> this session has < $R cores; request more"
    echo "       (e.g. acompile --ntasks=$R, or sinteractive --ntasks=$R), then retry."
    echo "     * OpenMPI 'not built with SLURM PMI' -> the launcher tried to use Slurm;"
    echo "       _faroe_env.sh now sets OMPI_MCA_plm=isolated to launch locally instead"
    echo "       -- make sure you re-pulled. If it persists, recreate the env with mpich"
    echo "       (conda create ... -c conda-forge mpich mpi4py ...)."
    echo "     * two size-1 singletons -> mpiexec and mpi4py are from different MPIs."
    fail=1
fi

echo "=== input data ==="
for f in faroe_workflow_outputs.npz faroe_block_maxima.npz \
         sigma_covariates_std_2stage_MCMC.csv xi_covariates_std_2stage_MCMC.csv \
         buffered_faroe_cliques.json buffered_faroe_separators.json; do
    if [ -f "$DATA_DIR/$f" ]; then echo "  data/$f OK"; else echo "  MISSING data/$f"; fail=1; fi
done
INP="${FAROE_OUT:-$FAROE_SCRIPTS/../output}/inputs.pkl"
if [ -f "$INP" ]; then
    echo "  output/inputs.pkl present (Task 1 has been run)"
else
    echo "  NOTE: output/inputs.pkl not found -- run Task 1 before the fits:"
    echo "        bash faroe_analysis/scripts/run_faroe_task1.sh"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "PREFLIGHT: PASS -- safe to launch the fits."
else
    echo "PREFLIGHT: FAIL -- resolve the items above before launching."
fi
exit "$fail"
