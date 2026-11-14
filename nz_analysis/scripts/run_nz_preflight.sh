#!/usr/bin/env bash
# Preflight sanity check -- run BEFORE launching the MPI fits (Tasks 2-5).
# Verifies, in the same env the fits use:
#   1. the conda env activates and the core Python deps import (geopandas REQUIRED
#      here -- the barrier is a shapefile, not a hardcoded polygon);
#   2. the mpiexec launcher and mpi4py library actually agree (a real 2-rank
#      allreduce -- catches the "size-1 singleton" wrong-MPI failure);
#   3. the input data files are present, and whether Task 1 has been run.
# Exits non-zero if deps are missing, data is missing, or the MPI pairing is broken.
#
#   bash nz_analysis/scripts/run_nz_preflight.sh          # 2-rank test
#   PREFLIGHT_RANKS=4 bash nz_analysis/scripts/run_nz_preflight.sh
source "$(dirname "$0")/_nz_env.sh"     # activates env, cd repo root, sets NZ_SCRIPTS
set +e                                  # report problems, don't abort on the first one
DATA_DIR="$NZ_SCRIPTS/../data"
fail=0

echo "=== conda env ==="
echo "env: ${CONDA_DEFAULT_ENV:-?}    python: $(command -v python)    PROJ_LIB=${PROJ_LIB:-unset}"

echo "=== core Python deps ==="
python - <<'PY'
mods = ["numpy", "scipy", "matplotlib", "shapely", "networkx", "mpi4py", "geopandas"]
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
PY
[ $? -ne 0 ] && fail=1

echo "=== barrier shapefile loads (geopandas + PROJ) ==="
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["NZ_SCRIPTS"]))
try:
    import nz_setup as N
    poly = N.barrier_polygon()
    print(f"  barrier OK: {poly.geom_type}, bounds {tuple(round(b,2) for b in poly.bounds)}")
except Exception as e:
    print(f"  BARRIER LOAD FAILED: {e}")
    raise SystemExit(1)
PY
[ $? -ne 0 ] && fail=1

echo "=== compiled RW extension (cpp/rw_inte_cpp.so) ==="
REPO_ROOT="$(cd "$NZ_SCRIPTS/../.." && pwd)"
if PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -c "import rw_inte" 2>/dev/null; then
    echo "  rw_inte_cpp.so builds/loads OK"
else
    echo "  RW EXTENSION DID NOT LOAD: cpp/rw_inte_cpp.so is missing OR was built"
    echo "     against a GSL this env can't load at runtime (e.g. 'libgsl.so.NN:"
    echo "     cannot open shared object file'). Rebuild it ON THIS MACHINE against"
    echo "     the env's GSL (clean forces a real rebuild, not an mtime skip):"
    echo "     conda install -c conda-forge gsl patchelf   # if the env lacks them"
    echo "     make -C \"$REPO_ROOT/cpp\" clean && make -C \"$REPO_ROOT/cpp\""
    fail=1
fi

echo "=== MPI launcher / library ==="
echo "mpiexec: $(command -v mpiexec || echo 'NOT FOUND')"
mpiexec --version 2>&1 | head -1
echo "cores available here: $(nproc 2>/dev/null || echo '?')   SLURM_NTASKS=${SLURM_NTASKS:-unset}"
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
    echo "     * 'not enough slots' -> this session has < $R cores; request more."
    echo "     * OpenMPI 'not built with SLURM PMI' -> _nz_env.sh sets"
    echo "       OMPI_MCA_plm=isolated to launch locally; recreate env with mpich if it persists."
    echo "     * two size-1 singletons -> mpiexec and mpi4py are from different MPIs."
    fail=1
fi

echo "=== input data ==="
for f in nz_workflow_outputs.npz sigma_covariates_std_NZ_precip.csv \
         xi_covariates_std_NZ_precip.csv upscaled_NZ_SI_GEV_mle.csv \
         NZ_SI_adjacency_matrix.csv barrier_1500m/NZ_SI_barrier_1500m.shp; do
    if [ -f "$DATA_DIR/$f" ]; then echo "  data/$f OK"; else echo "  MISSING data/$f"; fail=1; fi
done
if [ -f "$DATA_DIR/ne_10m_coastline/ne_10m_coastline.shp" ]; then
    echo "  data/ne_10m_coastline OK (coastline overlay available)"
else
    echo "  NOTE: data/ne_10m_coastline absent -- map overlays skip the coastline (harmless)"
fi
OUTDIR="${NZ_OUT:-$NZ_SCRIPTS/../output}"
if [ -f "$OUTDIR/inputs.pkl" ]; then
    echo "  output/inputs.pkl present (Task 1 has been run)"
else
    echo "  NOTE: output/inputs.pkl not found -- run Task 1 before the fits:"
    echo "        bash nz_analysis/scripts/run_nz_task1.sh"
fi
# Stage 1 is a hard prerequisite for tasks 2-5 (they hold the margins fixed at it).
if [ -f "${NZ_STAGE1_NPZ:-$OUTDIR/marginal/marginal_fit.npz}" ]; then
    echo "  marginal/marginal_fit.npz present (stage 1 has been run)"
else
    echo "  NOTE: stage-1 marginal fit not found -- REQUIRED by tasks 2-5:"
    echo "        bash nz_analysis/scripts/run_nz_marginal.sh"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "PREFLIGHT: PASS -- safe to launch the fits."
else
    echo "PREFLIGHT: FAIL -- resolve the items above before launching."
fi
exit "$fail"
