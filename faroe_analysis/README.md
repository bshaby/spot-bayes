# Faroe Islands Extreme-Wave Analysis

Self-contained spatial-extremes analysis of Faroe storm-season wave heights with the
`spot-bayes` package (HW / Wang / Shi models). Everything this analysis needs lives in
this directory; it depends only on the core package at the repo root.

```
faroe_analysis/
├── scripts/     all code: faroe_*.py, _common.py, run_faroe_task*.sh, _faroe_env.sh
├── data/        copied inputs (block maxima, threshold+beta npz, covariates, cliques, coastline)
├── output/      generated: inputs.{pkl,npz}, figures/, and the fitted runs
└── vignette/    faroe_modeling_vignette.md (the human-facing guide) + graphics/
```

## Quick start

Run the tasks in order (each `run_*` activates the `extremes` conda env, pins one thread
per rank, and defaults to `NRANKS=64`):

```bash
python scripts/faroe_preprocess.py  # (optional) Part I: regenerate the threshold + GPD
                                    # margins in data/faroe_workflow_outputs.npz -> output/preprocess/
./scripts/run_faroe_preflight.sh  # (recommended, esp. on a new machine) checks deps,
                                  # the mpiexec<->mpi4py pairing, and the input data
./scripts/run_faroe_task1.sh      # data prep -> output/inputs.* + review figures (serial)
./scripts/run_faroe_marginal.sh   # marginal run: standalone GPD margins -> output/marginal/ (serial)
./scripts/run_faroe_task2.sh      # fit Wang, full data     (margins FIXED from the marginal run)
./scripts/run_faroe_task3.sh      # fit HW,   training set  (margins FIXED from the marginal run)
./scripts/run_faroe_task4.sh      # fit Wang, training set  (margins FIXED from the marginal run)
./scripts/run_faroe_task5.sh      # fit Shi,  training set  (margins FIXED from the marginal run)
./scripts/run_faroe_task6.sh      # held-out model comparison (log-score/CRPS/TW-CRPS/WAIC)
./scripts/run_faroe_task7.sh      # assessment of the full Wang fit (Gumbel QQ + chi maps)
```

Tasks 2-5 fit only the DEPENDENCE structure: they freeze the GPD margins at the marginal
run's posterior-mean surface (`--fix-margins`, seeded from `output/marginal/marginal_fit.npz`
via `FAROE_STAGE1_NPZ`) and write to `*_fixmargins_stage1.*`. Run `run_faroe_marginal.sh`
first — Tasks 2-5 abort if its output is missing. To fit sampled margins instead, call
`faroe_fit.py` directly without `--fix-margins`.

Override defaults with env vars, e.g. `NRANKS=8 FAROE_NITERS=50000 ./scripts/run_faroe_task2.sh`.
On a laptop, drop `NRANKS` to your core count (or run a script's `python` line directly
for a serial chain). Fits checkpoint and auto-resume — relaunch the same command to continue.

## What each script is

| File | Role |
|---|---|
| `faroe_preprocess.py` | Part I (optional): 0.95 quantile-regression threshold and GPD regression from the block maxima. Writes to `output/preprocess/`; the shipped `data/faroe_workflow_outputs.npz` is its output. |
| `faroe_setup.py` | Shared setup: loads data, knots, split, model configs, junction trees. |
| `run_faroe_preflight.sh` | Pre-launch check: deps, mpiexec↔mpi4py pairing, input data. |
| `faroe_task1_prep.py` | Builds `output/inputs.pkl` + the split/threshold/knot figures. |
| `faroe_fit.py` | MPI fit driver (`--model {hw,wang,shi} --which {full,train}`). |
| `faroe_task6_compare.py` | Held-out scoring across the three training fits. |
| `faroe_task7_assess.py` | Gumbel QQ + extremal-dependence χ for the full Wang fit. |
| `faroe_diagnostics.py` | χ (empirical + model-based) and Gumbel-QQ implementations. |
| `_common.py` | Shared MCMC/scoring helpers. |

See **`vignette/faroe_modeling_vignette.md`** for the full narrative: the models, MPI
options, every sampler knob with recommendations, and the comparison/assessment strategy.
Part I of the vignette covers the data and marginal pre-processing that produced
`data/`; Part II covers the Bayesian modeling.

## Running on Alpine (CU Research Computing, Slurm)

Task 2 (and each fit) fits comfortably on **one `amilan` node = 64 cores**, so we run a
single-node job and launch the ranks with the conda env's own `mpiexec` in **local
(fork) mode** — no Slurm/PMI/`srun` integration needed (the most robust way to run a
conda-provided MPI under Slurm). `scripts/alpine_faroe.sbatch` does this.

**One-time setup on Alpine:**
```bash
module load anaconda
# create the env. gsl + patchelf are for building the native RW extension;
# mpi4py from conda-forge is self-contained.
conda create -n extremes -c conda-forge python=3.12 numpy scipy matplotlib \
    shapely networkx geopandas mpi4py gsl patchelf
conda activate extremes
# clone/copy this repo somewhere on /projects or scratch, then cd into it, and
# build the native RW-integration extension (links GSL; produces cpp/rw_inte_cpp.so):
module load gcc            # a C++ compiler (or add gxx to the conda create above)
make -C cpp
```

The conda MPI (OpenMPI or MPICH) runs **single-node, local launch** — the scripts set
`OMPI_MCA_plm=isolated` / `HYDRA_BOOTSTRAP=fork` in `_faroe_env.sh` so `mpiexec` does not
try to route through Slurm's PMI (which the conda OpenMPI isn't built for). Nothing extra
to export.

**Output location — one knob, `FAROE_OUT`.** `/home` has a small quota and the streamed
latent Z/S traces are large, so all output goes to scratch. `alpine_faroe.sbatch` defaults
`FAROE_OUT=/scratch/alpine/$USER/faroe_out` and creates it. The interactive prerequisites
below (preflight, Task 1, and the marginal run) MUST write to that **same** `FAROE_OUT`, so
export it in your interactive shell first — everything (`inputs.pkl`, `marginal/`, fits,
latent `.npy`, figures) then lives in one place.

**Preflight + Task 1 + marginal run** (all light — run interactively; request
**≥ PREFLIGHT_RANKS cores** for the preflight, since `acompile` gives only 1):
```bash
sinteractive --partition=amilan --ntasks=4 --time=00:30:00 --qos=normal --account=<your-allocation>
module load anaconda
export FAROE_OUT=/scratch/alpine/$USER/faroe_out        # match the sbatch default
PREFLIGHT_RANKS=4 bash faroe_analysis/scripts/run_faroe_preflight.sh   # expect PREFLIGHT_MPI_PEERS=4
bash faroe_analysis/scripts/run_faroe_task1.sh          # serial; builds $FAROE_OUT/inputs.pkl
bash faroe_analysis/scripts/run_faroe_marginal.sh       # serial; builds $FAROE_OUT/marginal/marginal_fit.npz
```
The marginal run fits the standalone GPD margins that Tasks 2–5 freeze; they **abort with a
clear error** if `marginal_fit.npz` is missing, so run it before submitting the fits.
If the preflight's N-rank test still fails after re-pulling, the simplest fix is to
recreate the env with **MPICH** instead of OpenMPI (its Hydra launcher does clean
single-node local launch): `conda create -n extremes -c conda-forge mpich mpi4py python=3.12
numpy scipy matplotlib shapely networkx geopandas`.

OpenMPI prints some harmless startup warnings on Alpine (InfiniBand "no device params",
etc.). `_faroe_env.sh` quiets them and — importantly — keeps OpenMPI's shared-memory
backing file on node-local RAM (`/dev/shm`) rather than Lustre (which OpenMPI warns is
slow); set `FAROE_OMPI_TMPDIR` to override if `/dev/shm` is unavailable.

**Submit the fits** — pass your allocation with `--account` (or edit the `--account`,
`--qos`/`--time` defaults in `alpine_faroe.sbatch` once), then submit **from the repo root**.
Tasks 2–5 fit the dependence structure with the margins **fixed** at the marginal run and
write to `$FAROE_OUT/…_fixmargins_stage1.*`:
```bash
ACCT=<your-allocation>   # e.g. ucb-general or ucbXXX_projectN
sbatch --account=$ACCT --export=ALL,FAROE_TASK=2 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 2 (wang full)
sbatch --account=$ACCT --export=ALL,FAROE_TASK=3 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 3 (hw   train)
sbatch --account=$ACCT --export=ALL,FAROE_TASK=4 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 4 (wang train)
sbatch --account=$ACCT --export=ALL,FAROE_TASK=5 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 5 (shi  train)
# post-processing (one-shot — run after 2–5; override the resume array with --array=1):
sbatch --account=$ACCT --array=1 --export=ALL,FAROE_TASK=6 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 6 (compare)
sbatch --account=$ACCT --array=1 --export=ALL,FAROE_TASK=7 faroe_analysis/scripts/alpine_faroe.sbatch  # Task 7 (assess)
```

- **`FAROE_OUT`**: the sbatch defaults it to `/scratch/alpine/$USER/faroe_out` (same as the
  interactive steps above). If you used a different path, forward it:
  `--export=ALL,FAROE_TASK=2,FAROE_OUT=/your/path`.
- **Wall time**: `normal` qos caps at 24 h; the full Wang fit may need longer. The default
  `--array=1-4%1` chains four consecutive 24 h windows that run one at a time, each resuming
  from the last checkpoint (the fits **checkpoint ~every 30 min and auto-resume**). Or use a
  single long window with `--qos=long --time=7-00:00:00`.
- **Account**: required. Use your CURC allocation; see the CURC allocations page.
- Output goes to `faroe_<jobname>_<jobid>_<arrayidx>.log`. Monitor with `tail -f` and
  `$FAROE_OUT/wang_full_fixmargins_stage1.trace.png` / `.trace.surfaces.png`.

## Data sources

Everything in `data/` was derived from public sources:

- **Wave heights** — Copernicus Marine Service, *Global Ocean Waves Reanalysis*
  (WAVERYS), product `GLOBAL_MULTIYEAR_WAV_001_032`, dataset
  `cmems_mod_glo_wav_my_0.2deg_PT3H-i` (3-hourly, 0.2° grid), variable `VHM0`
  (spectral significant wave height). DOI:
  [10.48670/moi-00022](https://doi.org/10.48670/moi-00022). Generated using E.U.
  Copernicus Marine Service Information. We subset it to 60–64°N, 9–5°W (a 21×21 grid),
  kept the October–March storm seasons 1980/81–2025/26, and reduced each grid cell to
  5-day block maxima (1656 blocks; `data/faroe_block_maxima.npz`). In leap-year seasons
  the block containing February 29 is 6 days, and the last 2 days of each season are
  dropped so the season splits evenly.
- **Threshold and starting margins** (`data/faroe_workflow_outputs.npz`) — a pooled 0.95
  quantile regression of the block maxima (covariates: spatial trend, smooth spatial
  terms, depth, distance to the island centroid, exposure angle) gives the threshold.
  A GPD regression fit by maximum likelihood gives the starting `beta_log_sigma` /
  `beta_logit_star_xi`. Part I of the vignette describes this pre-processing, and
  `scripts/faroe_preprocess.py` reproduces it (to optimizer tolerance).
- **GPD covariates, buffered island polygon, cliques and separators**
  (`data/*_covariates_std_2stage_MCMC.csv` for the threshold, σ and ξ;
  `data/buffered_faroe_{cliques,separators}.json`) — from Chien-Chung Wang's
  *Spatial Extremes on Domains with Physical Barriers* ([arXiv:2611.23565](https://arxiv.org/abs/2611.23565)) code. The cliques were converted from `.rds` to JSON.
- **Coastline** (map overlays only, not tracked) — [Natural Earth](https://www.naturalearthdata.com/)
  1:10m coastline, public domain.

## Notes

- **Native extension:** the package uses a compiled C++/GSL library (`cpp/rw_inte_cpp.so`)
  that is not in git. On any new machine, build it once: `make -C cpp` from the repo root
  (needs a C++ compiler + GSL — `conda install -c conda-forge gsl patchelf`). The preflight
  checks for it.
- **Inputs live in `data/`** so the analysis is self-contained; see [Data sources](#data-sources).
- The **global coastline shapefile** (`data/ne_10m_coastline/`, used only for the map
  overlays) is **not tracked in git** (it is large and redistributable). Download
  `ne_10m_coastline` from
  [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/)
  and unpack it into `data/ne_10m_coastline/` before running Task 1.
- The shared-library plotting hook (`diagnostics.plot_live_surfaces`) lives in the core
  package (`diagnostics.py` at the repo root), not here.
- To change the train/test holdout blocks, edit `BLOCK_E/BLOCK_NW/BLOCK_S` in
  `scripts/faroe_setup.py` and re-run Task 1.
