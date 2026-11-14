# New Zealand South Island Extreme-Precipitation Analysis

Self-contained spatial-extremes analysis of NZ South Island atmospheric-river
precipitation with the `spot-bayes` package (HW / Wang / Shi models) — the
precipitation analogue of `faroe_analysis/`. Everything this analysis needs lives in
this directory; it depends only on the core package at the repo root (`spot-bayes/`).

```
nz_analysis/
├── scripts/     all code: nz_*.py, _common.py, run_nz_task*.sh, _nz_env.sh
├── data/        copied inputs (workflow npz, covariates, 1500 m barrier + adjacency, coastline)
├── output/      generated: inputs.{pkl,npz}, figures/, and the fitted runs
└── vignette/    nz_modeling_vignette.md (the human-facing guide) + graphics/
```

The scientific story mirrors Faroe: the **Southern Alps shelter the leeward (SE)
side** from extreme precipitation, encoded by the Wang model's physical-barrier
visibility graph. The modelled series is the **3-day precipitation sum** (309 sites ×
3080 storm-scale blocks; see [Data sources](#data-sources)), above the fitted 0.95-quantile
threshold.

## Barrier: 1500 m, not 1200 m

The upstream pre-processing labelled the visibility graph a "1200 m barrier", **but the pre-built visibility graph is actually a 1500 m graph**
(verified: 0/309 sites inside the 1500 m contour and 0/26566 adjacency edges cross
it, versus 12 sites inside / 4908 edges crossing the 1200 m contour). This analysis
therefore uses the **1500 m** barrier (`data/barrier_1500m/`), which keeps all 309
sites and lets us reuse `data/NZ_SI_adjacency_matrix.csv` turnkey.

## Quick start

Run the tasks in order (each `run_*` activates the `extremes` conda env, pins one
thread per rank, and defaults to `NRANKS=64`):

```bash
python scripts/nz_preprocess.py  # (optional) Part I: regenerate the threshold + GPD
                                 # margins in data/nz_workflow_outputs.npz -> output/preprocess/
./scripts/run_nz_preflight.sh  # (recommended) checks deps, geopandas/PROJ, the
                               # mpiexec<->mpi4py pairing, and the input data
./scripts/run_nz_task1.sh      # data prep -> output/inputs.* + review figures (serial)
./scripts/run_nz_marginal.sh   # STAGE 1: GPD margins on the exceedances (serial)
./scripts/run_nz_task2.sh      # fit Wang, full data      | stage 2: margins FIXED
./scripts/run_nz_task3.sh      # fit HW,   training set    | at the stage-1 surface,
./scripts/run_nz_task4.sh      # fit Wang, training set    | only the dependence
./scripts/run_nz_task5.sh      # fit Shi,  training set    | field is sampled
./scripts/run_nz_task6.sh      # held-out model comparison (log-score/CRPS/TW-CRPS/WAIC)
./scripts/run_nz_task7.sh      # assessment of the full Wang fit (Gumbel QQ + chi maps)
```

**Stage 1 is a prerequisite**: tasks 2–5 refuse to start without
`output/marginal/marginal_fit.npz`. See [Two-stage margins](#two-stage-margins).

Override defaults with env vars, e.g. `NRANKS=64 NZ_NITERS=50000 ./scripts/run_nz_task2.sh`.
`NRANKS` defaults to **16**: four 16-core jobs schedule sooner than four 64-core ones, and
this sampler's ~150 allreduces per iteration mean parallel efficiency is already falling
off before 64 ranks. That is about allocation *size* — the jobs are independent
submissions and Slurm may scatter them across nodes, which is fine (the fit is
memory-bandwidth bound, so packing our own fits onto one node would make them compete).
On a laptop drop it to your core count. Fits checkpoint and auto-resume — relaunch the same command to
continue. A quick smoke run: `NZ_NITERS=200 NRANKS=4 ./scripts/run_nz_task3.sh`.

## Two-stage margins

`run_nz_marginal.sh` fits `marginal_model.py` — the GPD coefficients on the threshold
exceedances alone, conditionally independent, with no dependence field. Tasks 2–5 then
hold σ/ξ **frozen** at that posterior-mean surface (`--fix-margins`), so only
Z/S/φ/ρ/ν/α₀/geometry adapt.

This is not a convenience. In a sampled-margin fit the betas and the latent field weld to
each other through the nugget `X ≈ X*`, and the pair can settle far from a calibrated
margin with no single-block move able to come back (established on the Faroe data). Runs
are named `<model>_fixmargins_stage1_plug` so they never clobber a sampled-margin fit.

To sample the margins anyway, call `nz_fit.py` directly without `--fix-margins`; with
`NZ_STAGE1_NPZ` set it still warm-starts the betas from the stage-1 posterior mean.

## Starting values

Tasks 2–5 start from the **plug-in ("X-inversion") field** built by the package-level
`init_field.py`, not from the sampler's cold `S=1, Z=0, φ=1, α₀=5, ρ=1`. With the margins
fixed, the transformed data `X = qRW(pCGP(Y))` is computable, so the model
`X = R^φ·g(Z)·ε` can be *inverted* for a starting `(S, Z)`, and ν/ρ read off the resulting
correlogram. Measured on this dataset:

| start | data log-lik |
|---|---|
| cold default | −2,230,008 |
| scalars only (φ=0.5, α₀=2.05), S=1, Z=0 | −730,053 |
| **+ plug-in S/Z field** | **≈ −138,600** |

α₀ alone is worth ~1.2M nats (the cold default sits 3.0 above the `alpha0_min=2` wall).
Just as important, the plug-in Z lands **inside** the Gaussian typical set (quadratic form
825k against the expected `Ns·Nt` = 952k) rather than at the prior mode, where ν and ρ are
not identified at all — that inflation was the long transient.

The correlogram gives NZ **ρ ≈ 1.5**, consistently across all three models. Note this is
much shorter-ranged than Faroe's ρ = 3 *relative to the domain*: NZ's maximum site
separation is 12.0° against Faroe's 5.7°, and NZ's latent correlation is already ≈ 0 at
its median separation (4.7°) where Faroe's was still 0.49. Do not transfer ρ between the
two analyses.

**ν is not identified by this procedure**, and unlike ρ it is not stable: it is an
artifact of how the split apportions smooth structure between R and Z. HW's single-knot R
basis forces *all* spatial structure into Z, so the fit wants a very smooth field (ν → 24
unbounded); Wang's wide barrier-filtered Wendland basis (radius 4.33) absorbs that
structure and leaves Z rough (ν → 0.36). Only Shi's split lands interior. The sampler's
prior is Uniform[0.5, 2.0] — a *hard* bound, `-inf` outside — so a fit that runs into it
is clamped, not estimated, and starting nu against a wall is a poor start for the
parameter whose mixing is already the bottleneck. When the fit is censored the
initializer therefore falls back to **ν = 1.5** and says so; `--nu0` overrides.

φ is a **choice, not a fit** — it is not identifiable from the split itself, since a larger
φ simply moves variation out of Z and into R. The default `NZ_PHI0=0.5` is the
asymptotic-independence/dependence boundary. Knobs: `NZ_PHI0`, `NZ_ALPHA0`,
`NZ_PLUGIN_INIT=0` (or `--no-plugin-init`) to cold-start, `--nu0`/`--rho0` to pin the
covariance instead of estimating it.

## What each script is

| File | Role |
|---|---|
| `nz_preprocess.py` | Part I (optional): 3-day blocks, storm-duration EDA, 0.95 quantile-regression threshold, GPD regression. Writes to `output/preprocess/`; the shipped `data/nz_workflow_outputs.npz` is its output. |
| `nz_setup.py` | Shared setup: loads data, knots, split, model configs, junction trees. |
| `run_nz_preflight.sh` | Pre-launch check: deps, geopandas/PROJ, mpiexec↔mpi4py pairing, data. |
| `nz_task1_prep.py` | Builds `output/inputs.pkl` + the split/threshold/knot figures. |
| `nz_fit.py` | MPI fit driver (`--model {hw,wang,shi} --which {full,train}`). |
| `nz_task6_compare.py` | Held-out scoring across the three training fits. |
| `nz_task7_assess.py` | Gumbel QQ + extremal-dependence χ for the full Wang fit. |
| `nz_diagnostics.py` | χ (empirical + model-based) and Gumbel-QQ implementations. |
| `_common.py` | Shared MCMC/scoring helpers. |

See **`vignette/nz_modeling_vignette.md`** for the full narrative: the models, MPI
options, every sampler knob, and the comparison/assessment strategy. Part I of the vignette
covers the data and marginal pre-processing that produced `data/`; Part II covers the
Bayesian modeling.

## Train/test split

Two holdout blocks (~11% of the 309 sites), reviewed on `output/figures/split.png`:

- **NW open-ocean (windward)** — a rectangle `lon∈[166.5,168.5], lat∈[−42.5,−41.0]`
  pulled inward off the domain corner (20 sites).
- **SE leeward** — an explicit, hand-tuned strip hugging the SE/E flank of the 1500 m
  barrier (13 sites); trimmed of the far-offshore Pacific. Defined as a site list in
  `nz_setup.SE_SITES` (non-rectangular; its convex hull is drawn dotted on the map).

To change it, edit `BLOCK_NW` / `SE_SITES` in `scripts/nz_setup.py` and re-run Task 1.

## Running on Alpine (CU Research Computing, Slurm)

Each fit runs comfortably on **one `amilan` node = 64 cores**; `scripts/alpine_nz.sbatch`
launches the ranks with the conda env's own `mpiexec` in local (fork) mode — no
Slurm/PMI/`srun` integration needed.

**One-time setup on Alpine:**
```bash
module load anaconda
conda create -n extremes -c conda-forge python=3.12 numpy scipy matplotlib \
    shapely networkx geopandas mpi4py gsl patchelf
conda activate extremes
module load gcc            # a C++ compiler (or add gxx to the conda create above)
make -C cpp               # builds cpp/rw_inte_cpp.so (links GSL)
```

**Output goes to scratch.** `/home` has a small quota and the streamed latent Z/S
traces are large, so all output is written under **`NZ_OUT`** (which `nz_setup.py`
reads). `alpine_nz.sbatch` defaults `NZ_OUT=/scratch/alpine/$USER/nz_out` and
**creates it automatically** — you do **not** need to `mkdir` it or pass it at submit
time (`/scratch/alpine/$USER` is provisioned for every CURC user). Override with
`--export=ALL,NZ_OUT=/some/path`. **Task 1 must use the same `NZ_OUT`** so the fits
find `inputs.pkl` — export it in your interactive shell before running Task 1 (below),
or run Task 1 through the sbatch (`sbatch --array=1 --export=ALL,NZ_TASK=1 …`).

**Preflight + Task 1** (both light — run interactively; request **≥ PREFLIGHT_RANKS
cores**):
```bash
ACCT=<your-allocation>
sinteractive --partition=amilan --ntasks=4 --time=00:30:00 --qos=normal --account=$ACCT
module load anaconda
export NZ_OUT=/scratch/alpine/$USER/nz_out        # SAME path the sbatch uses (below)
PREFLIGHT_RANKS=4 bash nz_analysis/scripts/run_nz_preflight.sh   # expect PREFLIGHT_MPI_PEERS=4
bash nz_analysis/scripts/run_nz_task1.sh          # serial; builds $NZ_OUT/inputs.pkl
```

**Submit the fits** — pass your allocation with `--account` (or edit the `--account` line
in `alpine_nz.sbatch` once); `--qos`/`--time` default to the array-chained `normal` qos.
Submit **from the repo root**:
```bash
ACCT=<your-allocation>
sbatch --account=$ACCT nz_analysis/scripts/alpine_nz.sbatch                             # Task 2 (default)
sbatch --account=$ACCT --export=ALL,NZ_TASK=3 nz_analysis/scripts/alpine_nz.sbatch      # Task 3 (hw train)
sbatch --account=$ACCT --export=ALL,NZ_TASK=4 nz_analysis/scripts/alpine_nz.sbatch      # Task 4 (wang train)
sbatch --account=$ACCT --export=ALL,NZ_TASK=5 nz_analysis/scripts/alpine_nz.sbatch      # Task 5 (shi train)
sbatch --account=$ACCT --array=1 --export=ALL,NZ_TASK=6 nz_analysis/scripts/alpine_nz.sbatch  # Task 6 (compare)
sbatch --account=$ACCT --array=1 --export=ALL,NZ_TASK=7 nz_analysis/scripts/alpine_nz.sbatch  # Task 7 (assess)
```

- **Wall time**: `normal` qos caps at 24 h; the full Wang fit may need longer. Either use
  `--qos=long --time=7-00:00:00`, **or** keep `normal` and `sbatch` again when it ends —
  the fits **checkpoint (~every 30 min) and auto-resume**.
- Output goes to `nz_<jobname>_<jobid>.log`. Monitor `output/wang_full.trace.png`.

## Data sources

Everything in `data/` was derived from public sources:

- **Precipitation** — ECMWF ERA5 reanalysis (Hersbach et al., *ERA5 hourly data on single
  levels from 1940 to present*, Copernicus Climate Change Service (C3S) Climate Data
  Store, DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47); licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Contains modified Copernicus
  Climate Change Service information. Chien-Chung Wang aggregated ERA5 total
  precipitation over the South Island to daily totals and upscaled it to a coarser grid
  (323 cells, of which the 309 model sites are kept): `data/upscaled_era5_daily_precp_ARs.csv.gz`.
  It covers the January–April seasons 1950–2026, the period that Reid, K. J., Rosier, S. M., Harrington, L. J., King, A. D. and Lane, T. P. (2021), "Extreme rainfall in New Zealand and its association with Atmospheric Rivers", *Environmental Research Letters* 16, 044012, doi:[10.1088/1748-9326/abeae0](https://doi.org/10.1088/1748-9326/abeae0) found holds about
  60% of New Zealand's extreme rainfall events, most of them associated with
  atmospheric rivers (hence "ARs" in the file names). `scripts/nz_preprocess.py` sums
  non-overlapping 3-day blocks (3080 blocks; the leftover last day of each 121-day
  season is dropped). The block sums are `blocks` in `data/nz_workflow_outputs.npz`.
- **Threshold and starting margins** (`data/nz_workflow_outputs.npz`,
  `data/upscaled_NZ_SI_GEV_mle.csv`) — a spatial 0.95 quantile regression gives the
  threshold. A GPD regression fit by maximum likelihood gives the starting
  `beta_log_sigma` / `beta_logit_star_xi`. Per-site GEV MLEs are used only as a check
  on site order. Part I of the vignette describes this pre-processing, and
  `scripts/nz_preprocess.py` reproduces it (to optimizer tolerance).
- **Covariates and visibility graph** (`data/*_covariates_std_NZ_precip.csv`, identical
  for the threshold, σ and ξ; `data/NZ_SI_adjacency_matrix.csv`) — built by Chien-Chung
  Wang for the same 309 sites.
- **1500 m barrier** (`data/barrier_1500m/`) — the 1500 m elevation contour of the
  Southern Alps, derived from NASA SRTM elevation data (public domain).
- **Coastline** (map overlays only, not tracked) — [Natural Earth](https://www.naturalearthdata.com/)
  1:10m coastline, public domain. Download `ne_10m_coastline` and unpack it into
  `data/ne_10m_coastline/` if you want it on the maps; it is skipped with a warning if
  absent. The domain map in `nz_preprocess.py` can also show SRTM topography: put a
  South Island SRTM GeoTIFF at `data/NZ_SI_SRTM.tif` (not tracked; 22 MB).

## Notes

- **Parallelism** is MPI over the **3080 time replicates** (vs Faroe's 1656); use up to
  `NRANKS=64`.
- **Native extension:** the package uses a compiled `cpp/rw_inte_cpp.so` (C++/GSL), not
  in git. Build once with `make -C cpp` (needs a C++ compiler + GSL). The preflight
  checks for it.
- **Inputs live in `data/`** so the analysis is self-contained; see [Data sources](#data-sources).
- `data/nz_cliques.json` is generated by Task 1 (satisfies `ModelConfig.cliques_path`);
  the cliques actually used come from the junction trees built in `nz_setup.prepare_inputs`.
