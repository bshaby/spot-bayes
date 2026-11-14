# Contributing to spot-bayes

Contributions are welcome: bug reports, fixes, new models, new samplers, examples and
documentation.

## Before you start

- **Bugs and questions:** open an issue. For a bug, include the command or a minimal
  script, the full error, and your OS, Python and MPI versions.
- **Larger changes** (a new model type, a new sampler, a change to the likelihood):
  please open an issue to discuss the design first, so your work fits the code's
  structure and nobody spends weeks on something that can't be merged.
- Read [`docs/architecture.md`](docs/architecture.md). It explains how the model,
  the samplers and the serial/MPI drivers fit together, and has recipes for adding a
  model or an update block.

## Setting up

```bash
git clone https://github.com/<you>/spot-bayes.git     # your fork
cd spot-bayes
conda env create -f environment.yml
conda activate extremes
make -C cpp                                           # build the C++/GSL extension
```

Run one BLAS/OpenMP thread per process; the tests and samplers are written for that,
and oversubscribing cores is slow:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

For MPI, launch with the `mpiexec` from the same environment as your `mpi4py`
(`export PATH="$CONDA_PREFIX/bin:$PATH"`). A mismatched launcher runs every rank as an
independent single-process job without any error.

## Running the tests

```bash
python -m pytest tests/ -q                                  # everything, ~5 minutes
python -m pytest tests/ -q --ignore=tests/test_sampler.py   # skip the slower sampler runs
python -m pytest tests/test_kernels.py -q                   # one file while iterating
```

The whole suite must pass before a pull request is merged.

### The two tests that guard the samplers

- **`tests/test_sampler_golden.py`** runs short chains with both the serial and the
  MPI (2-rank) drivers and requires them to reproduce committed chains **byte for
  byte**. Any change to the update math, to the order of random draws, or to how
  per-time quantities are summed will break it.
- **`tests/test_checkpoint.py`** requires a resumed serial chain to be bit-identical
  to an uninterrupted one.

If your change is meant to change the numbers (a new default sampler, a likelihood
fix), regenerate the golden fixtures deliberately and say so in your pull request:

```bash
P=$(python -c "import sys; print(sys.executable)")
M=$(dirname "$P")/mpiexec          # the launcher next to this interpreter, not a bare mpiexec
$P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/serial.npz serial
$P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/serial_nuts.npz serial_nuts
$M -n 2 $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/mpi2.npz mpi
$M -n 2 $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/mpi2_nuts.npz mpi_nuts
```

If your change is **not** meant to change the numbers and the golden test fails, the
change has a side effect: find it rather than regenerating.

## Guidelines for code

- **One implementation for both drivers.** Update blocks live in `mcmc_updates.py` and
  are shared by the serial and MPI drivers. Draw global proposals from `ctx.grng`,
  local latent proposals from `ctx.lrng`, and reduce per-time values with `ctx.total`.
  [`docs/architecture.md`](docs/architecture.md#serial-and-mpi-share-one-implementation)
  explains why.
- **Keep existing defaults** when adding an alternative sampler or option. Add a
  setting for the new behaviour so existing results stay reproducible; changing a
  default is a separate, discussed decision.
- **Exactness first.** Performance changes to the likelihood should be exact (same
  numbers) and come with a test showing it. Approximations should be opt-in.
- **Tests with every change:** a unit test for new math, and a short run for a new
  model or sampler. Simulated data from `simulate.py` keeps tests self-contained.
- **Dependencies:** please discuss new ones in an issue first; the current ones are
  listed in `environment.yml`.
- Match the style of the surrounding code; comments should explain *why*, not restate
  *what*.

## Pull request checklist

- [ ] The test suite passes (`python -m pytest tests/ -q`).
- [ ] New behaviour has tests.
- [ ] If golden fixtures were regenerated, the pull request says so and why.
- [ ] User-visible changes are reflected in the README or the relevant docs.

## Data

If you contribute data (for example a new worked example), state its source, licence
and any processing in the example's README, the way the Faroe and New Zealand examples
do, and make sure its licence allows redistribution.

## Licence

By contributing, you agree that your contributions are licensed under the project's
[MIT licence](LICENSE).

## How changes reach this repository

This repository is published from the maintainer's development repository. When your
pull request is accepted, it is merged here as usual and also applied upstream, so
later releases keep it. You don't need to do anything differently.
