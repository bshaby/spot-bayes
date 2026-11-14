# Architecture guide

This guide is for people who want to change `spot-bayes`: add a model, add or replace
an MCMC update, or understand why the code is shaped the way it is. For using the
package, start with the [README](../README.md).

## 1. The model in one page

Data `Y(s,t)` are observed at sites `s` over time replicates `t`. Values above a
per-site threshold `u(s)` follow a generalized Pareto (GPD) margin; values below it are
censored. Spatial dependence in the tail comes from a latent scale-mixture process:

```
Z(s,t)  ~ N(0, K)                     latent Gaussian field (Matérn, or VisGP for wang)
W(s,t)  = 1 / (1 - Φ(Z(s,t)))         standard Pareto margin
R(s,t)  = (W_wend @ S)[s, t]          random scale; S_k(t) ~ Lévy(gamma_k) at knots k
X*(s,t) = R(s,t)^phi(s) · W(s,t)
eps     ~ LogLaplace(alpha0)          multiplicative nugget
X       = eps · X*
Y       = qCGP(pRW(X))                probability integral transform to the censored GPD
```

`pRW` is the marginal CDF of `X`. It has a closed form in incomplete gamma functions,
implemented in C++/GSL (`cpp/rw_inte_cpp.cpp`) together with its density `dRW` and
quantile `qRW` (a Brent root-find). Those three functions are what make the likelihood
tractable, and they dominate the run time.

Per time replicate, the log-likelihood has a censored part (the nugget CDF at the
censored sites) and an exceedance part (the nugget density plus the Jacobian
`log dCGP(Y) - log dRW(X)`). See `utilities.Y_loglik_1t_*`; the numba-compiled version
is used whenever numba is installed.

The three model types differ only in the dependence pieces:

| `model_type` | tail exponent `phi` | random scale `R` | covariance of `Z` |
|---|---|---|---|
| `shi` | field over knots | Wendland-weighted Lévy knots | nonstationary Matérn (range varies over knots) |
| `wang` | field over knots | Wendland weights filtered by line of sight across the barrier (optionally anisotropic) | VisGP: sparse precision on a visibility graph |
| `huser-wadsworth` | scalar | one global Lévy scale | stationary Matérn |

**Huser–Wadsworth is the knot machinery with a single all-ones knot,** so `R(s,t) =
S(t)` for every site and `phi` is a scalar. That is why one set of update blocks serves
all three models.

### Priors (`priors.py`)

| parameter | prior |
|---|---|
| `S` | Lévy(scale `gamma_k`) on `exp(log S)`, with the log-Jacobian |
| `Z` | `N(0, K)` (dense) or `N(0, Q⁻¹)` (VisGP) |
| `phi_k` | Beta(2, 2) on (0, 1) for `phi_domain="unit"`; N(0, 1) on `log phi` for `"positive"` |
| `rho_k` | half-normal, scale 10 |
| `alpha0` | `alpha0 = 2 + exp(eta)`, `eta ~ N(0, 1)` (keeps `alpha0` away from the pole at 1) |
| `nu` | Uniform[0.5, 2.0] |
| `radius_S` | half-t (df 1, scale 3) |
| GPD regression `beta` | `N(0, sd²)`, `sd` from `sigma_beta_sd` / `xi_beta_sd` |

## 2. Module map

| module | role |
|---|---|
| `config.py` | `ModelConfig`: model type and feature flags, as a plain dataclass (no class hierarchy). Validates combinations and round-trips through checkpoints. |
| `cpp/rw_inte_cpp.cpp`, `rw_inte.py` | C++/GSL `pRW`/`dRW`/`qRW`, and their ctypes bindings |
| `rw_numeric.py` | `dRW_safe`: the analytic density with a strictly positive floor, which is what the likelihood calls |
| `phi_transform.py` | the two tail-exponent domains (`unit` and `positive`) |
| `priors.py` | prior log-densities and parameter transforms |
| `kernels.py` | Wendland weights: standard, anisotropic, and barrier-filtered |
| `covariance.py` | Matérn, nonstationary Matérn, VisGP precision and prediction (some kernels physically live in `utilities.py` because of numba; import them through `covariance`) |
| `domain.py` | shapefile → visibility graph → cliques and junction tree (Wang model) |
| `utilities.py` | core math: GPD/censored GPD, log-Laplace, Lévy, the compiled per-time likelihood |
| `marginal.py`, `marginal_model.py`, `recursive_bayes.py` | GPD margin regression; stand-alone margin fit (stage 1); exact recursive-Bayes carry-over into the full model |
| `simulate.py` | forward simulation of all three models: the data contract for tests |
| `sampler.py` | `ModelState`, the serial driver `run_mcmc`, settings, save/load, checkpoints |
| `sampler_mpi.py` | the MPI driver `run_mcmc_mpi` |
| `mcmc_updates.py` | **every MCMC update block**, the adaptive RWMH primitive, the NUTS machinery, `Ctx` and `Caches` |
| `phi_gradient.py`, `gamma_order_deriv.py`, `bessel_nu.py` | derivatives the gradient samplers need (with respect to `phi`, incomplete-gamma order, Bessel order) |
| `init_field.py` | data-driven starting values for the latent field |
| `missing_data.py`, `predictive.py`, `diagnostics.py` | imputation; held-out scoring, WAIC and new-site prediction; trace plots and ESS |

## 3. How a run flows

```
ModelConfig + SamplerData + SamplerSettings
        │
        ▼
ModelState(config, data)        builds knots, kernels, covariance, latent S/Z, margins
        │
        ▼
run_mcmc  /  run_mcmc_mpi       per iteration:
        │                         update_S → update_Z → run_global_updates
        │                         → adapt (every adapt_size) → record (every thin)
        │                         → live plot / checkpoint
        ▼
saved run (.npz) + optional streamed latent traces (.Z.npy / .S.npy)
```

### `ModelState` (`sampler.py`)

Holds the current values and the derived quantities that are expensive to rebuild.
The pieces you are most likely to touch:

- `__init__`: knots (an isometric grid, or Huser–Wadsworth's single knot), Wendland
  weights, the Gaussian smoothers that map knot values to site fields, `gamma_bar`,
  distance caches, the covariance, and the latent `S`, `Z`, `R`.
- `_phi_field`: knot values → per-site `phi`, honouring the domain.
- `rebuild_cov`: chooses VisGP / stationary / nonstationary Matérn by model, and
  invalidates the cached Cholesky factor used by the non-centred `Z` updates
  (`zchol`).
- `Z_logprior`, `loglik_t`: the prior and per-time likelihood.
- `snapshot` / `restore`: checkpoint state; `restore` rebuilds all derived quantities.
- `domain_covered`: rejects geometry proposals that would leave a site with no
  kernel support.

### The update blocks (`mcmc_updates.py`)

Every block has the same signature and shape:

```python
def update_X(state, ad, ctx, caches, st) -> None:
    # propose -> check validity -> recompute only what changed -> MH accept -> record
```

`run_global_updates` calls the global blocks (`phi`, `rho`, `alpha0`, `nu`,
`radius_S`, anisotropy, GPD `beta`, ...) and `make_adaptives` / `adapt_all` create
and tune their proposal state. Several blocks have alternative samplers, chosen in
`SamplerSettings` (for example `z_sampler`, `s_sampler`, `phi_sampler`,
`nu_rho_sampler`, `beta_sampler`); the dispatcher for each block picks one.

### Serial and MPI share one implementation

There is exactly one copy of each update block. The two drivers differ only through
the `Ctx` object passed to every block:

- `ctx.times`: which time replicates this process owns (all of them when serial; a
  round-robin slice per rank under MPI).
- `ctx.grng`: the RNG for **global** proposals and their accept uniforms. Every MPI
  rank seeds it identically, so all ranks propose the same value and reach the same
  decision without communicating.
- `ctx.lrng`: the RNG for **local** latent updates (`S`, `Z` at owned times); one per
  rank.
- `ctx.total(...)` / `ctx.total_ll(...)`: reduce per-time quantities to a scalar
  (`np.sum` when serial, an `Allreduce` under MPI).

`Caches` holds per-time dictionaries keyed by `t` (`Yll`: likelihood, `Zlp`:
`Z` prior, `qd`: cached `qRW`/`dRW`), which the blocks update in place.

**Rules that keep this working:**

1. Draw global proposals and their accept uniforms from `ctx.grng`, local ones from
   `ctx.lrng`. Drawing from the wrong one desynchronises MPI ranks.
2. Every branch driven by `ctx.grng` must be taken identically on all ranks.
3. Reduce through `ctx.total`. Don't replace it with a Python `sum()` or a different
   summation order: the exact floating-point order is part of what the golden tests
   pin.

### The censoring shortcut

Typically about 95% of entries are censored. A censored entry's `qRW` depends only on
`(phi, gamma_bar, alpha0)`, not on time, so it is computed once per proposal rather
than once per (site, time), and `dRW` is needed only at exceedances. With fixed margins,
`pY = pCGP(Y)` never changes and is computed once. This is exact (bit-identical to the
naive computation) and removes most of the `qRW`/`dRW` work; when the GPD `beta`
changes, only the exceedance entries are patched.

### Checkpoints

Serial checkpoints store the state snapshot, every adaptive block, the RNG state and
the traces, so a resumed serial chain is bit-identical to one that never stopped. MPI
checkpoints are written per rank plus one global file. Settings are not stored in the
checkpoint, so `sampler.py` has guards that keep a resumed chain on the samplers it
started with even if a default has changed since.

## 4. Recipes

### Add a model type

1. Add the name to `MODEL_TYPES` in `config.py`, with validation for any flags it
   requires or forbids.
2. Branch in `ModelState.__init__` (knots and kernels) and in `rebuild_cov`
   (covariance) in `sampler.py`.
3. Add it to `simulate.simulate` so data can be generated from it.
4. If its latent structure is shaped like `S`/`Z`/`phi`, the existing update blocks
   should work unchanged; otherwise add blocks (below).
5. Check `predictive.py` for model-specific branches (new-site prediction).
6. Tests: a short run in `tests/test_sampler.py` style, a simulate-and-recover check,
   and, if the new model reduces to an existing one in some limit, an equivalence test
   in the style of `tests/test_model_equivalence.py` (Wang equals Shi with no barrier
   and a stationary covariance).

### Add or replace an MCMC update block

1. Write it in `mcmc_updates.py` with the signature `(state, ad, ctx, caches, st)`,
   following the RNG and reduction rules above.
2. Register it in `run_global_updates`, and in `make_adaptives` / `adapt_all` if it
   adapts.
3. If it is an alternative to an existing sampler, add a `SamplerSettings` option and
   keep the existing default, so existing results stay reproducible.
4. Run the golden tests. A new default changes the chains, so regenerate the fixtures
   deliberately (see [CONTRIBUTING.md](../CONTRIBUTING.md)) and confirm that serial
   and MPI both reproduce them.

### Change the tail-exponent parametrisation

Edit `phi_transform.py` **and** `ModelState._phi_field` (which applies the map inline),
plus any post-processing that converts `phi` back from the sampled scale. Extend
`tests/test_phi_transform.py`.

## 5. Invariants the tests enforce

- `tests/test_sampler_golden.py`: the serial and MPI (2-rank) drivers must reproduce
  committed chains byte for byte, for both the random-walk and the NUTS settings.
- `tests/test_checkpoint.py`: a resumed serial chain equals an uninterrupted one.
- `tests/test_rw.py`: the C++ transforms against high-precision references, including
  the far tail.
- `tests/test_model_equivalence.py`, `tests/test_covariance.py`: the Wang model reduces
  to Shi in the right limit; sparse VisGP equals the dense computation.
