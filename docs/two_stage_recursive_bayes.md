# Two-stage recursive Bayes for the spatial-extremes model

This documents the `two-stage` branch: a standalone **stage-1 marginal GPD model** and the
**stage-2 recursive-Bayes scaffold** that is meant to feed the stage-1 output into the full
dependence model. It also records, honestly, *why the clean recursive-Bayes cancellation
does not hold here* and what the realistic fallback is.

## Motivation

The production sampler fits the GPD margin coefficients `(beta_sigma, beta_xi)` **through**
the RW spatial-dependence likelihood (`mcmc_updates.update_beta*` → `loglik_t`). The margins
are therefore only weakly identified — the dependence structure absorbs signal that a pure
marginal analysis would attribute to `sigma`/`xi`. We want:

1. A **defensible marginal fit** obtained on its own terms, and
2. A principled way to carry that marginal information into the dependence fit — the
   two-stage **recursive Bayes** idea (Hooten, Johnson & Brost 2021, *TAS* 75(2):185–194,
   "Making Recursive Bayesian Inference Accessible").

## Stage 1 — the marginal model (`marginal_model.py`)

Model the threshold **exceedances only**, conditionally independent given the GPD
parameters:

> For every cell with `Y(s,t) > u(s)`,
> `Y(s,t) − u(s) ∼ GPD(scale = sigma(s), shape = xi(s))`, independently,
> with `log sigma = X_sigma beta_sigma`, `xi = scaled-logit(X_xi beta_xi)`
> (`marginal.MarginalDesign` + links), and `beta ∼ Normal(0, sd²)`.

The log target is

```
log p(beta) = Σ_{(s,t): Y>u}  log dGP(Y(s,t); u(s), sigma(s), xi(s))  +  log N(beta; 0, sd²).
```

This is **not** a censored GPD: the non-exceedances are not analyzed at all. (A censored GPD
would use every cell and plug the threshold probability into the non-exceedances. With a
*fixed* threshold probability the two often agree up to a `beta`-constant, but specifying the
model directly on the exceedances is cleaner and avoids implying the sub-threshold data carry
marginal-tail information they do not.)

- **Value + gradient**: `MarginalData.value_grad` is the RW-free half of
  `mcmc_updates._beta_value_grad` — the same analytic `d(log dGP)/d(sigma, xi)` chained
  through the links and the covariate design, with the entire `pCGP → qRW → copula` path
  removed. The `xi → 0` removable singularity is handled with the analytic limit
  `d(log dGP)/dxi → z²/2 − z`.
- **Sampler**: NUTS, reusing `mcmc_updates.NUTSWarmup` + `nuts_transition` (they take a plain
  `vg(theta) → (logp, grad)` and have no ModelState/MPI coupling). ~19 parameters, ~5% of
  cells are exceedances → cheap and serial.
- **Driver**: `faroe_analysis/scripts/faroe_marginal_fit.py` runs it on the real Faroe data
  (432 sites × 1656 blocks; `sigma` = intercept + 10 covariates, `xi` = intercept + 7),
  warm-started from the reference workflow coefficients, and writes
  `faroe_analysis/output/marginal/marginal_fit.npz` (posterior draws of both coefficient
  blocks) + a config json + a trace png.

The stage-1 posterior draws are the object the fallback and the (attempted) stage 2 consume.

## Stage 2 — recursive Bayes, and why it is hard here

### Hooten et al.'s Prior-Proposal Recursive Bayes (PP-RB)

Partition the data `y = (y₁, …, y_J)`. Fit stage 1 to `y₁` → posterior sample
`θ^(k) ∼ [θ | y₁]`. For stage `j`, run MCMC on the full model but **propose** `θ*` by drawing
(with replacement) from the previous stage's sample; the M-H ratio (their eqs. 5–7)
**collapses** to a ratio of *conditional data models*:

```
r = [y_j | θ*, y_{1:j−1}] / [y_j | θ^(k−1), y_{1:j−1}].
```

The prior `[θ]`, the earlier-stage likelihoods `[y_{1:j−1} | θ]`, **and the proposal density**
all cancel — because the proposal is drawn from exactly `[θ | y_{1:j−1}] ∝ [y_{1:j−1}|θ][θ]`,
and the full-model target factorizes as `[y_{1:j−1}|θ][y_j|θ,y_{1:j−1}][θ]`. That factorization
holds **because `y₁` and `y_j` are disjoint data partitions** (conditional independence across
partitions makes the first-stage likelihood an exact factor of the full likelihood).

### What transfers to our model, and what does not

Our two stages are **not** a data partition — they analyze the **same** responses `Y`:

- Stage 1 likelihood: `L_marg(beta) = [Y_exc | beta]` = product of GPD densities on the
  exceedances.
- Stage 2 likelihood: `L_full(beta, θ_dep) = [Y | beta, θ_dep]` = the RW dependence model,
  whose per-time term `loglik_t` is a marginal GPD-density term **plus** a copula/transform
  term, **entangled through the `X = qRW(pCGP(Y))` transform and its Jacobian** — not a clean
  product `L_marg · [dependence]`.

Because `L_marg` is **not an exact factor** of `L_full`, the Hooten *cancellation of the
full data model* does not happen: `L_full` must still be evaluated at every proposal, so the
**headline PP-RB compute saving is lost**. But — and this is the correction to an earlier,
overly pessimistic version of this note — the acceptance ratio is **still exact and fully
computable**. Propose `beta*` from the stage-1 posterior
`q(beta) = [beta | Y_exc] ∝ L_marg(beta) p₁(beta) / Z₁`; the independence-MH ratio for the
stage-2 full-conditional `π(beta) ∝ L_full(beta) p₂(beta)` is

```
r = π(beta*) q(beta_cur) / [π(beta_cur) q(beta*)]
  = L_full(beta*)  p₂(beta*)  L_marg(beta_cur) p₁(beta_cur)
    ----------------------------------------------------------
    L_full(beta_cur) p₂(beta_cur) L_marg(beta*)  p₁(beta*)
```

so the intractable stage-1 normaliser `Z₁` cancels and

```
log r = [ℓ_full(beta*)  + logp₂(beta*)  − ℓ_full(beta_cur)  − logp₂(beta_cur)]
      − [ℓ_marg(beta*)  + logp₁(beta*)  − ℓ_marg(beta_cur)  − logp₁(beta_cur)].
```

`q`'s density is **not** needed in some intractable form — it is `L_marg · p₁` up to `Z₁`,
and both `Z₁` and (if the stage priors match) the priors cancel; we keep the prior terms
explicit so a stage-2 prior `p₂ ≠ p₁` is still exact. Combined with the sampler's usual
`θ_dep` updates this targets the correct joint posterior `[beta, θ_dep | Y]`. There is **no
double-use of the data**: the stage-1 posterior is used only as a *proposal* (its normaliser
and prior cancel out of the target), not as an informative prior. This is exactly the
Proposal-RB construction (Lunn et al. 2013; Hooten et al. 2021, eq. 7 / App. A) instantiated
for a non-partition model.

**Implemented** in `recursive_bayes.py::update_beta_recursive` (the ratio above,
`ℓ_marg` = `_marginal_exc_loglik`, MPI-reduced via `ctx.total`), wired as
`beta_sampler="recursive"` (needs `settings.recursive_prior_npz` = the stage-1 fit and both
margins active). `tests/test_recursive_bayes.py` reconstructs `log r` from independent pieces
and checks the update feeds exactly that to `mh_accept`.

### The open (empirical) question

Correctness is settled; the payoff is not. Since `L_full` is evaluated every proposal, the
per-move cost is essentially that of the existing beta blocks plus a cheap `L_marg` pair. The
independence proposal helps **iff** the stage-1 posterior overlaps the full-model `beta`
conditional; if the dependence pulls `beta` far from the marginal fit, acceptance will be low
and the fixed-margin fallback is the better use of the stage-1 result. That comparison is what
this sampler exists to run.

## Fallback — defensible fixed margins (the likely path)

If recursive Bayes does not earn its keep, use the stage-1 fit directly. `faroe_fit.py
--fix-margins` with `FAROE_STAGE1_NPZ` set:

1. Fixes the training-site margins at **`surface(mean β)`** — the stage-1 posterior-mean
   coefficients pushed through the log / scaled-logit links. (The *mean of the per-draw
   surfaces* is the exact posterior mean and differs by ~0.1%, but a per-site surface has no
   coefficient representation, so `surface(mean β)` is used for consistency with prediction —
   see below.) Mechanically it overwrites `data.sigma_vec`/`xi_vec`, which is what the
   sampler reads when `update_GPD_sigma/xi` is off (`ModelState.__init__`; `restore` does
   **not** recompute `sigma_mat` from β when the update is off).
2. **Records the constant betas per draw** in the run. This is essential: prediction at
   held-out sites (`predictive._holdout_marginal`) forms new-site margins from
   `beta_sigma`/`beta_xi` + the new-site covariates — a fixed per-site surface cannot
   extrapolate. Without this the run would carry no betas and new-site scoring would fall
   back to externally-supplied fixed values.

Note: without `FAROE_STAGE1_NPZ`, `--fix-margins` pins the **reference workflow** surface
(`inp["sigma_vec"]`), *not* the stage-1 fit — so the env var is what makes it a genuine
two-stage run.

The margins are then defensible because they come from a proper Bayesian marginal analysis
rather than from the dependence likelihood pulling on the tail parameters.

**Propagating marginal uncertainty (a cut / modular posterior).** Fixing at a point
estimate discards the stage-1 posterior spread. The principled alternative feeds the stage-2
run a fresh `(β_sigma, β_xi)` *draw* from the stage-1 posterior each iteration, so margins
vary but are not updated by the dependence likelihood — the **cut** distribution (Plummer
2015, *Stat. Comput.* 25:37–43; Liu, Bayarri & Berger 2009, *Bayesian Anal.* 4:119–150;
Jacob et al. 2017, arXiv:1708.08719), with the two-stage sampling mechanics of Lunn et al.
(2013, *JRSS-C* 62:551–572) and the frequentist IFM analogue of Joe & Xu (1996). Not yet
implemented here; contrast with the recursive sampler above, which instead targets the full
joint posterior (dependence *does* feed back to the margins).

## Files

| File | Role |
|------|------|
| `marginal_model.py` | Stage-1 model: `MarginalData.value_grad` + NUTS `fit` (standalone). |
| `faroe_analysis/scripts/faroe_marginal_fit.py` | Faroe stage-1 driver → `marginal_fit.npz`. |
| `faroe_analysis/scripts/run_faroe_marginal.sh` | Stage-1 run script. |
| `recursive_bayes.py` | `RecursivePrior` + `update_beta_recursive` (the exact stage-2 move). |
| `mcmc_updates.py` (dispatch) | `beta_sampler="recursive"` hook (off by default). |
| `sampler.py` | `SamplerSettings.recursive_prior_npz` (path to the stage-1 fit). |
| `faroe_analysis/scripts/faroe_fit.py` | `--beta-sampler recursive` + `FAROE_STAGE1_NPZ` seeding. |
| `faroe_analysis/scripts/run_faroe_stage2_recursive.sh` | Stage-2 recursive-Bayes run script. |
| `faroe_analysis/scripts/run_faroe_stage2_fixmargins.sh` | Stage-2 fixed-margin fallback run script. |
| `tests/test_marginal_model.py` | Stage-1: gradient FD check, value-agreement, NUTS recovery. |
| `tests/test_recursive_bayes.py` | Stage-2: exact log-ratio reconstruction + cache consistency. |
