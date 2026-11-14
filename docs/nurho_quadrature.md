# The nu order channel is an ill-conditioned contraction

Status: **open accuracy issue in the production gradient**, found 2026-08-13 while
benchmarking `_nu_rho_cp_value_grad_fast` against the legacy implementation. Nothing
here is caused by the fast path; both implementations share the grid and the problem.

## What the block computes

The nu-gradient of the joint (nu, rho) CP target has an *order* channel

    order_ch = 2 * sum_{p in upper pairs} w_p * dM_p ,   w = (Gm * S)[iu]

where `dM_p = d/dnu of the Matern correlation` at pair distance `x_p`, supplied by
`bessel_nu.dmatern_dnu` via a trapezoidal quadrature of the integral representation
of `dK_nu/dnu` on a grid of `n_quad` nodes over `u in [u_min, u_max]`, `t = e^u`.

## The trap

It is natural to size that grid by the accuracy of `dkv_dnu` itself. That is the
wrong quantity. `order_ch` is a sum of ~68k terms of alternating sign, and it cancels
hard:

    sum|terms| / |sum terms|   ranges from ~2 to ~5000

across the (nu, rho) region this model visits, growing with both. A relative error
`eps` in `dM` therefore arrives in the gradient multiplied by up to ~5000. At Ns=369
the default grid resolves `dkv_dnu` to 2.8e-8 — and still delivers an `order_ch` that
is wrong by 8.6e-2 at nu=1.95, rho=3.

## Measured

Relative error of the **contracted scalar** against converged references (n=320–400 on
two different ranges, agreeing to 8 significant figures), Ns=369, k_rho=13:

| nu | rho | cond | n=128, u∈[-8,3] *(production)* | n=128, u∈[-5,2] |
|---|---|---|---|---|
| 0.55 | 1.5 | 2 | 2.2e-08 | 2.7e-08 |
| 0.55 | 3.0 | 13 | 6.7e-07 | 5.3e-08 |
| 0.55 | 6.0 | 257 | 3.3e-05 | 5.1e-03 |
| 1.3 | 1.5 | 41 | 2.2e-04 | 6.2e-07 |
| 1.3 | 3.0 | 218 | 2.0e-03 | 8.3e-07 |
| 1.3 | 6.0 | 1951 | 1.1e-02 | 6.3e-02 |
| 1.95 | 1.5 | 97 | 3.3e-03 | 4.5e-07 |
| 1.95 | 3.0 | 1443 | 8.6e-02 | 4.1e-05 |
| 1.95 | 6.0 | 4949 | **5.5e+00** | 2.3e-01 |

## Two conclusions

**A narrower u-range is not a saving, it is a bug.** `[-5,2]` looks better than
`[-8,3]` in the middle of the table, but it truncates real mass, so refining `n`
converges to the *wrong* integral. At nu=1.95, rho=6 the sequence n=128, 160, 192, 256
over `[-5,2]` gives −2.9e8, −1.2e8, **+1.0e8**, +5.8e8 — a sign flip — while the
converged truth is −3.798e8. Any apparent accuracy at low `n` on a narrow range is two
errors cancelling. Keep `[-8,3]` (or wider) and do not economise on nodes.

**The grid this block has always used is under-resolved in part of the space.** At
nu≈2 the nu-gradient can be wrong by 9% (rho=3) or a factor of 6.5 (rho=6). Whether
the sampler spends time there is an open question — Faroe's fitted rho is ≈3 and nu is
free over [0.5, 2] — but nu sits at the top of the block's prior support with a known
mixing problem, and a gradient that is 9% wrong there is at least a candidate
contributor rather than a curiosity.

## What to do about it

Not yet decided. Options, cheapest first:

1. **Raise `n_quad` where it matters.** Cost is exactly linear in nodes, so a blanket
   n=320 is 2.5x the quadrature cost — the opposite of a speedup, and the quadrature
   is ~70% of the gradient.
2. **Size the grid from the data.** The required range depends on `x.min()` (small x
   needs large t, i.e. larger `u_max`); the condition number is computable on the fly
   from `w` and `dM`. A grid chosen per call from those two would spend nodes only
   where the contraction is actually ill-conditioned.
3. **Reformulate to avoid the cancellation.** The cleanest fix if it exists, and the
   only one that removes the amplification rather than out-resolving it. Not explored.

Guardrails already in place: `tests/test_rho_nuts.py::test_order_channel_is_ill_conditioned`
fails if the contraction ever becomes well-conditioned (which would invalidate the
comment on `_NU_RHO_QUAD_DEFAULT`), and `test_order_channel_chunking_is_invariant`
pins the streamed contraction to be block-size independent.

The table was produced with a benchmark script that is not part of this release;
`tests/test_rho_nuts.py` checks the fast gradient against the validated original.
