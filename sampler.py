"""
Unified MCMC sampler for the three model types (shi / wang / huser-wadsworth),
all under Shi's GPD + multiplicative Log-Laplace nugget scheme.

Design
------
* Serial-first: by default all Nt time replicates are held in memory and looped
  (runs on a normal computer). An MPI backend (one rank per time slice, like
  Shi's original) can be layered on the same update functions; the serial path
  is what the replication tests exercise.
* Config dispatch: the covariance for Z(s) (nonstationary Matern for shi,
  stationary Matern for HW, VisGP precision for wang), the Wendland weights
  (standard/anisotropic/barrier), and the tail-exponent field (knot-based phi
  for shi/wang, scalar delta for HW; unit or positive domain) are all selected
  from the ModelConfig.
* Huser-Wadsworth is unified as the knot machinery with a single all-ones knot,
  so R = S(t) broadcast across sites and the same code paths apply.

The per-time conditional log-likelihood is utilities.Y_loglik_1t_qRWdRWout
(Shi's exact likelihood), with qRW/dRW precomputed outside the likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.spatial
from scipy.linalg import solve_triangular

import utilities as u
import kernels
import covariance as cov
from config import ModelConfig, MODEL_HW, PHI_UNIT
import mcmc_updates as mu
from mcmc_updates import Ctx
from rw_inte import qRW, qRW_warm
from rw_numeric import dRW_safe as dRW


# ---------------------------------------------------------------------------
@dataclass
class SamplerData:
    """Inputs to the sampler (a single dataset)."""
    Y: np.ndarray                  # (Ns, Nt), missing entries may be np.nan
    sites_xy: np.ndarray           # (Ns, 2)
    u_vec: np.ndarray              # (Ns,) thresholds
    sigma_vec: np.ndarray          # (Ns,) GPD scale (fixed marginal / init intercept)
    xi_vec: np.ndarray             # (Ns,) GPD shape (fixed marginal / init intercept)
    p_cens: float = 0.95
    # knot grids / weights (built by build_state if not provided)
    knots_S: Optional[np.ndarray] = None
    knots_phi: Optional[np.ndarray] = None
    knots_rho: Optional[np.ndarray] = None
    # wang extras
    cliques: Optional[list] = None
    separators: Optional[list] = None
    water_poly: object = None
    # --- optional GPD-marginal covariates (see marginal.MarginalDesign) -------
    # Each is spatial (Ns,p), temporal (Nt,p), or spatiotemporal (Ns,Nt,p).
    # When provided + the corresponding update flag is on, sigma/xi are modelled
    # as exp / scaled-logit of an intercept + these covariates' linear predictor.
    sigma_spatial: Optional[np.ndarray] = None
    sigma_temporal: Optional[np.ndarray] = None
    sigma_spatiotemporal: Optional[np.ndarray] = None
    xi_spatial: Optional[np.ndarray] = None
    xi_temporal: Optional[np.ndarray] = None
    xi_spatiotemporal: Optional[np.ndarray] = None


@dataclass
class SamplerSettings:
    n_iters: int = 2000
    adapt_size: int = 50
    thin: int = 1
    radius_S: float = 3.0
    # At init, bump radius_S up if the requested value (under the starting elliptical
    # anisotropy + barriers) leaves any site uncovered by the Wendland support, to
    # (min covering radius) * (1 + this margin) -- so the chain starts fully covered
    # with wiggle room (no zero-weight sites, and radius proposals down aren't
    # instantly rejected). Never shrinks a radius that already covers.
    radius_S_cover_margin: float = 0.2
    eff_range_phi: float = 3.0
    eff_range_rho: float = 3.0
    # phi smoothing kernel: 'gaussian' (Shi default, global support) or 'wendland'
    # (compact support -> a single phi-knot update only touches sites in its
    # support, so qRW/dRW are recomputed locally). 'wendland' changes the model.
    phi_kernel: str = "gaussian"
    phi_radius: Optional[float] = None   # Wendland support radius (default 2*eff_range_phi)
    nu: float = 1.0
    alpha0_min: float = 2.0
    alpha_stable: float = 0.5
    # rho (range) half-normal prior scale, as a multiple of the data's max pairwise
    # distance: scale = rho_prior_c * max_dist. Domain-aware -- a range much larger
    # than the domain is a near-constant (degenerate) field, so this auto-bounds rho
    # to sensible values regardless of the data's units/extent. c=1 -> scale = the
    # domain diameter (permissive); smaller c tightens it (c~0.5 centers a typical
    # range-half-the-domain field). Resolved to ModelState.rho_prior_scale at init.
    rho_prior_c: float = 1.0
    update_S: bool = True
    update_Z: bool = True
    update_phi: bool = True
    update_rho: bool = True
    update_alpha0: bool = True
    update_nu: bool = False           # update Matern smoothness nu
    # Joint (nu, rho) CP Metropolis instead of the one-at-a-time nu/rho blocks.
    # Only takes effect for scalar-rho models (k_rho == 1, i.e. HW / Wang) with
    # both update_nu and update_rho on; walks the correlated nu/rho ridge with a
    # 2x2 empirical-covariance-adapted move (see update_nu_rho_joint).
    joint_nu_rho: bool = False
    # ASIS interweaving of (nu, rho): run the joint CP move AND a non-centered (NCP)
    # move (Z' = L(theta') eta, accept on the data-likelihood ratio) each sweep, to
    # break the Z<->(nu,rho) coupling the CP-only block cannot. Scalar rho (HW/Wang);
    # implies the joint CP half. See update_nu_rho_ncp. DEFAULT OFF: the warm-HW A/B
    # (nuts+sep vs nuts+inr, 2026-07-05) shows the NCP half almost never fires
    # (accept ~0.003) and nu/rho ESS + throughput are a wash vs the properly-scaled
    # one-at-a-time blocks. The earlier "~2x" edge was an artefact of the OLD under-
    # scaled separate blocks; once make_adaptives seeds every block correctly (commit
    # 636fee6) that edge evaporated. Root cause is fundamental: co-moving the WHOLE
    # field with theta (Z' = L(theta') eta) swings a ~Ns*Nt-term data likelihood, so
    # any useful theta-step is rejected -- the "field can't be batched" wall (same as
    # the joint (eta,rho)-NUTS blow-up). Set True to re-enable (kept for the record /
    # low-rank co-move follow-up); auto-off for vector-rho (Shi, k_rho > 1) regardless.
    interweave_nu_rho: bool = False
    # Joint blocked adaptive-RWMH on the phi_k tail-exponent knots instead of the
    # one-at-a-time update_phi. The knots form a smooth spatial field (strongly
    # cross-correlated -- warm-Shi: mean |corr| ~0.5, leading mode ~48% of variance),
    # so per-knot moves crawl along their joint ridge. The joint move walks it with
    # an empirical-covariance proposal. Knots are split into contiguous blocks of at
    # most phi_block_max (adapting a full covariance in much higher dimension is
    # unreliable); each block is proposed/accepted as a unit on the data likelihood.
    joint_phi: bool = False
    phi_block_max: int = 15           # max knots per joint-phi block (blocked above this)
    # Block by SPATIAL proximity of the knot coords (balanced k-d median split) instead
    # of contiguous index. The default index blocking is only weakly spatial (row-major
    # grids + appended inner knots), so a contiguous block can span half the domain;
    # spatial blocking keeps each block's proposal covariance a genuinely LOCAL object.
    # With phi_block_max >= k_phi (one block) this has no effect.
    phi_block_spatial: bool = False
    # Optional (k_phi x k_phi) covariance to seed the joint-phi proposal Sigma_0
    # (e.g. the empirical phi_k covariance from a warm-start trace). Essential for
    # joint_phi: an isotropic start cannot self-organise onto phi's tight, strongly
    # correlated ridge. None -> fall back to a 1/d-scaled isotropic init.
    phi_proposal_cov: Optional[np.ndarray] = None
    # Optional per-margin covariances to seed the JOINT beta block Sigma_0 (the direct
    # analog of phi_proposal_cov, for beta_sampler="rwmh"). Essential there: from an
    # isotropic start the 11-D beta RWMH deadlocks (accept ~0 -> cov_est ~0 -> Sigma
    # stays isotropic; instrumented cond pinned at 1.0). NOTE this is a WARM-START
    # device (A/B / resume); a cold real analysis has no such covariance -- the cold
    # remedy is a design-derived seed or smaller blocks (see notes).
    beta_sigma_proposal_cov: Optional[np.ndarray] = None
    beta_xi_proposal_cov: Optional[np.ndarray] = None
    # ASIS non-centered interweave for VECTOR-rho models (Shi, k_rho > 1) -- the
    # generalisation of interweave_nu_rho (which is scalar-rho only). The component
    # CP update_nu/update_rho still run (the CP half); this adds an NCP half that
    # whitens eta = L(nu,rho)^-1 Z, proposes the covariance params, reforms
    # Z' = L(nu',rho') eta and accepts on the data likelihood, so the field follows
    # theta (breaks the Z<->theta coupling the CP-only moves cannot). Options:
    #   "off"   -> no NCP half (current Shi behaviour)
    #   "joint" -> propose [nu, rho_k...] together (blocked <= phi_block_max dims)
    #   "nu"    -> propose nu only (fallback if the big joint block mixes poorly)
    ncp_interweave_vec: str = "off"
    # Optional ((1+k_rho) x (1+k_rho)) covariance over [nu, rho_k...] to seed the
    # joint NCP proposal Sigma_0 (e.g. empirical from a warm trace). None -> isotropic.
    nu_rho_proposal_cov: Optional[np.ndarray] = None
    update_radius_S: bool = False     # update Wendland support radius for R's kernel
    update_anisotropy: bool = False   # update Wendland anisotropy (ratio, angle); needs config.use_anisotropy
    # Drive the Wendland-geometry parameters with ONE joint adaptive-RWMH block
    # instead of separate radius_S / ratio / angle moves. The triple is strongly
    # posterior-correlated (the coverage constraint forces a radius<->ratio ridge:
    # corr ~ -0.98 Shi, -0.62 Wang in exp1), so a covariance-adapted joint proposal
    # mixes along the ridge that one-at-a-time moves stall on -- and costs ONE global
    # W_wend_S/R/Y-likelihood rebuild per sweep instead of three. Block dim is 3 when
    # update_anisotropy is active (radius,ratio,angle), else 1 (radius only). No-op
    # for HW (no radius_S). Opt-in (A/B-gated); separate blocks remain the default.
    # Sampler for the phi knots:
    #   "rwmh" -> per-knot adaptive RWMH (update_phi), the default.
    #   "nuts" -> gradient NUTS on the analytic phi gradient (update_phi_nuts,
    #             phi_gradient.phi_value_grad). phi is the only block that has never
    #             reached stationarity in any run, at a correctly tuned acceptance --
    #             the signature the joint S block had before a gradient was worth 20x.
    #             The usual objection to a gradient here (a qRW inversion per
    #             leapfrog) does not bite, because the RWMH baseline pays k_phi full
    #             transforms per sweep: 272 ms against a trajectory's ~43 ms.
    #   "barker" -> Barker-proposal MALA (update_phi_barker): ONE gradient-informed
    #             joint move, two gradient evals a sweep against NUTS's whole
    #             trajectory. The per-knot RWMH is dear because it is COMPONENTWISE --
    #             k_phi full transforms a sweep -- so a joint move is ~2.8x cheaper and
    #             needs only ~35% of its ESS/iter to win, where NUTS needed 147%.
    phi_sampler: str = "rwmh"
    joint_geometry: bool = False
    update_GPD_sigma: bool = False   # update GPD scale regression coefficients
    update_GPD_xi: bool = False      # update GPD shape regression coefficients
    sigma_beta_sd: float = 5.0       # N(0, sd^2) prior on the beta coefficients
    xi_beta_sd: float = 5.0
    # Sampler for the GPD-regression coefficients:
    #   "component" -> per-COEFFICIENT 1-D adaptive-RWMH sweep (update_beta_component,
    #                  DEFAULT): each coef self-tunes, so a tightly-identified intercept
    #                  can't bottleneck the loosely-identified spatial coefs the way the
    #                  joint block does (which pinned beta_sigma at ~0.6% accept). Costs
    #                  ~d likelihood evals/sweep per margin. Best recovery in the warm
    #                  exp3-Shi A/B (sigma surfCorr +0.83, xi spRMSE 0.09) at ~4.5 s/it.
    #   "rwmh"      -> two per-margin JOINT adaptive-RWMH update_beta blocks.
    #   "nuts" -> ONE joint gradient-NUTS trajectory over [beta_sigma, beta_xi]
    #             (update_beta_nuts); requires BOTH update_GPD_sigma and
    #             update_GPD_xi. Shares nuts_warmup / nuts_target_accept. Gradient is
    #             analytic except the RW-transform log-derivative (exact closed-form
    #             rw_inte.dlog_dRW_dx). NOT RECOMMENDED: 30-55x slower than RWMH in the
    #             warm A/B and degrading -- each leapfrog is a full qRW-inversion sweep
    #             and the diagonal metric can't precondition the sigma/xi ridge, so
    #             trees hit max_depth (see mcmc_updates._beta_value_grad).
    #   "barker" -> ONE joint Barker-proposal MALA move over [beta_sigma, beta_xi]
    #             (update_beta_barker); requires BOTH margins active. Same analytic
    #             gradient as "nuts" but ONE proposal/iter (2 grad evals) instead of a
    #             tree, so it can't hit max_depth on the sigma/xi ridge -- a diagonal
    #             preconditioner + Barker's skew-adaptive sign flip carry the
    #             anisotropy (Livingstone & Zanella 2022; Vogrinc 2023). In A/B tests
    #             it did not beat component RWMH.
    #   "recursive" -> two-stage recursive-Bayes independence move: propose
    #             [beta_sigma, beta_xi] from a stage-1 marginal_fit posterior and
    #             accept against the FULL dependence likelihood corrected by the
    #             stage-1 marginal likelihood (update_beta_recursive; requires BOTH
    #             margins active + recursive_prior_npz). See
    #             docs/two_stage_recursive_bayes.md.
    beta_sampler: str = "component"
    # Path to the stage-1 marginal_fit.npz whose posterior drives the "recursive"
    # beta_sampler's independence proposal (ignored by every other beta_sampler).
    recursive_prior_npz: Optional[str] = None
    # Record the latent S and (large) Z traces. These are required ONLY for
    # out-of-sample model-comparison scoring (predictive log-lik / CRPS at
    # holdout or new sites). Set False to save disk + runtime memory when you
    # only need parameter posteriors. The Z trace is the big one: its size is
    # n_save * Ns * Nt * 8 bytes (e.g. ~35 MB at 150 saves, 225 sites, 128 times;
    # scales with how lightly you thin). latent_dtype='float32' halves it.
    save_latent: bool = True
    latent_dtype: str = "float64"   # 'float32' to halve the latent-trace storage
    # Stream the latent S/Z traces to disk via a memmap (written incrementally,
    # flushed every latent_chunk saves) instead of holding the whole trace in
    # RAM -- decouples runtime memory from chain length. Requires latent_out
    # (the run's output base path; the files become <latent_out>.Z.npy/.S.npy).
    stream_latent: bool = False
    latent_out: Optional[str] = None
    latent_chunk: int = 50
    # Checkpoint/resume: write a resumable checkpoint every `checkpoint_every`
    # iters to `checkpoint_path` (+ .pkl); set resume=True to continue an
    # interrupted run from the latest checkpoint. Restores state, adaptive
    # tuning, and the RNG so the resumed chain is identical to an uninterrupted one.
    checkpoint_every: int = 0
    checkpoint_path: Optional[str] = None
    resume: bool = False
    # Live diagnostics: write a multi-panel trace plot to plot_path every
    # plot_every iterations so the run can be monitored (headless-safe).
    plot_every: int = 0
    plot_path: Optional[str] = None
    # Progress feedback to stdout every `progress_every` iters (0 -> auto: ~50
    # updates over the run). Prints iter k/N, ll, elapsed, ETA, it/s.
    progress_every: int = 0
    # Latent Z sampler family:
    #   "rwmh" -> adaptive random-walk Metropolis (the z_update/z_block sub-mode
    #             below). Default; keeps existing runs byte-identical.
    #   "ess"  -> elliptical slice sampling on the exact Gaussian prior
    #             Z[:,t] ~ N(0,K); tuning-free, always moves (Phase 1).
    #   "nuts" -> gradient-based, per-time NUTS on the whitened coords eta_t
    #             (Z[:,t]=L eta_t); exact elementary gradient, dual-averaged step
    #             size warmed up then frozen (Phase 4).
    z_sampler: str = "rwmh"
    # NUTS step-size dual-averaging warmup length in sweeps (z_sampler=="nuts"
    # only); None -> n_iters // 2. After warmup the averaged step size is frozen.
    nuts_warmup: Optional[int] = None
    nuts_target_accept: float = 0.8
    # Adapt a DIAGONAL mass matrix (inverse metric = estimated posterior variance)
    # during warmup for the gradient blocks driven by NUTSWarmup. None (default) uses
    # a PER-BLOCK default: ON for beta (a fast-mixing data-likelihood block where the
    # warmup variance is reliable -> mean tree depth ~6->3), OFF for rho / nu-rho.
    # The nu/rho CP block mixes too slowly to self-estimate its scales, so a
    # warmup-adapted metric UNDERestimates them and over-shrinks the step (A/B on warm
    # Exp2 shi: nu-ESS x0.59) -- for nu/rho prefer a metric SEEDED from a warm trace
    # (nu_rho_mass_diag) over warmup adaptation. Set True/False to force it for all
    # NUTSWarmup blocks (e.g. the +mm/+nomm A/B arms); auto-off for nuts_warmup < 20.
    nuts_adapt_mass: Optional[bool] = None
    # Optional pre-seeded DIAGONAL inverse-metric for the (nu, rho) NUTS block over the
    # unconstrained coords [z_nu, log rho_k...] (length 1 + k_rho). When given, the
    # nu/rho arm uses this FIXED metric from the start instead of adapting one from its
    # own (slow-mixing) warmup -- seed it from a warm fit's posterior variance.
    # None -> fall back to nuts_adapt_mass.
    nu_rho_mass_diag: Optional[np.ndarray] = None
    # Sampler family for the rho range knots (Shi vector rho), on the CENTERED
    # conditional p(rho_k | Z, nu) with Z frozen:
    #   "rwmh" -> per-knot adaptive random-walk Metropolis (update_rho). Default;
    #             keeps existing runs byte-identical.
    #   "nuts" -> gradient-based NUTS on psi = log(rho_k) with the exact closed-form
    #             Gaussian-prior gradient (update_rho_nuts); shares nuts_warmup /
    #             nuts_target_accept. Prototype: serial only (target sums all times).
    rho_sampler: str = "rwmh"
    # Joint (nu, rho) sampler on the CENTERED conditional p(nu, rho | Z), Z frozen:
    #   "separate" -> the independent update_nu / update_rho blocks (default).
    #   "nuts"     -> ONE gradient NUTS trajectory over [z_nu, log rho_k] (logit nu,
    #                 log rho), using the ported dK_nu/dnu (bessel_nu). Requires
    #                 update_nu and update_rho; replaces both. Vector-rho safe (Shi).
    #                 Serial only (target sums all times). Shares nuts_warmup /
    #                 nuts_target_accept. Takes precedence over rho_sampler.
    nu_rho_sampler: str = "separate"
    # Record log S for this many time replicates (of the ones the recording worker
    # owns) so S mixing can be scored WITHOUT the full latent stream. 0 = off.
    # The full S trace is k_S * Nt per draw (1.1 GB at Faroe over 2000 draws) and the
    # A/B deliberately turns the latent stream off, since writing it perturbs the very
    # wall-clock being measured; k_S x s_trace_times is ~5 MB and costs nothing.
    s_trace_times: int = 0
    # Sampler for the latent Levy scales S:
    #   "joint"     -> one k_S-dimensional adaptive-covariance block per time
    #                  (default; what every run to date has used).
    #   "component" -> one knot at a time, using the Wendland compact support so
    #                  each move only touches its own support. Measured against the
    #                  production-tuned proposals at iteration 82,079: S-ESS min
    #                  3.4 -> 84.3 and median 10.3 -> 177.6 per 1000 draws, for
    #                  about +5% of total sampler wall (the S block is only ~0.12%
    #                  of an iteration). The joint block's learned CORRELATIONS were
    #                  measured to contribute nothing -- see update_S_component.
    #   "nuts"      -> per-time NUTS on log S with the exact elementary gradient
    #                  (update_S_nuts). One trajectory costs less than the
    #                  componentwise arm's k_S subset evaluations, because the S
    #                  gradient needs one Ns-pass plus two (Ns x k_S) matvecs.
    #   "indep"     -> per-knot mixture of an INDEPENDENCE proposal from the Levy
    #                  prior and the componentwise random walk (update_S_indep). The
    #                  prior cancels out of the independence ratio, so acceptance is
    #                  the likelihood ratio alone -- ~1 on the knots whose support
    #                  holds no exceedance, which is most of them in any one column.
    #
    # DEFAULT "nuts" since 2026-08-16, on exp4 (Ns=150, Nt=50, k_S=41, 4 chains x
    # 15000 iters per arm). ESS per 1000 iterations on the worst-mixing S knot --
    # the bottleneck, and the metric that decides:
    #
    #     arm         S-min   S-med   phi-min        (covariance fixed / freed)
    #     joint       0.225    1.37     2.387    |   0.248    1.41    1.371
    #     component   1.127   83.8      7.492    |
    #     indep       0.610  439.6      7.509    |
    #     both        0.674  441.7      7.936    |   1.262  448.9    4.899
    #     nuts        4.437  235.1     12.961    |   8.873  240.6    9.296
    #
    # 19.7x the joint block on S-min with the covariance fixed and 35.8x with it
    # freed -- consistent across two model families -- for +0.016 s/sweep, about 2%
    # of an iteration. indep/both beat it on the MEDIAN knot, but those are the
    # uninformed majority where a prior draw is nearly free; nuts is the only arm
    # that moves the hard ones. Scored per ITERATION, not per second: the globals
    # block's cost depends on where a chain sits (0.274 s at alpha0=2.2 vs 0.207 at
    # 6.0), so wall-clock comparisons between arms carry a location confound.
    s_sampler: str = "nuts"
    # Per-knot probability of an independence proposal when s_sampler == "indep".
    # 1.0 makes it pure independence (good on uninformed knots, poor on informed
    # ones); 0.0 reduces to the componentwise arm.
    s_prior_mix: float = 0.5
    # Metric for the per-time Z NUTS block:
    #   "unit" -> identity. THE DEFAULT, and measured to be the right one.
    #   "diag" -> Stan-style windowed diagonal-metric adaptation, shared across time
    #             replicates. MEASURED 2026-08-14: no gain. It raises the usable step
    #             size 46% and leaves the leapfrog count at 16.0 exactly, because tree
    #             depth is quantized and the conditioning gain (147 -> 58.5) is not
    #             enough to cross a doubling. 6-9% slower for nothing. Tested with the
    #             EXACT metric too, so it is not an estimation-quality problem.
    #             Retained default-off with the numbers in mcmc_updates.update_Z_nuts.
    z_metric: str = "unit"
    # Which implementation of the per-time Z gradient to drive update_Z_nuts with.
    #   "legacy" -> zblock_value_grad_np (default; keeps the goldens byte-exact)
    #   "trmv"   -> zblock_value_grad_trmv: same arithmetic, but the two matvecs with
    #               the Cholesky factor go through triangular BLAS instead of a dense
    #               matmul over a matrix that is half structural zeros. 1.37x on the
    #               gradient and 1.22x on a NUTS transition at Ns=369, agreeing to
    #               2e-16 -- which is still enough to diverge a chain, hence a switch.
    # DEFAULT "trmv" since 2026-08-14: 1.12x on a transition at 32 ranks, exact.
    # "legacy" remains as the reference the agreement tests are written against.
    z_grad: str = "trmv"
    # Which implementation of the joint (nu, rho) gradient to drive the block with.
    #   "fast"   -> _nu_rho_cp_value_grad_fast: identical target and closed forms, with
    #               the redundant work removed (one-pass covariance build, Bessel on
    #               unique pairs only, streamed order-channel quadrature). THE DEFAULT:
    #               1.76x at 32 ranks, agreeing with legacy to 3.6e-10. (2.05x serial, 1.77x at 4 ranks -- the Allreduce
    #               dilution saturates immediately and then holds.)
    #   "legacy" -> _nu_rho_cp_value_grad, the finite-difference-validated original.
    #               Retained as the reference the agreement tests are written against
    #               and as the control arm in benchmarks.
    nu_rho_grad: str = "fast"
    # Quadrature resolution for the nu order channel when nu_rho_grad == "fast".
    # None -> the module default, which is IDENTICAL to what legacy uses (n=128 over
    # u in [-8, 3]). Exposed so the accuracy of that grid can be studied, NOT as a
    # speed knob: the order channel amplifies dM error by up to ~5000x, so lowering it
    # produces a wrong gradient rather than a faster one. See docs/nurho_quadrature.md.
    nu_rho_quad_nodes: Optional[int] = None
    # Fire the joint (nu, rho) NUTS block only every k-th sweep (k=1 -> every sweep,
    # the default and byte-identical to before). Ignored unless nu_rho_sampler=="nuts".
    #
    # Why: at Faroe size the block is ~63% of the iteration (9.5 s of 15.1 s/iter at
    # 32 ranks -- it is replicated on every rank and does not parallelise), yet the
    # 2026-08-13 A/B showed it buys nu almost entirely per-CALL (nu ESS 51.1 vs 3.7 on
    # equal draws) while the blocks it crowds out (phi, alpha0) mix at the same rate
    # per sweep in both arms. Amortising the call restores those sweeps at a fraction
    # of the ESS cost.
    #
    # On a skipped sweep nu and rho are simply HELD -- there is NO fallback to the
    # RWMH blocks. This is a deterministic-scan cycle of period k over kernels that
    # each leave the target invariant, so the chain is still valid; a fallback would
    # instead make the kernel mix (not just its cost) differ between arms.
    #
    # NB nuts_warmup is counted in CALLS to the block, not sweeps, so with k>1 the
    # dual-averaging warmup spans k*nuts_warmup sweeps.
    nu_rho_nuts_every: int = 1
    # RWMH Z update parameterization (only used when z_sampler == "rwmh"):
    #   "ncp"      -> non-centered: propose in whitened coords e (Z = L e,
    #                 e ~ N(0,I), L = Cholesky of the Z covariance). Default;
    #                 mixes far better than the centered joint RWMH.
    #   "centered" -> the original adaptive joint RWMH on Z directly (for A/B).
    # z_block: NCP block size in whitened coords (None -> one joint block per
    # time, same cost as centered; small ints -> small-block/per-component moves,
    # better mixing at higher cost since each block needs a full loglik recompute).
    z_update: str = "ncp"
    z_block: Optional[int] = None
    seed: int = 0


def _covering_radius_S(sites, knots, config, ratio, angle, water_poly, requested,
                       lo, margin):
    """Starting Wendland radius that covers every site, with wiggle room.

    Returns ``max(requested, threshold * (1 + margin))`` where ``threshold`` is the
    smallest radius at which every site has >=1 knot in support (visibility- and
    anisotropy-aware, via ``kernels.domain_uncovered_mask``). Coverage is monotone
    in radius, so the threshold is found by bisection seeded from the Euclidean
    lower bound ``lo``. The starting anisotropy compresses the support elliptically,
    so the requested radius can leave a site isolated even when an isotropic support
    of the same radius would not -- this guarantees a covered start without shrinking
    a radius that already covers. If no radius covers (a site with no line-of-sight
    to any knot), returns the largest tried (the caller's uniform-weight fallback
    then applies, with a warning)."""
    k = knots.shape[0]

    def covered(r):
        return not kernels.domain_uncovered_mask(
            sites, knots, np.repeat(float(r), k), config=config,
            aniso_ratio=ratio, aniso_angle=angle, water_poly=water_poly).any()

    lo = max(float(lo), 1e-9)
    if covered(lo):
        thr = lo
    else:
        hi = max(float(requested), lo * 1.5)
        tries = 0
        while not covered(hi) and tries < 40:
            hi *= 1.5
            tries += 1
        if not covered(hi):
            return float(hi)                  # visibility-isolated: nothing covers
        a = lo
        for _ in range(30):                   # bisection to ~1e-9 relative
            mid = 0.5 * (a + hi)
            if covered(mid):
                hi = mid
            else:
                a = mid
        thr = hi
    return max(float(requested), thr * (1.0 + float(margin)))


# ---------------------------------------------------------------------------
class ModelState:
    """Holds current parameter values + cached derived quantities, with
    config-aware recomputation helpers shared across update blocks."""

    def __init__(self, config: ModelConfig, data: SamplerData,
                 settings: SamplerSettings):
        self.cfg = config
        self.data = data
        # Orientation of the zchol() square root M (M Mᵀ = Z covariance): lower for
        # the dense Cholesky of K (shi/HW), UPPER for visgp where M = R^{-T},
        # R = chol(Q). Used by the whitening solves (eta = M^{-1} Z).
        self._zfac_lower = not config.use_visgp
        self.st = settings
        self.rng = np.random.default_rng(settings.seed)
        Ns, Nt = data.Y.shape
        self.Ns, self.Nt = Ns, Nt

        # --- knots / weights -------------------------------------------------
        from simulate import make_isometric_knots
        minX, maxX = data.sites_xy[:, 0].min(), data.sites_xy[:, 0].max()
        minY, maxY = data.sites_xy[:, 1].min(), data.sites_xy[:, 1].max()
        dom = (minX - 1e-6, maxX + 1e-6, minY - 1e-6, maxY + 1e-6)

        if config.model_type == MODEL_HW:
            # single all-ones knot -> R(s,t)=S0(t), phi scalar
            self.knots_S = np.array([[(minX + maxX) / 2, (minY + maxY) / 2]])
            self.W_wend_S = np.ones((Ns, 1))
            self.knots_phi = self.knots_S
            self.W_gauss_phi = np.ones((Ns, 1))
            self.knots_rho = self.knots_S
            self.W_gauss_rho = np.ones((Ns, 1))
            self._min_radius_S = 0.0   # radius_S has no effect for HW (1 knot)
            self.aniso_ratio, self.aniso_angle = 1.0, 0.0   # N/A for HW
            self.radius_S = float(settings.radius_S)
            self._vis_mask_S = None    # no geometry block for HW
        else:
            self.knots_S = data.knots_S if data.knots_S is not None else \
                make_isometric_knots(*dom, 9)
            self.knots_phi = data.knots_phi if data.knots_phi is not None else \
                make_isometric_knots(*dom, 9)
            self.knots_rho = data.knots_rho if data.knots_rho is not None else \
                make_isometric_knots(*dom, 9)
            # anisotropy (elliptical Wendland support) for shi & wang; ratio in
            # (0,1), angle in [0,pi). Fixed at isotropic unless use_anisotropy.
            self.aniso_ratio = 0.5 if config.use_anisotropy else 1.0
            self.aniso_angle = 0.0
            # smallest radius that still covers every site (each site has >=1
            # S-knot within range); radius_S proposals below this are rejected.
            d_sk = scipy.spatial.distance.cdist(data.sites_xy, self.knots_S)
            self._min_radius_S = float(np.max(np.min(d_sk, axis=1)))
            # Fixed visibility mask (Wang barriers): metric-independent, so compute
            # the shapely line-of-sight tests ONCE here and reuse for every covering
            # -radius / coverage query (keeps shapely out of the per-proposal path).
            self._vis_mask_S = (kernels.visibility_mask(
                data.sites_xy, self.knots_S, data.water_poly,
                rule=config.barrier_visibility)
                if config.use_barrier_kernels else None)
            # start fully covered (visibility/anisotropy-aware) with wiggle room --
            # the elliptical init support can isolate a site at settings.radius_S.
            self.radius_S = _covering_radius_S(
                data.sites_xy, self.knots_S, config, self.aniso_ratio,
                self.aniso_angle, data.water_poly, float(settings.radius_S),
                self._min_radius_S, settings.radius_S_cover_margin)
            if self.radius_S > float(settings.radius_S):
                import warnings
                warnings.warn(
                    f"radius_S bumped {settings.radius_S:.3g} -> {self.radius_S:.3g} so "
                    "the starting Wendland support covers every site (some were "
                    "isolated by barriers/anisotropy at the requested radius).",
                    RuntimeWarning)
            self.W_wend_S = kernels.build_wendland_weights(
                data.sites_xy, self.knots_S,
                np.repeat(self.radius_S, self.knots_S.shape[0]),
                config=config, aniso_ratio=self.aniso_ratio,
                aniso_angle=self.aniso_angle, water_poly=data.water_poly)
            bw_phi = settings.eff_range_phi ** 2 / 6.0
            bw_rho = settings.eff_range_rho ** 2 / 6.0
            if settings.phi_kernel == "wendland":
                # compact-support phi surface: a single knot update only touches
                # sites within its support (exploited in the phi update block).
                radius = settings.phi_radius or (2.0 * settings.eff_range_phi)
                Wp = np.empty((self.data.sites_xy.shape[0], self.knots_phi.shape[0]))
                for j in range(self.knots_phi.shape[0]):
                    d = np.linalg.norm(data.sites_xy - self.knots_phi[j], axis=1)
                    Wp[:, j] = kernels._wendland_raw(d / radius)
                self.W_gauss_phi = kernels._normalize_rows(Wp)
            else:
                self.W_gauss_phi = u.build_weight_matrix(
                    data.sites_xy, self.knots_phi, kernel="gaussian",
                    radius=settings.radius_S, bandwidth=bw_phi, cutoff=False)
            self.W_gauss_rho = u.build_weight_matrix(
                data.sites_xy, self.knots_rho, kernel="gaussian",
                radius=settings.radius_S, bandwidth=bw_rho, cutoff=False)
            if config.use_visgp:
                # VisGP has a SCALAR range: visgp_precision consumes only
                # mean(rho_vec) (see rebuild_cov), so the per-knot rho field is
                # vestigial -- it leaves k_rho-1 directions unidentified by the
                # likelihood (only the prior moves them). Collapse rho to a single
                # knot so it is a genuine scalar (matches the model, and the
                # summary plot then drops the spurious rho(s) surface).
                self.knots_rho = np.array([[(minX + maxX) / 2, (minY + maxY) / 2]])
                self.W_gauss_rho = np.ones((Ns, 1))

        self.k_S = self.knots_S.shape[0]
        self.k_phi = self.knots_phi.shape[0]
        self.k_rho = self.knots_rho.shape[0]
        # which sites each phi-knot affects (compact support -> a strict subset)
        self._phi_compact = (config.model_type != MODEL_HW
                             and settings.phi_kernel == "wendland")
        self._phi_affected = [np.flatnonzero(self.W_gauss_phi[:, j] > 0.0)
                              for j in range(self.k_phi)]

        # --- gammas / gamma_bar ---------------------------------------------
        # (radius_S set per-branch above: covering-radius-with-margin for shi/wang,
        #  settings.radius_S for HW where the support is a single all-ones knot)
        self.update_aniso = (settings.update_anisotropy and config.use_anisotropy
                             and config.model_type != MODEL_HW)
        self.gamma_k = np.ones(self.k_S)
        self.gamma_bar = kernels.gamma_bar_from_weights(
            self.W_wend_S, self.gamma_k, alpha=settings.alpha_stable)

        # --- tail exponent (knot params on appropriate scale) ----------------
        self.phi_domain = config.phi_domain
        if self.phi_domain == PHI_UNIT:
            self.phi_k = np.full(self.k_phi, 0.5)        # constrained (0,1)
        else:
            self.phi_k = np.zeros(self.k_phi)            # unconstrained eta
        self.phi_vec = self._phi_field(self.phi_k)

        # --- covariance params ----------------------------------------------
        self.nu = settings.nu
        self.rho_k = np.full(self.k_rho, 1.0)
        self.rho_vec = self.W_gauss_rho @ self.rho_k
        self._dist_cache = u.ns_cov_precompute(data.sites_xy)
        self._DistMtx = cov.dist_matrix(data.sites_xy)
        # domain-aware rho prior scale: rho_prior_c * (max pairwise distance)
        self.rho_prior_scale = float(settings.rho_prior_c) * float(self._DistMtx.max())
        self.rebuild_cov()

        # --- alpha0 ----------------------------------------------------------
        self.alpha0 = settings.alpha0_min + 3.0

        # --- latent fields ---------------------------------------------------
        self.S = np.full((self.k_S, Nt), 1.0)
        self.Z = np.zeros((Ns, Nt))
        self.R = self.W_wend_S @ self.S

        # --- GPD marginal (sigma, xi) as (Ns, Nt) ---------------------------
        # Either fixed (broadcast the per-site vectors over time) or modelled by
        # covariate regression (intercept initialised from the per-site vectors).
        import marginal as _mg
        self._mg = _mg
        # Marginal regression activates on the update flag alone. With no
        # covariate groups this is a clean *intercept-only* estimate (the design
        # always includes the intercept); covariate groups add spatial/temporal
        # structure on top.
        self.update_sigma = bool(settings.update_GPD_sigma)
        self.update_xi = bool(settings.update_GPD_xi)
        if self.update_sigma:
            self.sigma_design = _mg.MarginalDesign(
                Ns, Nt, data.sigma_spatial, data.sigma_temporal, data.sigma_spatiotemporal)
            self.beta_sigma = np.zeros(self.sigma_design.n_params)
            self.beta_sigma[0] = np.log(np.median(data.sigma_vec))
            self.sigma_mat = _mg.sigma_from_eta(self.sigma_design.eta(self.beta_sigma))
        else:
            self.sigma_design = None
            self.beta_sigma = np.array([np.log(np.median(data.sigma_vec))])
            self.sigma_mat = np.repeat(np.asarray(data.sigma_vec)[:, None], Nt, axis=1)
        if self.update_xi:
            self.xi_design = _mg.MarginalDesign(
                Ns, Nt, data.xi_spatial, data.xi_temporal, data.xi_spatiotemporal)
            self.beta_xi = np.zeros(self.xi_design.n_params)
            self.beta_xi[0] = _mg.eta_from_xi(np.clip(np.median(data.xi_vec), -0.45, 0.95))
            self.xi_mat = _mg.xi_from_eta(self.xi_design.eta(self.beta_xi))
        else:
            self.xi_design = None
            self.beta_xi = np.array([_mg.eta_from_xi(np.clip(np.median(data.xi_vec), -0.45, 0.95))])
            self.xi_mat = np.repeat(np.asarray(data.xi_vec)[:, None], Nt, axis=1)

        # --- observed/missing masks per time --------------------------------
        self.miss = np.isnan(data.Y)
        self.Yfilled = np.where(self.miss, data.u_vec[:, None], data.Y)
        self.refresh_pY()   # pY + censored/exceedance index sets

    # -- derived-quantity helpers --------------------------------------------
    def _phi_field(self, phi_k):
        if self.phi_domain == PHI_UNIT:
            return self.W_gauss_phi @ phi_k
        return np.exp(self.W_gauss_phi @ phi_k)

    # -- checkpoint snapshot / restore ---------------------------------------
    # NUTS warmup/step adapters carried across a resume. Without them every restart
    # re-runs nuts_find_reasonable_eps + a full nuts_warmup of dual averaging, which
    # both wastes the warmup transitions and biases any timing calibrated on them
    # (warmup runs at a large eps => short trajectories => a misleadingly fast
    # s/iter). They hold only floats/arrays (NUTSWarmup does not retain its `vg`
    # closure), so they pickle cleanly. Keys are the ModelState attribute names.
    # Barker adapters ride in the same channel: they hold the same kind of state
    # (a dual-averaged eps + a diagonal preconditioner) and a warm start that drops
    # them re-runs the whole warmup, which is what this machinery exists to prevent.
    # Both are grng-driven, so neither is rank-local.
    _NUTS_ATTRS = ("_nuts_z", "_nuts_s", "_nuts_rho", "_nuts_nu_rho", "_nuts_beta",
                   "_nuts_phi", "_barker_beta", "_barker_phi")
    # per-RANK (seeded/updated from the LOCAL rng over this rank's own time columns)
    _NUTS_LOCAL = ("_nuts_z", "_nuts_s")

    def _nuts_snapshot(self):
        return {k: getattr(self, k) for k in self._NUTS_ATTRS
                if getattr(self, k, None) is not None}

    def _nuts_dims(self):
        """Expected theta length per adapter -- used to drop a restored adapter whose
        metric no longer matches the model (e.g. warm-starting across a different
        k_rho or beta length, where a stale inv_mass would be silently wrong)."""
        nb = self.beta_sigma.size + self.beta_xi.size
        return {"_nuts_rho": self.k_rho, "_nuts_nu_rho": 1 + self.k_rho,
                "_nuts_phi": self.k_phi, "_nuts_beta": nb,
                "_barker_phi": self.k_phi, "_barker_beta": nb}

    def _restore_nuts(self, snap):
        """Restore the adapters present in `snap`. Absent keys leave the current
        adapter untouched, so pre-existing checkpoints (which carry none) resume
        exactly as before and simply re-warm."""
        nuts = snap.get("nuts") or {}
        dims = self._nuts_dims()
        for k, ad in nuts.items():
            if k not in self._NUTS_ATTRS or ad is None:
                continue
            # NUTSWarmup carries inv_mass, BarkerWarmup a precond_sd; either is a
            # per-coordinate scale that is silently wrong if the block's dimension
            # changed since the checkpoint (e.g. a different k_phi or k_rho).
            im = getattr(ad, "inv_mass", None)
            if im is None:
                im = getattr(ad, "precond_sd", None)
            if im is not None and k in dims and np.size(im) != dims[k]:
                continue                                # stale scale -> re-warm
            setattr(self, k, ad)

    def snapshot(self):
        """All sampled values + the small derived kernels needed to resume.

        The "nuts" key is present only when this state actually has adapters (i.e.
        it has been run), so the cold-init snapshots that callers `np.savez` stay
        exactly as they were -- no 0-d object array for an empty dict."""
        s = dict(
            phi_k=self.phi_k.copy(), rho_k=self.rho_k.copy(), alpha0=self.alpha0,
            nu=self.nu, radius_S=self.radius_S, aniso_ratio=self.aniso_ratio,
            aniso_angle=self.aniso_angle, beta_sigma=self.beta_sigma.copy(),
            beta_xi=self.beta_xi.copy(), S=self.S.copy(), Z=self.Z.copy(),
            W_wend_S=self.W_wend_S.copy(), gamma_bar=self.gamma_bar.copy())
        nuts = self._nuts_snapshot()
        if nuts:
            s["nuts"] = nuts
        # Phase of the amortized nu/rho NUTS schedule (nu_rho_nuts_every > 1 only, so
        # every-sweep runs snapshot exactly as before). Without it a resume restarts
        # the cycle at 0 and the block fires one sweep early.
        if getattr(self, "_nu_rho_sweep", 0):
            s["nu_rho_sweep"] = int(self._nu_rho_sweep)
        return s

    def global_snapshot(self):
        """Snapshot of the global parameters only (no latent S/Z) -- for MPI,
        where the latent state is distributed and checkpointed per rank."""
        s = self.snapshot()
        del s["S"]; del s["Z"]
        # the Z adapter is per-rank (local rng, local time columns): it travels in
        # the per-rank checkpoint file, not this rank-0 global one.
        shared = {k: v for k, v in s.pop("nuts", {}).items()
                  if k not in self._NUTS_LOCAL}
        if shared:
            s["nuts"] = shared
        return s

    def restore(self, snap, latent=True, nuts=True):
        """Restore from a snapshot and rebuild all derived quantities. With
        latent=False the latent S/Z/R are left as-is (MPI restores them per rank).

        nuts=False skips the NUTS adapters -- the right choice for a WARM START from
        another fit, where the documented behaviour is cold tuning from a warm point:
        a frozen (post-warmup) adapter carried in from a different problem would never
        re-adapt. Checkpoint resume, which continues the same chain, keeps them."""
        self.nu = snap["nu"]
        self.rho_k = snap["rho_k"].copy()
        self.rho_vec = self.W_gauss_rho @ self.rho_k
        self.radius_S = snap["radius_S"]
        self.aniso_ratio = snap.get("aniso_ratio", self.aniso_ratio)
        self.aniso_angle = snap.get("aniso_angle", self.aniso_angle)
        self.W_wend_S = snap["W_wend_S"].copy()
        self.gamma_bar = snap["gamma_bar"].copy()
        if latent:
            self.S = snap["S"].copy()
            self.R = self.W_wend_S @ self.S
            self.Z = snap["Z"].copy()
        self.alpha0 = snap["alpha0"]
        self.beta_sigma = snap["beta_sigma"].copy()
        self.beta_xi = snap["beta_xi"].copy()
        if self.update_sigma:
            self.sigma_mat = self._mg.sigma_from_eta(self.sigma_design.eta(self.beta_sigma))
        if self.update_xi:
            self.xi_mat = self._mg.xi_from_eta(self.xi_design.eta(self.beta_xi))
        self.phi_k = snap["phi_k"].copy()
        self.phi_vec = self._phi_field(self.phi_k)
        if nuts:
            self._restore_nuts(snap)
            # scheduling state of the same block, so it travels with the adapters:
            # a resume continues the cycle, a WARM START (nuts=False) begins at 0.
            if "nu_rho_sweep" in snap:
                self._nu_rho_sweep = int(snap["nu_rho_sweep"])
        self.rebuild_cov()
        self.refresh_pY()

    def build_wendland(self, radius=None, ratio=None, angle=None):
        """Wendland weight matrix W_wend_S for proposed radius/anisotropy."""
        radius = self.radius_S if radius is None else radius
        ratio = self.aniso_ratio if ratio is None else ratio
        angle = self.aniso_angle if angle is None else angle
        return kernels.build_wendland_weights(
            self.data.sites_xy, self.knots_S,
            np.repeat(float(radius), self.k_S), config=self.cfg,
            aniso_ratio=ratio, aniso_angle=angle, water_poly=self.data.water_poly)

    def wendland_for_radius(self, radius):   # backward-compatible alias
        return self.build_wendland(radius=radius)

    def covering_radius(self, ratio=None, angle=None) -> float:
        """r_min(ratio, angle): the exact coverage threshold (radius > r_min covers
        every site), via the cached visibility mask. The joint-geometry proposal
        reparametrises to the unconstrained slack r - r_min so it never proposes an
        uncovered geometry. Equivalent to domain_covered as r > r_min, but returns
        the boundary value rather than a boolean, and uses no shapely (mask cached)."""
        ratio = self.aniso_ratio if ratio is None else ratio
        angle = self.aniso_angle if angle is None else angle
        return kernels.covering_radius(
            self.data.sites_xy, self.knots_S, aniso_ratio=ratio,
            aniso_angle=angle, vis_mask=getattr(self, "_vis_mask_S", None))

    def domain_covered(self, radius=None, ratio=None, angle=None) -> bool:
        """True iff the proposed Wendland geometry covers every site (each site
        has >=1 knot with positive raw weight, visibility-respecting). This is
        the anisotropy-aware coverage guarantee: a static minimum radius only
        works under isotropy, so radius_S / anisotropy proposals are checked
        against the actual (possibly elliptical) support each time they change."""
        radius = self.radius_S if radius is None else radius
        ratio = self.aniso_ratio if ratio is None else ratio
        angle = self.aniso_angle if angle is None else angle
        unc = kernels.domain_uncovered_mask(
            self.data.sites_xy, self.knots_S, np.repeat(float(radius), self.k_S),
            config=self.cfg, aniso_ratio=ratio, aniso_angle=angle,
            water_poly=self.data.water_poly)
        return not bool(unc.any())

    def rebuild_cov(self):
        """Build the covariance K (and precision Q for wang/VisGP).

        For VisGP, Q is assembled sparse and its log-determinant is cached from
        the clique factorizations, so visgp_logprob avoids any global O(d^3)
        slogdet and uses a sparse quadratic form. K (the dense inverse) is not
        formed for wang -- everything downstream uses Q.
        """
        self.Q_logdet = None
        if self.cfg.use_visgp:
            self.Q, self.Q_logdet = cov.visgp_precision(
                self.data.cliques, self.data.separators or [],
                self.data.sites_xy, float(np.mean(self.rho_vec)), self.nu,
                sigmasq=1.0, DistMtx=self._DistMtx, n_cores=self.cfg.n_cores,
                sparse=True, return_logdet=True)
            self.K = None
        elif self.cfg.model_type == MODEL_HW:
            # stationary Matern at the (canonical) scalar range and smoothness nu;
            # read from rho_vec so the rho update block (which sets rho_vec before
            # rebuilding) works for HW too. For HW rho_vec is constant == rho_k[0].
            self.K = cov.matern_wang(self._DistMtx, float(np.mean(self.rho_vec)),
                                     1.0, self.nu)
            self.Q = None
        else:
            # rho_vec is on the canonical (effective-range) scale; convert to
            # Shi's ns_cov internal range so rho means the same as wang/HW.
            range_internal = cov.canonical_to_ns_range(self.rho_vec, self.nu)
            self.K = u.ns_cov_cached(range_internal, np.ones(self.Ns),
                                     self._dist_cache, kappa=self.nu,
                                     cov_model="matern")
            self.Q = None
        self._Zchol = None   # invalidate the NCP Cholesky (rebuilt lazily)

    def Z_logprior(self, Zcol):
        if self.cfg.use_visgp:
            return cov.visgp_logprob(Zcol, self.Q, logdet=self.Q_logdet)
        return u.Z_logprior_1t(Zcol, _frozen_mvn(self.K))

    def zchol(self):
        """Lower-triangular Cholesky L of the marginal Z covariance (cov = L Lᵀ),
        computed lazily once per K/Q rebuild and reused by the non-centered Z
        update across all times. For VisGP the covariance is the dense inverse of
        the sparse precision Q (Ns is O(100s), so the one-off inverse is cheap)
        and only paid when z_update='ncp'."""
        L = getattr(self, "_Zchol", None)
        if L is None:
            if self.cfg.use_visgp:
                # Square root of the marginal covariance Sigma = Q^{-1}, WITHOUT ever
                # forming Q^{-1}. Factor the precision directly, Q = R Rᵀ (R lower);
                # then M = R^{-T} satisfies M Mᵀ = R^{-T} R^{-1} = Q^{-1} = Sigma. This
                # is both cheaper (one factorization + one triangular inverse vs a full
                # inverse + a second Cholesky) and far more robust: chol(Q) is stable
                # for any PD Q, whereas the old chol(inv(Q)) blew up when the explicit
                # inverse went numerically non-PD at ill-conditioned Q (the cluster
                # crash). M is UPPER triangular, so the whitening solve uses
                # _zfac_lower=False; the dense matmul Z = M @ eta (field update) and
                # M.T @ gradZ (gradient) are unchanged. _ensure_pd ridges Q only if the
                # factorization actually fails. NB: M is dense -- fine at Ns~O(100s);
                # for much larger Ns keep R sparse (CHOLMOD) and triangular-solve instead.
                Qd = self.Q.toarray() if hasattr(self.Q, "toarray") else np.asarray(self.Q)
                _, R = mu._ensure_pd(0.5 * (Qd + Qd.T))          # R = chol(Q), lower
                L = solve_triangular(R, np.eye(Qd.shape[0]), lower=True).T   # M = R^{-T}
            else:
                L = np.linalg.cholesky(self.K)                   # M = chol(Sigma), lower
            self._Zchol = L
        return L

    # -- per-time likelihood prep --------------------------------------------
    def refresh_pY(self):
        """Precompute the censored-GPD probabilities pY and the censored/
        exceedance index sets. The censored/exceedance partition depends only on
        Y vs u (fixed); pY depends on sigma/xi, so it is recomputed here whenever
        the marginal changes (and once up front). qRW for censored sites is
        pY-independent (= p_cens), so its dedup survives marginal updates.
        """
        Ns, Nt = self.Ns, self.Nt
        uu = self.data.u_vec
        self._pcens_vec = np.full(Ns, self.data.p_cens)
        self._pY = u.pCGP(self.Yfilled, self.data.p_cens, uu[:, None],
                          self.sigma_mat, self.xi_mat)
        self._cens_list, self._exc_list = [], []
        exc_i, exc_t = [], []
        for t in range(Nt):
            yc = self.Yfilled[:, t]
            c = np.flatnonzero(yc <= uu).astype(np.int64)
            e = np.flatnonzero(yc > uu).astype(np.int64)
            self._cens_list.append(c)
            self._exc_list.append(e)
            exc_i.append(e)
            exc_t.append(np.full(e.size, t, dtype=np.int64))
        self._exc_i = (np.concatenate(exc_i) if exc_i else np.array([], np.int64))
        self._exc_t = (np.concatenate(exc_t) if exc_t else np.array([], np.int64))
        self._pY_exc = self._pY[self._exc_i, self._exc_t]

    def cens_exc(self, t):
        return self._cens_list[t], self._exc_list[t]

    def qd_RW(self, t, phi_vec=None, gamma_bar=None, alpha0=None):
        """Per-time qRW/dRW (kept for callers that want a single column)."""
        qv, dv = self.qd_RW_all(phi_vec, gamma_bar, alpha0)
        return qv[:, t], dv[:, t]

    def qd_RW_all(self, phi_vec=None, gamma_bar=None, alpha0=None):
        """qRW/dRW for all (site, time), exploiting censoring redundancy.

        Censored entries (the great majority) all have pY = p_cens, so their qRW
        depends only on (phi, gamma_bar, alpha0) -- computed once per site rather
        than once per (site, time). dRW is needed only at exceedances (the
        censored likelihood term uses qRW alone), so it is skipped elsewhere.
        Exact: reproduces the per-(site,time) transforms bit-for-bit because for
        censored entries pY equals p_cens by construction.
        """
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        Ns, Nt = self.Ns, self.Nt
        qv_cens = qRW(self._pcens_vec, phi_vec, gamma_bar, alpha0)   # (Ns,)
        qv = np.repeat(qv_cens[:, None], Nt, axis=1)
        dv = np.zeros((Ns, Nt))
        if self._exc_i.size:
            ei, et = self._exc_i, self._exc_t
            qe = qRW(self._pY_exc, phi_vec[ei], gamma_bar[ei], alpha0)
            de = dRW(qe, phi_vec[ei], gamma_bar[ei], alpha0)
            qv[ei, et] = qe
            dv[ei, et] = de
        return qv, dv

    def qd_RW_cens(self, phi_vec=None, gamma_bar=None, alpha0=None):
        """Censored qRW value per site (same across all times). For MPI, where
        each rank only needs its own time columns."""
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        return qRW(self._pcens_vec, phi_vec, gamma_bar, alpha0)

    def qd_RW_one(self, t, qv_cens, phi_vec=None, gamma_bar=None, alpha0=None,
                  pY_col=None, qe_guess=None):
        """qRW/dRW for a single time column given the precomputed censored qRW.

        `pY_col` overrides the cached pY for this time (used by the GPD-beta
        update, which changes the exceedance pY).

        `qe_guess` (an Ns-length array, typically the PREVIOUS accepted qv column
        caches.qd[t][0]) warm-starts the exceedance root-finds from the last root:
        the MCMC inputs move slowly, so bracketing is near-instant. A stale/mismatched
        guess is safe -- qRW_warm falls back to the exact cold solve per cell."""
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        qv = qv_cens.copy()
        dv = np.zeros(self.Ns)
        e = self._exc_list[t]
        if e.size:
            pYe = self._pY[e, t] if pY_col is None else pY_col[e]
            if qe_guess is None:
                qe = qRW(pYe, phi_vec[e], gamma_bar[e], alpha0)
            else:
                qe = qRW_warm(pYe, phi_vec[e], gamma_bar[e], alpha0, qe_guess[e])
            qv[e] = qe
            dv[e] = dRW(qe, phi_vec[e], gamma_bar[e], alpha0)
        return qv, dv

    def qd_RW_subset(self, sites, phi_vec=None, gamma_bar=None, alpha0=None):
        """qRW/dRW over all times for a SUBSET of sites (for compact-support phi
        updates: only sites in the changed knot's support are recomputed).
        Returns (qv_sub, dv_sub), each (len(sites), Nt)."""
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        sites = np.asarray(sites, dtype=int)
        Nt = self.Nt
        qc = qRW(self._pcens_vec[sites], phi_vec[sites], gamma_bar[sites], alpha0)
        qv = np.repeat(qc[:, None], Nt, axis=1)
        dv = np.zeros((sites.size, Nt))
        mask = np.isin(self._exc_i, sites)
        if mask.any():
            ei, et = self._exc_i[mask], self._exc_t[mask]
            posmap = {int(s): k for k, s in enumerate(sites)}
            local = np.array([posmap[int(i)] for i in ei])
            qe = qRW(self._pY[ei, et], phi_vec[ei], gamma_bar[ei], alpha0)
            de = dRW(qe, phi_vec[ei], gamma_bar[ei], alpha0)
            qv[local, et] = qe
            dv[local, et] = de
        return qv, dv

    def loglik_subset_t(self, t, sites, phi_vec, qv_col, dv_col):
        """Log-likelihood contribution of a SUBSET of sites at time t (for
        compact-support phi delta updates). R, Z, gamma_bar, alpha0, sigma, xi
        are the current state; only phi_vec / qv / dv vary."""
        f = np.ascontiguousarray
        yc = self.Yfilled[sites, t]
        u_s = self.data.u_vec[sites]
        cens = np.flatnonzero(yc <= u_s).astype(np.int64)
        exc = np.flatnonzero(yc > u_s).astype(np.int64)
        return u.Y_loglik_1t_best(
            f(yc), float(self.data.p_cens), f(u_s), f(self.sigma_mat[sites, t]),
            f(self.xi_mat[sites, t]), f(self.R[sites, t]), f(self.Z[sites, t]),
            None, f(phi_vec[sites]), f(self.gamma_bar[sites]), float(self.alpha0),
            f(qv_col), f(dv_col), cens, exc)

    def loglik_subset_S(self, t, sites, R_sub, qv, dv):
        """Log-likelihood over a SUBSET of sites at time t with a PROPOSED R there.

        For the componentwise S update: moving one Wendland knot changes R only
        inside that knot's support, and R enters neither qRW nor dRW, so the MH
        ratio is a difference of these subset values -- every other site cancels.
        Everything except R is the live state; qv/dv are the caller's cached
        full-length qRW/dRW columns for time t (caches.qd[t])."""
        f = np.ascontiguousarray
        yc = self.Yfilled[sites, t]
        u_s = self.data.u_vec[sites]
        return u.Y_loglik_1t_best(
            f(yc), float(self.data.p_cens), f(u_s), f(self.sigma_mat[sites, t]),
            f(self.xi_mat[sites, t]), f(R_sub), f(self.Z[sites, t]), None,
            f(self.phi_vec[sites]), f(self.gamma_bar[sites]), float(self.alpha0),
            f(qv[sites]), f(dv[sites]),
            np.flatnonzero(yc <= u_s).astype(np.int64),
            np.flatnonzero(yc > u_s).astype(np.int64))

    def loglik_t(self, t, R=None, Z=None, phi_vec=None, gamma_bar=None,
                 alpha0=None, qv=None, dv=None, sigma_col=None, xi_col=None):
        R = self.R[:, t] if R is None else R
        Z = self.Z[:, t] if Z is None else Z
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        sigma_col = self.sigma_mat[:, t] if sigma_col is None else sigma_col
        xi_col = self.xi_mat[:, t] if xi_col is None else xi_col
        if qv is None or dv is None:
            qv, dv = self.qd_RW(t, phi_vec, gamma_bar, alpha0)
        cens, exc = self.cens_exc(t)
        # contiguous float64 inputs so the numba-compiled likelihood does not
        # recompile for column-slice layouts (R/Z are often non-contiguous).
        f = np.ascontiguousarray
        return u.Y_loglik_1t_best(
            f(self.Yfilled[:, t]), float(self.data.p_cens), f(self.data.u_vec),
            f(sigma_col), f(xi_col), f(R), f(Z), None,
            f(phi_vec), f(gamma_bar), float(alpha0), f(qv), f(dv), cens, exc)

    def export_zblock(self):
        """Export the fixed pieces of the Z-block conditional for a gradient-based
        (NUTS) update in white ``eta`` coords, matching the prototype potential
        ``nuts_implementation/prototype/zblock.make_potential``.

        In the Z conditional the transformed data ``X = qRW(pCGP(Y))`` is constant
        (it touches no RW-transform derivative), so the Z-gradient is elementary and
        the GSL/numba path is not needed for the move. Unlike ``export_state.py``
        (which rebuilds from a fresh simulation) this reads the *live* state, so the
        emitted conditional is exactly the one the Gibbs sweep currently targets.

        Returns numpy arrays (Z-independent unless noted):
          L      (Ns,Ns)  lower Cholesky of the marginal Z covariance (cov = L Lᵀ)
          logX   (Ns,Nt)  log transformed data X = qRW(pCGP(Y))
          logR   (Ns,Nt)  log R (the nugget field at sites)
          phi    (Ns,)    per-site tail exponent
          alpha0 scalar
          cens   (Ns,Nt)  bool: censored & observed
          exc    (Ns,Nt)  bool: exceedance & observed
        """
        Ns, Nt = self.Ns, self.Nt
        qv, _dv = self.qd_RW_all()
        logX = np.log(qv)
        cens = np.zeros((Ns, Nt), dtype=bool)
        exc = np.zeros((Ns, Nt), dtype=bool)
        for t in range(Nt):
            ci, ei = self.cens_exc(t)
            cens[ci, t] = True
            exc[ei, t] = True
        return dict(L=self.zchol(), logX=logX, logR=np.log(self.R),
                    phi=self.phi_vec.copy(), alpha0=float(self.alpha0),
                    cens=cens, exc=exc)

    # -- GPD marginal regression helpers -------------------------------------
    def gpd_support_ok(self, sigma_mat, xi_mat):
        """GPD support 1 + xi*(Y-u)/sigma > 0 at all exceedances."""
        ei, et = self._exc_i, self._exc_t
        if ei.size == 0:
            return True
        z = (self.Yfilled[ei, et] - self.data.u_vec[ei]) / sigma_mat[ei, et]
        return bool(np.all(1.0 + xi_mat[ei, et] * z > 0.0))

    def gpd_margin_evaluable(self, sigma_mat, xi_mat):
        """Cheap pre-likelihood guard, COMPANION to gpd_support_ok. A margin that
        passes the parametric support check can still push an exceedance CDF to the
        numerical boundary pY==1, where qRW(1)=+inf is undefined -- the datum would
        need an infinite latent X, i.e. zero density, so the proposal is a
        guaranteed MH reject. Catch it here (a cheap pCGP on the FULL exceedance
        set, so every MPI rank agrees, exactly like gpd_support_ok) and reject
        BEFORE the expensive per-time qRW/loglik loop -- otherwise a margin far from
        truth (cold or badly-scaled start) spills a flood of qRW/GSL warnings on
        proposals that were going to be rejected anyway. Returns True iff every
        exceedance CDF is strictly < 1 (safely evaluable)."""
        ei, et = self._exc_i, self._exc_t
        if ei.size == 0:
            return True
        pY = u.pCGP(self.Yfilled[ei, et], self.data.p_cens, self.data.u_vec[ei],
                    sigma_mat[ei, et], xi_mat[ei, et])
        return bool(np.all(pY < 1.0))

    def qd_exc_patch(self, qv, dv, pY_full, phi_vec=None, gamma_bar=None,
                     alpha0=None):
        """Recompute qRW/dRW at EXCEEDANCES from a new pY (censored entries are
        pY-independent and reused). For the GPD-beta update, which changes the
        marginal -> pY only at exceedances."""
        phi_vec = self.phi_vec if phi_vec is None else phi_vec
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        alpha0 = self.alpha0 if alpha0 is None else alpha0
        qv2, dv2 = qv.copy(), dv.copy()
        ei, et = self._exc_i, self._exc_t
        if ei.size:
            qe = qRW(pY_full[ei, et], phi_vec[ei], gamma_bar[ei], alpha0)
            qv2[ei, et] = qe
            dv2[ei, et] = dRW(qe, phi_vec[ei], gamma_bar[ei], alpha0)
        return qv2, dv2


_FROZEN_CACHE = {}


def _frozen_mvn(K):
    import scipy.stats
    key = id(K)
    fr = _FROZEN_CACHE.get(key)
    if fr is None:
        fr = scipy.stats.multivariate_normal(mean=None, cov=K, allow_singular=True)
        _FROZEN_CACHE.clear()
        _FROZEN_CACHE[key] = fr
    return fr


# ---------------------------------------------------------------------------
def run_mcmc(config: ModelConfig, data: SamplerData,
             settings: SamplerSettings, verbose: bool = False,
             init_snapshot: Optional[dict] = None) -> dict:
    """Run the serial MCMC and return trace arrays.

    Returns a dict of traces: phi_k, rho_k, alpha0, ll, plus final S, Z.

    init_snapshot : optional ModelState snapshot (see ModelState.snapshot) used to
        WARM-START the chain (e.g. from a previously fitted run). Applied via
        ModelState.restore before the caches are built; the adaptive blocks still
        start fresh. Ignored if a checkpoint is resumed (the checkpoint wins).
    """
    config.validate()
    keep_checkpoint_barrier_rule(config, settings,
                                 _checkpoint_file(settings.checkpoint_path or ""))
    state = ModelState(config, data, settings)
    rng = state.rng
    Ns, Nt = state.Ns, state.Nt
    st = settings

    # resume from a checkpoint if requested and present
    ckpt = (_load_checkpoint(st.checkpoint_path)
            if (st.resume and st.checkpoint_path) else None)
    if ckpt is not None:
        state.restore(ckpt["snapshot"])
        rng.bit_generator.state = ckpt["rng_state"]
    elif init_snapshot is not None:
        state.restore(init_snapshot, nuts=False)   # warm start = cold NUTS tuning

    # The serial worker owns every time replicate; one shared RNG drives both
    # global and latent proposals. reduce=None selects serial (np.sum) reduction.
    ctx = Ctx(range(Nt), rng, reduce=None, supports_compact=True,
              covers_all_times=True)

    # per-time caches: qRW/dRW, Y log-lik, Z log-prior (from restored state if
    # resuming). The shared update blocks read/write these in place.
    caches = mu.init_caches(state, ctx.times)

    # adaptive blocks (restored from the checkpoint, or built fresh)
    AD = ckpt["adaptives"] if ckpt is not None \
        else mu.make_adaptives(state, st, ctx.times, config)
    if ckpt is not None:
        keep_checkpoint_s_sampler(st, AD["S"], state.k_S)

    n_save = len(range(0, st.n_iters, st.thin))  # exactly the # of recorded iters
    TR = dict(
        phi_k=np.zeros((n_save, state.k_phi)),
        rho_k=np.zeros((n_save, state.k_rho)),
        alpha0=np.zeros(n_save),
        ll=np.zeros(n_save),
    )
    if getattr(st, "s_trace_times", 0):
        # log S for a few time replicates -- enough to score S mixing without the
        # 1.1 GB full latent trace. Serial owns every time, so take the first few.
        mon = np.arange(min(int(st.s_trace_times), state.Nt))
        TR["S_mon_times"] = mon
        TR["S_mon"] = np.zeros((n_save, state.k_S, mon.size))
    if st.update_nu:
        TR["nu"] = np.zeros(n_save)
    if st.update_radius_S and config.model_type != MODEL_HW:
        TR["radius_S"] = np.zeros(n_save)
    if state.update_aniso:
        TR["aniso_ratio"] = np.zeros(n_save)
        TR["aniso_angle"] = np.zeros(n_save)
    if state.update_sigma:
        TR["beta_sigma"] = np.zeros((n_save, state.beta_sigma.size))
    if state.update_xi:
        TR["beta_xi"] = np.zeros((n_save, state.beta_xi.size))
    # Latent-field traces (needed for out-of-sample / holdout prediction). These
    # are the large arrays; gated by settings.save_latent.
    if st.save_latent:
        # On resume the streamed memmaps are reopened and GROWN if this run asks
        # for more iters than the original (continuing an unconverged chain --
        # see _alloc_latent / the README). The non-streamed in-RAM traces are
        # re-allocated at the new n_save and refilled from the checkpoint below.
        TR["S"], TR["Z"] = _alloc_latent(st, n_save, state.k_S, Ns, Nt,
                                         resume=(ckpt is not None))

    start_iter = 0
    if ckpt is not None:
        start_iter = ckpt["it"] + 1
        # copy the recorded-so-far rows into the (possibly larger) trace arrays
        for k, v in ckpt["small_traces"].items():
            if k in TR:
                TR[k][:v.shape[0]] = v
        if st.save_latent and not st.stream_latent:
            TR["S"][:ckpt["S_trace"].shape[0]] = ckpt["S_trace"]
            TR["Z"][:ckpt["Z_trace"].shape[0]] = ckpt["Z_trace"]
        # (streamed latent traces persist in the memmap files, reopened above)

    def total_ll():
        return ctx.total_ll(caches.Yll, caches.Zlp)

    def _do_checkpoint(it):
        snap_traces = {k: TR[k] for k in TR
                       if k not in ("S", "Z", "state", "S_final", "Z_final")}
        payload = dict(it=it, snapshot=state.snapshot(), adaptives=AD,
                       rng_state=rng.bit_generator.state, small_traces=snap_traces)
        if st.save_latent and not st.stream_latent:
            payload["S_trace"] = TR["S"]; payload["Z_trace"] = TR["Z"]
        elif st.save_latent and st.stream_latent:
            TR["S"].flush(); TR["Z"].flush()
        _write_checkpoint(st.checkpoint_path, payload)
        # ALSO write the parameter traces to a readable .npz so progress is
        # inspectable (load_run / plots) without unpickling the checkpoint.
        npz_target = st.latent_out or (st.checkpoint_path + ".traces.npz")
        try:
            write_param_npz(npz_target, TR, state, st, it // st.thin + 1)
        except Exception as e:
            print(f"[checkpoint] readable trace npz write failed: {e}", flush=True)

    import diagnostics as _dg
    meter = (_dg.ProgressMeter(st.n_iters, st.progress_every or max(1, st.n_iters // 50),
                               start_iter=start_iter) if verbose else None)
    last_ll = float("nan")
    for it in range(start_iter, st.n_iters):
        # --- latent fields S, Z (local to this worker) ----------------------
        if st.update_S:
            mu.update_S(state, AD["S"], ctx, caches, st)
        if st.update_Z:
            mu.update_Z(state, AD["Z"], ctx, caches, st)

        # --- global parameters (phi, rho, alpha0, nu, radius_S, anisotropy,
        #     GPD marginal betas) -- the exact same blocks the MPI backend runs
        mu.run_global_updates(state, AD, ctx, caches, st, config)

        # --- adaptation ------------------------------------------------------
        if (it + 1) % st.adapt_size == 0:
            mu.adapt_all(AD, st.adapt_size, (it + 1) // st.adapt_size)

        # --- record ----------------------------------------------------------
        if it % st.thin == 0:
            s = it // st.thin
            TR["phi_k"][s] = state.phi_k
            TR["rho_k"][s] = state.rho_k
            TR["alpha0"][s] = state.alpha0
            if "S_mon" in TR:
                TR["S_mon"][s] = np.log(state.S[:, TR["S_mon_times"]])
            if "nu" in TR:
                TR["nu"][s] = state.nu
            if "radius_S" in TR:
                TR["radius_S"][s] = state.radius_S
            if "aniso_ratio" in TR:
                TR["aniso_ratio"][s] = state.aniso_ratio
                TR["aniso_angle"][s] = state.aniso_angle
            if state.update_sigma:
                TR["beta_sigma"][s] = state.beta_sigma
            if state.update_xi:
                TR["beta_xi"][s] = state.beta_xi
            TR["ll"][s] = total_ll()
            last_ll = TR["ll"][s]
            if st.save_latent:
                TR["S"][s] = state.S
                TR["Z"][s] = state.Z
                if st.stream_latent and (s + 1) % st.latent_chunk == 0:
                    TR["S"].flush(); TR["Z"].flush()   # checkpoint to disk

            # --- live diagnostic trace plot (record-based cadence) -----------
            # Triggered on the recording grid (this block runs only when
            # it % thin == 0), so plot_every is converted to a stride in
            # RECORDED samples -- matches the MPI backend and fires reliably for
            # any plot_every/thin combination (see sampler_mpi for the rationale).
            plot_stride = max(1, round(st.plot_every / st.thin)) if st.plot_every else 0
            if plot_stride and st.plot_path and (s + 1) % plot_stride == 0:
                try:
                    _dg.plot_traces(TR, s + 1, st.plot_path,
                                    thin=st.thin, iteration=it,
                                    phi_domain=config.phi_domain)
                    _dg.plot_live_surfaces(state, st.plot_path,
                                           phi_domain=config.phi_domain)
                except Exception as e:   # never let plotting kill a run, but be loud
                    print(f"[diagnostics] live trace plot failed: {e}", flush=True)

        # --- progress feedback (iter k/N, ll, elapsed, ETA, it/s) ------------
        if meter is not None:
            meter.update(it, last_ll)

        # --- checkpoint ------------------------------------------------------
        if st.checkpoint_every and st.checkpoint_path \
                and (it + 1) % st.checkpoint_every == 0:
            _do_checkpoint(it)

    if st.save_latent and st.stream_latent:
        TR["S"].flush(); TR["Z"].flush()
    TR["S_final"] = state.S
    TR["Z_final"] = state.Z
    TR["state"] = state
    TR["accept"] = mu.acceptance_summary(AD)   # run-level per-block MH acceptance
    try:
        import diagnostics
        TR["ess"] = diagnostics.ess_summary(TR, n_save, burn_frac=0.5)
        if verbose:
            diagnostics.print_ess(TR["ess"])
    except Exception:
        pass
    return TR


# ---------------------------------------------------------------------------
# Persist / reload a run for post-hoc prediction & holdout scoring
# ---------------------------------------------------------------------------
def _resume_memmap(path, n_save, tail_shape, dtype):
    """Reopen a streamed-latent memmap for resume, GROWING it if this run wants
    more recorded samples than the original (continuing an unconverged chain).

    The existing rows are preserved; new rows are zero-filled and written as the
    extended chain proceeds. Shrinking (fewer rows than already saved) is refused.
    """
    import os
    from numpy.lib.format import open_memmap
    existing = open_memmap(path, mode="r+")
    old_n = existing.shape[0]
    if old_n == n_save:
        return existing
    if old_n > n_save:
        raise ValueError(
            f"resume n_iters is SMALLER than the original ({path}: memmap has "
            f"{old_n} rows > requested {n_save}). Use n_iters >= the original.")
    # grow: write a larger memmap, copy the existing rows in, atomically replace
    tmp = path + ".grow.npy"
    big = open_memmap(tmp, mode="w+", dtype=dtype, shape=(n_save,) + tail_shape)
    big[:old_n] = existing[:]
    big.flush()
    del existing, big
    os.replace(tmp, path)
    return open_memmap(path, mode="r+")


def _alloc_latent(st: SamplerSettings, n_save, k_S, Ns, Nt, resume=False):
    """Allocate the latent S/Z trace storage: in-RAM arrays, or disk-backed
    memmaps (stream_latent) so runtime memory does not grow with chain length.
    With resume=True the streamed memmaps are reopened in 'r+' (keeping the data
    written before the checkpoint); if this run requests MORE iters than the
    original, the memmaps are grown to fit (chain continuation)."""
    if st.stream_latent:
        if not st.latent_out:
            raise ValueError("stream_latent=True requires latent_out (the run "
                             "output base path).")
        import os
        from numpy.lib.format import open_memmap
        zp, sp = _latent_paths(st.latent_out)
        os.makedirs(os.path.dirname(os.path.abspath(zp)) or ".", exist_ok=True)
        if resume:
            S = _resume_memmap(sp, n_save, (k_S, Nt), st.latent_dtype)
            Z = _resume_memmap(zp, n_save, (Ns, Nt), st.latent_dtype)
        else:
            S = open_memmap(sp, mode="w+", dtype=st.latent_dtype, shape=(n_save, k_S, Nt))
            Z = open_memmap(zp, mode="w+", dtype=st.latent_dtype, shape=(n_save, Ns, Nt))
        return S, Z
    return (np.zeros((n_save, k_S, Nt), dtype=st.latent_dtype),
            np.zeros((n_save, Ns, Nt), dtype=st.latent_dtype))


def _checkpoint_file(path: str) -> str:
    return path if path.endswith(".pkl") else path + ".pkl"


def keep_checkpoint_barrier_rule(config: ModelConfig, st: "SamplerSettings",
                                 ckpt_file: str) -> None:
    """On RESUME, keep the barrier-kernel visibility rule the chain was started
    with, rather than the one the caller built (mutates `config`).

    The default rule changed (ModelConfig.barrier_visibility, "domain" ->
    "obstacle"), and callers rebuild their config in code; resuming a checkpoint
    taken under the old rule would otherwise switch the Wendland kernels -- the
    model -- mid-chain. The rule in force is read from the config json every
    checkpoint already writes beside its trace npz (a json without the field
    loads as "domain")."""
    import os
    import warnings
    if not (config.use_barrier_kernels and st.resume and st.checkpoint_path
            and os.path.exists(ckpt_file)):
        return
    npz = st.latent_out or (st.checkpoint_path + ".traces.npz")
    cj = (npz if npz.endswith(".npz") else npz + ".npz") + ".config.json"
    if not os.path.exists(cj):
        return
    saved = ModelConfig.load(cj).barrier_visibility
    if saved != config.barrier_visibility:
        warnings.warn(f"resuming a chain fitted with barrier_visibility={saved!r}; "
                      f"keeping it (the caller asked for "
                      f"{config.barrier_visibility!r}). Start a fresh fit to use the "
                      f"new rule.")
        config.barrier_visibility = saved


_S_PER_KNOT_ARMS = ("component", "indep", "both")


def keep_checkpoint_s_sampler(st: "SamplerSettings", ad_S: dict, k_S: int) -> None:
    """On RESUME, keep the S sampler the checkpointed S adaptives were built for,
    rather than today's default (mutates `st.s_sampler`).

    The default changed ("joint" -> "nuts"), and the settings are not in the
    checkpoint, so a chain started under "joint" would resume under "nuts". That
    is a kernel switch mid-chain, and it also crashes at the first adaptation:
    "nuts" records a size-1 placeholder into each per-time k_S-dim adapter whose
    window still holds the joint block's k_S-vectors, and np.vstack refuses them.

    The arm is read off the adapters themselves: the per-knot arms hold a LIST of
    1-D adapters per time; joint and nuts both hold one k_S-dim adapter, and differ
    in what they record (the k_S log-scales vs a [0.0] placeholder). component,
    indep and both share one layout and record the same thing, so they cannot be
    told apart -- nor do they need to be: any of them resumes the others' adapters
    cleanly, so a choice within that family is honoured.

    A window can hold BOTH record sizes: a checkpoint written by a resume that ran
    "nuts" on a "joint" chain before this guard existed (it crashes at the next
    adaptation, but a checkpoint can land in between). Any k_S-vector means the
    chain is "joint"; the stray placeholders are then dropped so the joint block's
    covariance estimate sees only real draws. Dropping entries is safe: the window
    only feeds that estimate, and adapt() needs just >= 2 rows."""
    import warnings
    if not ad_S or k_S < 2:            # k_S == 1: joint and nuts records coincide
        return
    want = getattr(st, "s_sampler", "joint")
    v = next(iter(ad_S.values()))
    if isinstance(v, list):
        if want in _S_PER_KNOT_ARMS:
            return
        saved = "component"
    else:
        sizes = {np.asarray(w).size for w in v.window}
        if not sizes:
            return
        saved = "joint" if k_S in sizes else "nuts"
        if saved == "joint" and len(sizes) > 1:
            n_drop = 0
            for a in ad_S.values():
                keep = [w for w in a.window if np.asarray(w).size == k_S]
                n_drop += len(a.window) - len(keep)
                a.window = keep
            warnings.warn(f"S adaptive windows mixed joint draws with {n_drop} "
                          f"size-1 placeholders (a 'nuts' resume of this 'joint' "
                          f"chain was checkpointed); dropped the placeholders.")
        if saved == want:
            return
    warnings.warn(f"resuming a chain whose S adaptives were built for "
                  f"s_sampler={saved!r}; keeping it (the caller asked for {want!r}). "
                  f"Start a fresh fit to use the new sampler.")
    st.s_sampler = saved


def _write_checkpoint(path: str, payload: dict) -> None:
    """Atomically write a resume checkpoint (pickle)."""
    import os
    import pickle
    f = _checkpoint_file(path)
    tmp = f + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, f)


def _load_checkpoint(path):
    import os
    import pickle
    f = _checkpoint_file(path)
    if not os.path.exists(f):
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def _latent_paths(path: str):
    """Sibling .npy paths for the latent traces of a run saved at `path`."""
    base = path[:-4] if path.endswith(".npz") else path
    return base + ".Z.npy", base + ".S.npy"


def write_param_npz(path: str, TR: dict, state, settings: SamplerSettings,
                    nrec: int) -> None:
    """Write ONLY the parameter traces + geometry to `path` (a small .npz), using
    the first `nrec` recorded rows. Used mid-run at each checkpoint so progress is
    inspectable with sampler.load_run / diagnostics without unpickling the
    checkpoint (the latent S/Z .npy stream separately). Does not write the latent
    .npy or ess.json -- save_run does the full, final save."""
    data = state.data
    def cut(a):
        a = np.asarray(a)
        return a[:nrec]
    extra = {}
    if "nu" in TR:
        extra["nu_trace"] = cut(TR["nu"])
    if "radius_S" in TR:
        extra["radius_S_trace"] = cut(TR["radius_S"])
    if "aniso_ratio" in TR:
        extra["aniso_ratio_trace"] = cut(TR["aniso_ratio"])
        extra["aniso_angle_trace"] = cut(TR["aniso_angle"])
    for key in ("beta_sigma", "beta_xi"):
        if key in TR:
            extra[key] = cut(TR[key])
    tmp = path + ".tmp.npz"
    np.savez(tmp, phi_k=cut(TR["phi_k"]), rho_k=cut(TR["rho_k"]),
             alpha0=cut(TR["alpha0"]), ll=cut(TR["ll"]), sites_xy=data.sites_xy,
             u_vec=data.u_vec, sigma_vec=data.sigma_vec, xi_vec=data.xi_vec,
             knots_S=state.knots_S, knots_phi=state.knots_phi,
             knots_rho=state.knots_rho, gamma_k=state.gamma_k, nu=settings.nu,
             radius_S=settings.radius_S, eff_range_phi=settings.eff_range_phi,
             eff_range_rho=settings.eff_range_rho, p_cens=data.p_cens,
             has_latent=np.array(False), n_recorded=np.array(nrec), **extra)
    import os
    os.replace(tmp, path if path.endswith(".npz") else path + ".npz")
    config = getattr(state, "cfg", None)
    if config is not None:
        config.save((path if path.endswith(".npz") else path + ".npz") + ".config.json")


def save_run(path: str, TR: dict, config: ModelConfig, settings: SamplerSettings,
             data: SamplerData) -> None:
    """Save the traces + geometry needed to score/predict from a fitted run.

    Writes <path> (a small .npz: parameter traces + geometry), the large latent
    traces as sibling <path>.Z.npy / <path>.S.npy (memmap-friendly, so load is
    lazy), a <path>.config.json, and (for wang) a <path>.cliques.pkl. When the
    run streamed the latent traces (stream_latent), the .npy files are already on
    disk and are only flushed/relocated here -- never re-buffered through RAM.
    """
    import os
    state = TR["state"]
    extra = {}
    # traces use distinct keys so they don't clobber the scalar geometry values
    # (nu, radius_S) that build_draws reads as per-run defaults.
    if "nu" in TR:
        extra["nu_trace"] = TR["nu"]
    if "radius_S" in TR:
        extra["radius_S_trace"] = TR["radius_S"]
    if "aniso_ratio" in TR:
        extra["aniso_ratio_trace"] = TR["aniso_ratio"]
        extra["aniso_angle_trace"] = TR["aniso_angle"]
    for key in ("beta_sigma", "beta_xi", "S_mon", "S_mon_times"):
        if key in TR:
            extra[key] = TR[key]
    np.savez(path, phi_k=TR["phi_k"], rho_k=TR["rho_k"], alpha0=TR["alpha0"],
             ll=TR["ll"], sites_xy=data.sites_xy, u_vec=data.u_vec,
             sigma_vec=data.sigma_vec, xi_vec=data.xi_vec, knots_S=state.knots_S,
             knots_phi=state.knots_phi, knots_rho=state.knots_rho,
             gamma_k=state.gamma_k, nu=settings.nu, radius_S=settings.radius_S,
             eff_range_phi=settings.eff_range_phi,
             eff_range_rho=settings.eff_range_rho, p_cens=data.p_cens,
             has_latent=np.array("Z" in TR), **extra)
    config.save(path + ".config.json")
    if "ess" in TR:
        import json
        with open(path + ".ess.json", "w") as fh:
            json.dump(TR["ess"], fh, indent=2)
    if "Z" in TR:
        zp, sp = _latent_paths(path)
        if isinstance(TR["Z"], np.memmap):          # streamed: already on disk
            TR["Z"].flush(); TR["S"].flush()
            for mm, target in ((TR["Z"], zp), (TR["S"], sp)):
                cur = os.path.abspath(mm.filename)
                if cur != os.path.abspath(target):
                    os.replace(cur, target)          # move, no RAM copy
        else:                                        # in-RAM: write once
            np.save(zp, TR["Z"]); np.save(sp, TR["S"])
    if data.cliques is not None:
        import pickle
        with open(path + ".cliques.pkl", "wb") as fh:
            pickle.dump({"cliques": data.cliques,
                         "separators": data.separators or []}, fh)


def load_run(path: str) -> dict:
    """Load a run saved by save_run. The latent Z/S traces are memory-mapped
    (mmap_mode='r') so scoring streams them without loading the full arrays into
    RAM. Returns a dict with arrays, 'config', and (if present) cliques/seps."""
    import os
    with np.load(path) as npz:
        run = {k: npz[k] for k in npz.files}
    run["config"] = ModelConfig.load(path + ".config.json")
    zp, sp = _latent_paths(path)
    if os.path.exists(zp):
        run["Z"] = np.load(zp, mmap_mode="r")
        run["S"] = np.load(sp, mmap_mode="r")
    cpath = path + ".cliques.pkl"
    if os.path.exists(cpath):
        import pickle
        with open(cpath, "rb") as fh:
            cs = pickle.load(fh)
        run["cliques"] = cs["cliques"]; run["separators"] = cs["separators"]
    return run


def build_draws(run: dict, burn: int = 0, thin: int = 1) -> list:
    """Build the per-draw parameter dicts (for predictive.* new-site scoring)
    from a loaded run. Requires the latent S/Z traces (save_latent=True)."""
    if "S" not in run or "Z" not in run:
        raise ValueError("Run has no S/Z traces; refit with save_latent=True.")
    nu = float(run["nu"]); gamma_k = run["gamma_k"]
    # Gaussian rho-smoother (train) for the scalar canonical range (VisGP/HW)
    bw = float(run["eff_range_rho"]) ** 2 / 6.0
    W_rho = build_weight_matrix(run["sites_xy"], run["knots_rho"], kernel="gaussian",
                                radius=float(run["radius_S"]), bandwidth=bw, cutoff=False)
    idx = range(burn, run["phi_k"].shape[0], thin)
    has_bsig = "beta_sigma" in run
    has_bxi = "beta_xi" in run
    has_nu = "nu_trace" in run            # nu was updated -> per-draw value
    has_rad = "radius_S_trace" in run     # radius_S was updated -> per-draw value
    has_ar = "aniso_ratio_trace" in run   # anisotropy was updated -> per-draw value
    radius_fixed = float(run["radius_S"])
    draws = []
    for i in idx:
        rho_k = run["rho_k"][i]
        d = dict(
            # cast to float64 (traces may be stored as float32 to save space)
            Z_train=np.asarray(run["Z"][i], dtype=np.float64),
            S=np.asarray(run["S"][i], dtype=np.float64),
            phi_k=run["phi_k"][i], gamma_k=gamma_k, rho_k=rho_k,
            rho=float(np.mean(W_rho @ rho_k)),
            nu=float(run["nu_trace"][i]) if has_nu else nu,
            radius_S=float(run["radius_S_trace"][i]) if has_rad else radius_fixed,
            alpha0=float(run["alpha0"][i]))
        if has_ar:
            d["aniso_ratio"] = float(run["aniso_ratio_trace"][i])
            d["aniso_angle"] = float(run["aniso_angle_trace"][i])
        if has_bsig:
            d["beta_sigma"] = run["beta_sigma"][i]
        if has_bxi:
            d["beta_xi"] = run["beta_xi"][i]
        draws.append(d)
    return draws


# convenience re-export
from utilities import build_weight_matrix  # noqa: E402
