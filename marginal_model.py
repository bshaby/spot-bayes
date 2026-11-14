"""
Standalone Bayesian GPD regression on threshold exceedances (stage 1 of the
two-stage recursive-Bayes scheme).

This is the SIMPLE marginal model: the exceedances Y(s,t) > u(s) are modelled as
conditionally independent Generalized-Pareto given the GPD parameters, whose
(sigma, xi) surfaces are driven by the SAME spatially/temporally-varying covariates
the full dependence model uses (marginal.MarginalDesign) with Normal priors on the
coefficients. Crucially there is NO spatial-dependence field here (no Z/S/phi/rho/nu/
alpha0) and NO RW transform -- the likelihood is literally

    sum over {(s,t): Y(s,t) > u(s)} of log dGP(Y(s,t); loc=u(s), scale=sigma, shape=xi)
    + Normal(0, sd^2) priors on beta_sigma, beta_xi.

NOT a censored GPD: the non-exceedances are not analyzed at all (they contribute no
threshold-probability mass term). See docs/two_stage_recursive_bayes.md.

Why standalone matters. The production sampler DOES sample these betas, but through
the RW spatial-dependence likelihood (mcmc_updates._beta_value_grad / loglik_t), which
makes the marginal fit poorly identified. Here the analytic value+gradient are exactly
that machinery with the entire RW-transform path removed, leaving only the GPD-density
terms -- so the fit is a clean, defensible marginal analysis.

The module depends only on numpy/scipy, marginal.py, utilities.py, and the reusable
NUTS driver (mcmc_updates.NUTSWarmup / nuts_transition, which take a plain
vg(theta)->(logp, grad) callable). It never imports ModelState / MPI.

Dataset-generic: build a MarginalDesign for each margin (spatial / temporal /
spatiotemporal columns, as usual) and hand it Y and the per-site threshold u_vec.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import marginal as mg
import utilities as u
from mcmc_updates import NUTSWarmup, nuts_transition

# Below this |xi| the exact-formula d(log dGP)/dxi (which carries 1/xi^2 and 1/xi)
# is replaced by its analytic xi->0 limit (a removable singularity). The GPD log
# density is smooth in xi at 0; the limit is dlogdGP_dxi -> z^2/2 - z.
_XI_SMALL = 1e-6


# --------------------------------------------------------------------------- #
#  Per-margin design evaluated ONLY at the exceedance cells (efficient+generic) #
# --------------------------------------------------------------------------- #
class _DesignExc:
    """A marginal.MarginalDesign restricted to the exceedance cells (exc_s, exc_t).

    Precomputes the covariate rows needed to form eta and to fold a per-cell
    d(loglik)/d(eta) back into the coefficient gradient, matching the MarginalDesign
    coefficient layout [beta0, beta_sp, beta_tp, beta_st]. Works for any combination
    of spatial / temporal / spatiotemporal covariate groups."""

    def __init__(self, design: mg.MarginalDesign, exc_s, exc_t):
        self.n_params = design.n_params
        self.p_sp, self.p_tp, self.p_st = design.p_sp, design.p_tp, design.p_st
        self.n_exc = exc_s.size
        self.Xs = design.spatial[exc_s] if self.p_sp else None            # (n_exc, p_sp)
        self.Xt = design.temporal[exc_t] if self.p_tp else None           # (n_exc, p_tp)
        self.Xst = (design.spatiotemporal[exc_s, exc_t] if self.p_st else None)  # (n_exc,p_st)

    def eta(self, beta) -> np.ndarray:
        """Linear predictor at the exceedance cells, shape (n_exc,)."""
        out = np.full(self.n_exc, beta[0], dtype=float)
        i = 1
        if self.p_sp:
            out += self.Xs @ beta[i:i + self.p_sp]; i += self.p_sp
        if self.p_tp:
            out += self.Xt @ beta[i:i + self.p_tp]; i += self.p_tp
        if self.p_st:
            out += self.Xst @ beta[i:i + self.p_st]
        return out

    def grad_beta(self, dL_deta) -> np.ndarray:
        """Fold per-cell dL/d(eta), shape (n_exc,), into the coefficient gradient,
        shape (n_params,), in the [beta0, sp, tp, st] layout."""
        g = np.empty(self.n_params, dtype=float)
        g[0] = dL_deta.sum()
        i = 1
        if self.p_sp:
            g[i:i + self.p_sp] = self.Xs.T @ dL_deta; i += self.p_sp
        if self.p_tp:
            g[i:i + self.p_tp] = self.Xt.T @ dL_deta; i += self.p_tp
        if self.p_st:
            g[i:i + self.p_st] = self.Xst.T @ dL_deta
        return g


# --------------------------------------------------------------------------- #
#  The model container                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class MarginalData:
    """Everything the stage-1 GPD-regression target needs.

    Y            : (Ns, Nt) responses; NaN = missing (never an exceedance).
    u_vec        : (Ns,) per-site GP threshold (loc). Only used to build the
                   exceedance mask Y > u and as the GPD location.
    sigma_design : MarginalDesign for log(sigma).  eta -> sigma via mg.sigma_from_eta.
    xi_design    : MarginalDesign for the scaled-logit xi. eta -> xi via mg.xi_from_eta.
    sigma_beta_sd, xi_beta_sd : Normal(0, sd^2) prior scales on the coefficients.

    On construction the exceedance cells are located ONCE and each design is
    restricted to them (self._sx / self._xx), so value_grad touches only the
    ~exceedance-fraction of cells.
    """
    Y: np.ndarray
    u_vec: np.ndarray
    sigma_design: mg.MarginalDesign
    xi_design: mg.MarginalDesign
    sigma_beta_sd: float = 5.0
    xi_beta_sd: float = 5.0

    # populated in __post_init__
    exc_s: np.ndarray = field(init=False, repr=False)
    exc_t: np.ndarray = field(init=False, repr=False)
    y_exc: np.ndarray = field(init=False, repr=False)
    u_exc: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.Y = np.asarray(self.Y, dtype=float)
        self.u_vec = np.asarray(self.u_vec, dtype=float)
        Ns, Nt = self.Y.shape
        assert self.u_vec.shape == (Ns,), "u_vec must be (Ns,)"
        assert self.sigma_design.Ns == Ns and self.sigma_design.Nt == Nt, \
            "sigma_design shape mismatch"
        assert self.xi_design.Ns == Ns and self.xi_design.Nt == Nt, \
            "xi_design shape mismatch"
        exc = np.isfinite(self.Y) & (self.Y > self.u_vec[:, None])
        self.exc_s, self.exc_t = np.where(exc)
        self.y_exc = self.Y[self.exc_s, self.exc_t]
        self.u_exc = self.u_vec[self.exc_s]
        self._sx = _DesignExc(self.sigma_design, self.exc_s, self.exc_t)
        self._xx = _DesignExc(self.xi_design, self.exc_s, self.exc_t)
        self._ns = self.sigma_design.n_params        # split point of theta

    # -- dimensions ---------------------------------------------------------
    @property
    def n_sigma(self):
        return self.sigma_design.n_params

    @property
    def n_xi(self):
        return self.xi_design.n_params

    @property
    def n_params(self):
        return self.n_sigma + self.n_xi

    @property
    def n_exc(self):
        return self.exc_s.size

    # -- the NUTS target ----------------------------------------------------
    def value_grad(self, theta):
        """(logp, grad) of the stage-1 target at theta = [beta_sigma, beta_xi].

        Value: sum of plain GPD log-density over the exceedance cells (via
        utilities._gp_logpdf_np) + Normal priors. Gradient: the analytic
        d(log dGP)/d(sigma, xi) chained through the (log / scaled-logit) links and
        the covariate design -- the RW-free half of mcmc_updates._beta_value_grad --
        plus the prior gradient. Returns (-inf, 0) outside GPD support (sigma>0 is
        automatic; any 1+xi*z<=0 cell makes the log density -inf)."""
        ns = self._ns
        b_sig, b_xi = theta[:ns], theta[ns:]
        eta_s = self._sx.eta(b_sig)
        eta_x = self._xx.eta(b_xi)
        sig = mg.sigma_from_eta(eta_s)                     # (n_exc,) > 0
        xi = mg.xi_from_eta(eta_x)                         # (n_exc,) in (XI_LO, XI_HI)

        # value: plain GPD log density on exceedances (loc = threshold)
        logdens = u._gp_logpdf_np(self.y_exc, self.u_exc, sig, xi)
        if not np.all(np.isfinite(logdens)):              # some 1+xi*z <= 0
            return -np.inf, np.zeros_like(theta)
        ll = float(logdens.sum())

        # gradient of the data term (analytic; RW path removed)
        z = (self.y_exc - self.u_exc) / sig
        w = 1.0 + xi * z
        dlogdGP_ds = -1.0 / sig + z * (1.0 + xi) / (sig * w)          # d/d sigma
        small = np.abs(xi) < _XI_SMALL
        dlogdGP_dx = np.empty_like(xi)
        nz = ~small
        dlogdGP_dx[nz] = (np.log(w[nz]) / xi[nz] ** 2
                          - (1.0 / xi[nz] + 1.0) * z[nz] / w[nz])
        dlogdGP_dx[small] = 0.5 * z[small] ** 2 - z[small]           # xi -> 0 limit
        # link Jacobians: d sigma / d eta_s = sigma ; d xi / d eta_x = (HI-LO) g (1-g)
        LO, HI = mg.XI_LO, mg.XI_HI
        sg = (xi - LO) / (HI - LO)
        dL_deta_s = dlogdGP_ds * sig
        dL_deta_x = dlogdGP_dx * (HI - LO) * sg * (1.0 - sg)
        g_sig = self._sx.grad_beta(dL_deta_s)
        g_xi = self._xx.grad_beta(dL_deta_x)

        # Normal priors (value + gradient)
        ss, sx = self.sigma_beta_sd, self.xi_beta_sd
        ll += float((-0.5 * (b_sig / ss) ** 2 - np.log(ss)).sum())
        ll += float((-0.5 * (b_xi / sx) ** 2 - np.log(sx)).sum())
        ll += -0.5 * theta.size * np.log(2 * np.pi)
        g_sig -= b_sig / ss ** 2
        g_xi -= b_xi / sx ** 2
        return ll, np.concatenate([g_sig, g_xi])

    # -- map coefficients -> parameter surfaces -----------------------------
    def sigma_surface(self, beta_sigma):
        """sigma(s,t) over the full grid, shape (Ns, Nt)."""
        return mg.sigma_from_eta(self.sigma_design.eta(beta_sigma))

    def xi_surface(self, beta_xi):
        """xi(s,t) over the full grid, shape (Ns, Nt)."""
        return mg.xi_from_eta(self.xi_design.eta(beta_xi))


# --------------------------------------------------------------------------- #
#  NUTS fit                                                                      #
# --------------------------------------------------------------------------- #
def _save_ckpt_atomic(path, state):
    """Pickle the resume state to `path` atomically (write tmp, then os.replace)."""
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_ckpt(path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def fit(data: MarginalData, beta_sigma0, beta_xi0, n_iters, *, warmup=None,
        thin=1, seed=0, target_accept=0.8, adapt_mass=True, rng=None,
        progress=False, checkpoint_path=None, checkpoint_every=0, resume=True,
        on_checkpoint=None):
    """Fit the stage-1 marginal model by NUTS (analytic gradient).

    Reuses the exact production NUTS driver: NUTSWarmup dual-averages the step size
    and (adapt_mass) estimates a diagonal metric over a warmup window; nuts_transition
    does one Stan-style multinomial/generalised-U-turn transition. The metric matters
    here because the intercept and the (standardized) covariate coefficients live on
    different scales.

    Checkpointing (optional, for long/interruptible runs). If `checkpoint_path` is set
    and `checkpoint_every > 0`, the full resume state (theta, the NUTSWarmup adapter, the
    RNG state, and the draws so far) is pickled every `checkpoint_every` iters and at the
    end; `on_checkpoint(partial_result)` is then called so the driver can write inspectable
    artifacts (npz/png). On a later call, if `checkpoint_path` exists and `resume` is True,
    the run CONTINUES from it (extending to a larger `n_iters` if asked); an already-complete
    checkpoint returns immediately. The NUTSWarmup schedule (its `_m` counter) persists, so
    warmup/metric adaptation resume correctly.

    Returns a dict of stacked draws (post-thinning): beta_sigma (K, n_sigma),
    beta_xi (K, n_xi), plus per-iteration accept_stat and tree depth and run metadata.
    """
    beta_sigma0 = np.asarray(beta_sigma0, dtype=float)
    beta_xi0 = np.asarray(beta_xi0, dtype=float)
    assert beta_sigma0.size == data.n_sigma, "beta_sigma0 length mismatch"
    assert beta_xi0.size == data.n_xi, "beta_xi0 length mismatch"
    if warmup is None:
        warmup = max(1, n_iters // 2)
    ns = data.n_sigma
    vg = data.value_grad

    # -- resume from a checkpoint, or cold-start ---------------------------
    ck = None
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        ck = _load_ckpt(checkpoint_path)
        if ck.get("n_sigma") != data.n_sigma or ck.get("n_xi") != data.n_xi:
            raise ValueError(f"checkpoint {checkpoint_path} has different dims "
                             f"({ck.get('n_sigma')},{ck.get('n_xi')}) than this data "
                             f"({data.n_sigma},{data.n_xi}); delete it to refit.")
    if ck is not None:
        theta = ck["theta"]
        nb = ck["nb"]
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["rng_state"]
        draws_s, draws_x = ck["draws_s"], ck["draws_x"]
        accepts, depths = ck["accepts"], ck["depths"]
        start = ck["i"]
        if progress:
            print(f"[marginal.fit] resuming from {checkpoint_path} at iter {start}/{n_iters}",
                  flush=True)
    else:
        rng = np.random.default_rng(seed) if rng is None else rng
        theta = np.concatenate([beta_sigma0, beta_xi0])
        logp0 = vg(theta)[0]
        if not np.isfinite(logp0):
            raise ValueError("initial (beta_sigma0, beta_xi0) is outside GPD support "
                             "(logp = -inf); pick inits with 1 + xi*z > 0 everywhere.")
        nb = NUTSWarmup(vg, theta, rng, warmup, target_accept=target_accept,
                        adapt_mass=adapt_mass)
        draws_s, draws_x, accepts, depths = [], [], [], []
        start = 0

    def _partial():
        return dict(
            beta_sigma=np.array(draws_s), beta_xi=np.array(draws_x),
            accept=np.array(accepts), depth=np.array(depths),
            warmup=int(warmup), thin=int(thin), n_iters=int(n_iters),
            n_exc=int(data.n_exc), n_sigma=int(data.n_sigma), n_xi=int(data.n_xi))

    def _checkpoint(i_next):
        if checkpoint_path:
            _save_ckpt_atomic(checkpoint_path, dict(
                theta=theta, nb=nb, rng_state=rng.bit_generator.state,
                draws_s=draws_s, draws_x=draws_x, accepts=accepts, depths=depths,
                i=i_next, n_sigma=data.n_sigma, n_xi=data.n_xi))
        if on_checkpoint is not None:
            on_checkpoint(_partial())

    if start >= n_iters:                       # already complete -> idempotent no-op
        if progress:
            print(f"[marginal.fit] checkpoint already at {start} >= {n_iters}; nothing to do.",
                  flush=True)
        return _partial()

    for i in range(start, n_iters):
        theta, acc, depth = nuts_transition(vg, theta, nb.eps, rng,
                                            inv_mass=nb.inv_mass)
        nb.update(theta, acc, vg, rng)
        accepts.append(acc)
        depths.append(depth)
        if i % thin == 0:
            draws_s.append(theta[:ns].copy())
            draws_x.append(theta[ns:].copy())
        if progress and (i + 1) % max(1, n_iters // 20) == 0:
            print(f"[marginal.fit] {i + 1}/{n_iters}  accept~{np.mean(accepts[-100:]):.2f} "
                  f"depth~{np.mean(depths[-100:]):.1f}", flush=True)
        if checkpoint_every and (i + 1) % checkpoint_every == 0 and (i + 1) < n_iters:
            _checkpoint(i + 1)

    _checkpoint(n_iters)                        # final checkpoint + artifact emit
    return _partial()
