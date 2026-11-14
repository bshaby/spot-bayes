"""
Shared MCMC machinery for the serial and MPI samplers.

Two layers live here:

1. The adaptive random-walk Metropolis primitive (`Adaptive`, `mh_accept`) ---
   Shi's scheme: log-scale step-size adaptation toward an optimal acceptance
   rate, plus empirical covariance Sigma_0 adaptation --- reused by every block.

2. The parameter **update blocks** themselves (`update_S`, `update_phi`, ...),
   written once and called by *both* `sampler.run_mcmc` (serial) and
   `sampler_mpi.run_mcmc_mpi`. The only differences between the two backends are
   captured by a small `Ctx` object: which time replicates this worker owns, the
   RNG handle(s), and how per-time log-likelihoods are reduced to a scalar
   (a plain ``np.sum`` over all times for serial; an MPI Allreduce of the
   per-rank partial sum for MPI). The per-time caches (Yll, Zlp, qd) live in a
   `Caches` container, keyed by time index in both backends.

Both backends use the per-time qRW/dRW recompute (`qd_RW_cens` + `qd_RW_one`),
which is bit-for-bit identical to the serial batched `qd_RW_all` because the
RW transforms are elementwise -- so unifying the blocks does not change the
numerical output of either sampler (guarded by tests/test_sampler_golden.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve, LinAlgError
from scipy.linalg.blas import dtrmv as _dtrmv
import warnings as _warnings
from scipy.special import (log_ndtr, kv as _kv, gamma as _gamma, expit as _expit,
                           psi as _digamma)

try:  # numba is in the `extremes` env; the NUTS Z-block kernel is JIT-compiled.
    from numba import njit as _njit
except Exception as _numba_exc:  # pragma: no cover - fall back to a pure-python kernel
    # DO NOT let this fall back silently: a `extremes` env missing numba once cost a ~24x
    # Alpine slowdown that hid for weeks. chol/qRW stay fast (no numba), so nothing flags
    # it, yet _zblock_value_grad_kernel -- ~80%% of NUTS wall time -- runs interpreted.
    # Announce it loudly so the next missing-numba env is a one-line fix, not an autopsy.
    _warnings.warn(
        "numba unavailable ({}: {}) -- the NUTS Z-block kernel _zblock_value_grad_kernel "
        "will run in PURE PYTHON, ~20-30x slower per sweep. Run `conda install numba` in "
        "this env to restore JIT speed. Results are bit-identical either way (same float64 "
        "formulas), so this is purely a performance foot-gun.".format(
            type(_numba_exc).__name__, _numba_exc),
        RuntimeWarning, stacklevel=2)

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

import utilities as u
import priors as pr
import bessel_nu
import kernels
import phi_transform as pt
from config import MODEL_HW, PHI_UNIT
from rw_inte import dlog_dRW_dx


# Adaptation constants (match Shi's sampler defaults)
R_OPT = 0.234
C0, C1, OFFSET = 1.0, 0.8, 3.0


def _ensure_pd(M: np.ndarray):
    """Return (M', L): M' is M if positive-definite, else M plus a minimal
    scale-relative ridge that makes it PD; L is M''s lower Cholesky factor. The
    Robbins-Monro covariance blend keeps Sigma_0 PD in exact arithmetic, so for
    healthy covariances no ridge is added -- but a matrix can drift close enough
    to singular that Cholesky fails even when not strictly singular, so we trigger
    on Cholesky failure itself. The ridge is relative to the matrix scale (trace/d)
    so it never swamps small-magnitude covariances (phi eig ~1e-8). L is returned
    so the caller can reuse it to build the proposal -- the PD check costs nothing."""
    try:
        return M, np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        scale = max(np.trace(M) / M.shape[0], 1e-300)
        for k in range(1, 13):                       # escalate until PD (roundoff only)
            R = M + (10.0 ** (k - 12)) * scale * np.eye(M.shape[0])
            try:
                return R, np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                continue
        R = M + 1e-3 * scale * np.eye(M.shape[0])
        return R, np.linalg.cholesky(R)


@dataclass
class Adaptive:
    """Per-block adaptive state for random-walk Metropolis."""
    dim: int
    sigma_m_sq: float = 1.0
    Sigma_0: np.ndarray = None
    num_accepted: int = 0
    window: list = field(default_factory=list)
    # Target acceptance. 0.234 is the d->inf optimum; low-d optima are higher
    # (~0.44 at d=1). Default stays 0.234 so every legacy block is byte-unchanged;
    # callers that want the dimension-aware optimum pass r_opt=_target_accept(dim).
    r_opt: float = R_OPT
    _chol: np.ndarray = None      # cached lower Cholesky of Sigma_0 (proposal factor)
    n_prop: int = 0               # cumulative proposals (for run-level acceptance reporting)
    n_acc: int = 0                # cumulative accepts (NOT reset by adapt(), unlike num_accepted)

    def __post_init__(self):
        if self.Sigma_0 is None:
            self.Sigma_0 = np.eye(self.dim)
        self.Sigma_0, self._chol = _ensure_pd(np.atleast_2d(self.Sigma_0))

    def propose(self, current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # sample via the cached Cholesky (same as MVN(0, Sigma_0) in distribution);
        # L is the factor already computed for the PD check, so no extra work.
        L = getattr(self, "_chol", None)
        if L is None:
            self.Sigma_0, L = _ensure_pd(np.atleast_2d(self.Sigma_0))
            self._chol = L
        return current + np.sqrt(self.sigma_m_sq) * (L @ rng.standard_normal(self.dim))

    def record(self, accepted: bool, value: np.ndarray):
        if accepted:
            self.num_accepted += 1
        # cumulative run-level tally (getattr-safe: tolerates checkpoints predating
        # these fields, where the attribute is absent on the unpickled object)
        self.n_prop = getattr(self, "n_prop", 0) + 1
        self.n_acc = getattr(self, "n_acc", 0) + (1 if accepted else 0)
        self.window.append(np.asarray(value, dtype=float).copy())

    def adapt(self, adapt_size: int, t_adapt: int):
        gamma = C0 / ((t_adapt + OFFSET) ** C1)
        r_hat = self.num_accepted / float(adapt_size)
        self.sigma_m_sq = float(np.clip(
            np.exp(np.log(self.sigma_m_sq) + gamma * (r_hat - self.r_opt)), 1e-3, 1e3))
        self.num_accepted = 0
        if len(self.window) >= 2:
            buf = np.vstack(self.window[-adapt_size:])
            cov_est = np.atleast_2d(np.cov(buf, rowvar=False))
            if cov_est.shape == (self.dim, self.dim):
                # Robbins-Monro convex blend (gamma/C0 < 1): with Sigma_0 starting
                # PD and cov_est PSD, the recursion stays PD on its own -- no
                # regularisation needed. _ensure_pd only catches numerical roundoff
                # (and then with a SCALE-RELATIVE ridge); a fixed 1e-6 floor would
                # swamp small-scale params -- phi eig ~1e-8 -- and isotropise the
                # proposal, destroying the learned correlation. _ensure_pd also
                # returns the Cholesky factor, cached for the next propose() calls.
                self.Sigma_0, self._chol = _ensure_pd(
                    self.Sigma_0 + (gamma / C0) * (cov_est - self.Sigma_0))
        self.window = self.window[-adapt_size:]
        return r_hat


def mh_accept(log_r: float, rng: np.random.Generator) -> bool:
    """Standard Metropolis acceptance given a log acceptance ratio."""
    return bool(np.isfinite(log_r) and (np.log(rng.random()) < log_r))


# Elliptical slice sampling: shrinkage always brackets theta=0 (where the
# proposal returns to the current point with ll >= the slice level), so the loop
# terminates in a few evals. The cap only guards against numerical pathology
# (e.g. a stale cached ll or a NaN likelihood) turning into an infinite loop.
_ESS_MAX_EVAL = 40


def elliptical_slice(f, nu, loglik, ll_cur, rng, max_eval=_ESS_MAX_EVAL):
    """One elliptical slice sampling transition (Murray, Adams & MacKay 2010).

    Targets ``p(f) ∝ N(f; 0, K) * exp(loglik(f))`` and leaves the Gaussian prior
    ``N(0, K)`` invariant *exactly*, with no step-size tuning. The prior enters
    only through the auxiliary draw ``nu ~ N(0, K)``; the move always lands on the
    ellipse ``f cos θ + nu sin θ`` and always accepts (the bracket shrinks toward
    θ=0, which recovers ``f`` with ``loglik(f) = ll_cur > log_y``).

    Parameters
    ----------
    f : current state (an array drawn so its marginal is N(0, K)).
    nu : a fresh prior draw ~ N(0, K) (e.g. ``L @ standard_normal`` with K=L Lᵀ).
    loglik : callable(x) -> float, the log-likelihood only (NOT the prior).
    ll_cur : loglik(f), the cached current value (saves one evaluation).
    rng : numpy Generator (drives the slice level, angle, and shrinkage draws).

    Returns ``(f_new, ll_new, n_eval)``; ``n_eval`` is the number of loglik calls.
    """
    log_y = ll_cur + np.log(rng.random())          # slice level on the likelihood
    theta = rng.uniform(0.0, 2.0 * np.pi)           # initial angle
    theta_min, theta_max = theta - 2.0 * np.pi, theta
    for n_eval in range(1, max_eval + 1):
        f_prop = f * np.cos(theta) + nu * np.sin(theta)
        ll = loglik(f_prop)
        if ll > log_y:
            return f_prop, ll, n_eval
        if theta < 0.0:                              # shrink the bracket toward 0
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)
    return f, ll_cur, max_eval                       # safety: degenerate slice


# ===========================================================================
# Z-block conditional potential for gradient-based (NUTS) updates (Phase 4)
# ===========================================================================
# Pure-numpy mirror of nuts_implementation/prototype/zblock.make_potential. In the
# Z conditional X = qRW(pCGP(Y)) is constant, so the Z-dependence is elementary
# (no RW-transform derivatives): X* = R^phi / Phibar(Z), eps = X / X*, with a
# log-Laplace(alpha0) likelihood on eps. The Z-independent dCGP(Y)/dRW(X) terms
# that state.loglik_t also carries are dropped here -- they are additive constants
# for this conditional, so this reproduces loglik_t's Z-DEPENDENCE (not its
# absolute value); the faithfulness test
# (tests/test_sampler.py::test_zblock_potential_matches_loglik_t) gates exactly that. This is also the core of the production (non-JAX) NUTS potential.
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _norm_logsf_np(z):
    """log standard-normal survival function, via the compiled log_ndtr ufunc
    (logsf(z) = log Phi(-z)); ~10x cheaper than scipy.stats.norm.logsf, which the
    NUTS leapfrog calls tens of thousands of times per sweep."""
    return log_ndtr(-z)


def _norm_logpdf_np(z):
    """log standard-normal pdf (analytic; avoids the scipy.stats wrapper)."""
    return -0.5 * z * z - _LOG_SQRT_2PI


def _ll_logcdf_np(log_e, alpha0):
    """log-Laplace(alpha0) log-CDF as a function of log(eps)."""
    below = np.log(0.5) + alpha0 * log_e                       # eps < 1
    # clamp the (discarded) eps>=1 branch at log_e>=0 so exp() can't overflow to
    # NaN at deep-tail eps<<1 cells -- np.where evaluates both branches.
    above = np.log1p(-0.5 * np.exp(-alpha0 * np.maximum(log_e, 0.0)))  # eps >= 1
    return np.where(log_e < 0.0, below, above)


def _ll_logpdf_np(log_e, alpha0):
    """log-Laplace(alpha0) log-PDF as a function of log(eps)."""
    c = np.log(alpha0) - np.log(2.0)
    below = c + (alpha0 - 1.0) * log_e                         # eps < 1
    above = c - (alpha0 + 1.0) * log_e                         # eps >= 1
    return np.where(log_e < 0.0, below, above)


def zblock_loglik_np(arrays, Z):
    """Z-block conditional data log-likelihood (function of the correlated field
    ``Z``), up to a Z-independent constant. ``arrays`` is ``ModelState.export_zblock()``."""
    logX, logR = arrays["logX"], arrays["logR"]
    phi = arrays["phi"][:, None]                                # (Ns,1) over t
    alpha0 = float(arrays["alpha0"])
    cens, exc = arrays["cens"], arrays["exc"]
    log_g = -_norm_logsf_np(Z)                                # log g(Z) = -log Phibar(Z)
    log_Xstar = phi * logR + log_g
    log_e = logX - log_Xstar                                   # log(X / X*)
    ll_cens = np.where(cens, _ll_logcdf_np(log_e, alpha0), 0.0)
    ll_exc = np.where(exc, -log_Xstar + _ll_logpdf_np(log_e, alpha0), 0.0)
    return float(ll_cens.sum() + ll_exc.sum())


def zblock_logjoint_np(arrays, eta):
    """Full Z-block log-joint in the white non-centered coords ``eta`` (Z = L eta):
    the data loglik plus the standard-normal prior ``-½‖eta‖²``. Target for NUTS."""
    Z = arrays["L"] @ eta
    return zblock_loglik_np(arrays, Z) - 0.5 * float(np.sum(eta * eta))


@_njit(cache=True)
def _zblock_value_grad_kernel(Z, logsf, logX, logR, phi, alpha0, cens, exc):
    """Fused elementwise core of ``zblock_value_grad_np`` (returns ``loglik`` and the
    per-cell ``grad_Z``). A scalar-loop @njit kernel replaces ~45 tiny-array numpy
    calls (each ~1us of dispatch, ~68%% of the un-JITed cost; profiled at Ns=203) and
    sidesteps ``np.where`` evaluating both branches. The two BLAS matvecs (``L @ eta``,
    ``Lᵀ gZ``) and the tail-stable ``log_ndtr`` survival fn stay in numpy (callers).
    Same float64 formulas as the original, so the faithfulness test still gates it."""
    Ns, Nt = Z.shape
    gZ = np.empty_like(Z)
    loglik = 0.0
    log_half = np.log(0.5)
    c = np.log(alpha0) - np.log(2.0)
    am1 = alpha0 - 1.0
    ap1 = alpha0 + 1.0
    for i in range(Ns):
        pi = phi[i]
        for j in range(Nt):
            z = Z[i, j]
            lsf = logsf[i, j]
            log_Xstar = pi * logR[i, j] - lsf          # = phi logR + log g
            log_e = logX[i, j] - log_Xstar
            h = np.exp(-0.5 * z * z - _LOG_SQRT_2PI - lsf)   # hazard = d log g / dZ
            ce, ex = cens[i, j], exc[i, j]
            if log_e < 0.0:                             # eps < 1 branch
                if ce:
                    loglik += log_half + alpha0 * log_e
                if ex:
                    loglik += -log_Xstar + c + am1 * log_e
                dcdf, dpdf = alpha0, am1
            else:                                       # eps >= 1 branch
                a = np.exp(-alpha0 * log_e)
                if ce:
                    loglik += np.log1p(-0.5 * a)
                if ex:
                    loglik += -log_Xstar + c - ap1 * log_e
                dcdf, dpdf = 0.5 * alpha0 * a / (1.0 - 0.5 * a), -ap1
            g = 0.0
            if ce:
                g += -h * dcdf
            if ex:
                g += -h * (1.0 + dpdf)
            gZ[i, j] = g
    return loglik, gZ


def zblock_value_grad_trmv(arrays, eta):
    """`zblock_value_grad_np` with the two matvecs done as TRIANGULAR BLAS calls.

    The Z leapfrog is memory-bound on the Cholesky factor: `L @ eta` and `L.T @ gZ`
    together stream 2 * Ns^2 * 8 bytes per call (2.18 MB at Ns=369), and at 49.8 of
    the 70 us per call they are 69% of the gradient -- the numba kernel is 8 us and
    log_ndtr 6 us. But L is TRIANGULAR, so half of what those matmuls read is
    structural zeros that numpy has no way to know about. dtrmv reads one triangle.

    Measured at Ns=369, interleaved arms in one process (median of 7 rounds, ratio
    range 1.36-1.37): vg 70.0 -> 51.2 us, 1.37x; a whole NUTS transition 1.22x, the
    difference being the ~29% of transition wall the driver spends outside vg.
    Agreement with the numpy path is 2.3e-16 on logp and 4.5e-16 on the gradient --
    same arithmetic, different summation order.

    `arrays["Lf"]` must be the factor in FORTRAN order (else scipy copies it on every
    call, which costs far more than the win) and `arrays["lower"]` must say which
    triangle it is: chol(Sigma) is LOWER for Shi/HW, but the VisGP path builds
    M = R^{-T}, which is UPPER. ModelState carries that as `_zfac_lower`; getting it
    wrong silently returns a wrong gradient rather than failing."""
    lo = 1 if arrays["lower"] else 0
    e = eta[:, 0] if eta.ndim == 2 else eta
    Z = _dtrmv(arrays["Lf"], e, lower=lo)[:, None]
    logsf = _norm_logsf_np(Z)
    loglik, gZ = _zblock_value_grad_kernel(
        Z, logsf, arrays["logX"], arrays["logR"], arrays["phi"],
        float(arrays["alpha0"]), arrays["cens"], arrays["exc"])
    logp = float(loglik) - 0.5 * float(e @ e)
    grad = _dtrmv(arrays["Lf"], np.ascontiguousarray(gZ[:, 0]), lower=lo, trans=1) - e
    return logp, grad[:, None]


def zblock_value_grad_np(arrays, eta):
    """Value AND analytic gradient of ``zblock_logjoint_np`` at ``eta`` in one pass.

    The Z conditional touches no RW-transform derivative, so the gradient is
    elementary. With ``g(z)=1/Phibar(z)``, the hazard ``h(z)=phi(z)/Phibar(z)`` is
    ``d log g / dZ``; since ``log X* = phi logR + log g`` and ``log_e = logX - log X*``,
    ``d log_e/dZ = d(-log X*)/dZ = -h``. Per cell:
      censored:    dloglik/dZ = -h * (log-CDF)'(log_e)
      exceedance:  dloglik/dZ = -h * (1 + (log-PDF)'(log_e))
    then ``grad_eta = Lᵀ grad_Z - eta`` (the white-prior term). Returns ``(logp, grad)``.

    The elementwise work is a fused @njit kernel (see ``_zblock_value_grad_kernel``);
    only the two BLAS matvecs and ``log_ndtr`` stay in numpy here.
    """
    L = arrays["L"]
    Z = np.ascontiguousarray(L @ eta)
    logsf = _norm_logsf_np(Z)                          # tail-stable, stays in numpy
    loglik, gZ = _zblock_value_grad_kernel(
        Z, logsf, np.ascontiguousarray(arrays["logX"]),
        np.ascontiguousarray(arrays["logR"]), arrays["phi"], float(arrays["alpha0"]),
        np.ascontiguousarray(arrays["cens"]), np.ascontiguousarray(arrays["exc"]))
    logp = float(loglik) - 0.5 * float(np.sum(eta * eta))
    grad = L.T @ gZ - eta
    return logp, grad


# ===========================================================================
# Clean numpy NUTS driver (Stan-style: multinomial sampling + generalised U-turn)
# ===========================================================================
# Drives a value_and_grad callable ``vg(theta) -> (logp, grad_logp)`` on a unit
# (identity-mass) phase space. theta/grad are arrays of any shape; momentum is
# i.i.d. standard normal of the same shape. Used by update_Z_nuts on the white eta
# coords, where the Z-block gradient (zblock_value_grad_np) is exact and elementary.
#
# Originally the slice-sampler "efficient NUTS" (Hoffman & Gelman 2014). Now the
# Stan/AdvancedHMC design (Betancourt 2017): (1) MULTINOMIAL sampling -- weight every
# trajectory state by exp(H0-H) via log-sum-exp and select with biased progressive
# sampling, instead of a slice indicator -- which extracts more effective movement
# per leapfrog; (2) the GENERALISED U-turn -- momentum-sum ``rho`` plus the two inner
# sub-span checks (stan base_nuts.hpp compute_criterion) -- which terminates sooner.
# Both cut the LEAPFROG COUNT, the dominant cost (benchmarked:
# numerical work is ~33us/leapfrog, so depth/leapfrogs and Python overhead dominate).
# Unit metric => Stan's p_sharp == the momentum, so the criterion simplifies to
# ``rho·p > 0`` at the relevant endpoints. Sign convention here: ``joint = logp -
# ½‖r‖²`` so the log state weight is ``joint - joint0`` (= Stan's H0 - H).
_NUTS_MAX_DELTA = 1000.0          # divergence threshold on the Hamiltonian


# Optional DIAGONAL metric. ``inv_mass`` is the diagonal of the inverse mass matrix
# M^{-1} (== the estimated posterior variance; Stan's "inverse metric"). None => the
# unit metric (byte-identical to the original code). With a metric: momentum p~N(0,M),
# velocity M^{-1}p = inv_mass*p, kinetic ½ p^T M^{-1} p = ½ Σ inv_mass p², and the
# U-turn uses p_sharp = M^{-1}p = inv_mass*p.
def _nuts_kinetic(r, inv_mass):
    rr = r.ravel()
    if inv_mass is None:
        return 0.5 * float(rr @ rr)
    return 0.5 * float((inv_mass.ravel() * rr) @ rr)


def _nuts_draw_r(shape, rng, inv_mass):
    z = rng.standard_normal(shape)
    return z if inv_mass is None else z / np.sqrt(inv_mass)      # Var = 1/inv_mass = M


def _nuts_leapfrog(theta, r, grad, eps, vg, inv_mass=None):
    r = r + 0.5 * eps * grad
    theta = theta + eps * (r if inv_mass is None else inv_mass * r)   # + eps M^{-1} p
    logp, grad = vg(theta)
    r = r + 0.5 * eps * grad
    return theta, r, logp, grad


def _nuts_crit(rho, p_a, p_b, inv_mass=None):
    """Generalised no-U-turn criterion (Betancourt 2017; Stan ``compute_criterion``).
    A span is still expanding iff the summed momentum ``rho`` projects positively onto
    the p_sharp = M^{-1} p at its ends (unit metric => p_sharp == momentum)."""
    rr = rho.ravel()
    if inv_mass is None:
        return (rr @ p_a.ravel() > 0.0) and (rr @ p_b.ravel() > 0.0)
    im = inv_mass.ravel()
    return (rr @ (im * p_a.ravel()) > 0.0) and (rr @ (im * p_b.ravel()) > 0.0)


def _nuts_build_tree(theta, r, grad, v, j, eps, vg, joint0, rng, inv_mass=None):
    """Recursively build one balanced subtree of height ``j`` in direction ``v``,
    with MULTINOMIAL sampling (log-sum-exp state weights ``lsw``; no slice variable)
    and the GENERALISED U-turn test. Returns a dict with the two integration-order
    endpoints (b=beg/start side, e=end/frontier side), the momentum sum ``rho``, the
    sampled representative (ts/ls/gs), ``lsw``, validity ``s``, and the leapfrog
    counters ``sm`` (summed Metropolis prob) and ``n``. On an invalid subtree (s=0)
    only ``sm``/``n`` are meaningful -- they are still summed so the dual-averaging
    accept stat stays unbiased."""
    if j == 0:
        th, rr, lp, gg = _nuts_leapfrog(theta, r, grad, v * eps, vg, inv_mass)
        joint = lp - _nuts_kinetic(rr, inv_mass)
        logw = joint - joint0                              # log state weight (H0-H)
        if not np.isfinite(logw):
            logw = -np.inf
        valid = logw > -_NUTS_MAX_DELTA                    # else divergent -> stop
        sm = 1.0 if logw > 0.0 else float(np.exp(logw))    # exp(-inf)=0, safe
        return dict(tb=th, rb=rr, gb=gg, te=th, re=rr, ge=gg, rho=rr.copy(),
                    ts=th, ls=lp, gs=gg, lsw=logw, s=int(valid), sm=sm, n=1)
    # inner (start-side) subtree, then the outer (frontier-side) half from its end
    A = _nuts_build_tree(theta, r, grad, v, j - 1, eps, vg, joint0, rng, inv_mass)
    if A["s"] == 0:
        return A
    B = _nuts_build_tree(A["te"], A["re"], A["ge"], v, j - 1, eps, vg, joint0, rng, inv_mass)
    if B["s"] == 0:
        return dict(s=0, sm=A["sm"] + B["sm"], n=A["n"] + B["n"])
    # multinomial: keep B's representative with prob w_B / (w_A + w_B)
    lsw = np.logaddexp(A["lsw"], B["lsw"])
    if rng.random() < np.exp(B["lsw"] - lsw):
        ts, ls, gs = B["ts"], B["ls"], B["gs"]
    else:
        ts, ls, gs = A["ts"], A["ls"], A["gs"]
    rho = A["rho"] + B["rho"]
    # generalised U-turn: full span + the two inner sub-spans (stan build_tree)
    s = (_nuts_crit(rho, A["rb"], B["re"], inv_mass)
         and _nuts_crit(A["rho"] + B["rb"], A["rb"], B["rb"], inv_mass)
         and _nuts_crit(B["rho"] + A["re"], A["re"], B["re"], inv_mass))
    return dict(tb=A["tb"], rb=A["rb"], gb=A["gb"], te=B["te"], re=B["re"],
                ge=B["ge"], rho=rho, ts=ts, ls=ls, gs=gs, lsw=lsw, s=int(s),
                sm=A["sm"] + B["sm"], n=A["n"] + B["n"])


def nuts_transition(vg, theta0, eps, rng, inv_mass=None, max_depth=10):
    """One NUTS transition (multinomial sampling + generalised U-turn, à la Stan)
    from ``theta0`` at step size ``eps``. ``inv_mass`` is an optional diagonal metric
    (inverse mass = estimated posterior variance); None => unit metric.

    Returns ``(theta_new, accept_stat, depth)`` where ``accept_stat`` is the mean
    Metropolis probability over the trajectory (the dual-averaging target)."""
    logp0, grad0 = vg(theta0)
    r0 = _nuts_draw_r(theta0.shape, rng, inv_mass)
    joint0 = logp0 - _nuts_kinetic(r0, inv_mass)

    t_minus = t_plus = theta0
    r_minus = r_plus = r0
    g_minus = g_plus = grad0
    # four corner momenta (fwd/bck frontier + the two inner boundaries); unit metric
    # => p_sharp == momentum, all initialised to r0 (single-point trajectory).
    p_ff = p_fb = p_bf = p_bb = r0
    rho = r0.copy()

    theta = theta0
    lsw = 0.0                                    # log weight of the initial state
    sum_metro = 0.0
    n_lf = 0
    depth = 0
    while depth < max_depth:
        v = 1 if rng.random() < 0.5 else -1
        if v == 1:
            rho_bck = rho
            p_bf = p_ff                          # old fwd frontier -> bck_fwd corner
            T = _nuts_build_tree(t_plus, r_plus, g_plus, 1, depth, eps, vg, joint0, rng, inv_mass)
            sum_metro += T["sm"]; n_lf += T["n"]
            if T["s"] == 0:
                break
            t_plus, r_plus, g_plus = T["te"], T["re"], T["ge"]
            rho_fwd = T["rho"]; p_fb = T["rb"]; p_ff = T["re"]
        else:
            rho_fwd = rho
            p_fb = p_bb                          # old bck frontier -> fwd_bck corner
            T = _nuts_build_tree(t_minus, r_minus, g_minus, -1, depth, eps, vg, joint0, rng, inv_mass)
            sum_metro += T["sm"]; n_lf += T["n"]
            if T["s"] == 0:
                break
            t_minus, r_minus, g_minus = T["te"], T["re"], T["ge"]
            rho_bck = T["rho"]; p_bf = T["rb"]; p_bb = T["re"]
        depth += 1
        # biased progressive multinomial: prefer the newer (further) subtree
        if T["lsw"] > lsw or rng.random() < np.exp(T["lsw"] - lsw):
            theta = T["ts"]
        lsw = np.logaddexp(lsw, T["lsw"])
        rho = rho_bck + rho_fwd
        # generalised termination over the whole merged trajectory
        if not (_nuts_crit(rho, p_bb, p_ff, inv_mass)
                and _nuts_crit(rho_bck + p_fb, p_bb, p_fb, inv_mass)
                and _nuts_crit(rho_fwd + p_bf, p_bf, p_ff, inv_mass)):
            break
    return theta, (sum_metro / n_lf if n_lf else 0.0), depth


_EPS_SEARCH_MAX = 100     # doublings/halvings; 2^100 spans anything reachable
_EPS_FLOOR = 1e-12        # a step this small is useless but must not be ZERO
_EPS_CEIL = 1e6


def nuts_find_reasonable_eps(vg, theta, rng, inv_mass=None):
    """Heuristic initial step size (Hoffman & Gelman Alg. 4): double/halve eps until
    the single-leapfrog acceptance crosses 0.5. Metric-aware via ``inv_mass``."""
    eps = 1.0
    logp, grad = vg(theta)
    r = _nuts_draw_r(theta.shape, rng, inv_mass)
    joint0 = logp - _nuts_kinetic(r, inv_mass)
    _, r1, logp1, _ = _nuts_leapfrog(theta, r, grad, eps, vg, inv_mass)
    joint1 = logp1 - _nuts_kinetic(r1, inv_mass)
    a = 1 if (joint1 - joint0) > np.log(0.5) else -1
    # Bounded, and never returns 0. Without the bound this loop can halve until eps
    # underflows: a target whose proposal keeps landing outside the support returns
    # -inf, which flips `a` to -1 and then satisfies the loop condition forever until
    # the float bottoms out. eps=0 does not fail loudly -- it makes every leapfrog a
    # no-op, so no U-turn is ever detected and every trajectory runs to max_depth
    # (1024 evaluations that move nothing). Found on the S block, where most knot
    # directions carry no likelihood information and are pure heavy-tailed Levy
    # prior, but nothing about the trap is specific to that block.
    for _ in range(_EPS_SEARCH_MAX):
        if not (a * (joint1 - joint0) > a * np.log(0.5)):
            break
        eps *= 2.0 ** a
        if eps < _EPS_FLOOR or eps > _EPS_CEIL:
            break
        _, r1, logp1, _ = _nuts_leapfrog(theta, r, grad, eps, vg, inv_mass)
        joint1 = logp1 - _nuts_kinetic(r1, inv_mass)
        if not np.isfinite(joint1):
            a = -1                                    # overshot into divergence
    return float(np.clip(eps, _EPS_FLOOR, _EPS_CEIL))


class NUTSStepAdapter:
    """Nesterov dual-averaging of the NUTS step size toward ``target_accept``
    (Hoffman & Gelman Alg. 5). Call ``update(accept_stat)`` each warmup transition;
    ``finalize()`` freezes ``eps`` at the averaged value for the sampling phase."""

    def __init__(self, eps0, target_accept=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10.0 * eps0)
        self.eps = eps0
        self.log_eps_bar = 0.0
        self.h_bar = 0.0
        self.target = target_accept
        self.gamma, self.t0, self.kappa = gamma, t0, kappa
        self.m = 0

    def update(self, accept_stat):
        self.m += 1
        m = self.m
        self.h_bar = ((1.0 - 1.0 / (m + self.t0)) * self.h_bar
                      + (1.0 / (m + self.t0)) * (self.target - accept_stat))
        log_eps = self.mu - np.sqrt(m) / self.gamma * self.h_bar
        w = m ** (-self.kappa)
        self.log_eps_bar = w * log_eps + (1.0 - w) * self.log_eps_bar
        self.eps = float(np.exp(log_eps))

    def finalize(self):
        if self.m > 0:
            self.eps = float(np.exp(self.log_eps_bar))
        return self.eps


class NUTSWarmup:
    """Step-size dual-averaging + optional DIAGONAL-METRIC adaptation for a NUTS
    block (Stan-style windowed schedule). Call ``update(theta, accept_stat, vg, rng)``
    once per transition; ``eps`` and ``inv_mass`` feed the next ``nuts_transition``.

    The metric (``inv_mass`` = regularised posterior variance, Stan's inverse metric)
    is estimated by Welford over a middle window of warmup, set ONCE, after which the
    step size is re-found and dual-averaging restarts under the new geometry -- so a
    correlated/ill-scaled target (e.g. the beta coefficients: intercept vs spatial
    columns) is preconditioned to ~unit scale, cutting tree depth. Metric-off
    (adapt_mass=False or a short warmup) reduces to plain step-size adaptation on the
    unit metric. Deterministic given (vg, rng) -> identical on every MPI rank."""

    def __init__(self, vg, theta0, rng, warmup, target_accept=0.8, adapt_mass=True,
                 init_inv_mass=None):
        self.warmup = int(warmup)
        # A pre-seeded diagonal metric (e.g. from a warm trace's posterior variance)
        # is used FIXED -- no warmup estimation (the right choice for a slow-mixing
        # block that can't self-estimate its scales); step size still dual-averages.
        if init_inv_mass is not None:
            self.inv_mass = np.asarray(init_inv_mass, dtype=float)
            adapt_mass = False
        else:
            self.inv_mass = None
        self.step = NUTSStepAdapter(
            nuts_find_reasonable_eps(vg, theta0, rng, self.inv_mass),
            target_accept=target_accept)
        self._m = 0
        # metric window [w_start, w_set): skip an initial transient, leave a tail to
        # re-tune eps. Needs a non-trivial warmup to be worth it.
        self._adapt_mass = bool(adapt_mass) and self.warmup >= 20
        self._w_start = max(1, int(0.15 * self.warmup))
        self._w_set = max(self._w_start + 1, int(0.75 * self.warmup))
        self._n = 0
        self._mean = np.zeros_like(theta0, dtype=float)
        self._m2 = np.zeros_like(theta0, dtype=float)

    @property
    def eps(self):
        return self.step.eps

    def update(self, theta, accept_stat, vg, rng):
        self._m += 1
        m = self._m
        if m > self.warmup:
            return
        self.step.update(float(accept_stat))                 # dual-average eps
        if self._adapt_mass and self.inv_mass is None and m >= self._w_start:
            self._n += 1                                      # Welford variance
            d = theta - self._mean
            self._mean += d / self._n
            self._m2 += d * (theta - self._mean)
        if (self._adapt_mass and self.inv_mass is None
                and m >= self._w_set and self._n > 5):
            var = self._m2 / (self._n - 1)
            n = self._n                                       # Stan regularisation
            self.inv_mass = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            self.step = NUTSStepAdapter(
                nuts_find_reasonable_eps(vg, theta, rng, self.inv_mass),
                target_accept=self.step.target)
        if m == self.warmup:
            self.step.finalize()


def barker_transition(vg, theta0, eps, precond_sd, rng, logp0=None, grad0=None):
    """One Barker-proposal MALA transition (Livingstone & Zanella 2022) from
    ``theta0`` at scalar step ``eps`` with a diagonal preconditioner ``precond_sd``.

    Coordinate-wise: draw a symmetric step ``xi ~ N(0, (eps*precond_sd)^2)`` and keep
    its sign with probability ``sigmoid(xi * grad_i)`` (biasing the move uphill); the
    induced skewness is corrected EXACTLY in the Metropolis ratio via the softplus
    terms below, so the move is reversible w.r.t. the target. Robust to step-size
    misspecification (accept degrades gracefully), which is the whole point vs MALA.

    One fresh gradient eval at the proposal; the current-point value+grad may be
    passed in (``logp0``/``grad0``) to save a re-eval. Returns
    ``(theta_new, logp_new, grad_new, accept_prob)`` -- value+grad are AT ``theta_new``
    (so an accept lets the caller reuse them), and ``accept_prob = min(1, e^logratio)``
    is the dual-averaging target. Deterministic given ``rng`` (identical on every MPI
    rank when ``rng`` is the shared grng)."""
    if logp0 is None:
        logp0, grad0 = vg(theta0)
    sd = eps * precond_sd
    xi = rng.standard_normal(theta0.shape) * sd
    keep = rng.random(theta0.shape) < _expit(xi * grad0)   # keep sign w.p. sigmoid(xi*g)
    b = np.where(keep, xi, -xi)                            # signed increment c = y - x
    theta1 = theta0 + b
    logp1, grad1 = vg(theta1)
    if not np.isfinite(logp1):                            # outside support -> reject
        return theta0, logp0, grad0, 0.0
    # log q(x|y)/q(y|x) = sum_i [ softplus(-c_i g_x,i) - softplus(c_i g_y,i) ], where
    # softplus(z)=log(1+e^z)=logaddexp(0,z) is the Barker sign-flip normaliser.
    log_ratio = (logp1 - logp0
                 + np.logaddexp(0.0, -b * grad0).sum()
                 - np.logaddexp(0.0, b * grad1).sum())
    a = float(np.exp(min(0.0, log_ratio)))
    if np.log(rng.random()) < log_ratio:
        return theta1, logp1, grad1, a
    return theta0, logp0, grad0, a


# high-dimensional Barker optimal acceptance (Livingstone & Zanella 2022; the Vogrinc
# 2023 optimal-noise analysis lands at the same ~0.574 as MALA).
_BARKER_TARGET = 0.574


def barker_find_reasonable_eps(vg, theta, precond_sd, rng, max_eps=np.inf):
    """Initial Barker step size: double/halve eps until the transition's acceptance
    probability brackets the 0.574 optimum (cheap -- one proposal eval per trial).

    ``max_eps`` caps the doubling. Some targets are not merely improbable far from the
    mode but RUINOUSLY EXPENSIVE to evaluate there -- the phi block's qRW root-find
    crawls once phi leaves its plausible range -- so an unbounded search can walk the
    proposal into a region where a single trial takes minutes. Capping eps changes only
    the proposal scale, which is a free tuning constant: the target is untouched and
    the kernel stays reversible."""
    logp0, grad0 = vg(theta)
    eps = min(1.0, max_eps)
    _, _, _, a = barker_transition(vg, theta, eps, precond_sd, rng, logp0, grad0)
    d = 1 if a > _BARKER_TARGET else -1        # a>target -> steps too small -> grow eps
    for _ in range(50):
        eps = min(eps * 2.0 ** d, max_eps)
        _, _, _, a = barker_transition(vg, theta, eps, precond_sd, rng, logp0, grad0)
        if (d * (a - _BARKER_TARGET) <= 0 or not np.isfinite(eps) or eps == 0.0
                or (d > 0 and eps >= max_eps)):
            break
    return eps


class BarkerWarmup:
    """Step-size dual-averaging + optional DIAGONAL-PRECONDITIONER adaptation for a
    Barker-proposal MALA block. Mirrors ``NUTSWarmup`` but for a single Barker move:
    dual-average ``eps`` toward the 0.574 Barker optimum, and estimate a per-coordinate
    posterior sd (the preconditioner) by Welford over a middle warmup window, set ONCE.
    Barker's skew-adaptive sign flip already handles step-size misspecification, so a
    scalar step + diagonal preconditioner is enough for the correlated sigma/xi scales
    (no dense proposal needed). A fixed ``init_precond_sd`` (e.g. a warm trace's per-coef
    sd) skips estimation. Deterministic given (vg, rng) -> identical on every MPI rank."""

    def __init__(self, vg, theta0, rng, warmup, target_accept=_BARKER_TARGET,
                 adapt_precond=True, init_precond_sd=None, max_eps=np.inf):
        self.warmup = int(warmup)
        self.max_eps = float(max_eps)
        if init_precond_sd is not None:                  # fixed preconditioner, no est.
            self.precond_sd = np.asarray(init_precond_sd, dtype=float)
            adapt_precond = False
        else:
            self.precond_sd = np.ones_like(theta0, dtype=float)
        self.step = NUTSStepAdapter(
            barker_find_reasonable_eps(vg, theta0, self.precond_sd, rng,
                                       max_eps=self.max_eps),
            target_accept=target_accept)
        self._m = 0
        self._adapt = bool(adapt_precond) and self.warmup >= 20
        self._set = False
        self._w_start = max(1, int(0.15 * self.warmup))
        self._w_set = max(self._w_start + 1, int(0.75 * self.warmup))
        self._n = 0
        self._mean = np.zeros_like(theta0, dtype=float)
        self._m2 = np.zeros_like(theta0, dtype=float)

    @property
    def eps(self):
        # dual averaging may keep pushing past the cap when the cap binds (accept then
        # sits above target); clamp on READ so the used step never exceeds it
        return min(self.step.eps, self.max_eps)

    def update(self, theta, accept_stat, vg, rng):
        self._m += 1
        m = self._m
        if m > self.warmup:
            return
        self.step.update(float(accept_stat))             # dual-average eps
        if self._adapt and not self._set and m >= self._w_start:
            self._n += 1                                  # Welford per-coord variance
            d = theta - self._mean
            self._mean += d / self._n
            self._m2 += d * (theta - self._mean)
        if self._adapt and not self._set and m >= self._w_set and self._n > 5:
            var = self._m2 / (self._n - 1)
            n = self._n                                   # Stan-style regularisation
            var = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            self.precond_sd = np.sqrt(var)
            self._set = True                              # re-find eps under new scale
            self.step = NUTSStepAdapter(
                barker_find_reasonable_eps(vg, theta, self.precond_sd, rng,
                                           max_eps=self.max_eps),
                target_accept=self.step.target)
        if m == self.warmup:
            self.step.finalize()


# ===========================================================================
# Parallel context + per-time caches shared by both backends
# ===========================================================================
class Ctx:
    """Captures the serial-vs-MPI differences seen by the update blocks.

    Parameters
    ----------
    times : the time replicates this worker updates (range(Nt) serial;
        the round-robin local slice for MPI), kept as an ordered list.
    grng : RNG driving the *global*-parameter proposals/accepts (identical on
        every MPI rank so global moves stay in lock-step).
    lrng : RNG for the *local* latent fields S/Z (per-rank under MPI). Defaults
        to grng for serial.
    reduce : None for serial (the per-time values are summed locally with
        ``np.sum``, matching the original serial arithmetic exactly); a
        ``float -> float`` Allreduce-sum for MPI.
    reduce_arr : None for serial (identity); an ``ndarray -> ndarray``
        Allreduce-sum for MPI, used to combine per-time *matrix/vector* partial
        sums across ranks (e.g. the sum_t b_t b_t^T in the CP nu/rho gradient).
    supports_compact : whether the serial-only compact-support phi fast path may
        be used (False under MPI, which always recomputes per time).
    covers_all_times : True when ``times`` spans every replicate (serial), so
        whole-matrix derived state (e.g. _pY_exc) can be refreshed.
    """

    def __init__(self, times, grng, lrng=None, *, reduce=None, reduce_arr=None,
                 supports_compact=False, covers_all_times=True):
        self.times = list(times)
        self.grng = grng
        self.lrng = lrng if lrng is not None else grng
        self._reduce = reduce
        self._reduce_arr = reduce_arr
        self.supports_compact = supports_compact
        self.covers_all_times = covers_all_times

    @property
    def reduces_arrays(self) -> bool:
        """Whether areduce actually combines across ranks. A block whose target is a
        SUM OVER ALL TIMES (phi, nu/rho) is silently wrong without it -- each rank
        would sample a partial-times posterior and the ranks would drift apart with no
        error raised -- so those blocks assert on this rather than trusting the ctx."""
        return self._reduce_arr is not None

    def areduce(self, arr):
        """Allreduce-sum a per-rank partial-sum array to the global total (serial:
        identity, so the arithmetic is bit-identical to the single-process code)."""
        if self._reduce_arr is None:
            return arr
        return self._reduce_arr(arr)

    def total(self, vals_by_t) -> float:
        """Reduce a dict {t: value} to the global scalar total.

        Serial uses ``np.sum`` over the full ordered list (pairwise summation,
        bit-identical to the pre-merge code); MPI sums its local slice then
        Allreduces."""
        if self._reduce is None:
            return float(np.sum(np.array([vals_by_t[t] for t in self.times])))
        return self._reduce(float(sum(vals_by_t[t] for t in self.times)))

    def total_ll(self, Yll, Zlp) -> float:
        """Total log-likelihood for the trace (Y + Z), per-backend arithmetic."""
        if self._reduce is None:
            return float(np.sum(np.array([Yll[t] for t in self.times]))
                         + np.sum(np.array([Zlp[t] for t in self.times])))
        return self._reduce(float(sum(Yll[t] + Zlp[t] for t in self.times)))


class Caches:
    """Per-time MCMC caches, keyed by time index in both backends."""
    __slots__ = ("Yll", "Zlp", "qd")

    def __init__(self, Yll, Zlp, qd):
        self.Yll = Yll   # {t: float}            current Y log-lik per time
        self.Zlp = Zlp   # {t: float}            current Z log-prior per time
        self.qd = qd     # {t: (qv_col, dv_col)} current qRW/dRW per time


def init_caches(state, times) -> Caches:
    """Build the per-time caches from the current state (used at startup and
    after a checkpoint restore). Uses the per-time qRW/dRW recompute, which is
    bit-identical to the batched serial path."""
    qvc = state.qd_RW_cens()
    qd = {t: state.qd_RW_one(t, qvc) for t in times}
    Yll = {t: state.loglik_t(t, qv=qd[t][0], dv=qd[t][1]) for t in times}
    Zlp = {t: state.Z_logprior(state.Z[:, t]) for t in times}
    return Caches(Yll, Zlp, qd)


def _use_joint_nu_rho(state, st) -> bool:
    """The joint (nu, rho) CP block applies only when both are sampled and rho is
    scalar (k_rho == 1, i.e. HW / Wang); Shi's vector rho keeps the per-knot blocks."""
    return (getattr(st, "joint_nu_rho", False) and st.update_nu and st.update_rho
            and state.k_rho == 1)


def _nu_rho_nuts_due(state, st) -> bool:
    """Whether the joint (nu, rho) NUTS block fires on THIS sweep.

    With ``nu_rho_nuts_every == k`` it fires on sweeps 0, k, 2k, ... and nu/rho are
    held on the rest (no RWMH fallback -- see SamplerSettings.nu_rho_nuts_every).
    ``state._nu_rho_sweep`` counts sweeps rather than calls so that the phase, not
    just the tally, survives a checkpoint resume; it is left absent at k == 1 to keep
    the default path and its snapshots exactly as they were."""
    k = int(getattr(st, "nu_rho_nuts_every", 1) or 1)
    if k <= 1:
        return True
    i = int(getattr(state, "_nu_rho_sweep", 0))
    state._nu_rho_sweep = i + 1
    return i % k == 0


def _use_interweave_nu_rho(state, st) -> bool:
    """ASIS interweaving of (nu, rho): the CP half (update_nu_rho_joint) + the NCP
    half (update_nu_rho_ncp) each sweep. Scalar rho only (k_rho == 1). Implies the
    joint CP block as its CP half; takes precedence over plain ``joint_nu_rho``."""
    return (getattr(st, "interweave_nu_rho", False) and st.update_nu and st.update_rho
            and state.k_rho == 1)


def _use_ncp_vec(state, st) -> bool:
    """ASIS NCP interweave half for VECTOR rho (Shi, k_rho > 1) -- the generalisation
    of the scalar interweave. The component CP update_nu/update_rho still run; this
    adds the NCP half. Opt-in via ncp_interweave_vec in {"joint","nu"}."""
    return (getattr(st, "ncp_interweave_vec", "off") in ("joint", "nu")
            and st.update_nu and st.update_rho and state.k_rho > 1)


def _use_joint_geometry(state, st) -> bool:
    """Drive the Wendland-geometry parameters with one joint adaptive-RWMH block.
    Needs joint_geometry set and radius_S sampled; never for HW (no radius_S)."""
    return (getattr(st, "joint_geometry", False) and st.update_radius_S
            and state.cfg.model_type != MODEL_HW)


def _geom_dim(state, st) -> int:
    """Dimension of the joint-geometry block: 3 (radius, ratio, angle) when the
    anisotropy is live, else 1 (radius only)."""
    return 3 if state.update_aniso else 1


# Per-component initial proposal variances for the joint-geometry block, in the
# UNCONSTRAINED coordinates it samples: (log-slack, logit-ratio, angle). These are
# roughly commensurate (all O(0.01-0.1) given the exp1 posteriors), so a small
# isotropic-ish seed suffices -- the Robbins-Monro adaptation learns the true
# covariance (and the radius<->ratio ridge) quickly from a warm start.
_GEOM_INIT_VAR = np.array([0.04, 0.01, 0.01])


def _geom_adaptive(state, st) -> "Adaptive":
    """Joint-geometry RWMH block (dim 1 or 3) operating in coverage-respecting
    coordinates (log-slack, logit-ratio, angle); see update_geometry_joint. The
    coverage constraint and the (0,1) ratio box are absorbed by the transform, so
    the adapted Gaussian proposal no longer fights a hard boundary. sigma_m_sq
    starts at the Andrieu-Thoms optimum 2.38^2/d; off-diagonals (the ridge) are
    learned by the empirical-covariance adaptation."""
    d = _geom_dim(state, st)
    s = (2.38 ** 2) / d
    Sigma_0 = np.diag(_GEOM_INIT_VAR[:d] / s)
    return Adaptive(d, sigma_m_sq=s, Sigma_0=Sigma_0, r_opt=_target_accept(d))


def _ncp_vec_param_blocks(state, st):
    """Parameter-index blocks for the vector NCP move. Index 0 == nu, 1..k_rho ==
    rho knots. "nu" -> just [nu]; "joint" -> [nu, rho_k...] split into contiguous
    blocks of <= phi_block_max dims (a 14-D Shi proposal is one block; larger knot
    sets split, since adapting a much higher-D covariance is unreliable)."""
    if getattr(st, "ncp_interweave_vec", "off") == "nu":
        return [np.array([0])]
    return _phi_blocks(1 + state.k_rho, st.phi_block_max)


def _ncp_vec_adaptives(state, st):
    """Adapters for the vector NCP blocks, seeded from nu_rho_proposal_cov (the warm
    [nu, rho_k] covariance) when supplied -- like joint phi, an isotropic start
    struggles on the correlated nu/rho ridge. Falls back to a 1/d isotropic init."""
    cov = getattr(st, "nu_rho_proposal_cov", None)
    cov = np.asarray(cov) if cov is not None else None
    d_full = 1 + state.k_rho
    have = cov is not None and cov.shape == (d_full, d_full)
    out = []
    for idx in _ncp_vec_param_blocks(state, st):
        d = len(idx)
        if have:
            out.append(Adaptive(d, sigma_m_sq=(2.38 ** 2) / d,
                                Sigma_0=np.atleast_2d(cov[np.ix_(idx, idx)]),
                                r_opt=_target_accept(d)))
        else:
            out.append(_mk(d, 0.01 / d))
    return out


def _target_accept(dim: int) -> float:
    """Dimension-dependent optimal MH acceptance rate. The famous 0.234 is the
    d->inf limit; low-d optima are higher (Roberts/Gelman/Gilks; Sherlock). Our
    convention: 0.41 at d=1, 0.35 at d=2, ramp to 0.234 by d=10 (treated as
    effectively infinite), linear in between. Trivially retuned here."""
    if dim <= 1:
        return 0.41
    if dim >= 10:
        return R_OPT
    return 0.35 + (R_OPT - 0.35) * (dim - 2) / (10 - 2)  # 0.35 at d=2 -> 0.234 at d=10


# Historical per-block initial proposal variances (the legacy `sigma_m_sq` seeds
# when Sigma_0 == I). _mk reuses these as the Sigma_0 pre-scale so cold-start
# behaviour is preserved while the convergent scaling is corrected.
_NU_RHO_INIT_VAR = (2.38 ** 2) / 2   # -> Sigma_0 = I, sigma_m_sq = 2.38^2/2 (the A/B-validated inr+fix)


def _mk(dim: int, init_var: float) -> "Adaptive":
    """Build an RWMH block with the scale-degeneracy fix applied. The proposal is
    `sigma_m_sq * Sigma_0`; both used to carry scale, and seeding sigma_m_sq at
    0.01 while Sigma_0 collapsed ~100x toward the (small) empirical covariance
    left it chronically 3-5x under-scaled (acceptance pinned 0.5-0.9 -- the
    Robbins-Monro log-schedule decays too fast to climb to the optimum). Here
    sigma_m_sq is seeded AT the convergent Andrieu-Thoms optimum 2.38^2/d (so it
    only fine-tunes around it as Sigma_0 learns the covariance) and the target
    acceptance is dimension-aware (_target_accept). Sigma_0 is pre-scaled so the
    INITIAL proposal variance equals `init_var` (the historical per-block seed),
    keeping cold-start behaviour intact until the empirical-cov adaptation takes
    over."""
    s = (2.38 ** 2) / dim
    return Adaptive(dim, sigma_m_sq=s, Sigma_0=(init_var / s) * np.eye(dim),
                    r_opt=_target_accept(dim))


def _use_joint_phi(st) -> bool:
    """Whether to drive phi_k with the joint blocked move (update_phi_joint)."""
    return getattr(st, "joint_phi", False)


def _phi_blocks(k_phi: int, max_dim: int, coords=None):
    """Partition phi_k knot indices into ceil(k_phi/max_dim) blocks of <= max_dim.

    Default (coords=None): contiguous index ranges. Knots are often numbered
    roughly spatially, but not reliably -- e.g. a row-major outer grid with inner
    knots appended, so a contiguous block can span half the domain.

    With `coords` ((k_phi, 2) knot xy): SPATIALLY COMPACT blocks, via a balanced
    recursive median split along the longest-spread axis (a k-d partition into
    exactly ceil(k_phi/max_dim) groups). Nearby knots share a block regardless of
    index order, so each block's proposal covariance is a LOCAL (low-dim, well-
    determined) object -- the point of small blocks is that a 5x5 covariance is far
    better estimated from few effective samples than a 13x13 that chases noise."""
    n = max(1, -(-k_phi // max(1, max_dim)))              # ceil division
    if coords is None:
        return [np.asarray(b) for b in np.array_split(np.arange(k_phi), n)]
    coords = np.asarray(coords, dtype=float)

    def rec(ix, k):
        if k <= 1 or len(ix) <= 1:
            return [ix]
        pts = coords[ix]
        ax = int(np.argmax(pts.max(0) - pts.min(0)))      # longest-spread axis
        order = ix[np.argsort(pts[:, ax], kind="stable")]
        kl = k // 2                                       # leaves to the left half
        cut = int(round(len(order) * kl / k))
        cut = min(max(cut, kl), len(order) - (k - kl))    # keep both sides feasible
        return rec(order[:cut], kl) + rec(order[cut:], k - kl)

    return [np.sort(b) for b in rec(np.arange(k_phi), n)]


def _phi_block_coords(state, st):
    """phi-knot coordinates for spatial blocking, or None for the contiguous default
    (SamplerSettings.phi_block_spatial off, or no knot coords available)."""
    if not getattr(st, "phi_block_spatial", False):
        return None
    return getattr(getattr(state, "data", None), "knots_phi", None)


def _phi_adaptives(state, st):
    """Build the phi_k RWMH blocks: per-knot scalars by default, or -- when
    joint_phi is set -- one adapter per contiguous knot block (<= phi_block_max).
    A joint block seeds Sigma_0 from the warm-start phi covariance
    (st.phi_proposal_cov) when supplied: this is essential, since an isotropic
    start cannot self-organise onto phi's tight, strongly correlated ridge; the
    (fixed-jitter) Robbins-Monro adaptation then refines the seed. With no
    covariance (e.g. a cold start), falls back to a 1/d-scaled isotropic init."""
    if not _use_joint_phi(st):
        return [_mk(1, 0.01) for _ in range(state.k_phi)]
    pcov = getattr(st, "phi_proposal_cov", None)
    pcov = np.asarray(pcov) if pcov is not None else None
    have_cov = pcov is not None and pcov.shape == (state.k_phi, state.k_phi)
    out = []
    for b in _phi_blocks(state.k_phi, st.phi_block_max, coords=_phi_block_coords(state, st)):
        d = len(b)
        if have_cov:
            out.append(Adaptive(d, sigma_m_sq=(2.38 ** 2) / d,   # __post_init__ ensures PD
                                Sigma_0=np.atleast_2d(pcov[np.ix_(b, b)]),
                                r_opt=_target_accept(d)))
        else:
            out.append(_mk(d, 0.01 / d))
    return out


def _beta_adaptive(size, st, proposal_cov=None):
    """Adapter(s) for one GPD-margin coefficient vector. Component mode -> a LIST of
    1-D adapters (one per coef, each self-tuning); joint modes -> a single d-dim
    adapter. See update_beta_component.

    A joint block seeds Sigma_0 from `proposal_cov` (the warm-start beta covariance)
    when supplied -- ESSENTIAL for the joint block: from an isotropic start the 11-D
    RWMH deadlocks (accept ~0 -> the adapt window is all rejects -> cov_est ~0 ->
    Sigma stays isotropic and shrinks -> still ~0 accept; measured cond(Sigma) pinned
    at 1.0). A correct starting shape lets it accept from step 1 so the RM adaptation
    has moves to refine, exactly as for joint-phi (phi_proposal_cov). With no cov
    (cold start) falls back to the dimension-aware 0.01/d isotropic seed."""
    if getattr(st, "beta_sampler", "rwmh") == "component":
        return [_mk(1, 0.01) for _ in range(size)]
    pcov = np.asarray(proposal_cov) if proposal_cov is not None else None
    if pcov is not None and pcov.shape == (size, size):
        return Adaptive(size, sigma_m_sq=(2.38 ** 2) / size,   # __post_init__ ensures PD
                        Sigma_0=np.atleast_2d(pcov), r_opt=_target_accept(size))
    return _mk(size, 0.01 / max(1, size))


def make_adaptives(state, st, times, config) -> dict:
    """Construct the adaptive-RWMH blocks for one worker (S/Z keyed by `times`)."""
    # Only the centered RWMH path needs the full Ns x Ns empirical covariance
    # (dim == state.Ns); NCP adapts a scalar step in whitened coords and ESS
    # has no step at all (dim == 1, unused).
    centered = (getattr(st, "z_sampler", "rwmh") == "rwmh"
                and getattr(st, "z_update", "ncp") == "centered")
    z_dim = state.Ns if centered else 1
    return dict(
        S=({t: [_mk(1, 1.0) for _ in range(state.k_S)] for t in times}
           if getattr(st, "s_sampler", "joint") in ("component", "indep", "both")
           else {t: _mk(state.k_S, 1.0) for t in times}) if st.update_S else {},
        Z={t: _mk(z_dim, 1.0) for t in times} if st.update_Z else {},
        phi=_phi_adaptives(state, st) if st.update_phi else None,
        rho=[_mk(1, 0.01) for _ in range(state.k_rho)] if st.update_rho else None,
        alpha=_mk(1, 0.1) if st.update_alpha0 else None,
        nu=_mk(1, 0.01) if st.update_nu else None,
        nu_rho=_mk(2, _NU_RHO_INIT_VAR) if (_use_joint_nu_rho(state, st)
               or _use_interweave_nu_rho(state, st)) else None,
        nu_rho_ncp=_mk(2, _NU_RHO_INIT_VAR) if _use_interweave_nu_rho(state, st) else None,
        ncp_vec=_ncp_vec_adaptives(state, st) if _use_ncp_vec(state, st) else None,
        # Joint-geometry block (radius[,ratio,angle]) supersedes the separate
        # rad/ar/aa adapters when joint_geometry is on.
        geom=_geom_adaptive(state, st) if _use_joint_geometry(state, st) else None,
        rad=(_mk(1, 0.05) if st.update_radius_S and config.model_type != MODEL_HW
             and not _use_joint_geometry(state, st) else None),
        ar=_mk(1, 0.02) if state.update_aniso and not _use_joint_geometry(state, st) else None,
        aa=_mk(1, 0.05) if state.update_aniso and not _use_joint_geometry(state, st) else None,
        # beta blocks: "component" -> one self-tuning 1-D adapter per coefficient
        # (each reaches its own ~0.44 target, so a tight intercept can't bottleneck
        # the loose spatial coefs); "rwmh"/"nuts" -> one JOINT d-coef block with a
        # dimension-aware 0.01/d seed (mirrors joint-phi) so the aggregate step in
        # linear-predictor space stays ~O(0.1) for any d. (Seeding 0.01 per coord on
        # the joint block made the ~11-dim sigma step a ~33% jump -> ~0% accept.)
        bsig=_beta_adaptive(state.beta_sigma.size, st,
                            getattr(st, "beta_sigma_proposal_cov", None))
             if state.update_sigma else None,
        bxi=_beta_adaptive(state.beta_xi.size, st,
                           getattr(st, "beta_xi_proposal_cov", None))
            if state.update_xi else None,
    )


def adapt_all(ad: dict, adapt_size: int, t_adapt: int) -> None:
    """Run one adaptation step on every active block (no RNG draws)."""
    for v in ad["S"].values():                   # per time: one block, or k_S of them
        for a in (v if isinstance(v, list) else [v]):
            a.adapt(adapt_size, t_adapt)
    for a in ad["Z"].values():
        a.adapt(adapt_size, t_adapt)
    for a in (ad["phi"] or []):
        a.adapt(adapt_size, t_adapt)
    for a in (ad["rho"] or []):
        a.adapt(adapt_size, t_adapt)
    for a in (ad.get("ncp_vec") or []):          # vector NCP interweave blocks
        a.adapt(adapt_size, t_adapt)
    for key in ("alpha", "nu", "rad", "ar", "aa", "geom", "nu_rho", "nu_rho_ncp"):
        a = ad.get(key)            # .get: tolerate checkpoints predating a block
        if a is not None:
            a.adapt(adapt_size, t_adapt)
    for key in ("bsig", "bxi"):    # scalar (joint) OR list (component-wise)
        v = ad.get(key)
        for a in (v if isinstance(v, list) else [v] if v is not None else []):
            a.adapt(adapt_size, t_adapt)


def acceptance_summary(ad: dict) -> dict:
    """Run-level realized MH acceptance per block, pooled over a block's components
    (the phi/rho/ncp_vec lists and the per-time S/Z dicts). Reads the cumulative
    n_acc/n_prop counters; blocks that never proposed (e.g. the unused Z adapter
    under NUTS/ESS) are omitted. Lets the A/B / diagnostics see whether each block
    sits near its dimension-aware target instead of inferring it from change-rates
    (which thinning inflates ~2x and which conflate multiple moves)."""
    def pooled(items):
        items = [a for a in items if isinstance(a, Adaptive)]
        tp = sum(getattr(a, "n_prop", 0) for a in items)
        ta = sum(getattr(a, "n_acc", 0) for a in items)
        return (ta / tp) if tp else None
    out = {}
    for key, val in ad.items():
        items = (list(val.values()) if isinstance(val, dict)
                 else val if isinstance(val, list)
                 else [val] if val is not None else [])
        flat = []                                 # S may be {t: [Adaptive, ...]}
        for x in items:
            flat.extend(x if isinstance(x, list) else [x])
        items = flat
        rate = pooled(items)
        if rate is not None:
            out[key] = rate
    return out


def phi_logprior(state, phi_k) -> float:
    """Log-prior of the phi knot values, on the unconstrained scale."""
    uvec = (pt.to_unconstrained(phi_k, state.phi_domain)
            if state.phi_domain == PHI_UNIT else phi_k)
    return pt.logprior_unconstrained(uvec, state.phi_domain)


# ===========================================================================
# Update blocks (shared by serial run_mcmc and MPI run_mcmc_mpi)
# ===========================================================================
@_njit(cache=True)
def _sblock_value_grad_kernel(Z, logsf, logX, logR, phi, alpha0, cens, exc):
    """Fused elementary core for the S/R conditional: loglik and d(loglik)/d(logR).

    Structurally identical to `_zblock_value_grad_kernel`. Both blocks reach the
    likelihood through the same quantity, log_e = logX - (phi*logR + log g(Z)), and
    differ only in the chain-rule factor onto it:

        Z block:  d log_e / dZ     = -hazard(Z)
        S block:  d log_e / d logR = -phi

    so the two kernels share every branch and swap `h` for `phi`. Verified against
    finite differences of the production loglik_t to 3e-9. Kept as a separate @njit
    function rather than a flag on the Z kernel so the Z path stays byte-identical.
    """
    Ns, Nt = Z.shape
    gR = np.empty_like(Z)
    loglik = 0.0
    log_half = np.log(0.5)
    c = np.log(alpha0) - np.log(2.0)
    am1 = alpha0 - 1.0
    ap1 = alpha0 + 1.0
    for i in range(Ns):
        pi = phi[i]
        for j in range(Nt):
            lsf = logsf[i, j]
            log_Xstar = pi * logR[i, j] - lsf
            log_e = logX[i, j] - log_Xstar
            ce, ex = cens[i, j], exc[i, j]
            if log_e < 0.0:
                if ce:
                    loglik += log_half + alpha0 * log_e
                if ex:
                    loglik += -log_Xstar + c + am1 * log_e
                dcdf, dpdf = alpha0, am1
            else:
                a = np.exp(-alpha0 * log_e)
                if ce:
                    loglik += np.log1p(-0.5 * a)
                if ex:
                    loglik += -log_Xstar + c - ap1 * log_e
                dcdf, dpdf = 0.5 * alpha0 * a / (1.0 - 0.5 * a), -ap1
            g = 0.0
            if ce:
                g += -pi * dcdf
            if ex:
                g += -pi * (1.0 + dpdf)
            gR[i, j] = g
    return loglik, gR


def _levy_log_and_grad(logS, gamma_k):
    """log p(log S) and its gradient for the Levy prior, in LOG coordinates.

    levy.logpdf(x; c) + log x  =  const - (c/2) e^{-l} - l/2  at l = log x,
    so d/dl = (c/2) e^{-l} - 1/2. The constant is dropped from neither: it is
    included so the value matches pr.S_logprior_1t exactly (tests compare them).
    """
    inv = np.exp(-logS)
    val = float(np.sum(0.5 * np.log(gamma_k / (2.0 * np.pi))
                       - 0.5 * gamma_k * inv - 0.5 * logS))
    return val, 0.5 * gamma_k * inv - 0.5


def _sblock_value_grad(state, t, qv, dv, logsf, cens, exc, logS):
    """(logp, grad) of the S conditional at ONE time, in log-knot coordinates.

    R = W S is linear in S, so the chain rule from logR to logS is one matvec:
        d logR_i / d logS_j = W_ij S_j / R_i
    hence  grad_j = S_j * (W^T (gR / R))_j.  At Ns=369, k_S=41 that is a 369x41
    matvec against the Z block's 369x369 -- this gradient is CHEAPER than the Z one,
    which is the whole reason a gradient S sampler is affordable."""
    S = np.exp(logS)
    R = state.W_wend_S @ S
    if not (np.all(R > 0.0) and np.all(np.isfinite(R))):
        return -np.inf, np.zeros_like(logS)
    ll, gR = _sblock_value_grad_kernel(
        np.ascontiguousarray(state.Z[:, t:t + 1]), logsf,
        np.ascontiguousarray(np.log(qv)[:, None]),
        np.ascontiguousarray(np.log(R)[:, None]),
        state.phi_vec, float(state.alpha0), cens, exc)
    lp, gp = _levy_log_and_grad(logS, state.gamma_k)
    grad = S * (state.W_wend_S.T @ (gR[:, 0] / R)) + gp
    return float(ll) + lp, grad


def _levy_logprior_delta(lp, l0, c):
    """log pi_0(exp(lp)) - log pi_0(exp(l0)) for the Levy prior in LOG coordinates.

    log p(l) = const - (c/2) e^{-l} - l/2, so the constant cancels in the difference
    and this is three flops on Python floats. It replaces two calls to
    pr.S_logprior_1t per knot move, which go through scipy.stats.levy.logpdf --
    measured at 46% of the entire S block's runtime (0.319 s of 0.693 s over 4100
    calls), all of it dispatch overhead on one-element arrays."""
    return -0.5 * c * (np.exp(-lp) - np.exp(-l0)) - 0.5 * (lp - l0)


def _s_one_move(state, t, j, sj, W, gk, logS, ad, ctx, caches, independence):
    """One knot move, either an INDEPENDENCE draw from the Levy prior or the adaptive
    random walk. Split out so the two can be composed without duplicating the accept
    and bookkeeping. Updates state.S/state.R in place on acceptance; the caller
    refreshes Yll once per time."""
    qv, dv = caches.qd[t]
    if independence:
        y = ctx.lrng.standard_normal()
        if y == 0.0:
            ad.record(False, logS[j:j + 1])
            return
        lp = float(np.log(gk[j]) - 2.0 * np.log(abs(y)))    # Levy(scale=gamma_j)
        dprior = 0.0                    # the prior CANCELS against the proposal
    else:
        lp = ad.propose(logS[j:j + 1], ctx.lrng)[0]
        dprior = _levy_logprior_delta(lp, logS[j], gk[j])
    R_cur = state.R[sj, t]
    R_new = R_cur + W[sj, j] * (np.exp(lp) - np.exp(logS[j]))
    if np.all(R_new > 0.0) and np.all(np.isfinite(R_new)):
        log_r = (state.loglik_subset_S(t, sj, R_new, qv, dv)
                 - state.loglik_subset_S(t, sj, R_cur, qv, dv)) + dprior
        if mh_accept(log_r, ctx.lrng):
            logS[j] = lp
            state.S[j, t] = np.exp(lp)
            state.R[sj, t] = R_new
            ad.record(True, logS[j:j + 1])
            return
    ad.record(False, logS[j:j + 1])


def update_S_indep(state, ad_S, ctx, caches, st, arm_both=False) -> None:
    """Per-knot INDEPENDENCE proposals from the Levy prior, alone or alongside the
    componentwise random walk.

    Why. A knot is informed by the data only through exceedances in its support, or
    censored sites near enough the threshold to still be sensitive: the censored
    gradient factor is 0.5*alpha0*exp(-alpha0*log_e)/(1-...), decaying EXPONENTIALLY
    in how far the latent scale sits below the threshold. Measured on a 40-site column
    at alpha0=5: exceedance factor 5.0, censored factor 2.3e-9. So in any one time
    column most knots have a conditional that IS their prior, numerically.

    For those knots an independence proposal from that prior is near optimal, because
    the prior cancels exactly out of the Metropolis ratio -- with q(S') = pi_0(S'),
        alpha = [L(S') pi_0(S') pi_0(S)] / [L(S) pi_0(S) pi_0(S')] = L(S')/L(S)
    -- leaving the likelihood ratio alone, which is ~1 precisely where the likelihood
    is flat. Those coordinates then draw independently.

    Measured on a 40-site, 13-knot problem (S-ESS of 300 draws, min over knots then
    median): componentwise 21.7/54.2, pure independence 7.8/248.4. The two arms win on
    DIFFERENT knots -- independence on the uninformed majority, the random walk on the
    informed few -- and a probabilistic mixture inherits each one's weakness on the
    other's knots (at p=0.5, min 7.2). ``arm_both`` therefore takes BOTH moves at every
    knot, for two subset evaluations instead of one.

    Nothing here classifies a knot: both moves leave the conditional invariant, so any
    composition of them does too, and the kernel stays exact rather than approximately
    right. Sampling uses gamma/Y^2 ~ Levy(gamma) for Y ~ N(0,1), i.e. one normal
    deviate per proposal."""
    Yll = caches.Yll
    W = state.W_wend_S
    supp = [np.flatnonzero(W[:, j] > 0.0) for j in range(state.k_S)]
    gk = state.gamma_k
    p_ind = float(getattr(st, "s_prior_mix", 0.5))
    for t in ctx.times:
        qv, dv = caches.qd[t]
        blocks = ad_S[t]
        logS = np.log(state.S[:, t])
        for j in range(state.k_S):
            sj = supp[j]
            if sj.size == 0:
                blocks[j].record(False, logS[j:j + 1])
                continue
            moves = ((False, True) if arm_both
                     else (bool(ctx.lrng.random() < p_ind),))
            for independence in moves:
                _s_one_move(state, t, j, sj, W, gk, logS, blocks[j], ctx, caches,
                            independence)
        Yll[t] = state.loglik_t(t, qv=qv, dv=dv)


def _nuts_warmup_len(st):
    """Warmup length for a per-time NUTS block, read from the SETTINGS every sweep.

    Not stored on the adapter. Freezing it at creation makes a resumed run inherit
    the warmup of whatever run wrote the checkpoint: resume a 60-iteration run as a
    100-iteration one and it warms for 30 sweeps where an uninterrupted 100 would
    warm for 50, so the two are not the same chain. Caught by the checkpoint
    bit-identity test once NUTS became the default S sampler; update_Z_nuts had the
    same latent bug, unexercised because the goldens run RWMH on Z."""
    w = getattr(st, "nuts_warmup", None)
    return int(w) if w is not None else max(1, st.n_iters // 2)


def update_S_nuts(state, ad_S, ctx, caches, st) -> None:
    """Per-time NUTS on the latent Levy scales, in log-knot coordinates.

    The third S arm. The joint RWMH block is dimension-limited (a k_S-dimensional
    move must shrink each coordinate ~sqrt(k_S) to hold acceptance) and its learned
    covariance was measured to be noise; the componentwise arm fixes the step size at
    the cost of k_S likelihood evaluations per time. This replaces both with one
    gradient trajectory: `_sblock_value_grad` costs a single Ns-column pass plus two
    (Ns x k_S) matvecs, so a whole trajectory is cheaper than k_S subset evaluations.

    The cost objection that killed gradient sampling for the GPD betas -- every
    leapfrog needing a full qRW inversion sweep -- does not apply, for the same
    reason it did not apply to Z: the RW transform is constant in this conditional.

    One shared step adapter across the worker's times (as update_Z_nuts does), driven
    by the LOCAL rng, so this stays a per-rank local move under MPI."""
    Yll, qd = caches.Yll, caches.qd

    def make_vg(t):
        qv, dv = qd[t]
        ci, ei = state.cens_exc(t)
        cens = np.zeros((state.Ns, 1), dtype=bool)
        exc = np.zeros((state.Ns, 1), dtype=bool)
        cens[ci, 0] = True
        exc[ei, 0] = True
        logsf = _norm_logsf_np(np.ascontiguousarray(state.Z[:, t:t + 1]))

        def vg(l):
            # the driver requires grad to have theta's SHAPE, not just its size:
            # returning (k,) for a (k,1) theta broadcasts to (k,k) in the leapfrog
            # and fails later, inside the U-turn test, far from the cause.
            flat = l[:, 0] if l.ndim == 2 else l
            lp, g = _sblock_value_grad(state, t, qv, dv, logsf, cens, exc, flat)
            return lp, (g[:, None] if l.ndim == 2 else g)

        return vg

    ns = getattr(state, "_nuts_s", None)
    if ns is None:
        t0 = ctx.times[0]
        l0 = np.log(state.S[:, t0])[:, None]
        ns = {"ad": NUTSStepAdapter(
            nuts_find_reasonable_eps(make_vg(t0), l0, ctx.lrng),
            target_accept=st.nuts_target_accept)}
        state._nuts_s = ns
    ad = ns["ad"]
    warming = ad.m < _nuts_warmup_len(st)

    accs = []
    for t in ctx.times:
        vg = make_vg(t)
        l0 = np.log(state.S[:, t])[:, None]
        l1, a, depth = nuts_transition(vg, l0, ad.eps, ctx.lrng)
        state.S[:, t] = np.exp(l1[:, 0])
        state.R[:, t] = state.W_wend_S @ state.S[:, t]
        qv, dv = qd[t]
        Yll[t] = state.loglik_t(t, qv=qv, dv=dv)
        accs.append(a)
        ad_S[t].record(depth > 0, np.array([0.0]))
    if warming and accs:
        ad.update(float(np.mean(accs)))
        if ad.m >= _nuts_warmup_len(st):
            ad.finalize()


def update_S(state, ad_S, ctx, caches, st) -> None:
    """Latent Levy scales S(t): dispatch on ``st.s_sampler`` (local block, local RNG)."""
    arm = getattr(st, "s_sampler", "joint")
    if arm == "component":
        return update_S_component(state, ad_S, ctx, caches, st)
    if arm == "nuts":
        return update_S_nuts(state, ad_S, ctx, caches, st)
    if arm == "indep":
        return update_S_indep(state, ad_S, ctx, caches, st)
    if arm == "both":
        return update_S_indep(state, ad_S, ctx, caches, st, arm_both=True)
    return update_S_joint(state, ad_S, ctx, caches, st)


def update_S_component(state, ad_S, ctx, caches, st) -> None:
    """One knot at a time, exploiting the Wendland COMPACT SUPPORT.

    Why this exists. The joint per-time block proposes all k_S log-scales at once
    from an adaptive empirical covariance, and there is one such covariance PER TIME
    (make_adaptives). Measured against the production-tuned proposals from
    shi_fixmargins_stage1 at iteration 82,079 (Ns=369, k_S=41):

      * the learned CORRELATIONS are worthless. Zeroing every off-diagonal changes
        S-ESS by 0.85-1.04x, i.e. nothing. Across 40 different times the learned
        off-diagonal patterns correlate at median -0.001 -- the knot-coupling
        geometry is identical for every time, so a real structure would drive that
        toward 1. After 82,080 proposals each adapter has learned its own noise.
      * the block is NOT mistuned: acceptance sits at 0.24-0.36 against its
        dimension-aware target of 0.234. It is correctly tuned for what it is.
      * it is what it is that hurts. Independently tuned 1-D blocks want a per-knot
        step ~9.4x larger (the price a 41-d move pays to hold acceptance), and on
        top of that the joint diagonal mis-allocates across knots by ~1.9x, because
        one shared sigma_m_sq plus an empirical-covariance diagonal cannot give each
        knot its own scale.

    Result: S-ESS min 3.4 -> 84.3 and median 10.3 -> 177.6 per 1000 draws (25x/17x).
    S was effectively frozen. Cost is k_S subset likelihood evaluations per time
    instead of one full one -- about +5% of total sampler wall, because the S block
    is only ~0.12% of an iteration to begin with.

    Only R changes when a knot moves, and only at that knot's support; qRW/dRW do
    not depend on R at all (unlike the phi case), so nothing else is recomputed. The
    MH ratio uses the difference of SUBSET likelihoods, which equals the difference
    of full ones because every other site is untouched; Yll[t] is then refreshed
    exactly once per time from the production loglik_t, so the cache never carries
    accumulated drift (and tests/test_s_component.py checks the two agree)."""
    Yll = caches.Yll
    W = state.W_wend_S
    # recomputed per sweep on purpose: W_wend_S is rebuilt whenever radius_S or the
    # anisotropy moves, so a cached support list would go stale. ~50 us against a
    # block costing milliseconds.
    supp = [np.flatnonzero(W[:, j] > 0.0) for j in range(state.k_S)]
    gk = state.gamma_k
    for t in ctx.times:
        qv, dv = caches.qd[t]                    # R does not enter qRW/dRW
        blocks = ad_S[t]
        logS = np.log(state.S[:, t])
        for j in range(state.k_S):
            ad = blocks[j]
            lp = ad.propose(logS[j:j + 1], ctx.lrng)[0]
            sj = supp[j]
            if sj.size == 0:
                ad.record(False, logS[j:j + 1])
                continue
            R_cur = state.R[sj, t]
            R_new = R_cur + W[sj, j] * (np.exp(lp) - np.exp(logS[j]))
            if np.all(R_new > 0.0) and np.all(np.isfinite(R_new)):
                # the Levy prior factorises over knots, so only knot j's term moves
                dprior = _levy_logprior_delta(lp, logS[j], gk[j])
                log_r = (state.loglik_subset_S(t, sj, R_new, qv, dv)
                         - state.loglik_subset_S(t, sj, R_cur, qv, dv)) + dprior
                if mh_accept(log_r, ctx.lrng):
                    logS[j] = lp
                    state.S[j, t] = np.exp(lp)
                    state.R[sj, t] = R_new
                    ad.record(True, logS[j:j + 1])
                    continue
            ad.record(False, logS[j:j + 1])
        Yll[t] = state.loglik_t(t, qv=qv, dv=dv)    # exact refresh, once per time


def update_S_joint(state, ad_S, ctx, caches, st) -> None:
    """Latent Levy scales S(t) (local: one block per time, local RNG)."""
    Yll, qd = caches.Yll, caches.qd
    for t in ctx.times:
        ad = ad_S[t]
        logS = np.log(state.S[:, t])
        logS_prop = ad.propose(logS, ctx.lrng)
        R_prop = state.W_wend_S @ np.exp(logS_prop)
        if np.all(np.isfinite(R_prop)) and np.all(R_prop > 0):
            q, d = qd[t]
            ll_prop = state.loglik_t(t, R=R_prop, qv=q, dv=d)
            Sp_prop = pr.S_logprior_1t(logS_prop, state.gamma_k)
            Sp_cur = pr.S_logprior_1t(logS, state.gamma_k)
            log_r = (ll_prop + Sp_prop) - (Yll[t] + Sp_cur)
            if mh_accept(log_r, ctx.lrng):
                state.S[:, t] = np.exp(logS_prop)
                state.R[:, t] = R_prop
                Yll[t] = ll_prop
                ad.record(True, np.log(state.S[:, t]))
                continue
        ad.record(False, np.log(state.S[:, t]))


def update_Z(state, ad_Z, ctx, caches, st) -> None:
    """Latent Gaussian field Z(s,t): dispatch on the configured sampler.

    ``st.z_sampler`` selects the family; ``st.z_update`` / ``st.z_block`` are the
    RWMH sub-mode (kept for back-compat, so existing runs are byte-identical):

    * ``"rwmh"`` (default) -> adaptive random-walk Metropolis. ``z_update="ncp"``
      (default) is the non-centered whitened update (Z = L e, L the marginal-
      covariance Cholesky), which mixes far better in high dimension than the
      centered full-joint move (``"centered"``, kept for A/B).
    * ``"ess"`` -> elliptical slice sampling on the exact Gaussian prior
      Z[:,t] ~ N(0, K); tuning-free, always moves (Phase 1).
    * ``"nuts"`` -> per-time NUTS on the whitened coords with the exact elementary
      Z-block gradient; dual-averaged step size warmed up then frozen (Phase 4).

    All arms drive ``ctx.lrng`` per time, matching the local-update discipline so
    serial and MPI stay consistent."""
    sampler = getattr(st, "z_sampler", "rwmh")
    if sampler == "ess":
        update_Z_ess(state, ad_Z, ctx, caches, st)
    elif sampler == "nuts":
        update_Z_nuts(state, ad_Z, ctx, caches, st)
    elif getattr(st, "z_update", "ncp") == "centered":
        update_Z_centered(state, ad_Z, ctx, caches, st)
    else:
        update_Z_ncp(state, ad_Z, ctx, caches, st)


def update_Z_centered(state, ad_Z, ctx, caches, st) -> None:
    """Centered full-joint adaptive RWMH over the observed sites at each time
    (empirical Ns x Ns covariance). Mixes poorly in high dimension; kept for the
    A/B comparison against the non-centered update."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    Ns = state.Ns
    for t in ctx.times:
        ad = ad_Z[t]
        obs = np.flatnonzero(~state.miss[:, t])
        step = np.zeros(Ns)
        step[obs] = ctx.lrng.multivariate_normal(
            np.zeros(obs.size), ad.Sigma_0[np.ix_(obs, obs)])
        Zp = state.Z[:, t] + np.sqrt(ad.sigma_m_sq) * step
        q, d = qd[t]
        ll_prop = state.loglik_t(t, Z=Zp, qv=q, dv=d)
        zlp_prop = state.Z_logprior(Zp)
        if mh_accept((ll_prop + zlp_prop) - (Yll[t] + Zlp[t]), ctx.lrng):
            state.Z[:, t] = Zp
            Yll[t] = ll_prop
            Zlp[t] = zlp_prop
            ad.record(True, state.Z[:, t])
        else:
            ad.record(False, state.Z[:, t])


def _z_blocks(Ns, z_block):
    """Index blocks for the whitened (non-centered) Z sweep. ``z_block=None``
    (default) -> one joint block over all Ns whitened coords (cost parity with
    the centered joint move); a positive int -> contiguous blocks of that size
    (smaller blocks raise the per-block acceptance but cost more likelihoods)."""
    if not z_block or z_block >= Ns:
        return [np.arange(Ns)]
    return [np.arange(i, min(i + z_block, Ns)) for i in range(0, Ns, z_block)]


def update_Z_ncp(state, ad_Z, ctx, caches, st) -> None:
    """Non-centered Z update: Z(:,t) = L e with e ~ N(0, I), L the marginal Z
    covariance Cholesky (``state.zchol()``). We step the whitened coords e in
    blocks; the Gaussian prior on Z becomes a unit-Gaussian prior on e, so the
    prior ratio is the cheap local ``-0.5(||e_prop||^2 - ||e_cur||^2)`` and no
    Z_logprior call is needed inside the sweep. Updates ALL Ns sites (the
    centered path skipped missing sites, but missing entries are exactly the
    prior-driven ones the whitened proposal samples correctly). A scalar
    step-size per time is adapted toward R_OPT."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    L = state.zchol()
    blocks = _z_blocks(state.Ns, getattr(st, "z_block", None))
    for t in ctx.times:
        ad = ad_Z[t]
        q, d = qd[t]
        z_cur = state.Z[:, t].copy()
        # whitened current coords: e = L^{-1} z  (one triangular solve per time)
        e_cur = solve_triangular(L, z_cur, lower=state._zfac_lower)
        accepted_any = False
        for blk in blocks:
            de = np.sqrt(ad.sigma_m_sq) * ctx.lrng.standard_normal(blk.size)
            z_prop = z_cur + L[:, blk] @ de
            ll_prop = state.loglik_t(t, Z=z_prop, qv=q, dv=d)
            e_prop_blk = e_cur[blk] + de
            dlp = -0.5 * float(e_prop_blk @ e_prop_blk - e_cur[blk] @ e_cur[blk])
            if mh_accept((ll_prop - Yll[t]) + dlp, ctx.lrng):
                z_cur = z_prop
                e_cur[blk] = e_prop_blk
                Yll[t] = ll_prop
                accepted_any = True
        state.Z[:, t] = z_cur
        Zlp[t] = state.Z_logprior(z_cur)
        ad.record(accepted_any, np.array([0.0]))


def update_Z_ess(state, ad_Z, ctx, caches, st) -> None:
    """Elliptical slice sampling for the latent Gaussian field Z(:,t).

    The model has the exact Gaussian prior Z[:,t] ~ N(0, K) i.i.d. over time, so
    ESS (Murray, Adams & MacKay 2010) applies directly and tuning-free: per time
    t, draw an auxiliary nu ~ N(0, K) = L @ standard_normal (L = state.zchol()),
    set a likelihood slice level, and shrink the angle bracket on the ellipse
    ``f cos θ + nu sin θ`` until the proposal clears the level. The qRW/dRW cache
    qd[t] is Z-independent (Z enters the likelihood only through X* = R^φ g(Z)),
    so it is reused unchanged. Drives ``ctx.lrng`` per time like update_Z_ncp.
    The Adaptive block is unused (ESS has no step size); we record the per-time
    eval count via the accept counter as a cheap cost diagnostic."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    L = state.zchol()
    Ns = state.Ns
    for t in ctx.times:
        ad = ad_Z[t]
        q, d = qd[t]
        f = state.Z[:, t]
        nu = L @ ctx.lrng.standard_normal(Ns)        # nu ~ N(0, K)
        f_new, ll_new, n_eval = elliptical_slice(
            f, nu, lambda x: state.loglik_t(t, Z=x, qv=q, dv=d), Yll[t], ctx.lrng)
        state.Z[:, t] = f_new
        Yll[t] = ll_new
        Zlp[t] = state.Z_logprior(f_new)
        ad.record(n_eval == 1, np.array([0.0]))      # accept@1st-try rate ~ cheapness


def update_Z_nuts(state, ad_Z, ctx, caches, st) -> None:
    """Per-time NUTS update of the latent Gaussian field Z.

    Given the global parameters the Z columns are conditionally independent --
    the prior factorizes as Z[:,t] ~ N(0,K) i.i.d. over time and the likelihood
    factorizes over t -- so a per-time NUTS move on the whitened coords
    ``eta_t`` (Z[:,t] = L eta_t, eta_t ~ N(0,I)) samples each column's exact
    conditional, with no loss versus a joint Ns*Nt move, and stays MPI-compatible
    (one column per ``ctx.times`` entry, driving ``ctx.lrng``). The white-coord
    target and its exact elementary gradient come from ``zblock_value_grad_np`` (no
    RW-transform derivatives -- X is constant in the Z conditional). A single shared
    step size is dual-averaged over the first ``st.nuts_warmup`` sweeps, then frozen.

    Note: the qd[t] cache is Z-independent and reused; Yll/Zlp are refreshed with the
    *production* ``state.loglik_t`` / ``Z_logprior`` (the potential differs from
    loglik_t by a Z-independent constant, irrelevant to the eta dynamics). NUTS warmup
    state lives on ``state`` and is not checkpoint-restored (a resume re-warms up)."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    L = state.zchol()
    logR_full = np.log(state.R)
    # Triangular-BLAS gradient (see zblock_value_grad_trmv): 1.37x on the gradient,
    # 1.22x on a transition, exact to 2e-16. Behind a switch because that 2e-16 is
    # enough to diverge a chain, so the goldens move; default keeps them byte-exact.
    _zvg = (zblock_value_grad_trmv if getattr(st, "z_grad", "legacy") == "trmv"
            else zblock_value_grad_np)
    # one copy per sweep, amortised over len(ctx.times) * (leapfrogs per transition);
    # already a no-op for the VisGP factor, which is built transposed and so is
    # Fortran-ordered to begin with.
    Lf = np.asfortranarray(L) if _zvg is zblock_value_grad_trmv else None

    def make_vg(t):
        # per-column sliced arrays (Ns,1); reuses the full-field value_and_grad.
        q, _d = qd[t]
        ci, ei = state.cens_exc(t)
        cens = np.zeros((state.Ns, 1), dtype=bool)
        exc = np.zeros((state.Ns, 1), dtype=bool)
        cens[ci, 0] = True
        exc[ei, 0] = True
        # Contiguity is settled HERE, once per time replicate, not inside the
        # gradient once per leapfrog -- logR is a column slice of a C-ordered
        # (Ns, Nt) array, so it is genuinely non-contiguous and was being copied
        # on every single leapfrog (~0.7 us of a ~70 us call for all four).
        cols = dict(L=L, Lf=Lf, lower=state._zfac_lower,
                    logX=np.ascontiguousarray(np.log(q)[:, None]),
                    logR=np.ascontiguousarray(logR_full[:, t:t + 1]),
                    phi=state.phi_vec, alpha0=state.alpha0,
                    cens=np.ascontiguousarray(cens), exc=np.ascontiguousarray(exc))
        return lambda eta_col: _zvg(cols, eta_col)

    # Diagonal metric (z_metric="diag"). MEASURED 2026-08-14 AND NOT WORTH IT -- kept
    # default-off with the numbers, so the question does not get re-asked.
    #
    # The geometry argument for it is sound. Whitening makes the PRIOR contribute
    # exactly unit curvature, but not the target: the likelihood adds L' diag(h) L,
    # diagonal in Z and dense in eta. At Ns=369 the per-cell curvature h has median
    # ~1.0 -- comparable to the prior, not negligible even at 90% censoring -- and
    # reaches 26, leaving the condition number at 147 against 58.5 under an exact
    # diag(H^-1) metric. On the usual sqrt(cond) heuristic that is ~1.6x in leapfrogs.
    #
    # It buys nothing. At Ns=369, eps dual-averaged separately per arm to accept 0.8:
    #
    #     metric                       eps      mean leapfrogs   ms/transition
    #     unit                       0.2383          16.0            1.58
    #     exact diag(H^-1)           0.3484          16.0            1.74
    #     estimated (as here)        0.3007          16.0            1.67
    #
    # The metric does exactly what it should -- it raises the usable step size by 46%
    # -- and the leapfrog count does not move, because TREE DEPTH IS QUANTIZED. NUTS
    # stops on the U-turn criterion at a power-of-2 trajectory length, and sqrt(cond)
    # only falls from 12.1 to 7.6, which is inside the same bucket (depth 4 spans 8-16
    # leapfrogs). A metric must improve conditioning by ~4x to remove one doubling;
    # 2.5x buys a step size the termination rule then declines to spend. The arms are
    # identical in work and the metric ones are 6-9% SLOWER from the extra per-leapfrog
    # multiplies. Note the exact metric was tested too, so this is not an estimation-
    # quality problem (corr(estimated, exact) is only 0.438, but both give 16.0).
    #
    # Implication for anything similar: on this block, judge a preconditioner by
    # whether it can cross a depth boundary, not by the condition number it improves.
    #
    # One metric shared across time replicates, adapted from all of them; warmup is
    # counted in TRANSITIONS (len(times) per sweep), so nuts_warmup keeps its meaning.
    z_metric = getattr(st, "z_metric", "unit")
    nz = getattr(state, "_nuts_z", None)
    if nz is None:
        warmup = _nuts_warmup_len(st)
        t0 = ctx.times[0]
        eta0 = solve_triangular(L, state.Z[:, t0], lower=state._zfac_lower)[:, None]
        if z_metric == "diag":
            adapter = NUTSWarmup(make_vg(t0), eta0, ctx.lrng,
                                 max(20, int(warmup) * max(1, len(ctx.times))),
                                 target_accept=st.nuts_target_accept, adapt_mass=True)
        else:
            adapter = NUTSStepAdapter(
                nuts_find_reasonable_eps(make_vg(t0), eta0, ctx.lrng),
                target_accept=st.nuts_target_accept)
        nz = {"ad": adapter, "warmup": warmup, "metric": z_metric}
        state._nuts_z = nz
    ad = nz["ad"]
    per_transition = nz.get("metric", "unit") == "diag"
    warming = (not per_transition) and ad.m < _nuts_warmup_len(st)

    accs = []
    for t in ctx.times:
        vg = make_vg(t)
        eta0 = solve_triangular(L, state.Z[:, t], lower=state._zfac_lower)[:, None]
        eta1, a, depth = nuts_transition(vg, eta0, ad.eps, ctx.lrng,
                                         inv_mass=getattr(ad, "inv_mass", None))
        z_new = (L @ eta1)[:, 0]
        state.Z[:, t] = z_new
        Yll[t] = state.loglik_t(t, Z=z_new, qv=qd[t][0], dv=qd[t][1])
        Zlp[t] = state.Z_logprior(z_new)
        accs.append(a)
        ad_Z[t].record(depth > 0, np.array([0.0]))   # tree-depth>0 ~ moved diagnostic
        if per_transition:
            ad.update(eta1, a, vg, ctx.lrng)   # eps + Welford metric, per transition

    if warming and accs:
        ad.update(float(np.mean(accs)))
        if ad.m >= _nuts_warmup_len(st):
            ad.finalize()


def update_phi_nuts(state, ad_phi, ctx, caches, st) -> None:
    """Gradient NUTS over the phi knots, using the analytic phi gradient.

    phi is the last field block without a gradient option, and the only quantity that
    has never reached stationarity in any run: R-hat 1.50-2.46 across all six Faroe
    arms and above 1.01 in all seven exp4 arms, while its per-knot RWMH sits at a
    correctly tuned 0.420 acceptance against a 0.410 target. Correct tuning with no
    convergence is the signature the joint S block had, and there a gradient was worth
    20x.

    Cost. Unlike the GPD betas -- where a gradient lost 30-55x because each leapfrog
    needed a fresh qRW inversion against a CHEAP RWMH baseline -- phi's RWMH baseline
    is extravagant: k_phi knots each trigger a full transform, and the compact-support
    fast path never fires in production (it needs phi_kernel="wendland"; Faroe and
    exp4 both use the "gaussian" default). Measured: one full transform 5.4 ms against
    the RWMH sweep's 272 ms, so a depth-3 trajectory is ~6x cheaper than the block it
    replaces. The gradient itself adds no inversions -- dX/dphi comes from the
    implicit-function identity at the root already in hand.

    GLOBAL block: driven by ctx.grng so every rank walks the identical trajectory, and
    the per-time value/gradient sums are Allreduced inside the value_grad (one buffer
    carrying both), with the priors added once after the reduction.
    """
    import phi_gradient as _pg

    if not (ctx.covers_all_times or ctx.reduces_arrays):
        raise RuntimeError(
            "update_phi_nuts needs an array Allreduce (Ctx(reduce_arr=...)): phi's "
            "target sums over ALL times, so without it every rank samples its own "
            "partial-times posterior and they diverge silently.")

    def vg(theta):
        pk = theta[:, 0] if theta.ndim == 2 else theta
        # -inf reads as a divergence and stops that subtree, which is both the correct
        # behaviour for zero target density and the cheap-reject for a wild proposal
        if not _phi_in_support(state, pk):
            g = np.zeros_like(np.asarray(pk, float))
            return -np.inf, (g[:, None] if theta.ndim == 2 else g)
        v, g = _pg.phi_value_grad(state, np.asarray(pk, float),
                                  times=ctx.times, areduce=ctx.areduce,
                                  qguess={t: caches.qd[t][0] for t in ctx.times})
        return v, (g[:, None] if theta.ndim == 2 else g)

    theta0 = state.phi_k.copy()[:, None]
    np_ = getattr(state, "_nuts_phi", None)
    if np_ is None:
        np_ = {"ad": NUTSStepAdapter(
            nuts_find_reasonable_eps(vg, theta0, ctx.grng),
            target_accept=st.nuts_target_accept)}
        state._nuts_phi = np_
    ad = np_["ad"]
    warming = ad.m < _nuts_warmup_len(st)

    theta1, acc, _ = nuts_transition(vg, theta0, ad.eps, ctx.grng)
    state.phi_k = theta1[:, 0].copy()
    state.phi_vec = state._phi_field(state.phi_k)
    # phi moved, so the cached transform and per-time likelihoods are both stale
    qvc = state.qd_RW_cens()
    for t in ctx.times:
        q, d = state.qd_RW_one(t, qvc)
        caches.qd[t] = (q, d)
        caches.Yll[t] = state.loglik_t(t, qv=q, dv=d)
    if warming:
        ad.update(float(acc))
        if ad.m >= _nuts_warmup_len(st):
            ad.finalize()


# Numerical validity limit on phi(s) for the GRADIENT phi kernels. In the "unit"
# domain the model's own support (0,1) supplies this. In "positive", phi = exp(W phi_k)
# is unbounded, and far from the mode the target is not merely improbable but
# RUINOUSLY EXPENSIVE: qRW's deep-tail bracketing crawls, so a single wild proposal can
# stall a run for minutes (observed: a doubling eps search reached 4.0, sending phi to
# exp(4) ~ 55, and qd_RW_cens never returned). This wall lets such a proposal be
# rejected for the price of an exp() instead.
#
# It DOES truncate the target, so the bound is set far outside anything reachable: phi
# is a tail exponent whose posterior mass sits below 1, and Faroe/NZ fits run ~0.5.
# The per-knot RWMH does not need it -- it moves one knot at a time at a tuned scale --
# so its target is unchanged and the goldens are untouched.
_PHI_NUMERIC_MAX = 20.0
_PHI_BARKER_MAX_EPS = 1.0


def _phi_in_support(state, pk):
    """Is this phi_k somewhere the transform can actually be evaluated?"""
    fld = state._phi_field(np.asarray(pk, float))
    if state.phi_domain == PHI_UNIT:
        return bool(np.all(fld > 0.0) and np.all(fld < 1.0))
    return bool(np.all(fld > 0.0) and np.all(fld < _PHI_NUMERIC_MAX))


def update_phi_barker(state, ad_phi, ctx, caches, st) -> None:
    """Barker-proposal MALA over the phi knots -- one gradient-informed JOINT move.

    The cost case that NUTS failed. phi's per-knot RWMH is expensive for a structural
    reason: it is COMPONENTWISE, so a sweep pays k_phi full transforms (13 x 6.5 =
    84.9 ms at exp4 size, 57% of the whole iteration). Any joint move pays one. Barker
    takes two gradient evaluations per sweep rather than NUTS's whole trajectory -- and
    the current-point one reuses caches.qd outright, since the cache already holds the
    transform at exactly this phi -- so the block costs ~30 ms against the default's
    85 ms. It therefore needs only ~35% of the RWMH's ESS per iteration to win on
    wall-clock, where NUTS needed 147%.

    Why Barker rather than plain joint RWMH: the joint move's known failure at k_phi
    dimensions is step-size shrinkage under heterogeneous per-knot scales, and Barker's
    skew-adaptive sign flip is specifically the robust-to-misspecified-scale kernel
    (Livingstone & Zanella 2022). Whether that is enough is exactly what the A/B
    measures -- phi has been diagnosed COUPLING-limited rather than proposal-limited,
    and if that holds no within-block kernel can help, Barker included.

    GLOBAL block: ctx.grng, and the per-time sums are Allreduced inside the gradient.
    """
    import phi_gradient as _pg

    if not (ctx.covers_all_times or ctx.reduces_arrays):
        raise RuntimeError(
            "update_phi_barker needs an array Allreduce (Ctx(reduce_arr=...)): phi's "
            "target sums over ALL times, so without it every rank samples its own "
            "partial-times posterior and they diverge silently.")

    qg = {t: caches.qd[t][0] for t in ctx.times}

    def vg(theta, cached=None):
        pk = np.asarray(theta, float)
        if not _phi_in_support(state, pk):              # cheap reject, no transform
            return -np.inf, np.zeros_like(pk)
        return _pg.phi_value_grad(state, pk, times=ctx.times, areduce=ctx.areduce,
                                  qguess=qg, qd_cached=cached)

    theta0 = state.phi_k.copy()
    bk = getattr(state, "_barker_phi", None)
    if bk is None:
        warmup = (st.nuts_warmup if getattr(st, "nuts_warmup", None) is not None
                  else max(1, st.n_iters // 2))
        # phi_k is a priori O(1) in both domains (N(0,1) unconstrained; Beta(2,2),
        # sd 0.22, constrained), so a step of 1 prior sd is already large for a joint
        # k_phi-dimensional move. Without the cap the doubling search overshoots into
        # the region where each trial evaluation takes minutes.
        bk = state._barker_phi = BarkerWarmup(vg, theta0, ctx.grng, warmup,
                                              max_eps=_PHI_BARKER_MAX_EPS)
    # value+grad at the CURRENT phi costs no root-finds: caches.qd is the transform at
    # exactly this phi (the cache contract every block maintains)
    lp0, g0 = vg(theta0, cached={t: caches.qd[t] for t in ctx.times})

    theta1, _, _, acc = barker_transition(vg, theta0, bk.eps, bk.precond_sd, ctx.grng,
                                          logp0=lp0, grad0=g0)
    moved = not np.array_equal(theta1, theta0)
    # ORDER MATTERS. phi_value_grad MUTATES state.phi_k/phi_vec (that is how a proposal
    # gets evaluated), and BarkerWarmup.update calls vg during its one-time
    # preconditioner reset -- so on that sweep the state is left wherever the eps search
    # last probed, NOT at theta1. Adapt first, then commit, so the committed phi is
    # always the accepted one.
    bk.update(theta1, acc, vg, ctx.grng)
    state.phi_k = theta1.copy()
    state.phi_vec = state._phi_field(state.phi_k)
    if moved:                       # on a reject the caches already match theta0
        qvc = state.qd_RW_cens()
        for t in ctx.times:
            q, d = state.qd_RW_one(t, qvc, qe_guess=qg[t])
            caches.qd[t] = (q, d)
            caches.Yll[t] = state.loglik_t(t, qv=q, dv=d)


def update_phi(state, ad_phi, ctx, caches, st) -> None:
    """Tail-exponent knot values phi_k (global; per-knot blocks). The serial
    backend may use the compact-support fast path when phi has compact support."""
    Yll, qd = caches.Yll, caches.qd
    Ns = state.Ns
    for j in range(state.k_phi):
        ad = ad_phi[j]
        phi_k_prop = state.phi_k.copy()
        phi_k_prop[j] = ad.propose(state.phi_k[j:j + 1], ctx.grng)[0]
        valid = (0.0 < phi_k_prop[j] < 1.0) if state.phi_domain == PHI_UNIT else True
        if valid:
            phi_vec_prop = state._phi_field(phi_k_prop)
            aff = state._phi_affected[j]
            qd_prop = {}
            if ctx.supports_compact and state._phi_compact and aff.size < 0.5 * Ns:
                # only the affected sites' qRW/dRW change; patch them per time
                qs, ds = state.qd_RW_subset(aff, phi_vec=phi_vec_prop)
                for t in ctx.times:
                    qcol = qd[t][0].copy()
                    dcol = qd[t][1].copy()
                    qcol[aff] = qs[:, t]
                    dcol[aff] = ds[:, t]
                    qd_prop[t] = (qcol, dcol)
            else:
                qvc = state.qd_RW_cens(phi_vec=phi_vec_prop)
                for t in ctx.times:
                    qd_prop[t] = state.qd_RW_one(t, qvc, phi_vec=phi_vec_prop,
                                                 qe_guess=qd[t][0])
            ll_prop = {t: state.loglik_t(t, phi_vec=phi_vec_prop,
                                         qv=qd_prop[t][0], dv=qd_prop[t][1])
                       for t in ctx.times}
            log_r = ((ctx.total(ll_prop) + phi_logprior(state, phi_k_prop))
                     - (ctx.total(Yll) + phi_logprior(state, state.phi_k)))
            if mh_accept(log_r, ctx.grng):
                state.phi_k = phi_k_prop
                state.phi_vec = phi_vec_prop
                for t in ctx.times:
                    qd[t] = qd_prop[t]
                    Yll[t] = ll_prop[t]
                ad.record(True, state.phi_k[j:j + 1])
                continue
        ad.record(False, state.phi_k[j:j + 1])


def update_phi_joint(state, ad_phi, ctx, caches, st) -> None:
    """Blocked joint adaptive-RWMH on the phi_k tail-exponent knots (alternative to
    the one-at-a-time update_phi; selected by SamplerSettings.joint_phi). The knots
    form a smooth, strongly cross-correlated spatial field, so the per-knot move
    crawls along their joint ridge; here each contiguous block of <= phi_block_max
    knots is proposed as a unit with an empirical-covariance proposal that learns
    the correlation. Accept on the DATA-likelihood ratio (phi enters the X-transform
    via qRW/dRW), recomputing the full qd -- this forgoes the per-knot compact-
    support fast path, but a sweep now costs n_blocks (<= ceil(k_phi/15)) likelihood
    evals instead of k_phi. MPI-safe: grng proposals + ctx.total reduction, so every
    rank takes the same (synchronized-RNG) accept/reject, exactly like update_phi."""
    Yll, qd = caches.Yll, caches.qd
    for ad, blk in zip(ad_phi, _phi_blocks(state.k_phi, st.phi_block_max,
                                           coords=_phi_block_coords(state, st))):
        phi_k_prop = state.phi_k.copy()
        phi_k_prop[blk] = ad.propose(state.phi_k[blk], ctx.grng)
        if state.phi_domain == PHI_UNIT and not np.all(
                (phi_k_prop[blk] > 0.0) & (phi_k_prop[blk] < 1.0)):
            ad.record(False, state.phi_k[blk])
            continue
        phi_vec_prop = state._phi_field(phi_k_prop)
        qvc = state.qd_RW_cens(phi_vec=phi_vec_prop)
        qd_prop = {t: state.qd_RW_one(t, qvc, phi_vec=phi_vec_prop, qe_guess=qd[t][0])
                   for t in ctx.times}
        ll_prop = {t: state.loglik_t(t, phi_vec=phi_vec_prop,
                                     qv=qd_prop[t][0], dv=qd_prop[t][1])
                   for t in ctx.times}
        log_r = ((ctx.total(ll_prop) + phi_logprior(state, phi_k_prop))
                 - (ctx.total(Yll) + phi_logprior(state, state.phi_k)))
        if mh_accept(log_r, ctx.grng):
            state.phi_k = phi_k_prop
            state.phi_vec = phi_vec_prop
            for t in ctx.times:
                qd[t] = qd_prop[t]
                Yll[t] = ll_prop[t]
            ad.record(True, state.phi_k[blk])
        else:
            ad.record(False, state.phi_k[blk])


def update_rho(state, ad_rho, ctx, caches, st) -> None:
    """Covariance range knot values rho_k (global; rebuilds the Z covariance)."""
    Zlp = caches.Zlp
    for j in range(state.k_rho):
        ad = ad_rho[j]
        rho_k_prop = state.rho_k.copy()
        rho_k_prop[j] = ad.propose(state.rho_k[j:j + 1], ctx.grng)[0]
        if rho_k_prop[j] > 1e-3:
            rho_old, K_old, Q_old, ld_old = (state.rho_vec, state.K, state.Q,
                                             state.Q_logdet)
            state.rho_vec = state.W_gauss_rho @ rho_k_prop
            state.rebuild_cov()
            zlp_prop = {t: state.Z_logprior(state.Z[:, t]) for t in ctx.times}
            log_r = ((ctx.total(zlp_prop) + pr.rho_logprior(rho_k_prop, scale=state.rho_prior_scale))
                     - (ctx.total(Zlp) + pr.rho_logprior(state.rho_k, scale=state.rho_prior_scale)))
            if mh_accept(log_r, ctx.grng):
                state.rho_k = rho_k_prop
                for t in ctx.times:
                    Zlp[t] = zlp_prop[t]
                ad.record(True, state.rho_k[j:j + 1])
                continue
            state.rho_vec, state.K, state.Q, state.Q_logdet = (
                rho_old, K_old, Q_old, ld_old)
        ad.record(False, state.rho_k[j:j + 1])


def _cp_gaussian_core(nu, rho_vec, dist_cache, Dmat, Z, Nt, areduce=None):
    """Shared CENTERED-conditional core for the Z-prior target sum_t logN(Z_t;0,K)
    and its building blocks (Z, nu, rho all frozen at the passed values).

    Returns ``None`` on a non-PD K, else a dict with: the Gaussian log-density
    ``logp`` (no priors/Jacobians), the contraction matrix ``Gm = 1/2(sum_t b_t
    b_t^T - Nt K^{-1})`` (b_t = K^{-1}Z_t), the per-knot-range sensitivity matrix
    ``A`` (A_mj = dK_mj/dr_m), the Paciorek scale ``S``, the Mahalanobis ``Dist``,
    the row-contraction ``rowsum_m = sum_j Gm_mj A_mj`` (so dU_gauss/dr_m =
    2 rowsum_m), and ``r`` = range_internal. Both the rho-only and the joint
    (nu, rho) value_grads build their parameter gradients from these by chain rule.

    Closed form: ns_cov_cached uses arg12==0, arg11==arg22==range, so the Paciorek
    form collapses to K_ij = S_ij M(Dist_ij), depending only on (r_i, r_j); the
    Matern distance-derivative uses d/dx[x^nu K_nu]=-x^nu K_{nu-1} (one Bessel order
    down, NO dK_nu/dnu).

    `Nt` is the GLOBAL replicate count and `Z` this worker's columns. Every quantity
    but the two per-time sums -- the quad form ``sum_t Z_t^T K^{-1} Z_t`` and the
    contraction ``sum_t b_t b_t^T`` (b_t = K^{-1}Z_t) -- depends only on (nu, rho),
    identical on every rank. Those two are packed into one buffer and combined with
    `areduce` (Allreduce-sum under MPI; identity + bit-identical serially), so the
    result matches the single-process computation exactly. K/Kinv/logdet/A/... are
    recomputed redundantly per rank (cheap vs the communication) and never reduced."""
    Ns = rho_vec.shape[0]
    r = rho_vec ** 2 / (8.0 * nu)                      # = canonical_to_ns_range
    try:
        K = u.ns_cov_cached(r, np.ones(Ns), dist_cache, kappa=nu, cov_model="matern")
        cf = cho_factor(K, lower=True)
    except (LinAlgError, FloatingPointError, ValueError):
        return None
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    # K^{-1} is needed in full for the log|K| gradient trace (dK/dr_m has a row+col
    # "cross" support, so tr(K^{-1} dK/dr_m) reads every row of K^{-1}); B = K^{-1}Z
    # then falls out of the same inverse via one GEMM (no extra triangular solve).
    Kinv = cho_solve(cf, np.eye(Ns))
    B = Kinv @ Z                                       # K^{-1} Z   (Ns, Nt_local)
    # per-time partial sums (this rank's columns), reduced to the global totals: the
    # quad form (scalar) and sum_t b_t b_t^T (Ns x Ns) packed into a single Allreduce.
    if areduce is not None:
        buf = np.empty(Ns * Ns + 1)
        buf[:Ns * Ns] = (B @ B.T).ravel()
        buf[-1] = float(np.sum(Z * B))
        buf = areduce(buf)
        BBt = buf[:Ns * Ns].reshape(Ns, Ns)
        quad = float(buf[-1])
    else:
        BBt = B @ B.T
        quad = float(np.sum(Z * B))
    logp = -0.5 * Nt * logdet - 0.5 * quad - 0.5 * Nt * Ns * np.log(2 * np.pi)

    Gm = 0.5 * (BBt - Nt * Kinv)                       # contraction matrix
    rr = r[:, None] + r[None, :]
    Dist = Dmat * np.sqrt(2.0 / rr)
    S = 2.0 * np.sqrt(np.outer(r, r)) / rr
    # M(Dist) is recovered from K = S*M (K_ii=S_ii=1) -- no second K_nu eval (the one
    # inside ns_cov_cached already paid for it); only M'(x) = -c x^nu K_{nu-1}(x)
    # costs a Bessel eval. At Ns~200 these dominate the cost, so 3 evals -> 2.
    c = 2.0 ** (1 - nu) / _gamma(nu)
    M = K / S
    Mp = np.zeros_like(Dist)
    pos = Dist > 0
    xp = Dist[pos]
    Mp[pos] = -c * xp ** nu * _kv(nu - 1.0, xp)        # via d/dx[x^nu K_nu]=-x^nu K_{nu-1}
    dS = np.sqrt(r[None, :] / r[:, None]) * (r[None, :] - r[:, None]) / rr ** 2
    A = dS * M + S * Mp * (-Dist / (2.0 * rr))         # A_mj = dK_mj/dr_m
    np.fill_diagonal(A, 0.0)                            # K_mm const => no r_m dependence
    return dict(logp=logp, Gm=Gm, A=A, S=S, Dist=Dist, r=r, M=M,
                rowsum=np.sum(Gm * A, axis=1))


# Quadrature grid for the nu order channel (bessel_nu.dkv_dnu). These are the values
# `dmatern_dnu` has always used, kept IDENTICAL here on purpose so that "fast" is a
# pure-performance change and agrees with "legacy" to ~1e-11.
#
# Do not economise on this grid, and in particular do not trust the accuracy of
# dkv_dnu itself when choosing it. The order channel is not an evaluation, it is the
# CONTRACTION sum_p w_p dM_p over ~68k pairs of alternating sign, and that sum is
# ill-conditioned: sum|terms| / |sum terms| runs from ~2 to ~5000 across the (nu, rho)
# range this model visits, growing with both. So a relative error eps in dM arrives in
# the gradient multiplied by up to ~5000.
#
# Measured against converged references (n=320-400, two different ranges agreeing to
# 8 digits), relative error of the CONTRACTED scalar at Ns=369:
#
#     nu    rho    cond    n=128 [-8,3]      what a narrower range does
#     0.55  1.5       2         2.2e-08
#     1.3   3.0     218         2.0e-03      converges to a DIFFERENT value
#     1.95  3.0    1443         8.6e-02
#     1.95  6.0    4949         5.5e+00      n=192 [-5,2] flips sign
#
# Two things follow. (1) A narrower u-range is not a saving, it is a bug: [-5,2]
# truncates real mass, so refining n converges to the wrong integral -- at nu=1.95,
# rho=6 the sequence n=128,160,192,256 runs -2.9e8, -1.2e8, +1.0e8, +5.8e8 while the
# truth is -3.8e8. (2) The grid this block has always used is itself under-resolved in
# part of the parameter space; at nu~2 the nu-gradient can be wrong by 9% (rho=3) or a
# factor of 6.5 (rho=6). That is a pre-existing accuracy question, NOT something the
# fast path introduces, and it wants its own fix -- see docs/nurho_quadrature.md.
_NU_RHO_QUAD_DEFAULT = 128
_NU_RHO_QUAD_UMIN = -8.0
_NU_RHO_QUAD_UMAX = 3.0


def _sym_from_upper(vals, n, iu, diag):
    """(n, n) symmetric matrix from its upper-triangle values -- so a scalar special
    function is evaluated on the n(n-1)/2 UNIQUE pairs instead of all n^2 entries."""
    out = np.empty((n, n))
    out[iu[0], iu[1]] = vals
    out.T[iu[0], iu[1]] = vals
    np.fill_diagonal(out, diag)
    return out


def _cp_gaussian_core_fast(nu, rho_vec, Dmat, Z, Nt, iu, areduce=None):
    """Same return contract as `_cp_gaussian_core`, built in one pass.

    Three redundancies removed, none of them mathematical (profiled at Ns=369: the
    build drops 116.8 -> 42.0 ms, agreeing with the original to 1.5e-15):

      * the generic `ns_cov_cached` computes Dist and the Paciorek scale internally,
        discards both, and returns only K -- after which `_cp_gaussian_core` rebuilds
        the identical Dist and S from scratch. Here they are built once and kept.
      * `ns_cov_cached` applies its two diagonal scalings as FOUR dense N^3 GEMMs
        (`np.diag(v).dot(A).dot(np.diag(v))`), and the second pair is against
        sigsq_vec = ones, i.e. a multiplication by the identity. Both become
        outer-product scalings here (12.7 -> 1.1 ms) or vanish (12.0 -> 0 ms).
      * K_nu and K_{nu-1} were each evaluated over the FULL symmetric matrix. Here
        they run on the unique pairs and are mirrored (77 -> 36 and 72 -> 37 ms).

    Deliberately NOT a change to `ns_cov_cached` itself: that function is on every
    model's hot path, so touching it moves results at 1e-15 everywhere and would
    force a golden regeneration for a 24 ms win outside the bottleneck.

    Also returns `x`/`xnu`/`Mu` (the unique-pair distance, x**nu and correlation),
    which the joint gradient's order channel would otherwise recompute."""
    Ns = rho_vec.shape[0]
    r = rho_vec ** 2 / (8.0 * nu)
    rr = r[:, None] + r[None, :]
    Dist = Dmat * np.sqrt(2.0 / rr)
    S = 2.0 * np.sqrt(np.outer(r, r)) / rr
    c = 2.0 ** (1 - nu) / _gamma(nu)
    x = Dist[iu[0], iu[1]]
    xnu = x ** nu
    Mu = c * xnu * _kv(nu, x)                          # correlation, unique pairs
    M = _sym_from_upper(Mu, Ns, iu, 1.0)
    K = S * M
    try:
        cf = cho_factor(K, lower=True)
    except (LinAlgError, FloatingPointError, ValueError):
        return None
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    Kinv = cho_solve(cf, np.eye(Ns))
    B = Kinv @ Z
    if areduce is not None:
        buf = np.empty(Ns * Ns + 1)
        buf[:Ns * Ns] = (B @ B.T).ravel()
        buf[-1] = float(np.sum(Z * B))
        buf = areduce(buf)
        BBt = buf[:Ns * Ns].reshape(Ns, Ns)
        quad = float(buf[-1])
    else:
        BBt = B @ B.T
        quad = float(np.sum(Z * B))
    logp = -0.5 * Nt * logdet - 0.5 * quad - 0.5 * Nt * Ns * np.log(2 * np.pi)

    Gm = 0.5 * (BBt - Nt * Kinv)
    Mp = _sym_from_upper(-c * xnu * _kv(nu - 1.0, x), Ns, iu, 0.0)
    dS = np.sqrt(r[None, :] / r[:, None]) * (r[None, :] - r[:, None]) / rr ** 2
    A = dS * M + S * Mp * (-Dist / (2.0 * rr))
    np.fill_diagonal(A, 0.0)
    return dict(logp=logp, Gm=Gm, A=A, S=S, Dist=Dist, r=r, M=M,
                rowsum=np.sum(Gm * A, axis=1), x=x, xnu=xnu, Mu=Mu, c=c)


def _order_channel(nu, x, xnu, Mu, w, n_quad, u_min, u_max, c, chunk=2048):
    """The nu ORDER-channel contraction ``sum_p w_p dM_p`` (the caller doubles it for
    the symmetric pair sum), computed WITHOUT materialising the quadrature table.

    `bessel_nu.dkv_dnu` builds a (n_pairs x n_quad) array -- 70 MB at Faroe size, on
    every leapfrog -- which the caller then immediately contracts to a single scalar.
    Streaming it in cache-sized blocks and accumulating the scalar directly is the
    same arithmetic to 2e-14 and drops 252 -> 152 ms; blocks of 16384 and above show
    no gain over the unchunked version, so this is a cache effect, and it should pay
    MORE under 32 ranks contending for bandwidth than it does on one core."""
    A, C, h = bessel_nu._trap_nodes(nu, n_quad, u_min, u_max)
    lead = -np.log(2.0) - _digamma(nu)
    # The monitor must not perturb what it monitors: keep np.dot for the value (an
    # explicit terms.sum() differs from it at 1e-16, which was enough to break the
    # goldens) and only pay for the |.| pass on the sampled calls.
    check = _order_cond_due()
    tot = 0.0
    abstot = 0.0
    for s in range(0, x.size, chunk):
        xb = x[s:s + chunk]
        wb = w[s:s + chunk]
        dkv = h * (np.exp(-xb[:, None] * C) @ A)
        base = Mu[s:s + chunk] * (lead + np.log(xb)) + c * xnu[s:s + chunk] * dkv
        tot += float(np.dot(wb, base))
        if check:
            abstot += float(np.abs(wb * base).sum())
    if check:
        _warn_if_ill_conditioned(abstot, tot, nu, n_quad)
    return tot


# The quadrature grid is adequate at the (nu, rho) this model has actually been fitted
# at -- Faroe sits at nu=1.28, rho~3.1, where the contracted order channel is accurate
# to 2e-3 -- but it degrades fast with the cancellation, and there is nothing in a fit
# that would tell you it had wandered somewhere the grid cannot handle. Measured
# against converged references: cond ~218 -> 2.0e-3, ~1443 -> 8.6e-2, ~4949 -> 5.5
# (i.e. the nu-gradient wrong by a factor of 6.5). Threshold set to flag the regime
# where the error passes ~1e-2. See docs/nurho_quadrature.md.
#
# NB this cannot bias the posterior: the Metropolis accept uses the exact density
# assembled from scipy.kv, and leapfrog stays volume-preserving and reversible under
# ANY force field, so a wrong gradient makes a worse PROPOSAL, never a wrong answer.
# What it can do is stop nu exploring, which matters because nu is the standing
# bottleneck -- hence a warning rather than an error.
_ORDER_COND_WARN = 500.0
_ORDER_COND_EVERY = 200       # sample the monitor; conditioning drifts slowly in (nu, rho)
_order_cond_warned = False
_order_cond_calls = 0


def reset_order_cond_monitor():
    """Re-arm the cancellation monitor (it warns once per process, and samples one
    call in _ORDER_COND_EVERY). Exposed so tests can exercise both states without
    reaching into two module globals and getting one of them wrong."""
    global _order_cond_warned, _order_cond_calls
    _order_cond_warned = False
    _order_cond_calls = 0


def _order_cond_due():
    """True on the first call and every _ORDER_COND_EVERY-th after, so the extra |.|
    pass costs ~0.02% instead of a few percent of the gradient. Sampling is enough:
    the cancellation ratio is a smooth function of (nu, rho), so a fit that drifts
    into the bad region stays there for far more than 200 gradient calls."""
    global _order_cond_calls
    if _order_cond_warned:
        return False
    due = (_order_cond_calls % _ORDER_COND_EVERY) == 0
    _order_cond_calls += 1
    return due


def _warn_if_ill_conditioned(abstot, tot, nu, n_quad):
    global _order_cond_warned
    if _order_cond_warned or tot == 0.0:
        return
    cond = abstot / abs(tot)
    if cond > _ORDER_COND_WARN:
        _order_cond_warned = True
        _warnings.warn(
            f"nu order-channel contraction is ill-conditioned (cancellation ratio "
            f"{cond:.0f} at nu={nu:.3f}); the n_quad={n_quad} quadrature grid is "
            f"likely giving a nu-gradient accurate to worse than 1%. This degrades "
            f"nu MIXING only -- the posterior is unaffected, since the accept uses "
            f"the exact density. Raise nu_rho_quad_nodes if nu mixes badly. "
            f"See docs/nurho_quadrature.md.", RuntimeWarning, stacklevel=3)


def _nu_rho_cp_value_grad_fast(theta, W, Dmat, Z, Nt, rho_scale, nu_loc, nu_scale, iu,
                               n_quad=_NU_RHO_QUAD_DEFAULT, u_min=_NU_RHO_QUAD_UMIN,
                               u_max=_NU_RHO_QUAD_UMAX, areduce=None):
    """Optimised `_nu_rho_cp_value_grad`: identical target, identical closed forms,
    same (logp, grad) contract. Profiled 470.6 -> 326.6 ms at Ns=369 with the default
    quadrature, ~200 ms at n_quad=64, agreeing with the original to 1e-11 relative.

    The quadrature grid defaults to exactly what the legacy path uses, so this is a
    pure-performance change; `n_quad` is exposed only so the accuracy of that grid can
    be studied separately. Do NOT lower it to buy speed -- the order channel is an
    ill-conditioned contraction and amplifies dM error by up to ~5000x. See
    `_NU_RHO_QUAD_DEFAULT`."""
    z_nu, psi = theta[0], theta[1:]
    sig = 1.0 / (1.0 + np.exp(-z_nu))
    nu = nu_loc + nu_scale * sig
    rho_k = np.exp(psi)
    rho_vec = W @ rho_k
    core = _cp_gaussian_core_fast(nu, rho_vec, Dmat, Z, Nt, iu, areduce)
    if core is None:
        return -np.inf, np.zeros_like(theta)
    Gm, S, r, rowsum = core["Gm"], core["S"], core["r"], core["rowsum"]

    logp = (core["logp"] + pr.rho_logprior(rho_k, scale=rho_scale)
            + pr.nu_logprior(nu, loc=nu_loc, scale=nu_scale) + float(np.sum(psi))
            + np.log(nu_scale) + np.log(sig) + np.log1p(-sig))

    dU_drhovec = (rho_vec / (2.0 * nu)) * rowsum
    g_psi = (W.T @ dU_drhovec - rho_k / rho_scale ** 2) * rho_k + 1.0

    w = (Gm * S)[iu[0], iu[1]]
    order_ch = 2.0 * _order_channel(nu, core["x"], core["xnu"], core["Mu"], w,
                                    n_quad, u_min, u_max, core["c"])
    range_ch = -(2.0 / nu) * float(np.sum(rowsum * r))
    g_znu = (order_ch + range_ch) * (nu_scale * sig * (1.0 - sig)) + (1.0 - 2.0 * sig)
    return logp, np.concatenate(([g_znu], g_psi))


def _rho_cp_value_grad(psi, W, nu, dist_cache, Dmat, Z, Nt, rho_scale, areduce=None):
    """``(logp, grad)`` of the CENTERED rho-knot target in psi = log(rho_k) coords.

    Target (Z frozen, exactly the conditional update_rho samples):

        U(psi) = sum_{t<Nt} logN(Z_t; 0, K(rho_k)) + halfnormal(rho_k) + sum(psi)

    with rho_k = exp(psi), rho_vec = W @ rho_k. See `_cp_gaussian_core` for the
    closed-form covariance gradient; here it is chained through r(rho_vec) and
    rho_vec = W rho_k. The +sum(psi) is the log-Jacobian of rho_k = exp(psi)."""
    rho_k = np.exp(psi)
    rho_vec = W @ rho_k
    core = _cp_gaussian_core(nu, rho_vec, dist_cache, Dmat, Z, Nt, areduce)
    if core is None:
        return -np.inf, np.zeros_like(psi)
    logp = core["logp"] + pr.rho_logprior(rho_k, scale=rho_scale) + float(np.sum(psi))
    # dU/drho_vec_m = (sum_{i,j} G_ij dK_ij/dr_m) dr_m/drho_vec_m; by symmetry the
    # inner sum is 2 rowsum_m, and dr_m/drho_vec_m = rho_vec_m/(4nu).
    dU_drhovec = (rho_vec / (2.0 * nu)) * core["rowsum"]
    dU_drhok = W.T @ dU_drhovec - rho_k / rho_scale ** 2   # + halfnormal prior grad
    g_psi = dU_drhok * rho_k + 1.0                     # chain exp + Jacobian d(sum psi)
    return logp, g_psi


def _nu_rho_cp_value_grad(theta, W, dist_cache, Dmat, Z, Nt, rho_scale,
                          nu_loc, nu_scale, iu, n_quad=128, areduce=None):
    """``(logp, grad)`` of the JOINT (nu, rho) CENTERED target. State is
    ``theta = [z_nu, psi_1..psi_krho]`` with rho_k = exp(psi) (log, positivity) and
    nu = nu_loc + nu_scale * sigmoid(z_nu) (logit, keeps nu inside its uniform
    support so the trajectory never hits a hard wall).

    nu enters K through THREE channels, all assembled from `_cp_gaussian_core`:
      * order:  S_ij * dM(Dist_ij;nu)/dnu  -- the one piece needing dK_nu/dnu, via
                `bessel_nu.dmatern_dnu` (computed on the unique pairs ``iu`` only);
      * range:  r = rho_vec^2/(8 nu) so dr_m/dnu = -r_m/nu, reusing the SAME A-matrix
                contraction (rowsum) as rho -- range-channel grad = -(2/nu) sum_m
                rowsum_m r_m.
    The uniform nu prior is flat (zero gradient) inside support; its log-Jacobian
    log(nu_scale)+log(sig)+log(1-sig) supplies the only z_nu prior-side gradient."""
    z_nu = theta[0]
    psi = theta[1:]
    sig = 1.0 / (1.0 + np.exp(-z_nu))
    nu = nu_loc + nu_scale * sig
    rho_k = np.exp(psi)
    rho_vec = W @ rho_k
    core = _cp_gaussian_core(nu, rho_vec, dist_cache, Dmat, Z, Nt, areduce)
    if core is None:
        return -np.inf, np.zeros_like(theta)
    Gm, S, Dist, r, rowsum, M = (core["Gm"], core["S"], core["Dist"], core["r"],
                                 core["rowsum"], core["M"])
    logp = (core["logp"] + pr.rho_logprior(rho_k, scale=rho_scale)
            + pr.nu_logprior(nu, loc=nu_loc, scale=nu_scale) + float(np.sum(psi))
            + np.log(nu_scale) + np.log(sig) + np.log1p(-sig))   # + nu logit Jacobian

    dU_drhovec = (rho_vec / (2.0 * nu)) * rowsum
    dU_drhok = W.T @ dU_drhovec - rho_k / rho_scale ** 2
    g_psi = dU_drhok * rho_k + 1.0

    # nu gradient: order channel (full pair sum, diag 0) + range channel. Reuse the
    # already-built correlation M = K/S (unique pairs) so dmatern_dnu skips a kv eval.
    dM = bessel_nu.dmatern_dnu(nu, Dist[iu], n=n_quad, M=M[iu])
    order_ch = 2.0 * float(np.sum((Gm * S)[iu] * dM))       # symmetric -> 2x upper-tri
    range_ch = -(2.0 / nu) * float(np.sum(rowsum * r))
    dU_dnu = order_ch + range_ch                            # uniform prior: no grad term
    # chain nu = loc + scale*sig(z) (dnu/dz = scale*sig*(1-sig)) + d/dz logit-Jacobian.
    g_znu = dU_dnu * (nu_scale * sig * (1.0 - sig)) + (1.0 - 2.0 * sig)
    return logp, np.concatenate(([g_znu], g_psi))


def update_rho_nuts(state, ctx, caches, st) -> None:
    """NUTS update of the rho range knots on the CENTERED conditional
    p(rho_k | Z, nu) (Z frozen), in unconstrained psi = log(rho_k) coords.

    Prototype gradient-based replacement for the per-knot RWMH ``update_rho``: same
    target, but driven by the exact closed-form gradient (`_rho_cp_value_grad`) so a
    single trajectory walks the correlated knot ridge instead of crawling it one
    knot at a time. Shares the ``nuts_warmup`` / ``nuts_target_accept`` dual-averaging
    knobs; the step adapter lives on ``state._nuts_rho`` (re-warms on resume).

    Like ``update_rho`` this is a CP move (Z fixed): the data likelihood Yll is
    unchanged and only the Z-prior cache Zlp is refreshed. MPI-safe: the target's
    per-time sums (quad form + sum_t b_t b_t^T) are Allreduced inside the value_grad
    via ``ctx.areduce``, and Nt is the global replicate count -- so every rank walks
    the identical trajectory (shared grng + identical value/grad) and the result
    matches the serial computation."""
    Zlp = caches.Zlp
    W, nu = state.W_gauss_rho, state.nu
    dist_cache = state._dist_cache
    Dmat = np.sqrt(dist_cache['d1_sq'] + dist_cache['d2_sq'])   # euclidean dist
    Z = state.Z[:, ctx.times]
    Nt = int(round(ctx.total({t: 1.0 for t in ctx.times})))    # GLOBAL replicate count
    rho_scale = state.rho_prior_scale

    def vg(psi):
        return _rho_cp_value_grad(psi, W, nu, dist_cache, Dmat, Z, Nt, rho_scale,
                                  ctx.areduce)

    psi0 = np.log(state.rho_k)
    nr = getattr(state, "_nuts_rho", None)
    if nr is None:
        warmup = (st.nuts_warmup if getattr(st, "nuts_warmup", None) is not None
                  else max(1, st.n_iters // 2))
        adapt = getattr(st, "nuts_adapt_mass", None)      # None -> per-block default
        nr = state._nuts_rho = NUTSWarmup(
            vg, psi0, ctx.grng, warmup, target_accept=st.nuts_target_accept,
            adapt_mass=(False if adapt is None else adapt))   # rho: off unless forced

    psi1, acc, depth = nuts_transition(vg, psi0, nr.eps, ctx.grng, inv_mass=nr.inv_mass)
    rho_k_new = np.exp(psi1)
    if not np.array_equal(rho_k_new, state.rho_k):
        state.rho_k = rho_k_new
        state.rho_vec = state.W_gauss_rho @ state.rho_k
        state.rebuild_cov()
        for t in ctx.times:
            Zlp[t] = state.Z_logprior(state.Z[:, t])

    nr.update(psi1, acc, vg, ctx.grng)          # eps + diagonal-metric warmup schedule


def update_nu_rho_nuts(state, ctx, caches, st) -> None:
    """JOINT NUTS update of (nu, rho) on the CENTERED conditional p(nu, rho | Z),
    Z frozen -- one trajectory over the whole correlated nu/rho ridge in unconstrained
    coords [z_nu, log rho_k] (nu via logit into its uniform support, rho via log).

    Replaces the separate update_nu + update_rho (RWMH) blocks with a single
    gradient move whose nu-gradient uses the ported dK_nu/dnu (`bessel_nu`). Exact
    closed-form gradient (`_nu_rho_cp_value_grad`); step adapter on
    ``state._nuts_nu_rho``. CP move: Yll unchanged, only Zlp refreshed. MPI-safe: the
    target's per-time sums (quad form + sum_t b_t b_t^T) are Allreduced inside the
    value_grad via ``ctx.areduce`` with the global Nt, so every rank walks the
    identical trajectory and the result matches the serial computation."""
    Zlp = caches.Zlp
    W = state.W_gauss_rho
    dist_cache = state._dist_cache
    Dmat = np.sqrt(dist_cache['d1_sq'] + dist_cache['d2_sq'])
    Z = state.Z[:, ctx.times]
    Nt = int(round(ctx.total({t: 1.0 for t in ctx.times})))   # GLOBAL replicate count
    rho_scale = state.rho_prior_scale
    nu_loc, nu_scale = pr.nu_logprior.__defaults__       # uniform [loc, loc+scale]
    iu = np.triu_indices(state.Ns, k=1)

    if getattr(st, "nu_rho_grad", "legacy") == "fast":
        nq = getattr(st, "nu_rho_quad_nodes", None) or _NU_RHO_QUAD_DEFAULT

        def vg(theta):
            return _nu_rho_cp_value_grad_fast(theta, W, Dmat, Z, Nt, rho_scale,
                                              nu_loc, nu_scale, iu, n_quad=int(nq),
                                              areduce=ctx.areduce)
    else:
        def vg(theta):
            return _nu_rho_cp_value_grad(theta, W, dist_cache, Dmat, Z, Nt, rho_scale,
                                         nu_loc, nu_scale, iu, areduce=ctx.areduce)

    sig = np.clip((state.nu - nu_loc) / nu_scale, 1e-6, 1 - 1e-6)
    theta0 = np.concatenate(([np.log(sig / (1 - sig))], np.log(state.rho_k)))
    nr = getattr(state, "_nuts_nu_rho", None)
    if nr is None:
        warmup = (st.nuts_warmup if getattr(st, "nuts_warmup", None) is not None
                  else max(1, st.n_iters // 2))
        adapt = getattr(st, "nuts_adapt_mass", None)      # None -> per-block default
        # optional metric SEEDED from a warm trace (preferred over adapting one from
        # this block's slow-mixing warmup); ignored if the length doesn't match theta.
        seed = getattr(st, "nu_rho_mass_diag", None)
        if seed is not None:
            seed = np.asarray(seed, dtype=float)
            if seed.size != theta0.size:
                seed = None
        nr = state._nuts_nu_rho = NUTSWarmup(
            vg, theta0, ctx.grng, warmup, target_accept=st.nuts_target_accept,
            adapt_mass=(False if adapt is None else adapt),   # nu/rho: off unless forced
            init_inv_mass=seed)

    theta1, acc, depth = nuts_transition(vg, theta0, nr.eps, ctx.grng, inv_mass=nr.inv_mass)
    if not np.array_equal(theta1, theta0):
        sig1 = 1.0 / (1.0 + np.exp(-theta1[0]))
        state.nu = float(nu_loc + nu_scale * sig1)
        state.rho_k = np.exp(theta1[1:])
        state.rho_vec = state.W_gauss_rho @ state.rho_k
        state.rebuild_cov()
        for t in ctx.times:
            Zlp[t] = state.Z_logprior(state.Z[:, t])

    nr.update(theta1, acc, vg, ctx.grng)        # eps + diagonal-metric warmup schedule


def update_alpha0(state, ad_alpha, ctx, caches, st) -> None:
    """Log-Laplace nugget tail parameter alpha0 (global, soft-wall)."""
    Yll, qd = caches.Yll, caches.qd
    eta = pr.eta_from_alpha0(state.alpha0, alpha0_min=st.alpha0_min)
    eta_prop = ad_alpha.propose(np.array([eta]), ctx.grng)[0]
    a0_prop = pr.alpha0_from_eta(eta_prop, alpha0_min=st.alpha0_min)
    if a0_prop != 1.0:
        qvc = state.qd_RW_cens(alpha0=a0_prop)
        qd_prop = {}
        ll_prop = {}
        for t in ctx.times:
            qd_prop[t] = state.qd_RW_one(t, qvc, alpha0=a0_prop, qe_guess=qd[t][0])
            ll_prop[t] = state.loglik_t(t, alpha0=a0_prop,
                                        qv=qd_prop[t][0], dv=qd_prop[t][1])
        log_r = ((ctx.total(ll_prop) + pr.eta_logprior(eta_prop))
                 - (ctx.total(Yll) + pr.eta_logprior(eta)))
        if mh_accept(log_r, ctx.grng):
            state.alpha0 = a0_prop
            for t in ctx.times:
                qd[t] = qd_prop[t]
                Yll[t] = ll_prop[t]
            ad_alpha.record(True, np.array([eta_prop]))
        else:
            ad_alpha.record(False, np.array([eta]))


def update_nu(state, ad_nu, ctx, caches, st) -> None:
    """Matern smoothness nu (global; rebuilds the Z covariance like rho)."""
    Zlp = caches.Zlp
    nu_prop = ad_nu.propose(np.array([state.nu]), ctx.grng)[0]
    if np.isfinite(pr.nu_logprior(nu_prop)):
        nu_old, K_old, Q_old, ld_old = (state.nu, state.K, state.Q, state.Q_logdet)
        state.nu = nu_prop
        state.rebuild_cov()
        zlp_prop = {t: state.Z_logprior(state.Z[:, t]) for t in ctx.times}
        log_r = ((ctx.total(zlp_prop) + pr.nu_logprior(nu_prop))
                 - (ctx.total(Zlp) + pr.nu_logprior(nu_old)))
        if mh_accept(log_r, ctx.grng):
            for t in ctx.times:
                Zlp[t] = zlp_prop[t]
            ad_nu.record(True, np.array([nu_prop]))
        else:
            state.nu, state.K, state.Q, state.Q_logdet = (
                nu_old, K_old, Q_old, ld_old)
            ad_nu.record(False, np.array([state.nu]))
    else:
        ad_nu.record(False, np.array([state.nu]))


def update_nu_rho_joint(state, ad, ctx, caches, st) -> None:
    """Joint adaptive-RWMH on (nu, rho) for scalar-rho models (HW / Wang), on the
    CENTERED conditional p(nu, rho | Z) ∝ p(nu) p(rho) ∏_t N(Z_t; 0, K(nu,rho)).

    Z is held fixed, so -- exactly like update_nu / update_rho -- there is NO data
    likelihood; the move is accepted on the Gaussian Z-prior ratio alone. The single
    2x2 (empirical-covariance-adapted) proposal walks the strongly correlated
    nu/rho ridge that the one-at-a-time blocks crawl along. Replaces both of those
    blocks when ``_use_joint_nu_rho`` holds. Note this targets the nu<->rho
    correlation, NOT the Z<->(nu,rho) coupling (that needs the NCP/interweaving half).
    """
    Zlp = caches.Zlp
    nu_old = state.nu
    rho_k_old = state.rho_k                       # length-1 array (scalar rho)
    theta_cur = np.array([nu_old, float(rho_k_old[0])])
    nu_prop, rho_prop = ad.propose(theta_cur, ctx.grng)

    lp_prop = pr.nu_logprior(nu_prop) + pr.rho_logprior(np.array([rho_prop]), scale=state.rho_prior_scale)
    if rho_prop > 1e-3 and np.isfinite(lp_prop):
        rho_vec_old, K_old, Q_old, ld_old = (state.rho_vec, state.K, state.Q,
                                             state.Q_logdet)
        state.nu = nu_prop
        state.rho_k = np.array([rho_prop])
        state.rho_vec = state.W_gauss_rho @ state.rho_k
        state.rebuild_cov()
        zlp_prop = {t: state.Z_logprior(state.Z[:, t]) for t in ctx.times}
        lp_cur = pr.nu_logprior(nu_old) + pr.rho_logprior(rho_k_old, scale=state.rho_prior_scale)
        log_r = (ctx.total(zlp_prop) + lp_prop) - (ctx.total(Zlp) + lp_cur)
        if mh_accept(log_r, ctx.grng):
            for t in ctx.times:
                Zlp[t] = zlp_prop[t]
            ad.record(True, np.array([nu_prop, rho_prop]))
            return
        state.nu = nu_old                         # reject: restore params + cov
        state.rho_k = rho_k_old
        state.rho_vec, state.K, state.Q, state.Q_logdet = (
            rho_vec_old, K_old, Q_old, ld_old)
    ad.record(False, theta_cur)


def update_nu_rho_ncp(state, ctx, caches, ad) -> None:
    """ASIS NON-CENTERED (ancillary) half of the joint (nu, rho) update for scalar-rho
    models. Whitens the ancillary coords ``eta_t = L(theta)^{-1} Z_t``, proposes
    ``theta'``, and forms ``Z'_t = L(theta') eta_t`` so the WHOLE field co-adapts with
    theta in one move; accepts on the DATA-likelihood ratio (eta's N(0,I) prior is
    theta-free and cancels -- no Z-prior term here, unlike the CP half).

    This is the half that breaks the Z<->theta coupling: under the centered move
    p(theta|Z) is razor-sharp (Nt factors) so theta crawls even with a ridge-aligned
    proposal (the warm-HW A/B confirmed the CP-only joint block does not help); here
    the field follows theta, so theta can travel. Interwoven with update_nu_rho_joint
    (the CP half) once per sweep => ASIS (Yu & Meng 2011). qd is Z- and theta-
    independent, so the cached transforms are reused. MPI-safe: operates per owned
    column (ctx.times) and reduces the accept ratio via ctx.total, so every rank takes
    the same (synchronized-RNG) decision, exactly like update_rho/update_nu."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    L_old = state.zchol()
    eta = {t: solve_triangular(L_old, state.Z[:, t], lower=state._zfac_lower) for t in ctx.times}

    nu_old = state.nu
    rho_k_old = state.rho_k                        # length-1 array (scalar rho)
    theta_cur = np.array([nu_old, float(rho_k_old[0])])
    nu_prop, rho_prop = ad.propose(theta_cur, ctx.grng)

    lp_prop = pr.nu_logprior(nu_prop) + pr.rho_logprior(np.array([rho_prop]), scale=state.rho_prior_scale)
    if rho_prop > 1e-3 and np.isfinite(lp_prop):
        rho_vec_old, K_old, Q_old, ld_old = (state.rho_vec, state.K, state.Q,
                                             state.Q_logdet)
        state.nu = nu_prop
        state.rho_k = np.array([rho_prop])
        state.rho_vec = state.W_gauss_rho @ state.rho_k
        state.rebuild_cov()                        # invalidates _Zchol
        Lp = state.zchol()                         # chol at theta' (now cached)
        z_prop = {t: Lp @ eta[t] for t in ctx.times}
        yll_prop = {t: state.loglik_t(t, Z=z_prop[t], qv=qd[t][0], dv=qd[t][1])
                    for t in ctx.times}
        lp_cur = pr.nu_logprior(nu_old) + pr.rho_logprior(rho_k_old, scale=state.rho_prior_scale)
        log_r = (ctx.total(yll_prop) + lp_prop) - (ctx.total(Yll) + lp_cur)
        if mh_accept(log_r, ctx.grng):
            for t in ctx.times:
                state.Z[:, t] = z_prop[t]
                Yll[t] = yll_prop[t]
                Zlp[t] = state.Z_logprior(z_prop[t])
            ad.record(True, np.array([nu_prop, rho_prop]))
            return
        state.nu = nu_old                          # reject: restore params + cov + chol
        state.rho_k = rho_k_old
        state.rho_vec, state.K, state.Q, state.Q_logdet = (
            rho_vec_old, K_old, Q_old, ld_old)
        state._Zchol = L_old                       # zchol() cached Lp; restore old L
    ad.record(False, theta_cur)


def update_nu_rho_ncp_vec(state, ad_list, ctx, caches, st) -> None:
    """ASIS non-centered half for nu and VECTOR rho_k (Shi) -- generalises
    update_nu_rho_ncp. Whitens eta_t = L(nu,rho)^-1 Z_t ONCE: eta is invariant under
    the NCP transform (Z' = L' eta => L'^-1 Z' = eta), so it stays valid across
    blocks. For each parameter block (index 0 == nu, 1..k_rho == rho knots; "nu"
    mode = just [nu], "joint" mode = [nu, rho_k...] split <= phi_block_max) it
    proposes the subset, rebuilds L(nu',rho'), forms Z'_t = L' eta_t, and accepts on
    the DATA-likelihood ratio so the field follows theta (breaks the Z<->theta
    coupling the CP-only update_nu/update_rho cannot). Those component CP moves still
    run this sweep -- this is the complementary NCP half (ASIS, Yu & Meng 2011).
    MPI-safe: grng proposals + ctx.total reduction (every rank takes the same
    decision), exactly like update_nu_rho_ncp."""
    Yll, Zlp, qd = caches.Yll, caches.Zlp, caches.qd
    L_cur = state.zchol()
    eta = {t: solve_triangular(L_cur, state.Z[:, t], lower=state._zfac_lower) for t in ctx.times}
    for ad, idx in zip(ad_list, _ncp_vec_param_blocks(state, st)):
        p_cur = np.concatenate(([state.nu], state.rho_k))   # [nu, rho_k...]
        p_blk = ad.propose(p_cur[idx], ctx.grng)
        nu_prop = float(p_blk[0]) if idx[0] == 0 else state.nu
        rho_k_prop = state.rho_k.copy()
        for pos, j in enumerate(idx):
            if j >= 1:
                rho_k_prop[j - 1] = p_blk[pos]
        lp_prop = pr.nu_logprior(nu_prop) + pr.rho_logprior(rho_k_prop, scale=state.rho_prior_scale)
        if not (np.all(rho_k_prop > 1e-3) and np.isfinite(lp_prop)):
            ad.record(False, p_cur[idx])
            continue
        nu_old, rho_k_old = state.nu, state.rho_k
        L_before = state.zchol()                            # current chol (prior blocks may have moved it)
        rho_vec_old, K_old, Q_old, ld_old = (state.rho_vec, state.K, state.Q,
                                             state.Q_logdet)
        state.nu = nu_prop
        state.rho_k = rho_k_prop
        state.rho_vec = state.W_gauss_rho @ rho_k_prop
        state.rebuild_cov()                                 # invalidates _Zchol
        Lp = state.zchol()                                  # chol at theta'
        z_prop = {t: Lp @ eta[t] for t in ctx.times}
        yll_prop = {t: state.loglik_t(t, Z=z_prop[t], qv=qd[t][0], dv=qd[t][1])
                    for t in ctx.times}
        lp_cur = pr.nu_logprior(nu_old) + pr.rho_logprior(rho_k_old, scale=state.rho_prior_scale)
        log_r = (ctx.total(yll_prop) + lp_prop) - (ctx.total(Yll) + lp_cur)
        if mh_accept(log_r, ctx.grng):
            for t in ctx.times:
                state.Z[:, t] = z_prop[t]
                Yll[t] = yll_prop[t]
                Zlp[t] = state.Z_logprior(z_prop[t])
            ad.record(True, p_blk)
        else:
            state.nu = nu_old                               # reject: restore params + cov + chol
            state.rho_k = rho_k_old
            state.rho_vec, state.K, state.Q, state.Q_logdet = (
                rho_vec_old, K_old, Q_old, ld_old)
            state._Zchol = L_before
            ad.record(False, p_cur[idx])


def _update_wendland_global(state, ctx, caches, W_prop, lp_prop, lp_cur, st):
    """Shared accept/reject for a global move that rebuilds W_wend_S (radius_S
    and anisotropy): recompute gamma_bar, R and the full Y-likelihood, then
    commit W_wend_S/gamma_bar/R on accept. Returns whether the move was accepted
    (the caller commits the specific parameter it proposed)."""
    Yll, qd = caches.Yll, caches.qd
    gbar_prop = kernels.gamma_bar_from_weights(
        W_prop, state.gamma_k, alpha=st.alpha_stable)
    R_prop = W_prop @ state.S
    qvc = state.qd_RW_cens(gamma_bar=gbar_prop)
    qd_prop = {}
    ll_prop = {}
    for t in ctx.times:
        qd_prop[t] = state.qd_RW_one(t, qvc, gamma_bar=gbar_prop, qe_guess=qd[t][0])
        ll_prop[t] = state.loglik_t(t, R=R_prop[:, t], gamma_bar=gbar_prop,
                                    qv=qd_prop[t][0], dv=qd_prop[t][1])
    log_r = (ctx.total(ll_prop) + lp_prop) - (ctx.total(Yll) + lp_cur)
    if mh_accept(log_r, ctx.grng):
        state.W_wend_S = W_prop
        state.gamma_bar = gbar_prop
        state.R = R_prop
        for t in ctx.times:
            qd[t] = qd_prop[t]
            Yll[t] = ll_prop[t]
        return True
    return False


def update_radius_S(state, ad_rad, ctx, caches, st) -> None:
    """Wendland support radius for R's kernel (global; rebuilds W_wend_S)."""
    r_prop = ad_rad.propose(np.array([state.radius_S]), ctx.grng)[0]
    lp_prop = pr.radius_S_logprior(r_prop)
    # `_min_radius_S` is a cheap isotropic pre-filter; `domain_covered` is the
    # real (anisotropy-aware) guarantee that no site is left uncovered.
    if (np.isfinite(lp_prop) and r_prop > state._min_radius_S
            and state.domain_covered(radius=r_prop)):
        W_prop = state.wendland_for_radius(r_prop)
        if _update_wendland_global(state, ctx, caches, W_prop, lp_prop,
                                   pr.radius_S_logprior(state.radius_S), st):
            state.radius_S = r_prop
            ad_rad.record(True, np.array([r_prop]))
        else:
            ad_rad.record(False, np.array([state.radius_S]))
    else:
        ad_rad.record(False, np.array([state.radius_S]))


def update_anisotropy(state, ad_ar, ad_aa, ctx, caches, st) -> None:
    """Wendland anisotropy (ratio in (0,1) Beta(2,2); angle in [0,pi) uniform)."""
    import scipy.stats as _sps
    # ratio (reject any shape that leaves part of the domain uncovered)
    rat = ad_ar.propose(np.array([state.aniso_ratio]), ctx.grng)[0]
    if 0.0 < rat < 1.0 and state.domain_covered(ratio=rat) and _update_wendland_global(
            state, ctx, caches, state.build_wendland(ratio=rat),
            _sps.beta.logpdf(rat, 2, 2),
            _sps.beta.logpdf(state.aniso_ratio, 2, 2), st):
        state.aniso_ratio = rat
        ad_ar.record(True, np.array([rat]))
    else:
        ad_ar.record(False, np.array([state.aniso_ratio]))
    # angle (periodic, uniform prior cancels)
    ang = float(ad_aa.propose(np.array([state.aniso_angle]), ctx.grng)[0]) % np.pi
    if state.domain_covered(angle=ang) and _update_wendland_global(
            state, ctx, caches,
            state.build_wendland(angle=ang), 0.0, 0.0, st):
        state.aniso_angle = ang
        ad_aa.record(True, np.array([ang]))
    else:
        ad_aa.record(False, np.array([state.aniso_angle]))


def update_geometry_joint(state, ad_geom, ctx, caches, st) -> None:
    """One joint adaptive-RWMH move on the Wendland geometry, in COVERAGE-RESPECTING
    coordinates. The hard constraint is ``radius > r_min(ratio, angle)`` (every site
    must keep a covering knot); proposing (radius, ratio, angle) directly wastes most
    moves against that boundary, which runs PARALLEL to the strong radius<->ratio
    posterior ridge. Instead we sample the unconstrained triple

        u = ( log(radius - r_min(ratio, angle)),  logit(ratio),  angle )

    so every proposal is covered by construction (radius = r_min + exp(u0) > r_min)
    and the ratio lands in (0,1) -- the only rejections left are the half-t radius
    prior, the Beta(2,2) ratio prior, and the MH ratio itself. Angle is periodic on
    [0,pi) (uniform prior cancels). The change-of-variables Jacobian is triangular,
    so r_min's dependence on (ratio, angle) drops out of the determinant:

        log|J| = log(radius - r_min) + log(ratio) + log(1 - ratio)

    (the angle term is 1). Reduces to a radius-only slack move when anisotropy is off
    (dim 1, no ratio terms). Costs ONE W_wend_S/R/Y-likelihood rebuild. r_min uses the
    cached visibility mask (no shapely in the per-proposal path, incl. Wang)."""
    from scipy.special import expit, logit
    import scipy.stats as _sps
    has_aniso = state.update_aniso

    # --- current state -> unconstrained u (slack clamped only for a degenerate,
    #     on-boundary current that should never arise from a covered init) ----------
    rmin_c = state.covering_radius(state.aniso_ratio, state.aniso_angle)
    slack_c = max(state.radius_S - rmin_c, 1e-12)
    u_cur = np.asarray([np.log(slack_c)] + (
        [logit(state.aniso_ratio), state.aniso_angle] if has_aniso else []))

    # --- propose in u-space, invert to (radius, ratio, angle) -----------------------
    u_p = ad_geom.propose(u_cur, ctx.grng)
    rat_p = float(expit(u_p[1])) if has_aniso else state.aniso_ratio
    ang_p = (float(u_p[2]) % np.pi) if has_aniso else state.aniso_angle
    rmin_p = state.covering_radius(rat_p, ang_p)
    slack_p = float(np.exp(u_p[0]))
    r_p = rmin_p + slack_p
    if not (np.isfinite(rmin_p) and slack_p > 0.0):    # uncoverable / underflow -> reject
        ad_geom.record(False, u_cur)
        return

    # --- priors + change-of-variables Jacobian --------------------------------------
    lp_prop = pr.radius_S_logprior(r_p) + np.log(slack_p)
    lp_cur = pr.radius_S_logprior(state.radius_S) + np.log(slack_c)
    if has_aniso:
        lp_prop += _sps.beta.logpdf(rat_p, 2, 2) + np.log(rat_p) + np.log1p(-rat_p)
        lp_cur += (_sps.beta.logpdf(state.aniso_ratio, 2, 2)
                   + np.log(state.aniso_ratio) + np.log1p(-state.aniso_ratio))
    if not np.isfinite(lp_prop):
        ad_geom.record(False, u_cur)
        return

    # --- single global rebuild + accept/reject (coverage guaranteed by r_p > r_min) -
    W_prop = state.build_wendland(radius=r_p, ratio=rat_p, angle=ang_p)
    if _update_wendland_global(state, ctx, caches, W_prop, lp_prop, lp_cur, st):
        state.radius_S = r_p
        if has_aniso:
            state.aniso_ratio = rat_p
            state.aniso_angle = ang_p
        # record the WRAPPED angle so the adaptation coordinate stays in [0,pi)
        u_acc = np.array([u_p[0], u_p[1], ang_p]) if has_aniso else np.array([u_p[0]])
        ad_geom.record(True, u_acc)
    else:
        ad_geom.record(False, u_cur)


def _beta_value_grad(theta, state, ctx, st, qvc):
    """``(logp, grad)`` of the JOINT GPD-regression target over theta =
    [beta_sigma, beta_xi], for a gradient (NUTS) update of the marginal blocks.

    Target: sum_t loglik_t(beta) + N(0, sd^2) priors on the coefficients. Only
    exceedances (Y>u) depend on beta -- the censored transform X=qRW(p_cens) is
    beta-independent -- so the gradient is a sum over exceedance cells. The chain is
    beta -> eta -> (sigma, xi) [link] -> p=pCGP(Y) -> X=qRW(p), plus the direct GPD
    density term. dX/dp = 1/dRW(X) (dRW already computed for the value); the one
    non-elementary piece d/dX log dRW(X) uses the exact closed-form `dlog_dRW_dx`
    (analytic-density derivative -- smooth+accurate where the density is reliable,
    see rw_inte). `qvc` is the beta-independent censored qRW (state.qd_RW_cens()).

    MPI-safe: the per-time loglik and the likelihood gradient are reduced across
    ranks (ctx.total / ctx.areduce); the Gaussian priors are added once."""
    mg = state._mg
    ns = state.sigma_design.n_params
    b_sig, b_xi = theta[:ns], theta[ns:]
    sig_mat = mg.sigma_from_eta(state.sigma_design.eta(b_sig))
    xi_mat = mg.xi_from_eta(state.xi_design.eta(b_xi))
    if not state.gpd_support_ok(sig_mat, xi_mat):
        return -np.inf, np.zeros_like(theta)
    a0, pc, uu = state.alpha0, state.data.p_cens, state.data.u_vec
    Xsp, Xxp = state.sigma_design.spatial, state.xi_design.spatial
    LO, HI = mg.XI_LO, mg.XI_HI
    g_lik = np.zeros_like(theta)                       # likelihood grad (reduced)
    ll = {}
    bad_by_t = {}
    for t in ctx.times:
        pYc = u.pCGP(state.Yfilled[:, t], pc, uu, sig_mat[:, t], xi_mat[:, t])
        e = state._exc_list[t]
        # Boundary guard: a leapfrog step can push a deep-tail exceedance to
        # pCGP -> 1 (GPD saturation), where qRW/dRW are undefined. Flag it and skip
        # the transform here; a single post-loop collective (so all ranks agree)
        # then returns -inf (outside support) for NUTS to reject/diverge.
        bad = (not np.all(np.isfinite(pYc))) or (e.size and np.any(pYc[e] >= 1.0))
        bad_by_t[t] = 1.0 if bad else 0.0
        if bad:
            ll[t] = 0.0
            continue
        q, d = state.qd_RW_one(t, qvc, pY_col=pYc)
        ll[t] = state.loglik_t(t, qv=q, dv=d, sigma_col=sig_mat[:, t], xi_col=xi_mat[:, t])
        if e.size == 0:
            continue
        Y_e, u_e, s_e, x_e = state.Yfilled[e, t], uu[e], sig_mat[e, t], xi_mat[e, t]
        X_e, dX_e = q[e], d[e]
        Xstar = (state.R[e, t] ** state.phi_vec[e]) * u.g(state.Z[e, t])
        r = X_e / Xstar
        z = (Y_e - u_e) / s_e
        w = 1.0 + x_e * z
        gpdf = (1.0 / s_e) * w ** (-1.0 / x_e - 1.0)                 # dGP pdf
        A = w ** (-1.0 / x_e)
        dpGP_ds = -z * gpdf                                          # d pGP / d sigma
        dpGP_dx = A * (z / (x_e * w) - np.log(w) / x_e ** 2)         # d pGP / d xi
        dlogdGP_ds = -1.0 / s_e + z * (1.0 + x_e) / (s_e * w)        # d log dGP / d sigma
        dlogdGP_dx = np.log(w) / x_e ** 2 - (1.0 / x_e + 1.0) * z / w
        # d/dX [ log f_eps(X/X*) - log dRW(X) ]; the log-Laplace part is elementary
        # (X* cancels), the dRW part is the exact analytic-density log-derivative.
        dfeps_dX = np.where(r < 1.0, (a0 - 1.0) / X_e, -(a0 + 1.0) / X_e)
        bracket = dfeps_dX - dlog_dRW_dx(X_e, state.phi_vec[e], state.gamma_bar[e], a0)
        inv = (1.0 - pc) / dX_e                                      # dX/dp * (1-pc)/... : dp/d(.)=(1-pc)*dpGP
        dL_ds = bracket * inv * dpGP_ds + dlogdGP_ds                 # dL / d sigma
        dL_dx = bracket * inv * dpGP_dx + dlogdGP_dx                 # dL / d xi
        dL_deta_s = dL_ds * s_e                                      # d sigma / d eta = sigma
        sg = (x_e - LO) / (HI - LO)
        dL_deta_x = dL_dx * (HI - LO) * sg * (1.0 - sg)             # d xi / d eta
        g_lik[0] += dL_deta_s.sum()
        g_lik[1:ns] += Xsp[e].T @ dL_deta_s
        g_lik[ns] += dL_deta_x.sum()
        g_lik[ns + 1:] += Xxp[e].T @ dL_deta_x
    # single collective (identical on every rank) -> unanimous boundary decision
    if ctx.total(bad_by_t) > 0.0:
        return -np.inf, np.zeros_like(theta)
    g_lik = ctx.areduce(g_lik)                          # sum likelihood grad over ranks
    logp = ctx.total(ll)
    # Gaussian priors (added once, after the reduction)
    logp += (-0.5 * (b_sig / st.sigma_beta_sd) ** 2 - np.log(st.sigma_beta_sd)).sum()
    logp += (-0.5 * (b_xi / st.xi_beta_sd) ** 2 - np.log(st.xi_beta_sd)).sum()
    logp += -(b_sig.size + b_xi.size) * 0.5 * np.log(2 * np.pi)
    g_prior = np.concatenate([-b_sig / st.sigma_beta_sd ** 2, -b_xi / st.xi_beta_sd ** 2])
    return logp, g_lik + g_prior


def _beta_try(state, which, beta_prop, ad, ctx, caches, st, rec=None) -> None:
    """MH-evaluate a proposed GPD-margin coefficient vector `beta_prop` for margin
    `which` and accept/reject, committing the per-time caches on acceptance.

    Shared by the JOINT (update_beta) and COMPONENT-wise (update_beta_component)
    RWMH paths -- only the proposal differs. The full-vector Gaussian prior is
    summed at both endpoints; for a 1-D component move the unchanged coefficients
    cancel in the ratio, so it is still exactly correct.

    `rec`: coordinate index for a component (1-D) move -> record ONLY that coord to
    the 1-D adapter, so its window/cov stays 1-D and Sigma_0 actually adapts (a
    full-vector record makes cov_est (d,d) != (1,1) and the shape update is skipped).
    None -> record the full vector (joint block)."""
    import scipy.stats
    def _rec(accepted, vec):
        ad.record(accepted, vec if rec is None else vec[rec:rec + 1])
    Yll, qd = caches.Yll, caches.qd
    if which == "sigma":
        beta = state.beta_sigma
        design = state.sigma_design
        beta_sd = st.sigma_beta_sd
        mk_mat = lambda b: state._mg.sigma_from_eta(design.eta(b))
    else:
        beta = state.beta_xi
        design = state.xi_design
        beta_sd = st.xi_beta_sd
        mk_mat = lambda b: state._mg.xi_from_eta(design.eta(b))
    mat_prop = mk_mat(beta_prop)
    sig_prop = mat_prop if which == "sigma" else state.sigma_mat
    xi_prop = mat_prop if which == "xi" else state.xi_mat
    if not (state.gpd_support_ok(sig_prop, xi_prop)
            and state.gpd_margin_evaluable(sig_prop, xi_prop)):
        _rec(False, beta)
        return
    qvc = state.qd_RW_cens()
    qd_prop, ll_prop, pYcols = {}, {}, {}
    for t in ctx.times:
        pYc = u.pCGP(state.Yfilled[:, t], state.data.p_cens, state.data.u_vec,
                     sig_prop[:, t], xi_prop[:, t])
        q, d = state.qd_RW_one(t, qvc, pY_col=pYc, qe_guess=qd[t][0])
        qd_prop[t] = (q, d)
        pYcols[t] = pYc
        ll_prop[t] = state.loglik_t(t, qv=q, dv=d,
                                    sigma_col=sig_prop[:, t], xi_col=xi_prop[:, t])
    lp_p = np.sum(scipy.stats.norm.logpdf(beta_prop, 0.0, beta_sd))
    lp_c = np.sum(scipy.stats.norm.logpdf(beta, 0.0, beta_sd))
    if mh_accept((ctx.total(ll_prop) + lp_p) - (ctx.total(Yll) + lp_c), ctx.grng):
        if which == "sigma":
            state.beta_sigma = beta_prop
            state.sigma_mat = mat_prop
        else:
            state.beta_xi = beta_prop
            state.xi_mat = mat_prop
        for t in ctx.times:
            qd[t] = qd_prop[t]
            Yll[t] = ll_prop[t]
            state._pY[:, t] = pYcols[t]
        if ctx.covers_all_times:   # whole pY refreshed -> keep _pY_exc consistent
            state._pY_exc = state._pY[state._exc_i, state._exc_t]
        _rec(True, beta_prop)
    else:
        _rec(False, beta)


def update_beta(state, which, ad, ctx, caches, st) -> None:
    """GPD marginal regression coefficients (global), JOINT RWMH over all coefs.
    Only the exceedance likelihood changes (censored qRW is pY-independent)."""
    beta = state.beta_sigma if which == "sigma" else state.beta_xi
    _beta_try(state, which, ad.propose(beta, ctx.grng), ad, ctx, caches, st)


def update_beta_component(state, which, ad_list, ctx, caches, st) -> None:
    """Component-wise RWMH for a GPD-regression margin: a 1-D adaptive per
    coefficient, swept in order, each proposing a change to only its own coef.

    Each coef self-tunes its own step, sidestepping the joint block's failure to
    learn an anisotropic proposal from near-zero acceptance (measured: the joint
    sigma block pinned at ~0.6% accept, freezing the spatial coefs, while the
    adapter collapsed to an isotropic tiny step). Costs one full exceedance-
    likelihood eval per coefficient (~d x the joint block); the fallback until the
    gradient NUTS-beta block is validated. See [[adaptive-rwmh-scale-degeneracy]]."""
    for j, adj in enumerate(ad_list):
        beta = state.beta_sigma if which == "sigma" else state.beta_xi
        beta_prop = beta.copy()
        beta_prop[j] = adj.propose(beta[j:j + 1], ctx.grng)[0]
        _beta_try(state, which, beta_prop, adj, ctx, caches, st, rec=j)


def update_beta_nuts(state, ctx, caches, st) -> None:
    """JOINT NUTS update of the GPD-regression coefficients theta =
    [beta_sigma, beta_xi] on the marginal data likelihood -- replaces the two
    per-margin RWMH ``update_beta`` blocks with one gradient trajectory over the
    correlated sigma/xi coefficient ridge. Gradient: `_beta_value_grad` (analytic
    through the link/design + GPD density; the RW-transform piece uses the exact
    closed-form analytic-density log-derivative rw_inte.dlog_dRW_dx). Step adapter
    on ``state._nuts_beta`` (re-warms on resume). Only the exceedance likelihood
    moves, so on accept we rebuild sigma/xi and refresh qd/Yll/_pY exactly as
    ``update_beta`` does. MPI-safe: `_beta_value_grad` reduces the per-time loglik
    and gradient via ctx.total / ctx.areduce (shared grng keeps ranks in lock-step)."""
    Yll, qd = caches.Yll, caches.qd
    ns = state.sigma_design.n_params
    qvc = state.qd_RW_cens()                        # censored qRW (beta-independent)

    def vg(theta):
        return _beta_value_grad(theta, state, ctx, st, qvc)

    theta0 = np.concatenate([state.beta_sigma, state.beta_xi])
    nb = getattr(state, "_nuts_beta", None)
    if nb is None:
        warmup = (st.nuts_warmup if getattr(st, "nuts_warmup", None) is not None
                  else max(1, st.n_iters // 2))
        adapt = getattr(st, "nuts_adapt_mass", None)      # None -> per-block default
        nb = state._nuts_beta = NUTSWarmup(
            vg, theta0, ctx.grng, warmup, target_accept=st.nuts_target_accept,
            adapt_mass=(True if adapt is None else adapt))     # beta: on (validated win)

    theta1, acc, depth = nuts_transition(vg, theta0, nb.eps, ctx.grng,
                                         inv_mass=nb.inv_mass)
    if not np.array_equal(theta1, theta0):
        state.beta_sigma = theta1[:ns]
        state.beta_xi = theta1[ns:]
        state.sigma_mat = state._mg.sigma_from_eta(state.sigma_design.eta(state.beta_sigma))
        state.xi_mat = state._mg.xi_from_eta(state.xi_design.eta(state.beta_xi))
        for t in ctx.times:
            pYc = u.pCGP(state.Yfilled[:, t], state.data.p_cens, state.data.u_vec,
                         state.sigma_mat[:, t], state.xi_mat[:, t])
            q, d = state.qd_RW_one(t, qvc, pY_col=pYc)
            qd[t] = (q, d)
            Yll[t] = state.loglik_t(t, qv=q, dv=d,
                                    sigma_col=state.sigma_mat[:, t], xi_col=state.xi_mat[:, t])
            state._pY[:, t] = pYc
        if ctx.covers_all_times:      # whole pY refreshed -> keep _pY_exc consistent
            state._pY_exc = state._pY[state._exc_i, state._exc_t]

    nb.update(theta1, acc, vg, ctx.grng)                 # eps + metric warmup schedule


def update_beta_barker(state, ctx, caches, st) -> None:
    """JOINT Barker-proposal MALA update of the GPD-regression coefficients theta =
    [beta_sigma, beta_xi] -- a robust one-gradient move over the correlated sigma/xi
    ridge (Livingstone & Zanella 2022; Vogrinc 2023 optimal noise), an alternative to
    the joint NUTS (``update_beta_nuts``) and the per-margin RWMH blocks. Requires BOTH
    margins active (one joint trajectory). Same gradient (`_beta_value_grad`, MPI-safe)
    and the same post-move cache rebuild as ``update_beta_nuts``.

    Unlike NUTS, Barker takes ONE proposal per sweep, so it can't hit max_depth on the
    ill-conditioned ridge; the diagonal preconditioner + skew-adaptive sign flip carry
    the anisotropy. Because the block's target shifts every sweep (Z/phi/alpha0 move),
    the current-point gradient is re-evaluated each call (2 grad evals/iter) -- still far
    cheaper than component RWMH's one likelihood eval per coefficient."""
    Yll, qd = caches.Yll, caches.qd
    ns = state.sigma_design.n_params
    qvc = state.qd_RW_cens()                         # censored qRW (beta-independent)

    def vg(theta):
        return _beta_value_grad(theta, state, ctx, st, qvc)

    theta0 = np.concatenate([state.beta_sigma, state.beta_xi])
    bk = getattr(state, "_barker_beta", None)
    if bk is None:
        warmup = (st.nuts_warmup if getattr(st, "nuts_warmup", None) is not None
                  else max(1, st.n_iters // 2))
        # Warm-seed the diagonal preconditioner from the block's proposal covariances
        # (the SAME beta_*_proposal_cov the rwmh arm uses) when supplied -- their
        # per-coef sd fixes the sigma/xi scale spread that a cold single-step warmup
        # can't self-estimate. Both covs required; otherwise self-estimate (None).
        cs = getattr(st, "beta_sigma_proposal_cov", None)
        cx = getattr(st, "beta_xi_proposal_cov", None)
        pre = None
        if cs is not None and cx is not None:
            pre = np.sqrt(np.concatenate([np.diag(np.atleast_2d(cs)),
                                          np.diag(np.atleast_2d(cx))]))
        bk = state._barker_beta = BarkerWarmup(vg, theta0, ctx.grng, warmup,
                                               init_precond_sd=pre)

    theta1, _, _, acc = barker_transition(vg, theta0, bk.eps, bk.precond_sd, ctx.grng)
    if not np.array_equal(theta1, theta0):
        state.beta_sigma = theta1[:ns]
        state.beta_xi = theta1[ns:]
        state.sigma_mat = state._mg.sigma_from_eta(state.sigma_design.eta(state.beta_sigma))
        state.xi_mat = state._mg.xi_from_eta(state.xi_design.eta(state.beta_xi))
        for t in ctx.times:
            pYc = u.pCGP(state.Yfilled[:, t], state.data.p_cens, state.data.u_vec,
                         state.sigma_mat[:, t], state.xi_mat[:, t])
            q, d = state.qd_RW_one(t, qvc, pY_col=pYc)
            qd[t] = (q, d)
            Yll[t] = state.loglik_t(t, qv=q, dv=d,
                                    sigma_col=state.sigma_mat[:, t], xi_col=state.xi_mat[:, t])
            state._pY[:, t] = pYc
        if ctx.covers_all_times:      # whole pY refreshed -> keep _pY_exc consistent
            state._pY_exc = state._pY[state._exc_i, state._exc_t]

    bk.update(theta1, acc, vg, ctx.grng)                 # eps + preconditioner warmup


def run_global_updates(state, ad, ctx, caches, st, config) -> None:
    """Run every *global*-parameter block in the canonical order shared by both
    backends. Latent (S, Z) blocks are driven separately by each backend because
    only those differ in their RNG/locality handling beyond what Ctx captures."""
    interweave_nr = _use_interweave_nu_rho(state, st)
    joint_nr = _use_joint_nu_rho(state, st) or interweave_nr
    # joint (nu, rho) gradient NUTS on the CP conditional (replaces the separate
    # nu + rho blocks). SHI ONLY: it builds the dense nonstationary ns_cov Matern,
    # so it is invalid for HW (stationary matern_wang) and Wang (visgp precision);
    # HW is already excluded by the joint_nr precedence, Wang by use_visgp here.
    joint_nr_nuts = (getattr(st, "nu_rho_sampler", "separate") == "nuts"
                     and st.update_nu and st.update_rho and not joint_nr
                     and not config.use_visgp and config.model_type != MODEL_HW)
    if st.update_phi:
        _ps = getattr(st, "phi_sampler", "rwmh")
        if _ps == "nuts":
            update_phi_nuts(state, ad["phi"], ctx, caches, st)
        elif _ps == "barker":
            update_phi_barker(state, ad["phi"], ctx, caches, st)
        elif _use_joint_phi(st):
            update_phi_joint(state, ad["phi"], ctx, caches, st)
        else:
            update_phi(state, ad["phi"], ctx, caches, st)
    if joint_nr:
        a = ad.get("nu_rho")                       # lazily create on a flag-on resume
        if a is None:
            a = ad["nu_rho"] = _mk(2, _NU_RHO_INIT_VAR)
        update_nu_rho_joint(state, a, ctx, caches, st)   # ASIS centered (CP) half
        if interweave_nr:
            anc = ad.get("nu_rho_ncp")
            if anc is None:
                anc = ad["nu_rho_ncp"] = _mk(2, _NU_RHO_INIT_VAR)
            update_nu_rho_ncp(state, ctx, caches, anc)   # ASIS non-centered (NCP) half
    elif joint_nr_nuts:
        # joint (nu, rho) gradient NUTS, optionally amortized over nu_rho_nuts_every
        # sweeps (see SamplerSettings.nu_rho_nuts_every). The counter lives on the
        # state and advances in lock-step on every MPI rank, since run_global_updates
        # is called once per sweep everywhere; it is only maintained for k > 1, so the
        # every-sweep default keeps existing runs (and their snapshots) untouched.
        if _nu_rho_nuts_due(state, st):
            update_nu_rho_nuts(state, ctx, caches, st)
    elif st.update_rho:
        if getattr(st, "rho_sampler", "rwmh") == "nuts":
            update_rho_nuts(state, ctx, caches, st)
        else:
            update_rho(state, ad["rho"], ctx, caches, st)
    if st.update_alpha0:
        update_alpha0(state, ad["alpha"], ctx, caches, st)
    if st.update_nu and not joint_nr and not joint_nr_nuts:
        update_nu(state, ad["nu"], ctx, caches, st)
    if _use_ncp_vec(state, st):                  # ASIS NCP half for vector rho (Shi)
        av = ad.get("ncp_vec")                   # lazily create on a flag-on resume
        if av is None:
            av = ad["ncp_vec"] = _ncp_vec_adaptives(state, st)
        update_nu_rho_ncp_vec(state, av, ctx, caches, st)
    if _use_joint_geometry(state, st):
        ag = ad.get("geom")                            # lazily create on a flag-on resume
        if ag is None:
            ag = ad["geom"] = _geom_adaptive(state, st)
        update_geometry_joint(state, ag, ctx, caches, st)
    else:
        if st.update_radius_S and config.model_type != MODEL_HW:
            update_radius_S(state, ad["rad"], ctx, caches, st)
        if state.update_aniso:
            update_anisotropy(state, ad["ar"], ad["aa"], ctx, caches, st)
    # JOINT gradient NUTS over [beta_sigma, beta_xi] (replaces both RWMH blocks);
    # requires BOTH margins active (it is one joint trajectory). Falls back to the
    # per-margin RWMH blocks otherwise.
    _bs = getattr(st, "beta_sampler", "rwmh")
    if (_bs == "nuts" and state.update_sigma and state.update_xi):
        update_beta_nuts(state, ctx, caches, st)
    elif (_bs == "barker" and state.update_sigma and state.update_xi):
        update_beta_barker(state, ctx, caches, st)
    elif (_bs == "recursive" and state.update_sigma and state.update_xi):
        # Stage-2 two-stage recursive Bayes (scaffold; opt-in, off by default).
        from recursive_bayes import update_beta_recursive
        update_beta_recursive(state, ctx, caches, st)
    elif _bs == "component":
        if state.update_sigma:
            update_beta_component(state, "sigma", ad["bsig"], ctx, caches, st)
        if state.update_xi:
            update_beta_component(state, "xi", ad["bxi"], ctx, caches, st)
    else:
        if state.update_sigma:
            update_beta(state, "sigma", ad["bsig"], ctx, caches, st)
        if state.update_xi:
            update_beta(state, "xi", ad["bxi"], ctx, caches, st)
