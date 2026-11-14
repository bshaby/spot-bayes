"""
Prior log-densities and parameter transforms for the MCMC sampler.

These are the small, self-contained prior/Jacobian pieces shared by every
update block (latent S/Z, the tail exponent phi, the covariance range rho and
smoothness nu, the nugget tail alpha0, and the Wendland support radius). They
depend only on numpy + scipy.stats, so they live apart from the heavier core
math in utilities.py.
"""
from __future__ import annotations

import numpy as np
import scipy.stats


# S ~ Levy(scale=gamma_at_knots); log S Jacobian added
def S_logprior_1t(logS_vec, gamma_at_knots):
    return np.sum(scipy.stats.levy.logpdf(np.exp(logS_vec), scale=gamma_at_knots) + logS_vec)


# Z ~ MVN(0, K) via frozen distribution
def Z_logprior_1t(Z_vec, K_frozen):
    return K_frozen.logpdf(Z_vec)


def phi_logprior(phi_k_vec, a: float = 2.0, b: float = 2.0):
    """Independent Beta(a,b) prior on phi knots in (0,1). Default Beta(2,2)."""
    return np.sum(scipy.stats.beta.logpdf(phi_k_vec, a=a, b=b))


def rho_logprior(rho_k_vec, scale: float = 10.0):
    """Independent half-normal prior on rho knot coefficients (positivity), scale=10."""
    return np.sum(scipy.stats.halfnorm.logpdf(rho_k_vec, loc=0.0, scale=scale))


def logrho_logprior(log_rho_k_vec, scale: float = 10.0):
    return np.sum(scipy.stats.norm.logpdf(log_rho_k_vec, loc=0.0, scale=scale))


# --- Soft-wall reparameterization for alpha0 ---------------------------------
# alpha0 = alpha0_min + exp(eta)  with  eta in R  =>  alpha0 > alpha0_min.

def alpha0_from_eta(eta, alpha0_min=2.0):
    return alpha0_min + float(np.exp(eta))


def eta_from_alpha0(alpha0, alpha0_min=2.0):
    if alpha0 <= alpha0_min:
        raise ValueError("alpha0 must be > alpha0_min for eta=log(alpha0-alpha0_min).")
    return float(np.log(alpha0 - alpha0_min))


def eta_logprior(eta: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Normal(mu, sigma^2) prior on eta. Because the prior is defined on eta (the
    parameter we sample), there is NO Jacobian term in the MH ratio.
    """
    return float(scipy.stats.norm.logpdf(eta, loc=mu, scale=sigma))


def nu_logprior(nu, loc: float = 0.5, scale: float = 1.5):
    # Uniform prior on nu in [loc, loc+scale] = [0.5, 2.0]. The 2.0 upper bound is
    # deliberate: nu > 2 is a field smoother than twice-differentiable (rarely
    # identifiable from data), and for the Wang/visgp model the Gilbert-Datta
    # precision Q = sum_c inv(Matern_c) - sum_s inv(Matern_s) becomes INDEFINITE at
    # high nu & rho (the Matern stops being ~Markov w.r.t. the visibility graph),
    # which crashed the Z-update Cholesky. Capping nu keeps Q PD by construction --
    # no runtime PD check needed. Shared by all three models (the truth is nu=1.0).
    return float(scipy.stats.uniform.logpdf(nu, loc=loc, scale=scale))


def radius_S_logprior(r: float, df: float = 1.0, loc: float = 0.0, scale: float = 3.0):
    """Half-t prior on r>0: log[ 2 * t_pdf(r; df, scale)]."""
    if not np.isfinite(r) or r <= 0.0:
        return -np.inf
    return np.log(2.0) + scipy.stats.t.logpdf(r, df=df, loc=loc, scale=scale)


def sum_logparts(*terms):
    """Fast sum of provided log terms, skipping None."""
    total = 0.0
    for t in terms:
        if t is not None:
            total += float(t)
    return total
