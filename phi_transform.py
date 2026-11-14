"""
Tail-exponent (phi / delta) domain handling, shared by all model types.

The tail exponent is phi(s) for the shi/wang models (a smooth field built from
knot values) and a scalar delta for Huser-Wadsworth. Either may live in:

  * "unit"     -> (0, 1)    : Shi's original parameterisation. Knot/scalar
                              values are constrained to (0,1) via a logit map;
                              prior Beta(2,2) on the constrained value.
  * "positive" -> (0, inf)  : Wang's wider domain. Values are exp() of an
                              unconstrained real; prior Normal(0,1) on the log.

The MCMC samples on the *unconstrained* scale in both cases, so the update
block itself is domain-agnostic: it proposes an unconstrained step, maps to the
constrained value to evaluate the likelihood, and adds this module's log-prior
(which already includes the change-of-variables Jacobian for the constrained
prior where relevant).

For the field models (shi/wang) the smoothing is applied on the *knot* values:
  unit:     phi(s) = W_gauss @ phi_k        with phi_k in (0,1)
  positive: phi(s) = exp(W_gauss @ eta_k)   with eta_k in R
"""
from __future__ import annotations

import numpy as np
import scipy.stats

from config import PHI_UNIT, PHI_POSITIVE


# -- scalar / elementwise transforms -----------------------------------------
def to_constrained(u: np.ndarray, phi_domain: str) -> np.ndarray:
    """Map unconstrained real value(s) to the constrained tail exponent."""
    u = np.asarray(u, dtype=float)
    if phi_domain == PHI_UNIT:
        return 1.0 / (1.0 + np.exp(-u))      # logistic -> (0,1)
    elif phi_domain == PHI_POSITIVE:
        return np.exp(u)                     # exp -> (0, inf)
    raise ValueError(f"unknown phi_domain {phi_domain!r}")


def to_unconstrained(c: np.ndarray, phi_domain: str) -> np.ndarray:
    """Inverse of to_constrained."""
    c = np.asarray(c, dtype=float)
    if phi_domain == PHI_UNIT:
        return np.log(c) - np.log1p(-c)      # logit
    elif phi_domain == PHI_POSITIVE:
        return np.log(c)
    raise ValueError(f"unknown phi_domain {phi_domain!r}")


def logprior_unconstrained(u: np.ndarray, phi_domain: str,
                           beta_a: float = 2.0, beta_b: float = 2.0,
                           normal_sd: float = 1.0) -> float:
    """Log-prior evaluated on the unconstrained scale (Jacobian included).

    unit:     constrained ~ Beta(beta_a, beta_b); add log|d c/d u| for the
              logistic transform: log c + log(1-c).
    positive: unconstrained ~ Normal(0, normal_sd^2) directly (the prior is
              defined on the log scale, so no extra Jacobian).
    """
    u = np.asarray(u, dtype=float)
    if phi_domain == PHI_UNIT:
        c = to_constrained(u, phi_domain)
        lp = scipy.stats.beta.logpdf(c, a=beta_a, b=beta_b)
        log_jac = np.log(c) + np.log1p(-c)   # d c/d u = c (1-c)
        return float(np.sum(lp + log_jac))
    elif phi_domain == PHI_POSITIVE:
        return float(np.sum(scipy.stats.norm.logpdf(u, loc=0.0, scale=normal_sd)))
    raise ValueError(f"unknown phi_domain {phi_domain!r}")


def field_from_knots(W_gauss: np.ndarray, knot_unconstrained: np.ndarray,
                     phi_domain: str) -> np.ndarray:
    """Smooth tail-exponent field phi(s) from unconstrained knot values.

    unit:     phi(s) = sigmoid( W_gauss @ eta_k )  -- but Shi historically
              smooths the *constrained* knot values, so for exact
              compatibility we smooth the constrained values when unit.
    positive: phi(s) = exp( W_gauss @ eta_k ).
    """
    if phi_domain == PHI_UNIT:
        # smooth constrained knot values (Shi-compatible)
        phi_k = to_constrained(knot_unconstrained, phi_domain)
        return W_gauss @ phi_k
    elif phi_domain == PHI_POSITIVE:
        return np.exp(W_gauss @ knot_unconstrained)
    raise ValueError(f"unknown phi_domain {phi_domain!r}")
