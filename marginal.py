"""
GPD marginal regression for the scale (sigma) and shape (xi) parameters.

Each parameter's linear predictor eta(s,t) is built from an intercept plus three
optional covariate groups (Wang/Shi convention, "option 1": the user supplies the
groups separately and the code assembles the design without ever forming a giant
(Ns*Nt, P) matrix):

    eta(s,t) = beta0
             + spatial[s, :]        @ beta_sp       (varies in space, const in time)
             + temporal[t, :]       @ beta_tp       (varies in time,  const in space)
             + spatiotemporal[s,t,:] @ beta_st       (varies in both)

Transformations (link functions):
    sigma(s,t) = exp(eta_sigma)                         (log link, sigma > 0)
    xi(s,t)    = -0.5 + 1.5 * sigmoid(eta_xi)           (scaled logit -> (-0.5, 1))

The coefficient vector is laid out as
    beta = [beta0, beta_sp(0..p_sp-1), beta_tp(0..p_tp-1), beta_st(0..p_st-1)].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

XI_LO, XI_HI = -0.5, 1.0          # the (open) interval xi is mapped onto


# -- link transforms ----------------------------------------------------------
def sigma_from_eta(eta):
    return np.exp(eta)


def eta_from_sigma(sigma):
    return np.log(sigma)


def xi_from_eta(eta):
    return XI_LO + (XI_HI - XI_LO) / (1.0 + np.exp(-np.asarray(eta, dtype=float)))


def eta_from_xi(xi):
    z = (np.asarray(xi, dtype=float) - XI_LO) / (XI_HI - XI_LO)
    return np.log(z) - np.log1p(-z)


@dataclass
class MarginalDesign:
    """Covariate design for one GPD parameter (sigma or xi).

    spatial        : (Ns, p_sp) or None
    temporal       : (Nt, p_tp) or None
    spatiotemporal : (Ns, Nt, p_st) or None
    An intercept column is always included implicitly (beta[0]).
    """
    Ns: int
    Nt: int
    spatial: Optional[np.ndarray] = None
    temporal: Optional[np.ndarray] = None
    spatiotemporal: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.spatial is not None:
            self.spatial = np.atleast_2d(np.asarray(self.spatial, dtype=float))
            assert self.spatial.shape[0] == self.Ns, "spatial must be (Ns, p_sp)"
        if self.temporal is not None:
            self.temporal = np.atleast_2d(np.asarray(self.temporal, dtype=float))
            assert self.temporal.shape[0] == self.Nt, "temporal must be (Nt, p_tp)"
        if self.spatiotemporal is not None:
            self.spatiotemporal = np.asarray(self.spatiotemporal, dtype=float)
            assert self.spatiotemporal.shape[:2] == (self.Ns, self.Nt), \
                "spatiotemporal must be (Ns, Nt, p_st)"

    @property
    def p_sp(self):
        return 0 if self.spatial is None else self.spatial.shape[1]

    @property
    def p_tp(self):
        return 0 if self.temporal is None else self.temporal.shape[1]

    @property
    def p_st(self):
        return 0 if self.spatiotemporal is None else self.spatiotemporal.shape[2]

    @property
    def n_params(self):
        return 1 + self.p_sp + self.p_tp + self.p_st

    def eta(self, beta) -> np.ndarray:
        """Linear predictor eta(s,t), shape (Ns, Nt). beta has length n_params."""
        beta = np.asarray(beta, dtype=float)
        out = np.full((self.Ns, self.Nt), beta[0])
        i = 1
        if self.p_sp:
            out += (self.spatial @ beta[i:i + self.p_sp])[:, None]; i += self.p_sp
        if self.p_tp:
            out += (self.temporal @ beta[i:i + self.p_tp])[None, :]; i += self.p_tp
        if self.p_st:
            out += np.einsum("stp,p->st", self.spatiotemporal, beta[i:i + self.p_st])
        return out
