"""
Stage-2 two-stage recursive Bayes for the spatial-extremes model.

Stage 1 (marginal_model.py) fits the GPD margins to the exceedances alone and yields a
posterior sample of the coefficients theta_marg = (beta_sigma, beta_xi). Stage 2 fits the
FULL dependence model and uses that stage-1 posterior as an INDEPENDENCE PROPOSAL for the
beta block -- the Proposal-Recursive-Bayes idea of Lunn et al. (2013) / Hooten, Johnson &
Brost (2021, TAS 75(2):185-194).

## The acceptance ratio (exact -- earlier notes claiming "no cancellation" were wrong)

Let (writing beta for (beta_sigma, beta_xi), theta for the dependence params Z/S/phi/...):
  * L_full(beta)  = the full RW dependence likelihood  = sum_t loglik_t   (expensive)
  * L_marg(beta)  = the stage-1 marginal GPD likelihood = sum_{Y>u} dGP   (cheap)
  * p2(beta)      = the STAGE-2 coefficient prior (SamplerSettings.*_beta_sd)
  * p1(beta)      = the STAGE-1 coefficient prior (marginal_fit config *_beta_sd)

The stage-2 full-conditional for beta is  pi(beta) ~ L_full(beta) p2(beta).  We propose
beta* from the stage-1 posterior  q(beta) = [beta | Y_exc]_1 ~ L_marg(beta) p1(beta) / Z1
(realised by drawing, with replacement, a stored stage-1 sample). The independence-MH
ratio is

    r = [pi(beta*) q(beta_cur)] / [pi(beta_cur) q(beta*)]
      = { L_full(beta*) p2(beta*) L_marg(beta_cur) p1(beta_cur) }
        / { L_full(beta_cur) p2(beta_cur) L_marg(beta*) p1(beta*) },

so the intractable stage-1 normaliser Z1 cancels and the ratio is EXACTLY computable:

    log r = [ l_full(beta*)  + logp2(beta*)  - l_full(beta_cur)  - logp2(beta_cur) ]
          - [ l_marg(beta*)  + logp1(beta*)  - l_marg(beta_cur)  - logp1(beta_cur) ].

If p1 == p2 the prior terms cancel; we keep them general so a stage-2 prior different from
stage 1's is still exact. Combined with the sampler's usual theta updates this targets the
correct joint posterior [beta, theta | Y].

## What actually fails here (vs Hooten's headline result)

Hooten's PP-RB gets a CHEAP ratio r = [y_j|theta*]/[y_j|theta_cur] because, for a DISJOINT
DATA PARTITION, the first-stage likelihood is an exact factor of the full likelihood and
CANCELS, so the expensive full data model is never evaluated. Our two stages analyze the
SAME data and L_marg is not a factor of L_full (the RW transform entangles margins and
dependence), so nothing cancels away: L_full is still evaluated at every proposal. The
method stays CORRECT (exact independence sampler) but yields NO compute saving -- its only
value is a well-informed beta proposal. Whether that mixes better than sampling beta
through the dependence likelihood directly is the empirical question this enables. The
fixed-margin fallback (freeze beta at the stage-1 mean) remains the conservative option.
"""
from __future__ import annotations

import json
import os

import numpy as np

import utilities as u


def _config_path(npz_path):
    return (npz_path[:-4] if npz_path.endswith(".npz") else npz_path) + ".config.json"


def _normal_logprior(beta_sigma, beta_xi, sd_sigma, sd_xi):
    """log N(0, sd^2) over the two coefficient blocks (constant terms included so the
    stage-1 and stage-2 priors are comparable when their sds differ)."""
    bs, bx = np.asarray(beta_sigma, float), np.asarray(beta_xi, float)
    lp = (-0.5 * (bs / sd_sigma) ** 2 - np.log(sd_sigma)).sum()
    lp += (-0.5 * (bx / sd_xi) ** 2 - np.log(sd_xi)).sum()
    lp += -0.5 * (bs.size + bx.size) * np.log(2 * np.pi)
    return float(lp)


class RecursivePrior:
    """The stage-1 posterior of (beta_sigma, beta_xi), used to drive stage-2 proposals.

    Draws are stored as a single (K, n_params) array theta = [beta_sigma | beta_xi] (the
    same concatenation order marginal_model.fit uses), with the split `n_sigma` and the
    stage-1 coefficient prior sds (needed in the exact M-H ratio) remembered.
    """

    def __init__(self, beta_sigma_draws, beta_xi_draws, sigma_beta_sd=5.0, xi_beta_sd=5.0):
        bs = np.asarray(beta_sigma_draws, float)
        bx = np.asarray(beta_xi_draws, float)
        assert bs.ndim == 2 and bx.ndim == 2 and bs.shape[0] == bx.shape[0], \
            "beta_sigma/beta_xi draws must be (K, .) with matching K"
        assert bs.shape[0] > 0, "no stage-1 draws after burn-in"
        self.n_sigma = bs.shape[1]
        self.n_xi = bx.shape[1]
        self.beta_sigma = bs
        self.beta_xi = bx
        self.K = bs.shape[0]
        self.sigma_beta_sd = float(sigma_beta_sd)
        self.xi_beta_sd = float(xi_beta_sd)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_npz(cls, path, burn=0, sigma_beta_sd=5.0, xi_beta_sd=5.0):
        d = np.load(path, allow_pickle=True)
        return cls(d["beta_sigma"][burn:], d["beta_xi"][burn:],
                   sigma_beta_sd=sigma_beta_sd, xi_beta_sd=xi_beta_sd)

    @classmethod
    def from_marginal_fit(cls, path):
        """Load a marginal_model / faroe_marginal_fit output, dropping the warmup portion
        (warmup // thin recorded draws, from the sibling config json) and picking up the
        stage-1 coefficient prior sds so the M-H ratio's prior terms are exact."""
        d = np.load(path, allow_pickle=True)
        n_draws = int(d["beta_sigma"].shape[0])
        burn, sd_s, sd_x = 0, 5.0, 5.0
        cfgp = _config_path(path)
        if os.path.exists(cfgp):
            with open(cfgp) as fh:
                c = json.load(fh)
            burn = int(c.get("warmup", 0)) // max(1, int(c.get("thin", 1)))
            sd_s = float(c.get("sigma_beta_sd", 5.0))
            sd_x = float(c.get("xi_beta_sd", 5.0))
        burn = min(burn, n_draws // 2)
        return cls(d["beta_sigma"][burn:], d["beta_xi"][burn:],
                   sigma_beta_sd=sd_s, xi_beta_sd=sd_x)

    # -- proposals (Hooten et al. 2021, Appendix A) -------------------------
    def propose(self, rng):
        """One proposal (beta_sigma*, beta_xi*) drawn uniformly at random, WITH
        replacement, from the stored stage-1 posterior sample. Sampling with replacement
        (not sweeping the chain in order) approximates independent draws from the stage-1
        posterior; the stage-1 chain should already be thinned. `rng` MUST be the shared
        global RNG (ctx.grng) so every MPI rank proposes the identical vector."""
        k = int(rng.integers(self.K))
        return self.beta_sigma[k].copy(), self.beta_xi[k].copy()

    def logprior(self, beta_sigma, beta_xi):
        """The STAGE-1 coefficient log prior at (beta_sigma, beta_xi)."""
        return _normal_logprior(beta_sigma, beta_xi, self.sigma_beta_sd, self.xi_beta_sd)


# --------------------------------------------------------------------------- #
#  the stage-2 update                                                          #
# --------------------------------------------------------------------------- #
def _marginal_exc_loglik(state, ctx, sig_mat, xi_mat) -> float:
    """Stage-1 marginal GPD log-likelihood over the exceedance cells, reduced across
    ranks: sum_{(s,t): Y>u} log dGP(Y; u, sigma, xi). Same function stage 1 fit, so
    exp(this) * prior is (up to a constant) the stage-1 posterior density the proposal is
    drawn from -- which is what makes the M-H ratio exact."""
    uu = state.data.u_vec
    d = {}
    for t in ctx.times:
        e = state._exc_list[t]
        if e.size:
            d[t] = float(u._gp_logpdf_np(state.Yfilled[e, t], uu[e],
                                         sig_mat[e, t], xi_mat[e, t]).sum())
        else:
            d[t] = 0.0
    return ctx.total(d)


def _get_recursive_prior(state, st) -> RecursivePrior:
    rp = getattr(state, "_recursive_prior", None)
    if rp is None:
        path = getattr(st, "recursive_prior_npz", None)
        if not path:
            raise ValueError(
                "beta_sampler='recursive' requires settings.recursive_prior_npz "
                "(the stage-1 marginal_fit.npz). Run stage 1 first.")
        rp = state._recursive_prior = RecursivePrior.from_marginal_fit(path)
    return rp


def update_beta_recursive(state, ctx, caches, st) -> None:
    """Stage-2 recursive-Bayes independence update of [beta_sigma, beta_xi].

    Proposes beta* from the stage-1 posterior (RecursivePrior) and accepts against the
    EXACT ratio derived in the module docstring: the full dependence likelihood corrected
    by the stage-1 marginal likelihood (+ the two coefficient priors). On acceptance the
    per-time caches (qd, Yll, _pY, _pY_exc) are committed exactly as update_beta_nuts does.

    MPI-safe: the proposal is drawn on ctx.grng (identical on every rank), the GPD support
    guard is a pure function of the global betas+design (so the decision is unanimous), and
    L_full / L_marg are reduced with ctx.total; the accept draw uses ctx.grng."""
    from mcmc_updates import mh_accept    # local import: dispatch loads us lazily

    prior = _get_recursive_prior(state, st)
    Yll, qd = caches.Yll, caches.qd

    b_sig_cur, b_xi_cur = state.beta_sigma, state.beta_xi
    b_sig_p, b_xi_p = prior.propose(ctx.grng)
    sig_prop = state._mg.sigma_from_eta(state.sigma_design.eta(b_sig_p))
    xi_prop = state._mg.xi_from_eta(state.xi_design.eta(b_xi_p))

    # GPD support / evaluability of the proposal (unanimous across ranks by construction)
    if not (state.gpd_support_ok(sig_prop, xi_prop)
            and state.gpd_margin_evaluable(sig_prop, xi_prop)):
        return

    # full dependence likelihood at the proposal (rebuild pCGP -> qRW -> loglik_t)
    qvc = state.qd_RW_cens()
    qd_prop, ll_prop, pYcols = {}, {}, {}
    for t in ctx.times:
        pYc = u.pCGP(state.Yfilled[:, t], state.data.p_cens, state.data.u_vec,
                     sig_prop[:, t], xi_prop[:, t])
        q, dd = state.qd_RW_one(t, qvc, pY_col=pYc, qe_guess=qd[t][0])
        qd_prop[t] = (q, dd)
        pYcols[t] = pYc
        ll_prop[t] = state.loglik_t(t, qv=q, dv=dd,
                                    sigma_col=sig_prop[:, t], xi_col=xi_prop[:, t])

    lfull_prop = ctx.total(ll_prop)
    lfull_cur = ctx.total(Yll)
    lmarg_prop = _marginal_exc_loglik(state, ctx, sig_prop, xi_prop)
    lmarg_cur = _marginal_exc_loglik(state, ctx, state.sigma_mat, state.xi_mat)
    # stage-2 prior (from settings) and stage-1 prior (from the RecursivePrior)
    lp2_prop = _normal_logprior(b_sig_p, b_xi_p, st.sigma_beta_sd, st.xi_beta_sd)
    lp2_cur = _normal_logprior(b_sig_cur, b_xi_cur, st.sigma_beta_sd, st.xi_beta_sd)
    lp1_prop = prior.logprior(b_sig_p, b_xi_p)
    lp1_cur = prior.logprior(b_sig_cur, b_xi_cur)

    log_r = ((lfull_prop + lp2_prop) - (lfull_cur + lp2_cur)) \
        - ((lmarg_prop + lp1_prop) - (lmarg_cur + lp1_cur))

    if mh_accept(log_r, ctx.grng):
        state.beta_sigma = b_sig_p
        state.beta_xi = b_xi_p
        state.sigma_mat = sig_prop
        state.xi_mat = xi_prop
        for t in ctx.times:
            qd[t] = qd_prop[t]
            Yll[t] = ll_prop[t]
            state._pY[:, t] = pYcols[t]
        if ctx.covers_all_times:
            state._pY_exc = state._pY[state._exc_i, state._exc_t]
