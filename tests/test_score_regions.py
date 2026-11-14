"""Region scoring: can the per-block holdout scores be extracted from ONE pooled pass?

The pooled and per-block passes used to be run separately, which cost as much again as
the pooled run itself (Faroe hw_warm: 41.5 ks pooled + 43.0 ks for three blocks). The
blocks are NOT additive -- the log-score is a joint density and the Monte-Carlo
logsumexp over latent-field draws does not distribute over the site-sum -- but they ARE
extractable, because conditional on Z the sites are independent, so the per-site
log-densities the pooled pass already computes can be summed over any index set.

These tests pin that claim from three directions: the exact per-site decomposition, the
exact agreement of the pooled number with the old code path, and -- the one that
actually matters -- statistical agreement between a block extracted from the pooled pass
and the same block scored on its own.
"""
import numpy as np
import pytest

import predictive as pr
import utilities as u
import simulate as sm
from config import ModelConfig
from rw_inte import qRW, dRW

BLOCKS = {"A": np.arange(0, 3), "B": np.arange(3, 6), "C": np.arange(6, 9)}
N_NEW = 9


def _setup(seed=5, n_draws=6):
    """A small shi problem with 9 new sites in 3 blocks of 3."""
    sc = sm.SimConfig(Ns=24, Nt=5, seed=seed, rho_const=4.0)
    cfg = ModelConfig(model_type="shi")
    out = sm.simulate(cfg, sc)
    rng = np.random.default_rng(0)
    new = rng.uniform(0, 10, (N_NEW, 2))
    T = sc.Nt
    k_S = out["knots_S"].shape[0]
    k_phi = out["knots_phi"].shape[0]
    k_rho = out["knots_rho"].shape[0]
    draws = [dict(Z_train=out["Z"] + 0.01 * rng.standard_normal(out["Z"].shape),
                  S=np.abs(rng.standard_normal((k_S, T))) + 0.5,
                  phi_k=np.full(k_phi, 0.5), gamma_k=np.ones(k_S),
                  rho_k=np.full(k_rho, 4.0), rho=4.0, nu=out["nu"],
                  alpha0=out["alpha0"]) for _ in range(n_draws)]
    geom = pr.NewSiteGeometry(
        new_coords=new, train_coords=out["sites_xy"], knots_S=out["knots_S"],
        knots_phi=out["knots_phi"], knots_rho=out["knots_rho"], radius_S=3.0,
        eff_range_phi=3.0, eff_range_rho=3.0, config=cfg,
        cliques=None, separators=[], water_poly=None)
    hrng = np.random.default_rng(2)
    hold = pr.HoldoutData(
        Y=60 + np.abs(hrng.standard_normal((N_NEW, T))) * 5,
        u=np.full(N_NEW, 60.0), sigma=np.full(N_NEW, 20.0), xi=np.full(N_NEW, 0.15))
    return geom, draws, hold, T


def _sub(geom, hold, idx):
    """The same problem restricted to a subset of new sites -- what scoring a block on
    its own looks like (its own geometry, its own kriging)."""
    g = pr.NewSiteGeometry(
        new_coords=geom.new_coords[idx], train_coords=geom.train_coords,
        knots_S=geom.knots_S, knots_phi=geom.knots_phi, knots_rho=geom.knots_rho,
        radius_S=geom.radius_S, eff_range_phi=geom.eff_range_phi,
        eff_range_rho=geom.eff_range_rho, config=geom.config,
        cliques=geom.cliques, separators=geom.separators, water_poly=geom.water_poly)
    h = pr.HoldoutData(Y=hold.Y[idx], u=np.asarray(hold.u)[idx],
                       sigma=np.asarray(hold.sigma)[idx],
                       xi=np.asarray(hold.xi)[idx], p_cens=hold.p_cens)
    return g, h


# --------------------------------------------------------------------- exactness

def test_pointwise_sums_to_summed_kernel():
    """The identity everything rests on: given Z, the log-density is additive over
    sites, so a block's conditional density is a sub-sum of the pooled one."""
    rng = np.random.default_rng(7)
    n, p, alpha0 = 20, 0.95, 3.0
    uvec = np.full(n, 5.0)
    sig = rng.uniform(1.0, 3.0, n)
    xi = rng.uniform(-0.1, 0.2, n)
    Y = np.where(rng.random(n) < 0.6, uvec - rng.random(n),
                 uvec + rng.exponential(2.0, n))
    R = rng.uniform(0.5, 2.0, n)
    Z = rng.normal(size=n)
    phi = rng.uniform(0.2, 0.6, n)
    gb = rng.uniform(0.8, 1.2, n)
    q = qRW(u.pCGP(Y, p, uvec, sig, xi), phi, gb, alpha0)
    dv = dRW(q, phi, gb, alpha0)
    cens = np.flatnonzero(Y <= uvec).astype(np.int64)
    exc = np.flatnonzero(Y > uvec).astype(np.int64)
    f = np.ascontiguousarray
    args = (f(Y), p, f(uvec), f(sig), f(xi), f(R), f(Z), None, f(phi), f(gb),
            alpha0, f(q), f(dv), cens, exc)
    tot = u.Y_loglik_1t_best(*args)
    pw = u.Y_loglik_1t_pointwise(*args)
    assert abs(tot - pw.sum()) < 1e-9
    # and the sub-sums that the region projection takes
    parts = [pw[np.arange(a, b)].sum() for a, b in ((0, 7), (7, 13), (13, 20))]
    assert abs(sum(parts) - tot) < 1e-9


def test_regions_all_reproduces_the_no_regions_path():
    """Asking for a single region spanning every site must give EXACTLY what the
    original (regions=None) code path gives -- this is what pins the pointwise-kernel
    swap and the region-axis refactor to the pre-existing numbers."""
    geom, draws, hold, T = _setup()
    kw = dict(M=8, seed=3, tail_threshold=60.0, tail_smooths=(5.0,))
    base = pr.score_holdout_all(geom, draws, hold, **kw)
    reg = pr.score_holdout_all(geom, draws, hold, regions={"all": np.arange(N_NEW)},
                               **kw)["by_region"]["all"]
    assert reg["loglik"]["total"] == pytest.approx(base["loglik"]["total"], rel=1e-9)
    np.testing.assert_allclose(reg["loglik"]["per_time"], base["loglik"]["per_time"],
                               rtol=1e-9)
    np.testing.assert_allclose(reg["loglik"]["per_draw"], base["loglik"]["per_draw"],
                               rtol=1e-9)
    assert reg["crps_vanilla_mean"] == pytest.approx(base["crps_vanilla_mean"], rel=1e-12)
    assert reg["twcrps_hard_mean"] == pytest.approx(base["twcrps_hard_mean"], rel=1e-12)


def test_crps_pooled_is_the_weighted_mean_of_blocks():
    """CRPS is a MEAN of per-site scores, so unlike the log-score it does combine --
    exactly. If this ever fails, the region projection is slicing the wrong sites."""
    geom, draws, hold, T = _setup()
    regions = {"all": np.arange(N_NEW), **BLOCKS}
    out = pr.score_holdout_all(geom, draws, hold, M=8, seed=3, tail_threshold=60.0,
                               tail_smooths=(5.0,), regions=regions)["by_region"]
    for key in ("crps_vanilla_mean", "twcrps_hard_mean"):
        pooled = sum(len(idx) * out[nm][key] for nm, idx in BLOCKS.items()) / N_NEW
        assert out["all"][key] == pytest.approx(pooled, rel=1e-12)


def test_loglik_is_NOT_additive_over_blocks():
    """A guard on the reason this machinery exists: the log-score is joint, so summing
    block scores is NOT the pooled score. Anyone 'simplifying' the pooled pass away
    should fail here."""
    geom, draws, hold, T = _setup()
    regions = {"all": np.arange(N_NEW), **BLOCKS}
    out = pr.score_holdout_all(geom, draws, hold, M=8, seed=3, tail_threshold=60.0,
                               regions=regions)["by_region"]
    summed = sum(out[nm]["loglik"]["total"] for nm in BLOCKS)
    assert abs(summed - out["all"]["loglik"]["total"]) > 1.0


# ------------------------------------------------------------------- statistical

def test_extracted_blocks_match_separately_scored_blocks():
    """THE test: a block extracted from the pooled pass must agree with the same block
    scored on its own geometry.

    They cannot agree exactly -- the two use different Monte-Carlo realisations of the
    latent field (different draw shapes => different RNG streams) -- so this compares
    the MEANS over several seeds against the Monte-Carlo spread. A bug that summed the
    wrong sites, or used the wrong marginal, shifts the mean far beyond that spread."""
    seeds = [1, 2, 3, 4, 5, 6]
    geom, draws, hold, T = _setup()
    got = {nm: [] for nm in BLOCKS}
    sep = {nm: [] for nm in BLOCKS}
    for sd in seeds:
        pooled = pr.score_holdout_all(geom, draws, hold, M=10, seed=sd,
                                      tail_threshold=60.0,
                                      regions=BLOCKS)["by_region"]
        for nm, idx in BLOCKS.items():
            got[nm].append(pooled[nm]["loglik"]["total"])
            g_s, h_s = _sub(geom, hold, idx)
            sep[nm].append(pr.score_holdout_all(g_s, draws, h_s, M=10, seed=sd,
                                                tail_threshold=60.0)["loglik"]["total"])
    for nm in BLOCKS:
        a, b = np.array(got[nm]), np.array(sep[nm])
        se = np.hypot(a.std(ddof=1), b.std(ddof=1)) / np.sqrt(len(seeds))
        gap = abs(a.mean() - b.mean())
        # tolerate 4 SE, with a small floor so a near-zero-variance case cannot make
        # the bound vacuously tight
        assert gap < 4 * se + 0.05 * abs(b.mean()) + 1e-6, (
            f"{nm}: extracted {a.mean():.4f} vs separate {b.mean():.4f}, "
            f"gap {gap:.4f}, 4SE {4*se:.4f}")


def test_extracted_block_crps_matches_separately_scored():
    """Same comparison for CRPS, where the per-site marginal predictive is identical
    whether or not the other new sites were in the batch."""
    geom, draws, hold, T = _setup()
    pooled = pr.score_holdout_all(geom, draws, hold, M=12, seed=11,
                                  tail_threshold=60.0, regions=BLOCKS)["by_region"]
    for nm, idx in BLOCKS.items():
        g_s, h_s = _sub(geom, hold, idx)
        one = pr.score_holdout_all(g_s, draws, h_s, M=12, seed=11, tail_threshold=60.0)
        assert pooled[nm]["crps_vanilla_mean"] == pytest.approx(
            one["crps_vanilla_mean"], rel=0.25)
