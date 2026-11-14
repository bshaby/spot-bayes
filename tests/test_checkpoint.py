"""Checkpoint/resume: a resumed run must reproduce an uninterrupted one -- the
parameter chain and latent state BIT-IDENTICALLY, and the loglik trace to root-find
tolerance (qRW's last ULP is bracket-dependent under warm-starting; see below)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as sm  # noqa: E402
from config import ModelConfig  # noqa: E402
from sampler import SamplerData, SamplerSettings, run_mcmc  # noqa: E402


def _data(seed=11, Ns=30, Nt=10):
    out = sm.simulate(ModelConfig("shi"), sm.SimConfig(Ns=Ns, Nt=Nt, seed=seed))
    data = SamplerData(
        Y=out["Y_missing"], sites_xy=out["sites_xy"], u_vec=out["u_mat"][:, 0],
        sigma_vec=out["sigma_mat"][:, 0], xi_vec=out["xi_mat"][:, 0],
        p_cens=out["p_cens"], knots_S=out["knots_S"], knots_phi=out["knots_phi"],
        knots_rho=out["knots_rho"])
    return out, data


def _equal_traces(A, B, keys):
    return all(np.array_equal(A[k], B[k]) for k in keys)


def test_resume_is_bit_identical_to_uninterrupted():
    out, data = _data()
    # nuts_warmup is pinned EXPLICITLY. The comparison runs 60 iterations and then
    # resumes to 100, against a straight 100, so any schedule keyed to n_iters makes
    # the two differ before the checkpoint is even written: the default NUTS warmup is
    # n_iters//2, so the 60-run finalises its step size at iteration 30 while the
    # 100-run is still warming. That is correct behaviour, not a resume failure -- it
    # only became visible when NUTS became the default S sampler, since until then no
    # block had an n_iters-dependent schedule (the RWMH adapters key off adapt_size).
    # Holding it fixed is what isolates the thing under test.
    common = dict(adapt_size=20, thin=5, nu=out["nu"], seed=1,
                  update_nu=True, update_radius_S=True, nuts_warmup=10)
    full = run_mcmc(ModelConfig("shi"), data, SamplerSettings(n_iters=100, **common))
    ck = os.path.join(tempfile.mkdtemp(), "ck")
    run_mcmc(ModelConfig("shi"), data,
             SamplerSettings(n_iters=60, checkpoint_every=60, checkpoint_path=ck, **common))
    res = run_mcmc(ModelConfig("shi"), data,
                   SamplerSettings(n_iters=100, checkpoint_every=60,
                                   checkpoint_path=ck, resume=True, **common))
    # Params and latent state resume BIT-IDENTICALLY. `ll` is relaxed to root-find
    # tolerance: it sums over qRW root-finds whose last ULP is bracket-dependent under
    # warm-starting (qRW_warm seeds the bracket from the previous accepted qv, which the
    # resumed cache rebuilds cold) -- an ~8e-16 relative wobble that never flips an
    # accept/reject, so the chain itself stays exact (asserted below).
    assert _equal_traces(full, res, ["phi_k", "rho_k", "alpha0", "nu", "radius_S"])
    assert np.allclose(full["ll"], res["ll"], rtol=1e-9, atol=1e-9)
    assert np.array_equal(full["Z_final"], res["Z_final"])
    assert np.array_equal(full["S_final"], res["S_final"])


def test_resume_with_streaming_latent():
    """Streamed-latent resume (same n_iters) reopens the memmap r+ without
    truncating, continues from the mid-run checkpoint, and matches a reference
    run. checkpoint_every=30 with n_iters=40 leaves the last checkpoint at it=29,
    so the resume genuinely continues iters 30..39 into the full-size memmap."""
    out, data = _data()
    tmp = tempfile.mkdtemp()
    common = dict(adapt_size=20, thin=5, nu=out["nu"], seed=1, save_latent=True)
    # reference: uninterrupted, separate memmap
    ref = run_mcmc(ModelConfig("shi"), data,
                   SamplerSettings(n_iters=40, stream_latent=True,
                                   latent_out=os.path.join(tmp, "ref.npz"), **common))
    ref_Z = np.array(ref["Z"])

    ck = os.path.join(tmp, "ck")
    run_out = os.path.join(tmp, "run.npz")
    # leg 1: full-size memmap (n_iters=40), last checkpoint at it=29
    run_mcmc(ModelConfig("shi"), data,
             SamplerSettings(n_iters=40, stream_latent=True, latent_out=run_out,
                             checkpoint_every=30, checkpoint_path=ck, **common))
    # resume same n_iters -> continues 30..39 into the same full-size memmap
    res = run_mcmc(ModelConfig("shi"), data,
                   SamplerSettings(n_iters=40, stream_latent=True, latent_out=run_out,
                                   checkpoint_every=30, checkpoint_path=ck,
                                   resume=True, **common))
    assert np.allclose(ref["ll"], res["ll"], rtol=1e-9, atol=1e-9)  # 1-ULP qRW jitter
    assert np.array_equal(ref_Z, np.array(res["Z"]))                # latent bit-identical


def test_no_checkpoint_file_starts_fresh():
    out, data = _data()
    ck = os.path.join(tempfile.mkdtemp(), "missing")
    # resume=True but no checkpoint present -> starts from scratch, runs fine
    TR = run_mcmc(ModelConfig("shi"), data,
                  SamplerSettings(n_iters=30, adapt_size=15, thin=5, nu=out["nu"],
                                  seed=1, resume=True, checkpoint_path=ck))
    assert np.all(np.isfinite(TR["ll"]))


def test_nuts_adapters_survive_resume():
    """A NUTS resume must continue the SAME chain, which requires the step-size /
    metric adapters to travel in the checkpoint. Without them the resumed leg
    re-runs nuts_find_reasonable_eps and a fresh warmup, so it walks different
    trajectories from the uninterrupted reference (and mis-times any calibration
    done on those cheap large-eps warmup iterations)."""
    out, data = _data()
    common = dict(adapt_size=20, thin=5, nu=out["nu"], seed=1,
                  update_nu=True, update_rho=True, update_radius_S=True,
                  z_sampler="nuts", nuts_warmup=10, nu_rho_sampler="nuts")
    cfg = ModelConfig("shi", use_anisotropy=True)
    full = run_mcmc(cfg, data, SamplerSettings(n_iters=60, **common))
    ck = os.path.join(tempfile.mkdtemp(), "ck")
    run_mcmc(cfg, data, SamplerSettings(n_iters=30, checkpoint_every=30,
                                        checkpoint_path=ck, **common))
    res = run_mcmc(cfg, data, SamplerSettings(n_iters=60, checkpoint_every=30,
                                              checkpoint_path=ck, resume=True,
                                              **common))
    assert _equal_traces(full, res, ["phi_k", "rho_k", "alpha0", "nu", "radius_S"])
    assert np.array_equal(full["Z_final"], res["Z_final"])


def test_nuts_adapters_are_in_the_snapshot_and_survive_a_dim_change():
    """The adapters are stored under snapshot()['nuts']; a stale metric whose length
    no longer matches the model is DROPPED on restore (re-warms) rather than used."""
    from sampler import ModelState
    out, data = _data()
    st = SamplerSettings(n_iters=30, adapt_size=15, thin=5, nu=out["nu"], seed=1,
                         update_nu=True, update_rho=True, z_sampler="nuts",
                         nuts_warmup=5, nu_rho_sampler="nuts")
    cfg = ModelConfig("shi", use_anisotropy=True)
    TR = run_mcmc(cfg, data, st)
    snap = TR["state"].snapshot()
    assert "_nuts_nu_rho" in snap["nuts"] and "_nuts_z" in snap["nuts"]
    # the MPI global snapshot drops the per-rank Z adapter, keeps the shared ones
    g = TR["state"].global_snapshot()
    assert "_nuts_z" not in g["nuts"] and "_nuts_nu_rho" in g["nuts"]

    fresh = ModelState(cfg, data, st)
    snap["nuts"]["_nuts_nu_rho"].inv_mass = np.ones(99)   # wrong length
    fresh.restore(snap)
    assert getattr(fresh, "_nuts_nu_rho", None) is None   # dropped, will re-warm

    # an OLD checkpoint carries no "nuts" key at all -> restore is a no-op there
    old = {k: v for k, v in snap.items() if k != "nuts"}
    ModelState(cfg, data, st).restore(old)                # must not raise


def test_warm_start_does_not_inherit_nuts_adapters():
    """init_snapshot is a warm start (cold tuning from a warm point): a frozen
    post-warmup adapter from another fit must not be carried in."""
    from sampler import ModelState
    out, data = _data()
    st = SamplerSettings(n_iters=30, adapt_size=15, thin=5, nu=out["nu"], seed=1,
                         update_nu=True, z_sampler="nuts", nuts_warmup=5)
    cfg = ModelConfig("shi", use_anisotropy=True)
    snap = run_mcmc(cfg, data, st)["state"].snapshot()
    assert snap["nuts"]                                    # the fit produced adapters
    fresh = ModelState(cfg, data, st)
    fresh.restore(snap, nuts=False)
    assert getattr(fresh, "_nuts_z", None) is None


def test_resume_keeps_the_checkpoint_s_sampler():
    """A chain checkpointed under s_sampler="joint" and resumed under today's default
    ("nuts") must keep "joint". Switching crashed at the first adaptation after the
    resume (nuts records size-1 placeholders into a window of k_S-vectors) -- the
    Faroe wang_full_fixmargins_stage1 extension -- and would in any case change the
    kernel mid-chain. Kept, the resumed chain is the uninterrupted joint one."""
    import warnings
    out, data = _data()
    common = dict(adapt_size=20, thin=5, nu=out["nu"], seed=1, update_nu=True)
    full = run_mcmc(ModelConfig("shi"), data,
                    SamplerSettings(n_iters=80, s_sampler="joint", **common))
    ck = os.path.join(tempfile.mkdtemp(), "ck")
    run_mcmc(ModelConfig("shi"), data,
             SamplerSettings(n_iters=50, checkpoint_every=50, checkpoint_path=ck,
                             s_sampler="joint", **common))
    st = SamplerSettings(n_iters=80, checkpoint_every=50, checkpoint_path=ck,
                         resume=True, **common)
    assert st.s_sampler == "nuts"                       # today's default
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = run_mcmc(ModelConfig("shi"), data, st)
    assert st.s_sampler == "joint"
    assert any("s_sampler='joint'" in str(x.message) for x in w)
    assert _equal_traces(full, res, ["phi_k", "rho_k", "alpha0", "nu"])
    assert np.array_equal(full["S_final"], res["S_final"])
    assert np.array_equal(full["Z_final"], res["Z_final"])


def test_resume_repairs_a_mixed_joint_nuts_s_window():
    """The Faroe extension's second failure: a 'nuts' resume of a 'joint' chain ran
    long enough to CHECKPOINT before it crashed, so the saved S windows hold joint
    k_S-vectors AND nuts [0.0] placeholders. The guard must keep 'joint' and drop
    the placeholders, or the next adaptation vstacks mixed sizes again."""
    import pickle
    import warnings
    out, data = _data()
    common = dict(adapt_size=20, thin=5, nu=out["nu"], seed=1, update_nu=True)
    ck = os.path.join(tempfile.mkdtemp(), "ck")
    run_mcmc(ModelConfig("shi"), data,
             SamplerSettings(n_iters=50, checkpoint_every=50, checkpoint_path=ck,
                             s_sampler="joint", **common))
    with open(ck + ".pkl", "rb") as fh:
        payload = pickle.load(fh)
    for a in payload["adaptives"]["S"].values():       # simulate the stray nuts records
        a.window += [np.array([0.0])] * 15
    with open(ck + ".pkl", "wb") as fh:
        pickle.dump(payload, fh)
    st = SamplerSettings(n_iters=80, checkpoint_every=50, checkpoint_path=ck,
                         resume=True, **common)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = run_mcmc(ModelConfig("shi"), data, st)      # crashed at it=59 before
    assert st.s_sampler == "joint"
    assert any("dropped the placeholders" in str(x.message) for x in w)
    assert np.all(np.isfinite(res["ll"]))
