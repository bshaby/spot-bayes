"""
MPI (cluster) variant of the sampler.

Parallelises over time replicates: each rank owns a subset of the Nt times and
updates their latent fields (S, Z) locally. Global parameters (phi, rho, alpha0)
are kept identical across ranks by driving their proposals and accept/reject
decisions from a *shared* RNG (same seed on every rank) and summing the
per-time log-likelihood contributions with an Allreduce. This mirrors Shi's
original MPI structure while reusing the serial sampler's ModelState and update
math (see sampler.py).

Run with:  mpirun -n <P> python -m mpi4py scripts/run_sampler.py --mpi ...
or call run_mcmc_mpi directly from an mpirun-launched script.

The serial run_mcmc and run_mcmc_mpi share ModelState, so the model definition,
covariance dispatch, kernels and censoring-deduplicated likelihood are identical;
only the parallel orchestration differs.
"""
from __future__ import annotations

import numpy as np

import mcmc_updates as mu
from mcmc_updates import Ctx
from config import ModelConfig, MODEL_HW
from sampler import ModelState, SamplerData, SamplerSettings, _alloc_latent


def _mpi_write_checkpoint(path, comm, rank, it, state, local_t, grng, lrng,
                          AD, TR, st):
    """Per-rank latent + adaptives, and a rank-0 global file (params, grng,
    traces). Pragmatic resume: continues from a valid state (re-warming is
    avoided by also restoring adaptives), though MPI float-reduction ordering
    means it is not guaranteed bit-identical to an uninterrupted run.

    The unified adaptive dict AD is split into its local (S/Z, per-rank) and
    global (everything else, identical across ranks) parts for storage."""
    import os
    import pickle
    loc_ad = {"S": AD["S"], "Z": AD["Z"]}
    glob_ad = {k: v for k, v in AD.items() if k not in ("S", "Z")}
    rp = f"{path}.rank{rank}.pkl"
    tmp = rp + ".tmp"
    with open(tmp, "wb") as fh:
        # nuts_z is the per-RANK Z step adapter (seeded and dual-averaged from the
        # local rng over this rank's time columns), so it belongs here rather than in
        # the rank-0 global file that carries the shared nu_rho/rho/beta adapters.
        pickle.dump(dict(S_local=state.S[:, local_t].copy(),
                         Z_local=state.Z[:, local_t].copy(),
                         loc_ad=loc_ad, lrng=lrng.bit_generator.state,
                         nuts_z=getattr(state, "_nuts_z", None)), fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, rp)
    if rank == 0:
        small = {k: TR[k] for k in TR
                 if k not in ("S", "Z", "state", "S_final", "Z_final")}
        payload = dict(it=it, size=comm.Get_size(), gsnap=state.global_snapshot(),
                       grng=grng.bit_generator.state, glob_ad=glob_ad,
                       small_traces=small)
        if st.save_latent and not st.stream_latent:
            payload["S_trace"] = TR["S"]; payload["Z_trace"] = TR["Z"]
        elif st.save_latent and st.stream_latent:
            TR["S"].flush(); TR["Z"].flush()
        gp = f"{path}.global.pkl"
        tmp = gp + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, gp)
        # ALSO write a readable parameter-trace .npz (load_run / plots) so the run
        # is inspectable mid-flight without unpickling the checkpoint.
        from sampler import write_param_npz
        npz_target = st.latent_out or (path + ".traces.npz")
        try:
            write_param_npz(npz_target, TR, state, st, it // st.thin + 1)
        except Exception as e:
            print(f"[checkpoint] readable trace npz write failed: {e}", flush=True)
    comm.Barrier()


def _mpi_load_checkpoint(path, comm, rank):
    import os
    import pickle
    gp = f"{path}.global.pkl"
    if not os.path.exists(gp):
        return None, None
    with open(gp, "rb") as fh:
        g = pickle.load(fh)
    if g["size"] != comm.Get_size():
        # Report BOTH counts: quoting only the checkpoint's reads like an instruction
        # ("needs 16") without saying what was actually launched, which is the one
        # fact needed to fix it. Also point at the repartitioner -- changing a run's
        # rank count no longer means restarting it.
        raise ValueError(
            f"MPI rank-count mismatch: this checkpoint was written by {g['size']} "
            f"rank(s), but the job launched with {comm.Get_size()}.\n"
            f"  Either relaunch with {g['size']} ranks (check NRANKS / "
            f"--cpus-per-task; an exported NRANKS overrides --cpus-per-task),\n"
            f"  or repartition the checkpoint, which preserves the field, the "
            f"per-time adaptives and the iteration count:\n"
            f"    python faroe_analysis/scripts/repartition_checkpoint.py "
            f"--ckpt {path} --ranks {comm.Get_size()}")
    with open(f"{path}.rank{rank}.pkl", "rb") as fh:
        r = pickle.load(fh)
    return g, r


def run_mcmc_mpi(config: ModelConfig, data: SamplerData,
                 settings: SamplerSettings, verbose: bool = False,
                 init_snapshot: dict = None) -> dict:
    """MPI MCMC. Returns the trace dict on rank 0 (None-valued traces on others).

    init_snapshot : optional ModelState snapshot (full S/Z + globals, see
    ModelState.snapshot) to warm-start from a prior fit's final state. Unlike
    checkpoint resume it is rank-count INDEPENDENT -- every rank restores the same
    global snapshot then operates on its own time slice -- so a fit can be
    warm-started at any number of ranks. A checkpoint, if present, takes
    precedence (mirrors the serial run_mcmc)."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    config.validate()
    from sampler import keep_checkpoint_barrier_rule
    keep_checkpoint_barrier_rule(config, settings,
                                 f"{settings.checkpoint_path}.global.pkl")
    state = ModelState(config, data, settings)
    Ns, Nt = state.Ns, state.Nt
    st = settings

    local_t = list(range(rank, Nt, size))          # round-robin time partition

    # shared RNG (global params; identical on every rank) + local RNG (S/Z)
    grng = np.random.default_rng(st.seed)
    lrng = np.random.default_rng(st.seed * 100003 + rank + 1)

    def allsum(x):
        return comm.allreduce(float(x), op=MPI.SUM)

    def allsum_arr(a):
        # Allreduce-sum a per-rank partial-sum array (e.g. the sum_t b_t b_t^T +
        # quad-form buffer in the CP nu/rho gradient) to the global total.
        a = np.ascontiguousarray(a, dtype=float)
        out = np.empty_like(a)
        comm.Allreduce(a, out, op=MPI.SUM)
        return out

    # This rank owns the time slice `local_t`. Global proposals use the shared
    # grng (identical on every rank); S/Z use the per-rank lrng; the reduction
    # is an Allreduce-sum over ranks (scalar via `allsum`, arrays via `allsum_arr`).
    ctx = Ctx(local_t, grng, lrng, reduce=allsum, reduce_arr=allsum_arr,
              supports_compact=False, covers_all_times=False)

    # resume: restore global params (all ranks) + this rank's latent columns
    ck_g, ck_r = (_mpi_load_checkpoint(st.checkpoint_path, comm, rank)
                  if (st.resume and st.checkpoint_path) else (None, None))
    if ck_g is not None:
        state.restore(ck_g["gsnap"], latent=False)
        for k, t in enumerate(local_t):
            state.S[:, t] = ck_r["S_local"][:, k]
            state.Z[:, t] = ck_r["Z_local"][:, k]
            state.R[:, t] = state.W_wend_S @ state.S[:, t]
        grng.bit_generator.state = ck_g["grng"]
        lrng.bit_generator.state = ck_r["lrng"]
        # per-rank Z adapter (absent in checkpoints written before this was stored,
        # which then simply re-warm as they always did)
        if ck_r.get("nuts_z") is not None:
            state._nuts_z = ck_r["nuts_z"]
    elif init_snapshot is not None:
        # warm-start: every rank restores the same full snapshot (latent=True),
        # then owns its local_t columns. Fresh adaptives + RNG below (cold tuning
        # from a warm point); rank-count independent (no per-rank checkpoint).
        state.restore(init_snapshot, nuts=False)   # warm start = cold NUTS tuning

    # per-time caches over this rank's slice (from restored state if resuming)
    caches = mu.init_caches(state, local_t)

    # adaptive blocks (restored from the checkpoint, or built fresh). S/Z are
    # per-rank (local_t); the rest are global and identical across ranks.
    if ck_g is not None:
        AD = {**ck_g["glob_ad"], "S": ck_r["loc_ad"]["S"], "Z": ck_r["loc_ad"]["Z"]}
        from sampler import keep_checkpoint_s_sampler
        keep_checkpoint_s_sampler(st, AD["S"], state.k_S)
    else:
        AD = mu.make_adaptives(state, st, local_t, config)

    n_save = len(range(0, st.n_iters, st.thin))
    TR = None
    if rank == 0:
        TR = dict(phi_k=np.zeros((n_save, state.k_phi)),
                  rho_k=np.zeros((n_save, state.k_rho)),
                  alpha0=np.zeros(n_save), ll=np.zeros(n_save))
        if getattr(st, "s_trace_times", 0):
            # S is distributed, so record only the columns THIS (recording) rank
            # owns -- no communication, and rank 0's slice is a fair sample of the
            # round-robin partition.
            mon = np.asarray(local_t[:int(st.s_trace_times)], dtype=int)
            # state.S is full-width on every rank with only local columns valid,
            # so these GLOBAL indices index it directly.
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
        if st.save_latent:
            TR["S"], TR["Z"] = _alloc_latent(st, n_save, state.k_S, Ns, Nt,
                                             resume=(ck_g is not None))
        if ck_g is not None:                       # refill traces so far
            for k, v in ck_g["small_traces"].items():
                if k in TR:
                    TR[k][:v.shape[0]] = v
            if st.save_latent and not st.stream_latent:
                TR["S"][:ck_g["S_trace"].shape[0]] = ck_g["S_trace"]
                TR["Z"][:ck_g["Z_trace"].shape[0]] = ck_g["Z_trace"]

    start_iter = (ck_g["it"] + 1) if ck_g is not None else 0

    meter = None
    if rank == 0 and verbose:
        import diagnostics as _dg
        meter = _dg.ProgressMeter(st.n_iters,
                                  st.progress_every or max(1, st.n_iters // 50),
                                  tag=f"mpi size={size}", start_iter=start_iter)
    last_ll = float("nan")
    for it in range(start_iter, st.n_iters):
        # --- latent fields S, Z (local to this rank, per-rank RNG) ----------
        if st.update_S:
            mu.update_S(state, AD["S"], ctx, caches, st)
        if st.update_Z:
            mu.update_Z(state, AD["Z"], ctx, caches, st)

        # --- global parameters (phi, rho, alpha0, nu, radius_S, anisotropy,
        #     GPD marginal betas) -- the exact same blocks the serial backend
        #     runs; ctx.total turns each per-time sum into an Allreduce here.
        mu.run_global_updates(state, AD, ctx, caches, st, config)

        # --- adaptation ------------------------------------------------------
        if (it + 1) % st.adapt_size == 0:
            mu.adapt_all(AD, st.adapt_size, (it + 1) // st.adapt_size)

        # --- record (gather to rank 0) --------------------------------------
        if it % st.thin == 0:
            s = it // st.thin
            ll_tot = ctx.total_ll(caches.Yll, caches.Zlp)
            S_loc = {t: state.S[:, t].copy() for t in local_t} if st.save_latent else None
            Z_loc = {t: state.Z[:, t].copy() for t in local_t} if st.save_latent else None
            gathered = comm.gather((S_loc, Z_loc), root=0)
            if rank == 0:
                TR["phi_k"][s] = state.phi_k
                TR["rho_k"][s] = state.rho_k
                TR["alpha0"][s] = state.alpha0
                TR["ll"][s] = ll_tot
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
                if st.save_latent:
                    for Sd, Zd in gathered:
                        for t, col in Sd.items():
                            TR["S"][s][:, t] = col
                        for t, col in Zd.items():
                            TR["Z"][s][:, t] = col
                    if st.stream_latent and (s + 1) % st.latent_chunk == 0:
                        TR["S"].flush(); TR["Z"].flush()
                last_ll = ll_tot
                # Live trace plot. The cadence is measured in RECORDED samples
                # (this block only runs on recording steps, it % thin == 0), so
                # the old `(it+1) % plot_every` trigger could never align with the
                # recording grid when plot_every wasn't `thin*k - 1` -- it simply
                # never fired under MPI. Convert plot_every (iterations) to a
                # record stride so it fires reliably for any plot_every/thin.
                plot_stride = max(1, round(st.plot_every / st.thin)) if st.plot_every else 0
                if plot_stride and st.plot_path and (s + 1) % plot_stride == 0:
                    try:
                        import diagnostics
                        diagnostics.plot_traces(TR, s + 1, st.plot_path,
                                                thin=st.thin, iteration=it,
                                                phi_domain=config.phi_domain)
                        diagnostics.plot_live_surfaces(state, st.plot_path,
                                                       phi_domain=config.phi_domain)
                    except Exception as e:   # never kill the run, but be loud
                        print(f"[diagnostics] live trace plot failed: {e}", flush=True)

        # --- progress feedback (rank 0; iter k/N, ll, elapsed, ETA, it/s) ----
        if meter is not None:
            meter.update(it, last_ll)

        # --- checkpoint (collective: every rank writes its piece) -----------
        if st.checkpoint_every and st.checkpoint_path \
                and (it + 1) % st.checkpoint_every == 0:
            _mpi_write_checkpoint(st.checkpoint_path, comm, rank, it, state,
                                  local_t, grng, lrng, AD, TR, st)

    if rank == 0:
        if st.save_latent and st.stream_latent:
            TR["S"].flush(); TR["Z"].flush()
        TR["state"] = state
        TR["S_final"] = state.S
        TR["Z_final"] = state.Z
        # global blocks are replicated (synchronized RNG); S/Z are rank 0's owned cols
        TR["accept"] = mu.acceptance_summary(AD)
        try:
            import diagnostics
            TR["ess"] = diagnostics.ess_summary(TR, n_save, burn_frac=0.5)
            if verbose:
                diagnostics.print_ess(TR["ess"])
        except Exception:
            pass
    return TR
