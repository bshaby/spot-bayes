#!/usr/bin/env bash
# STAGE 1 -- standalone conditionally-independent GPD marginal fit (SERIAL, no MPI).
#
# Fits marginal_model.py to the Faroe threshold exceedances (NUTS) and writes the
# stage-1 posterior draws that stage 2 / the fixed-margin fallback consume:
#     <MARG_OUT>/marginal_fit.npz          (beta_sigma / beta_xi draws + inputs)
#     <MARG_OUT>/marginal_fit.config.json  (run settings)
#     <MARG_OUT>/marginal_fit.trace.png    (beta traces + accept + tree depth)
# Default MARG_OUT = faroe_analysis/output/marginal.
#
# DIAGNOSTICS: the run STREAMS numeric diagnostics to stdout -- a progress line every
# ~5% with the running NUTS accept rate and mean tree depth, then a final block with the
# posterior-mean betas next to the reference warm-start values and the tail accept/depth.
# The parameter TRACES are written to marginal_fit.trace.png (not drawn in the terminal);
# with checkpointing on, that png + the npz are REFRESHED every MARG_CKPT_EVERY iters, so a
# long run is inspectable while it runs. Watch progress live with e.g. `... | tee stage1.log`.
#
# CHECKPOINT / RESUME (crash-safe): every MARG_CKPT_EVERY(1000) iters the driver writes a
# resume checkpoint (marginal_fit.ckpt.pkl) alongside the npz/png. Re-running the SAME
# command CONTINUES from the last checkpoint (and EXTENDS the chain if MARG_NITERS is
# larger); an already-complete run is an idempotent no-op. Set MARG_CKPT_EVERY=0 to disable.
# MARG_FORCE=1 wipes the npz/ckpt and refits from scratch.
#
# Serial: a single core suffices (~5% of cells are exceedances).
#
# Env overrides (defaults in faroe_marginal_fit.py): MARG_OUT, MARG_NITERS(20000),
# MARG_WARMUP(2000), MARG_THIN(5), MARG_SEED(20260719), MARG_TARGET_ACCEPT(0.8),
# MARG_SIGMA_BETA_SD(5.0), MARG_XI_BETA_SD(5.0), MARG_CKPT_EVERY(1000), MARG_FORCE(0).
#
# Usage:
#     bash faroe_analysis/scripts/run_faroe_marginal.sh
#     MARG_NITERS=40000 MARG_FORCE=1 bash faroe_analysis/scripts/run_faroe_marginal.sh
source "$(dirname "$0")/_faroe_env.sh"     # activates `extremes`, GSL, thread caps, cd repo root

python "$FAROE_SCRIPTS/faroe_marginal_fit.py"
