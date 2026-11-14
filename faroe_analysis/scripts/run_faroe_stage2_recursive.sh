#!/usr/bin/env bash
# STAGE 2 (recursive Bayes) -- fit the full dependence model with the GPD-margin
# coefficients proposed from the stage-1 posterior (beta_sampler="recursive").
#
# The beta block is an INDEPENDENCE MH move: it draws (beta_sigma, beta_xi) from the
# stage-1 marginal_fit posterior and accepts against the EXACT ratio
#   log r = [l_full(b*) + logp2(b*)] - [l_full(b_cur) + logp2(b_cur)]
#         - [l_marg(b*) + logp1(b*)] - ... (stage-1 marginal likelihood + priors),
# which targets the correct full posterior [beta, theta | Y]. NOTE: unlike Hooten's
# disjoint-partition case there is NO compute saving -- the full dependence likelihood is
# still evaluated at each proposal (our stages share the data; l_marg is not a factor of
# l_full). See docs/two_stage_recursive_bayes.md. If this mixes no better than sampling
# beta through the dependence likelihood, use run_faroe_stage2_fixmargins.sh instead.
#
# PREREQUISITE: stage 1 finished ->
#     bash faroe_analysis/scripts/run_faroe_marginal.sh
# FAROE_STAGE1_NPZ supplies BOTH the warm-start betas and the RecursivePrior proposals.
# Margins are SAMPLED here (not fixed), so --fix-margins is intentionally absent. Output:
#     faroe_analysis/output/wang_full_recursive_stage1.*   (distinct; never clobbers)
# Checkpointed + auto-resuming: re-run the same command to extend/continue.
#
# Env overrides: NRANKS(64), FAROE_STAGE1_NPZ(<output>/marginal/marginal_fit.npz), and the
# standard faroe_setup knobs.
source "$(dirname "$0")/_faroe_env.sh"     # activates `extremes`, GSL, MPI env, cd repo root

export FAROE_STAGE1_NPZ="${FAROE_STAGE1_NPZ:-${FAROE_OUT:-$FAROE_SCRIPTS/../output}/marginal/marginal_fit.npz}"
if [ ! -f "$FAROE_STAGE1_NPZ" ]; then
  echo "ERROR: stage-1 output not found at $FAROE_STAGE1_NPZ" >&2
  echo "       run  bash faroe_analysis/scripts/run_faroe_marginal.sh  first." >&2
  exit 1
fi
echo "[stage2-recursive] proposing margins from $FAROE_STAGE1_NPZ"

# shellcheck disable=SC2086
mpiexec ${FAROE_MPIEXEC_ARGS:-} -n "$NRANKS" ${FAROE_RANKWRAP:-} \
    python "$FAROE_SCRIPTS/faroe_fit.py" --model wang --which full \
    --beta-sampler recursive "$@"
