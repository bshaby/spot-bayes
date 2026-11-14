#!/usr/bin/env python3
"""Build a per-model warm-start snapshot for the tasks 3-5 full-model (sampled-margin)
fits, seeded from the goodinit run's converged state.

Source: goodinit_field.npz (reassemble_goodinit.py) -- the exact final Z (432,1656),
S (41,1656) and globals (phi_k, rho_k[scalar], alpha0, nu, radius_S, anisotropy,
beta_sigma/xi, W_wend_S[432,41]) at iter 34999.

Common to all three models: Z (subset to the 369 train sites), beta_sigma/beta_xi,
alpha0, nu. Model-specific mapping (agreed with the user):
  wang : phi field, S field, scalar rho, radius_S, anisotropy -- all direct.
  shi  : phi field, S field, radius_S, anisotropy from goodinit; 13-knot rho FIELD from
         the tail-mean of shi_fixmargins_stage1's rho_k.
  hw   : scalar phi = R-WEIGHTED mean of goodinit's phi field; scalar rho direct; single
         S(t) = train-site spatial mean of goodinit R (= R[train].mean(0)/mean(w_HW)).

goodinit's Z was fit under Wang; transferring the values to HW/Shi is a warm start (the
covariance is reinterpreted under each model's rho/geometry and re-equilibrates).
"""
import os
import numpy as np
import kernels
import utilities as u
from sampler import ModelState


def _globals(fld):
    return {k[2:]: fld[k] for k in fld.files if k.startswith("g_")}


def build_warm_snapshot(model, cfg, data, settings, inp, field_npz, idx=None,
                        shi_rho_npz=None, shi_rho_burn=0.5, hw_phi_weighted=True):
    """Return a ModelState.snapshot() dict warm-starting `model` from goodinit.

    `idx` = the GLOBAL site indices this fit covers (inp['train_idx'] for a train fit,
    inp['full_idx'] for a full fit); defaults to train. goodinit's field is on all 432
    sites, so Z is subset to `idx`."""
    fld = np.load(field_npz)
    Z_full = np.asarray(fld["Z"], float)                 # (432, Nt)
    S_gi = np.asarray(fld["S"], float)                   # (41, Nt)
    g = _globals(fld)
    ti = inp["train_idx"] if idx is None else np.asarray(idx)
    Z_train = Z_full[ti]                                 # (len(idx), Nt)
    alpha_stable = getattr(settings, "alpha_stable", 0.5)

    st = ModelState(cfg, data, settings)                 # train state for this model
    # -- common globals -----------------------------------------------------
    st.alpha0 = float(g["alpha0"]); st.nu = float(g["nu"])
    st.beta_sigma = np.asarray(g["beta_sigma"], float).copy()
    st.beta_xi = np.asarray(g["beta_xi"], float).copy()
    st.Z = Z_train.copy()

    if model in ("wang", "shi"):
        # R-field transfer: radius/anisotropy/S from goodinit -> identical W_wend_S@S on
        # the shared knots, so R at the train sites matches goodinit's.
        st.radius_S = float(g["radius_S"])
        st.aniso_ratio = float(g["aniso_ratio"]); st.aniso_angle = float(g["aniso_angle"])
        st.W_wend_S = st.build_wendland(st.radius_S, st.aniso_ratio, st.aniso_angle)
        st.gamma_bar = kernels.gamma_bar_from_weights(st.W_wend_S, st.gamma_k, alpha=alpha_stable)
        st.phi_k = np.asarray(g["phi_k"], float).copy()  # 41-knot field
        st.S = S_gi.copy()                               # 41-knot, no site subset
        if model == "wang":
            st.rho_k = np.asarray(g["rho_k"], float).copy()          # scalar (1 knot)
        else:                                            # shi: 13-knot rho field
            if shi_rho_npz is None or not os.path.exists(shi_rho_npz):
                raise SystemExit(f"shi warm start needs shi_rho_npz (got {shi_rho_npz})")
            rd = np.asarray(np.load(shi_rho_npz)["rho_k"], float)     # (Ndraw, 13)
            lo = int(shi_rho_burn * rd.shape[0])
            st.rho_k = rd[lo:].mean(0).copy()                        # (13,)
            if st.rho_k.shape[0] != st.k_rho:
                raise SystemExit(f"shi rho knots {st.rho_k.shape[0]} != model k_rho {st.k_rho}")
    elif model == "hw":
        # scalar phi = R-weighted mean of goodinit's phi field (built on the FULL domain
        # via the same gaussian phi basis; cutoff=False so the radius arg is inert).
        W_phi_full = u.build_weight_matrix(
            inp["sites_xy"], inp["knots_phi"], kernel="gaussian",
            radius=settings.radius_S, bandwidth=settings.eff_range_phi ** 2 / 6.0,
            cutoff=False)
        # phi_k are in LOG space (phi field = exp(W @ phi_k)); average the ACTUAL phi.
        phi_actual = np.exp(W_phi_full @ np.asarray(g["phi_k"], float))   # (432,) actual phi
        R_full = np.asarray(g["W_wend_S"], float) @ S_gi             # (432, Nt)
        if hw_phi_weighted:
            w = R_full.sum(1)                                        # per-site total scale
            phi_hw = float(np.sum(w * phi_actual) / w.sum())        # R-weighted arithmetic mean
        else:
            phi_hw = float(phi_actual.mean())
        st.phi_k = np.array([np.log(phi_hw)])                       # HW stores log-phi (1 knot)
        st.rho_k = np.asarray(g["rho_k"], float).copy()             # scalar rho, direct
        w_hw = st.W_wend_S[:, 0]                                    # HW single-knot weight
        st.S = (R_full[ti].mean(0) / w_hw.mean())[None, :].copy()   # (1, Nt), all > 0
    else:
        raise SystemExit(f"unknown model {model}")

    # -- derived quantities the snapshot/restore chain expects --------------
    st.phi_vec = st._phi_field(st.phi_k)
    st.rho_vec = st.W_gauss_rho @ st.rho_k
    st.R = st.W_wend_S @ st.S
    if st.update_sigma:
        st.sigma_mat = st._mg.sigma_from_eta(st.sigma_design.eta(st.beta_sigma))
    if st.update_xi:
        st.xi_mat = st._mg.xi_from_eta(st.xi_design.eta(st.beta_xi))
    return st.snapshot()
