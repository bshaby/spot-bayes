"""Helper for test_sampler_golden (and for capturing the golden fixtures).

Runs a tiny, fixed problem that exercises every update block (S, Z, phi, rho,
alpha0, nu, radius_S, anisotropy, GPD sigma/xi) and saves the global-parameter
traces to argv[1]. argv[2] selects the driver: "serial" (run_mcmc) or "mpi"
(run_mcmc_mpi, launch under mpirun). Both drivers are deterministic run-to-run,
so the saved traces serve as golden fixtures: any change to the sampler that
alters numerical behaviour will change them. The serial/MPI update-block
unification must reproduce these byte-for-byte.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import simulate as sm
from config import ModelConfig
from sampler import SamplerData, SamplerSettings, run_mcmc

GLOBAL_KEYS = ["phi_k", "rho_k", "alpha0", "nu", "radius_S", "aniso_ratio",
               "aniso_angle", "beta_sigma", "beta_xi", "ll"]


def build():
    cfg = ModelConfig(model_type="shi", phi_domain="unit", use_anisotropy=True)
    SIM = sm.simulate(cfg, sm.SimConfig(Ns=18, Nt=6, seed=7))
    knots = sm.make_isometric_knots(0, 10, 0, 10, 4)
    data = SamplerData(
        Y=SIM["Y_missing"], sites_xy=SIM["sites_xy"], u_vec=SIM["u_mat"][:, 0],
        sigma_vec=SIM["sigma_mat"][:, 0], xi_vec=SIM["xi_mat"][:, 0],
        p_cens=SIM["p_cens"], knots_S=knots, knots_phi=knots, knots_rho=knots)
    st = SamplerSettings(
        n_iters=30, thin=2, seed=12345, adapt_size=10,
        update_S=True, update_Z=True, update_phi=True, update_rho=True,
        update_alpha0=True, update_nu=True, update_radius_S=True,
        update_anisotropy=True, update_GPD_sigma=True, update_GPD_xi=True)
    return cfg, data, st


def build_nuts():
    """Second fixture, covering the GRADIENT blocks. The original `build` runs
    z_sampler="rwmh" and nu_rho_sampler="separate", so neither NUTS gradient is
    exercised by it at all -- which was discovered the hard way when flipping
    z_grad/nu_rho_grad to their fast implementations changed no golden. This one
    drives both: per-time NUTS on Z (zblock_value_grad_*) and joint NUTS on
    (nu, rho) (_nu_rho_cp_value_grad*), which is Shi-only and needs vector rho."""
    cfg = ModelConfig(model_type="shi", phi_domain="unit", use_anisotropy=True)
    SIM = sm.simulate(cfg, sm.SimConfig(Ns=18, Nt=6, seed=7))
    knots = sm.make_isometric_knots(0, 10, 0, 10, 4)
    data = SamplerData(
        Y=SIM["Y_missing"], sites_xy=SIM["sites_xy"], u_vec=SIM["u_mat"][:, 0],
        sigma_vec=SIM["sigma_mat"][:, 0], xi_vec=SIM["xi_mat"][:, 0],
        p_cens=SIM["p_cens"], knots_S=knots, knots_phi=knots, knots_rho=knots)
    st = SamplerSettings(
        n_iters=24, thin=2, seed=12345, adapt_size=10,
        z_sampler="nuts", nu_rho_sampler="nuts", nuts_warmup=6,
        update_S=True, update_Z=True, update_phi=True, update_rho=True,
        update_alpha0=True, update_nu=True, update_radius_S=True,
        update_anisotropy=True, update_GPD_sigma=True, update_GPD_xi=True)
    return cfg, data, st


def run(mode):
    if mode.endswith("_nuts"):
        cfg, data, st = build_nuts()
        mode = mode[:-5]
    else:
        cfg, data, st = build()
    if mode == "serial":
        return run_mcmc(cfg, data, st), 0
    from sampler_mpi import run_mcmc_mpi
    from mpi4py import MPI
    return run_mcmc_mpi(cfg, data, st), MPI.COMM_WORLD.Get_rank()


def main(out_path, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR, rank = run(mode)
    if rank == 0:
        np.savez(out_path, **{k: np.asarray(TR[k]) for k in GLOBAL_KEYS if k in TR})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mpi")
