"""
ctypes bindings to the RW-integration shared library (rw_inte_cpp.so).

Provides the standard-Pareto-with-multiplicative-Log-Laplace-nugget transforms
used by Shi's model (closed-form, fast), plus the no-nugget standard-Pareto
transforms. The library is built from cpp/rw_inte_cpp.cpp; see cpp/Makefile.

Accuracy note: pRW/qRW here are accurate across the whole positive phi domain
(Monte-Carlo validated to phi=10). The closed-form DENSITY dRW USED TO suffer
catastrophic upper-tail cancellation (bias growing with alpha0), which is why the
sampler differenced pRW numerically instead -- but the incomplete-gamma accuracy fix
(cpp/rw_inte_cpp.cpp: downward recurrence for negative-order upper gammas +
regularised lower gamma P(a,x)) made this dRW MACHINE-PRECISE (validated <=3e-15 vs
mpmath across phi in [0.1,10], alpha0 in [2,5], p up to 1-1e-10). The sampler now
uses it via rw_numeric.dRW_safe, which just adds a strictly-positive floor; the old
numeric-difference path has been removed. dlog_dRW_dx is the exact log-derivative of
this same closed form (so density value and gradient are consistent).
"""
from __future__ import annotations

import ctypes
import os

import numpy as np


def _find_library() -> str:
    """Locate rw_inte_cpp.so, searching cpp/ next to this file, then CWD."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "cpp", "rw_inte_cpp.so"),
        os.path.join(here, "rw_inte_cpp.so"),
        os.path.abspath("./rw_inte_cpp.so"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "rw_inte_cpp.so not found. Build it with `make -C cpp` from the repo root "
        "(needs a C++ compiler + GSL; e.g. `conda install -c conda-forge gsl patchelf`). "
        f"Searched: {candidates}"
    )


_RW = ctypes.CDLL(_find_library())


def _bind(name):
    """Look up a C symbol, turning a STALE .so into an actionable message.

    cpp/*.so is a gitignored BUILD ARTIFACT while this file is tracked, so a pull that
    adds a binding lands on whatever binary the machine last built. ctypes then raises
    a bare "undefined symbol" at import time -- before any user code runs, on every
    rank at once -- which reads like a broken install rather than "run make". This has
    now cost a Faroe task-6 submission; say the actual fix instead."""
    try:
        return getattr(_RW, name)
    except AttributeError:
        raise ImportError(
            f"{_RW._name}: missing symbol {name!r}.\n"
            f"The compiled library is OLDER than rw_inte.py -- cpp/*.so is gitignored, "
            f"so `git pull` never updates it. Rebuild it:\n"
            f"    cd {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp')} "
            f"&& make\n"
            f"(with the conda env active, so GSL is found). The change is additive: "
            f"rebuilding does not alter any existing function, and the golden fixtures "
            f"are unchanged across it."
        ) from None

# Standard Pareto with multiplicative Log-Laplace nugget ----------------------
# Signature: (x_or_p, phi, gamma, alpha0)
_bind("pRW_standard_Pareto_multiplicative_nugget_C").restype = ctypes.c_double
_bind("pRW_standard_Pareto_multiplicative_nugget_C").argtypes = (
    ctypes.c_double,) * 4
_bind("qRW_standard_Pareto_multiplicative_nugget_C_brent").restype = ctypes.c_double
_bind("qRW_standard_Pareto_multiplicative_nugget_C_brent").argtypes = (
    ctypes.c_double,) * 4
_bind("dRW_standard_Pareto_multiplicative_nugget_C").restype = ctypes.c_double
_bind("dRW_standard_Pareto_multiplicative_nugget_C").argtypes = (
    ctypes.c_double,) * 4
# Exact d/dx log dRW(x) (closed-form derivative of the analytic density). See the
# cpp: the analytic density is tail-biased in magnitude but its log-derivative is
# smooth+accurate, so it powers the beta/phi NUTS gradients (dlog_dRW_dx below).
_bind("dlogdRW_dx_standard_Pareto_multiplicative_nugget_C").restype = ctypes.c_double
_bind("dlogdRW_dx_standard_Pareto_multiplicative_nugget_C").argtypes = (
    ctypes.c_double,) * 4
# Warm-start quantile: extra 5th arg x_guess (a previous accepted qRW for the cell).
# Seeds the Brent bracket from the guess; falls back to the exact cold solve if the
# guess can't bracket the root, so a stale guess only ever costs speed, never accuracy.
_bind("qRW_standard_Pareto_multiplicative_nugget_C_brent_warm").restype = ctypes.c_double
_bind("qRW_standard_Pareto_multiplicative_nugget_C_brent_warm").argtypes = (
    ctypes.c_double,) * 5
# d/dphi pRW -- the fused analytic derivative for the phi gradient's dX/dphi term
# (reuses the density's incomplete gammas + their order-derivatives; no qRW re-inversion).
_bind("dpRW_dphi_standard_Pareto_multiplicative_nugget_C").restype = ctypes.c_double
_bind("dpRW_dphi_standard_Pareto_multiplicative_nugget_C").argtypes = (ctypes.c_double,) * 4
# d/dphi log dRW -- the phi analogue of dlog_dRW_dx (the ∂logdRW/∂phi gradient term).
_bind("dlogdRW_dphi_standard_Pareto_multiplicative_nugget_C").restype = ctypes.c_double
_bind("dlogdRW_dphi_standard_Pareto_multiplicative_nugget_C").argtypes = (ctypes.c_double,) * 4
# incomplete-gamma ORDER derivatives: (a, z) -> d/da {Gamma,gamma}(a, z).
_bind("dupper_gamma_da_C").restype = ctypes.c_double
_bind("dupper_gamma_da_C").argtypes = (ctypes.c_double,) * 2
_bind("dlower_gamma_da_C").restype = ctypes.c_double
_bind("dlower_gamma_da_C").argtypes = (ctypes.c_double,) * 2

pRW = np.vectorize(_RW.pRW_standard_Pareto_multiplicative_nugget_C, otypes=[float])
qRW = np.vectorize(_RW.qRW_standard_Pareto_multiplicative_nugget_C_brent, otypes=[float])
qRW_warm = np.vectorize(
    _RW.qRW_standard_Pareto_multiplicative_nugget_C_brent_warm, otypes=[float])
dRW = np.vectorize(_RW.dRW_standard_Pareto_multiplicative_nugget_C, otypes=[float])
dlog_dRW_dx = np.vectorize(
    _RW.dlogdRW_dx_standard_Pareto_multiplicative_nugget_C, otypes=[float])
dpRW_dphi = np.vectorize(
    _RW.dpRW_dphi_standard_Pareto_multiplicative_nugget_C, otypes=[float])
dlog_dRW_dphi = np.vectorize(
    _RW.dlogdRW_dphi_standard_Pareto_multiplicative_nugget_C, otypes=[float])
dupper_gamma_da = np.vectorize(_RW.dupper_gamma_da_C, otypes=[float])
dlower_gamma_da = np.vectorize(_RW.dlower_gamma_da_C, otypes=[float])

# Backward-compatible long names (match Shi's RW_inte module) ------------------
pRW_standard_Pareto_multiplicative_nugget_vec = pRW
qRW_standard_Pareto_multiplicative_nugget_vec = qRW
dRW_standard_Pareto_multiplicative_nugget_vec = dRW

# Standard Pareto, no nugget (for reference / GEV block maxima) ----------------
_bind("pRW_standard_Pareto_C").restype = ctypes.c_double
_bind("pRW_standard_Pareto_C").argtypes = (ctypes.c_double,) * 3
_bind("dRW_standard_Pareto_C").restype = ctypes.c_double
_bind("dRW_standard_Pareto_C").argtypes = (ctypes.c_double,) * 3
_bind("qRW_standard_Pareto_C_brent").restype = ctypes.c_double
_bind("qRW_standard_Pareto_C_brent").argtypes = (ctypes.c_double,) * 3

pRW_no_nugget = np.vectorize(_RW.pRW_standard_Pareto_C, otypes=[float])
dRW_no_nugget = np.vectorize(_RW.dRW_standard_Pareto_C, otypes=[float])
qRW_no_nugget = np.vectorize(_RW.qRW_standard_Pareto_C_brent, otypes=[float])
