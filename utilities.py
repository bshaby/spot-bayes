"""
20250826 - Muyang
    utilities for the MCMC sampler using multiplicative log-laplace nugget
"""
# %% imports

import sys
import scipy
import math
import numpy         as np
import scipy.special as sc
from   scipy.spatial import distance
import rw_inte as RW_inte
try:
    import numba as nb
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
# %% ScaleMixture functions

# Link function g(Z) --------------------------------------

def norm_to_stdPareto(Z):
    pNorm = scipy.stats.norm.cdf(x = Z)
    return(scipy.stats.pareto.ppf(q = pNorm, b = 1))

def stdPareto_to_Norm(W):
    pPareto = scipy.stats.pareto.cdf(x = W, b = 1)
    return(scipy.stats.norm.ppf(q = pPareto))

g    = np.vectorize(norm_to_stdPareto)
ginv = np.vectorize(stdPareto_to_Norm)

# Numba-friendly g(Z) for compiled likelihoods
if _HAVE_NUMBA:
    _SQRT2 = math.sqrt(2.0)

    @nb.njit(cache=True)
    def g_nb(Z):
        # Z is expected to be 1-D array
        out = np.empty(Z.size, dtype=np.float64)
        for i in range(Z.size):
            z = Z[i]
            # survival function of N(0,1): 0.5*erfc(z/sqrt(2))
            sf = 0.5 * math.erfc(z / _SQRT2)
            out[i] = np.inf if sf == 0.0 else (1.0 / sf)
        return out
else:
    def g_nb(Z):
        return g(Z)

# RW distribution functions -------------------------------

pRW = RW_inte.pRW_standard_Pareto_multiplicative_nugget_vec
qRW = RW_inte.qRW_standard_Pareto_multiplicative_nugget_vec
dRW = RW_inte.dRW_standard_Pareto_multiplicative_nugget_vec

# %% distribution functions ---------------------------------------------------

# Lévy distribution ---------------------------------------

def rlevy(n, m = 0, s = 1):
    if np.any(s < 0):
        sys.exit("s must be positive")
    return s/scipy.stats.norm.ppf(1-scipy.stats.uniform.rvs(0,1,n)/2)**2 + m

def qlevy(p, m = 0, s = 1):
    return m + s/(2*(scipy.special.erfcinv(p))**2)

# Generalized Pareto distribution -------------------------
# (dGP/pGP/qGP are the building blocks of the censored-GPD response pCGP/dCGP/qCGP)

def dGP(y, loc, scale, shape):
    return scipy.stats.genpareto.pdf(y, c=shape, loc=loc, scale=scale)
def pGP(y, loc, scale, shape):
    return scipy.stats.genpareto.cdf(y, c=shape, loc=loc, scale=scale)
def qGP(p, loc, scale, shape):
    return scipy.stats.genpareto.ppf(p, c=shape, loc=loc, scale=scale)


def _gp_logpdf_np(y, loc, scale, shape):
    # log pdf of Generalized Pareto distribution
    z = (y - loc) / scale
    small = np.abs(shape) < 1e-8
    t = 1.0 + shape * z
    out = np.full_like(z, -np.inf, dtype=np.float64)
    # xi \approx 0
    out[small] = -np.log(scale[small]) - z[small]
    # xi \not 0 and t > 0
    ok = (~small) & (t > 0)
    out[ok] = -np.log(scale[ok]) - (1.0/shape[ok] + 1.0) * np.log(t[ok])
    return out

# Numba-friendly CGP log-pdf
if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _gp_logpdf_nb(y, loc, scale, shape):
        out = np.empty_like(y)
        for i in range(y.size):
            z = (y[i] - loc[i]) / scale[i]
            xi = shape[i]
            if abs(xi) < 1e-8:
                out[i] = -math.log(scale[i]) - z
            else:
                t = 1.0 + xi * z
                if t <= 0.0:
                    out[i] = -math.inf
                else:
                    out[i] = -math.log(scale[i]) - (1.0/xi + 1.0) * math.log(t)
        return out

# Censored Generalized Pareto distribution ----------------
# censored at probability p
# the loc should be the threshold u: p = P(Y <= u)
# pX is the CDF of X: pX = P(X <= x)
# p_thres is the censoring threshold probability 

def pCGP(y, p, loc, scale, shape):
    return np.where(y <= loc, p,
                              p + (1-p) * pGP(y, loc, scale, shape))
    # return p + (1-p) * pGP(y, loc, scale, shape)
def dCGP(y, p, loc, scale, shape):
    return np.where(y <= loc, 0.0,
                              (1-p) * dGP(y, loc, scale, shape))
    # return (1-p) * dGP(y, loc, scale, shape)
def qCGP(pX, p_thres, loc, scale, shape):
    return np.where(pX <= p_thres, loc,
                                   qGP((pX - p_thres)/(1-p_thres), loc, scale, shape))
    # return qGP((pX - p_thres)/(1-p_thres), loc, scale, shape)

def _cgp_logpdf_np(y, p, loc, scale, shape):
    # log dCGP: 0 for y <= loc; log(1-p) + log GP pdf for y > loc
    out = np.full_like(y, -np.inf, dtype=np.float64)
    exc = y > loc
    if np.any(exc):
        out[exc] = np.log1p(-p) + _gp_logpdf_np(y[exc], loc[exc], scale[exc], shape[exc])
    return out

if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def _cgp_logpdf_nb(y, p, loc, scale, shape):
        out = np.empty_like(y)
        log1mp = math.log1p(-p)
        for i in range(y.size):
            if y[i] <= loc[i]:
                out[i] = -math.inf
            else:
                z = (y[i] - loc[i]) / scale[i]
                xi = shape[i]
                if abs(xi) < 1e-8:
                    gp = -math.log(scale[i]) - z
                else:
                    t = 1.0 + xi * z
                    if t <= 0.0:
                        gp = -math.inf
                    else:
                        gp = -math.log(scale[i]) - (1.0/xi + 1.0) * math.log(t)
                out[i] = log1mp + gp
        return out

# Log–Laplace distribution ---------------------------------------------------
# Parameterization: location 0, scale = 1/alpha0  (same as notes)
# Support: epsilon > 0

def _asarray_strict(x):
    """Ensure ndarray float64 for vectorized math."""
    return np.asarray(x, dtype=np.float64)

def loglaplace_pdf(eps, alpha0):
    """
    f_ε(ε) =
      0,                      ε = 0,
      (α0/2) * ε^{α0-1},      0 < ε < 1,
      (α0/2) * ε^{-α0-1},     ε ≥ 1,
    """
    eps = _asarray_strict(eps)
    a0  = float(alpha0)
    out = np.zeros_like(eps, dtype=np.float64)
    pos = eps > 0.0
    lt1 = eps < 1.0
    ge1 = ~lt1
    if np.any(pos):
        out[lt1] = 0.5 * a0 * np.power(eps[lt1], a0-1.0)
        out[ge1] = 0.5 * a0 * np.power(eps[ge1], -a0-1.0)
    return out

def loglaplace_logpdf(eps, alpha0):
    """
    log f_ε(ε) =
      -inf,                          ε ≤ 0,
      log(α0/2) + (α0-1) log ε,      0 < ε < 1,
      log(α0/2) + (-(α0+1)) log ε,   ε ≥ 1.
    """
    eps = _asarray_strict(eps)
    a0  = float(alpha0)
    out = np.full_like(eps, fill_value=-np.inf, dtype=np.float64)
    pos = eps > 0.0
    lt1 = eps < 1.0
    ge1 = ~lt1
    if np.any(pos):
        c        = np.log(a0) - np.log(2.0)
        out[lt1] = c + (a0 - 1.0) * np.log(eps[lt1])
        out[ge1] = c + (-(a0 + 1.0)) * np.log(eps[ge1])
    return out

def loglaplace_cdf(eps, alpha0):
    """
    F_ε(ε) =
      0                                 ε ≤ 0,
      0.5 * ε^{α0},                     0 < ε < 1,
      1 - 0.5 * ε^{-α0},                ε ≥ 1.
    """
    eps = _asarray_strict(eps)
    a0  = float(alpha0)
    out = np.zeros_like(eps, dtype=np.float64)
    pos = eps > 0.0
    lt1 = eps < 1.0
    ge1 = ~lt1
    if np.any(pos):
        out[lt1] = 0.5 * np.power(eps[lt1], a0)
        out[ge1] = 1.0 - 0.5 * np.power(eps[ge1], -a0)
    return out

def loglaplace_logcdf(eps, alpha0):
    """
    log F_ε(ε) =
      -inf,                                               ε ≤ 0,
      log(0.5) + α0 log ε,                                0 < ε < 1,
      log(1 - 0.5 ε^{-α0}) = log1p(-0.5 e^{-α0 log ε}),   ε ≥ 1.
    """
    eps = _asarray_strict(eps)
    a0  = float(alpha0)
    out = np.full_like(eps, fill_value=-np.inf, dtype=np.float64)
    pos = eps > 0.0
    lt1 = eps < 1.0
    ge1 = ~lt1
    if np.any(pos):
        out[lt1] = np.log(0.5) + a0 * np.log(eps[lt1])
        out[ge1] = np.log1p(-0.5 * np.power(eps[ge1], -a0))
    return out

# Numba-friendly log-CDF / log-PDF for log-laplace
if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def loglaplace_logcdf_nb(eps, alpha0):
        out = np.empty_like(eps)
        for i in range(eps.size):
            e = eps[i]
            if e <= 0.0:
                out[i] = -np.inf
            elif e < 1.0:
                out[i] = math.log(0.5) + alpha0 * math.log(e)
            else:
                out[i] = math.log1p(-0.5 * (e ** (-alpha0)))
        return out

    @nb.njit(cache=True)
    def loglaplace_logpdf_nb(eps, alpha0):
        out = np.empty_like(eps)
        c = math.log(alpha0) - math.log(2.0)
        for i in range(eps.size):
            e = eps[i]
            if e <= 0.0:
                out[i] = -np.inf
            elif e < 1.0:
                out[i] = c + (alpha0 - 1.0) * math.log(e)
            else:
                out[i] = c + (-(alpha0 + 1.0)) * math.log(e)
        return out

# Backward-compatible names (density/cdf with optional log= flag)
def dloglaplace(eps, alpha0, log=False):
    return loglaplace_logpdf(eps, alpha0) if log else loglaplace_pdf(eps, alpha0)

def ploglaplace(eps, alpha0, log=False):
    return loglaplace_logcdf(eps, alpha0) if log else loglaplace_cdf(eps, alpha0)


# Log–Laplace random variate generator --------------------
def rloglaplace(size, alpha0, rng=None):
    """
    Draw random samples from Log-Laplace(α0).

    Equivalent to: if U ~ Uniform(0,1),
        if U < 0.5:  eps = U^{1/α0}
        else:        eps = (1-U)^{-1/α0}

    Parameters
    ----------
    size : int or tuple
        Number of samples (or shape).
    alpha0 : float
        Tail parameter (>0, ≠1).
    rng : np.random.Generator, optional
        Source of randomness. Pass a seeded Generator for REPRODUCIBLE draws
        (e.g. the predictive CRPS path); defaults to numpy's global RNG.

    Returns
    -------
    eps : ndarray
        Samples from Log–Laplace distribution.
    """
    draw = (rng.uniform if rng is not None else np.random.uniform)
    U = draw(0.0, 1.0, size=size)
    out = np.empty_like(U)
    mask = U < 0.5
    out[mask] = np.power(U[mask] * 2.0, 1.0/alpha0)
    out[~mask] = np.power(0.5 / (1.0 - U[~mask]), 1.0/alpha0)
    return out

# %% smoothing kernels ========================================================

# Gaussian Smoothing Kernel -------------------------------

def weights_fun(d,radius,h=1, cutoff=True):
    # When d > fit radius, the weight will be zero
    # h is the bandwidth parameter
    if(isinstance(d, (int, np.int64, float))): 
        d=np.array([d])
    tmp = np.exp(-d**2/(2*h))
    if cutoff: 
        tmp[d>radius] = 0
    return tmp/np.sum(tmp)

# Wendland compactly-supported basis ----------------------

def wendland_weights_fun(d, theta, k=0, dimension=2, derivative=0):
    # fields_Wendland(d, theta = 1, dimension, k, derivative=0, phi=NA)
    # theta: the range where the basis value is non-zero, i.e. [0, theta]
    # dimension: dimension of locations 
    # k: smoothness of the function at zero.
    if(isinstance(d, (int, np.int64, float))): 
        d=np.array([d])      
    d = d/theta
    l = np.floor(dimension/2) + k + 1
    if (k==0): 
        res = np.where(d < 1, (1-d)**l, 0)
    if (k==1):
        res = np.where(d < 1, (1-d)**(l+k) * ((l+1)*d + 1), 0)
    if (k==2):
        res = np.where(d < 1, (1-d)**(l+k) * ((l**2+4*l+3)*d**2 + (3*l+6) * d + 3), 0)
    if (k==3):
        res = np.where(d < 1, (1-d)**(l+k) * ((l**3+9*l**2+23*l+15)*d**3 + 
                                            (6*l**2+36*l+45) * d**2 + (15*l+45) * d + 15), 0)
    if (k>3):
        sys.exit("k must be less than 4")
    return res/np.sum(res)

# Building weight matrices --------------------------------

def build_weight_matrix(sites_xy,
                        knots_xy,
                        kernel='gaussian',
                        *,
                        radius=None,
                        bandwidth=None,
                        wendland_theta=None,
                        cutoff=True):
    """Compute a single site-to-knot weight matrix.

    Parameters
    ----------
    sites_xy : (Ns, 2)
    knots_xy : (k, 2)
    kernel   : 'gaussian' or 'wendland'
    radius   : float, Gaussian kernel radius parameter (passed to weights_fun)
    bandwidth: float, Gaussian kernel bandwidth (passed to weights_fun)
    wendland_theta : float or (k,), range parameter(s) for Wendland basis, passed to wendland_weights_fun
    cutoff   : bool, cutoff flag for Gaussian kernel

    Returns
    -------
    W : (Ns, k) ndarray with per-site normalized weights
    """
    Ns = int(sites_xy.shape[0])
    k  = int(knots_xy.shape[0])
    W  = np.empty((Ns, k), dtype=float)

    for i in range(Ns):
        d = scipy.spatial.distance.cdist(sites_xy[i, :].reshape(1, 2), knots_xy)  # (1, k) distance from knots
        if kernel == 'wendland':
            theta = wendland_theta if wendland_theta is not None else radius
            W[i, :] = wendland_weights_fun(d, theta)
        elif kernel == 'gaussian':
            if bandwidth is None: raise ValueError("bandwidth must be provided for Gaussian kernel")
            rad = 0.0 if radius is None else radius
            W[i, :] = weights_fun(d, rad, bandwidth, cutoff=cutoff)
        else:
            raise ValueError("kernel must be 'gaussian' or 'wendland'")

    return W

# Isometric Grid ------------------------------------------

def _iso_grid(minX, maxX, minY, maxY, N_outer_grid):
    """Builds the usual outer/inner isometric grid and clips to domain."""
    h = (maxX - minX) / (int(2 * np.sqrt(N_outer_grid)) - 1)
    v = (maxY - minY) / (int(2 * np.sqrt(N_outer_grid)) - 1)
    x_pos = np.linspace(minX + h/2, maxX + h/2, num=int(2 * np.sqrt(N_outer_grid)))
    y_pos = np.linspace(minY + v/2, maxY + v/2, num=int(2 * np.sqrt(N_outer_grid)))
    x_outer, x_inner = x_pos[0::2], x_pos[1::2]
    y_outer, y_inner = y_pos[0::2], y_pos[1::2]
    Xo, Yo = np.meshgrid(x_outer, y_outer)
    Xi, Yi = np.meshgrid(x_inner, y_inner)
    knots  = np.vstack([np.c_[Xo.ravel(), Yo.ravel()], np.c_[Xi.ravel(), Yi.ravel()]])
    keep   = (knots[:, 0] > minX) & (knots[:, 0] < maxX) & (knots[:, 1] > minY) & (knots[:, 1] < maxY)
    return knots[keep]

# %% Likelihood functions =============================================================================================

# =========================================================
# Priors ==================================================
# =========================================================

# Prior log-densities + parameter transforms now live in priors.py. They are
# re-exported here so existing `utilities.<name>` call sites keep working.
from priors import (  # noqa: E402,F401
    S_logprior_1t, Z_logprior_1t, phi_logprior, rho_logprior, logrho_logprior,
    alpha0_from_eta, eta_from_alpha0, eta_logprior, nu_logprior,
    radius_S_logprior, sum_logparts,
)

# =========================================================
# Likelihood ==============================================
# =========================================================

# Y|X^* log likelihood
def Y_loglik_1t_qRWdRWout(Y_vec, p, u_vec, sigma_vec, xi_vec,
                          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, alpha0,
                          qRW_vec, dRW_vec,censored_idx, exceed_idx):
    """
    Conditional log-likelihood for the multiplicative Log-Laplace nugget model.

    Data/indices
    ------------
    Y_vec        : (Ns,) observed responses
    p            : scalar censoring probability (P(Y ≤ u))
    u_vec        : (Ns,) GP thresholds
    sigma_vec    : (Ns,) GP scales
    xi_vec       : (Ns,) GP shapes
    censored_idx : 1-D integer array of indices with Y ≤ u
    exceed_idx   : 1-D integer array of indices with Y > u

    Model pieces (already prepared outside for efficiency)
    ------------------------------------------------------
    X_star       : (Ns,) latent smooth process  = (R ** phi) * g(Z)
    X            : (Ns,) transformed quantiles  = qRW(pCGP(Y), phi, γ̄, α0)
    dX           : (Ns,) marginal pdf of X at X  = dRW(X; phi, γ̄, α0)

    Returns
    -------
    scalar log-likelihood:
        sum_censored  log F_ε( X / X_star )
      + sum_exceed    [ log f_{X|X*}(X|X*) + log dCGP(Y) - log dX(X) ]
    """
    # Precompute smooth process and transformed X passed in by caller
    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW_vec
    dX     = dRW_vec

    # Guard: positivity (model implies positivity, but be safe)
    if np.any(X_star <= 0) or np.any(X <= 0):
        raise ValueError("X and X_star must be strictly positive under the multiplicative model.")

    # Vectorize indices
    censored_idx = np.asarray(censored_idx, dtype=int)
    exceed_idx   = np.asarray(exceed_idx,   dtype=int)
    # censored_idx = np.where(Y_vec <= u_vec)[0]
    # exceed_idx   = np.where(Y_vec >  u_vec)[0]

    # ========================
    # Censored contribution
    # ========================
    # At censored sites, pCGP(Y)=p so X equals x0 := qRW(p; phi, γ̄, α0).
    # Likelihood contribution: F_ε( x0 / X* ) = F_ε( X / X* )
    if censored_idx.size:
        ratio_c = X[censored_idx] / X_star[censored_idx]
        censored_ll = np.sum(ploglaplace(ratio_c, alpha0, log=True))
    else:
        censored_ll = 0.0

    # ========================
    # Exceedance contribution
    # ========================
    # At exceed sites, contribution:
    #   log f_{X|X*}(X|X*) + log dCGP(Y; u,σ,ξ,p) - log dX(X)
    # where
    #   f_{X|X*}(x|x*) = (1/x*) f_ε(x/x*)
    if exceed_idx.size:
        ratio_e = X[exceed_idx] / X_star[exceed_idx]
        log_f_cond = -np.log(X_star[exceed_idx]) + dloglaplace(ratio_e, alpha0, log=True)

        # GP part for Y (density of the exceedance under CGP)
        log_gY = np.log(dCGP(Y_vec[exceed_idx],
                             p,
                             u_vec[exceed_idx],
                             sigma_vec[exceed_idx],
                             xi_vec[exceed_idx]))

        # Jacobian from Y→X
        log_jac = log_gY - np.log(dX[exceed_idx])

        exceed_ll = np.sum(log_f_cond + log_jac)
    else:
        exceed_ll = 0.0

    return censored_ll + exceed_ll

def Y_loglik_1t_qRWdRWout_np(Y_vec, p, u_vec, sigma_vec, xi_vec,
                            R_vec, Z_vec, K, phi_vec, gamma_bar_vec, alpha0,
                            qRW_vec, dRW_vec, censored_idx, exceed_idx):
    # same as original, but uses NumPy-only logpdf
    X_star = (R_vec ** phi_vec) * g_nb(Z_vec)
    X      = qRW_vec
    dX     = dRW_vec

    censored_idx = np.asarray(censored_idx, dtype=np.int64)
    exceed_idx   = np.asarray(exceed_idx,   dtype=np.int64)

    if censored_idx.size:
        ratio_c = X[censored_idx] / X_star[censored_idx]
        censored_ll = np.sum(loglaplace_logcdf_nb(ratio_c, alpha0))

    else:
        censored_ll = 0.0

    if exceed_idx.size:
        ratio_e = X[exceed_idx] / X_star[exceed_idx]
        log_f_cond = -np.log(X_star[exceed_idx]) + loglaplace_logpdf_nb(ratio_e, alpha0)
        log_gY = _cgp_logpdf_nb(Y_vec[exceed_idx], p, u_vec[exceed_idx],
                                sigma_vec[exceed_idx], xi_vec[exceed_idx])
        log_jac = log_gY - np.log(dX[exceed_idx])
        exceed_ll = np.sum(log_f_cond + log_jac)
    else:
        exceed_ll = 0.0

    return censored_ll + exceed_ll

if _HAVE_NUMBA:
    Y_loglik_1t_qRWdRWout_np = nb.njit(cache=True)(Y_loglik_1t_qRWdRWout_np)

# Preferred per-time likelihood: the numba-compiled version when available
# (numerically identical to the numpy version, ~1000x faster because the numpy
# g() calls scipy per element), else the pure-numpy fallback. Used by the
# sampler and predictive scoring for both the Shi and Wang model paths.
Y_loglik_1t_best = (Y_loglik_1t_qRWdRWout_np if _HAVE_NUMBA
                    else Y_loglik_1t_qRWdRWout)


def Y_loglik_1t_pointwise(Y_vec, p, u_vec, sigma_vec, xi_vec,
                          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, alpha0,
                          qRW_vec, dRW_vec, censored_idx, exceed_idx):
    """PER-SITE conditional log-likelihood vector (sums to Y_loglik_1t_best).

    Identical model to Y_loglik_1t_qRWdRWout_np but returns the length-n vector of
    per-site contributions rather than their sum -- the pointwise quantity the
    conditional WAIC needs. Used only in post-processing, so pure NumPy."""
    X_star = (R_vec ** phi_vec) * g_nb(Z_vec)
    X = qRW_vec; dX = dRW_vec
    out = np.zeros(np.asarray(Y_vec).shape[0])
    censored_idx = np.asarray(censored_idx, dtype=np.int64)
    exceed_idx = np.asarray(exceed_idx, dtype=np.int64)
    if censored_idx.size:
        out[censored_idx] = loglaplace_logcdf_nb(
            X[censored_idx] / X_star[censored_idx], alpha0)
    if exceed_idx.size:
        ratio_e = X[exceed_idx] / X_star[exceed_idx]
        log_f_cond = -np.log(X_star[exceed_idx]) + loglaplace_logpdf_nb(ratio_e, alpha0)
        log_gY = _cgp_logpdf_nb(Y_vec[exceed_idx], p, u_vec[exceed_idx],
                                sigma_vec[exceed_idx], xi_vec[exceed_idx])
        out[exceed_idx] = log_f_cond + log_gY - np.log(dX[exceed_idx])
    return out


# %% spatial covariance functions copied from ns_cov ==========================

## -------------------------------------------------------------------------- ##
##               Implement the Matern correlation function (stationary)
## -------------------------------------------------------------------------- ##

def cov_spatial(r, cov_model = "exponential", cov_pars = np.array([1,1]), kappa = 0.5):
    # Input from a matrix of pairwise distances and a vector of parameters
    if type(r).__module__!='numpy' or isinstance(r, np.float64):
        r = np.array(r)
    if np.any(r<0):
        sys.exit('Distance argument must be nonnegative.')
    r[r == 0] = 1e-10
    
    if cov_model != "matern" and cov_model != "gaussian" and cov_model != "exponential" :
        sys.exit("Please specify a valid covariance model (matern, gaussian, or exponential).")
    
    if cov_model == "exponential":
        C = np.exp(-r)
    
    if cov_model == "gaussian" :
        C = np.exp(-(r^2))
  
    if cov_model == "matern" :
        range = 1
        nu = kappa
        part1 = 2 ** (1 - nu) / sc.gamma(nu)
        part2 = (r / range) ** nu
        part3 = sc.kv(nu, r / range)
        C = part1 * part2 * part3
    return C
## -------------------------------------------------------------------------- ##

def ns_cov_precompute(coords):
    """Precompute pairwise coordinate distance pieces for ns_cov_cached.

    Returns a dict with:
        d1_sq: (N,N) (x_i - x_j)^2
        d2_sq: (N,N) (y_i - y_j)^2
        d12  : (N,N) (x_i - x_j)*(y_i - y_j)
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        sys.exit('coords must be N x 2 array.')
    x = coords[:, 0]
    y = coords[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return {
        'd1_sq': dx * dx,
        'd2_sq': dy * dy,
        'd12': dx * dy,
    }

def ns_cov_cached(range_vec, sigsq_vec, dist_cache, kappa = 0.5, cov_model = "matern"):
    """Nonstationary covariance using precomputed distance pieces.

    dist_cache should be produced by ns_cov_precompute(coords).
    """
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
        range_vec = np.array(range_vec)
        sigsq_vec = np.array(sigsq_vec)

    N = range_vec.shape[0] # Number of spatial locations
    d1_sq = dist_cache['d1_sq']
    d2_sq = dist_cache['d2_sq']
    d12   = dist_cache['d12']
    if d1_sq.shape[0] != N:
        sys.exit('dist_cache size does not match range_vec.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0, N)
    ones = np.repeat(1, N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) --- 
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones  
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones 
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix pieces (precomputed)
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    Dist_mat_sqd = inv11*d1_sq + 2*inv12*d12 + inv22*d2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = cov_spatial(Dist_mat, cov_model = cov_model, cov_pars = np.array([1,1]), kappa = kappa)
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec)) 
    return(Spatial_cov)

# --- Numba-compiled kernels and cached covariance (fixed kernels) ------------
if _HAVE_NUMBA:
    @nb.njit(cache=True)
    def cov_spatial_exp_nb(r):
        return np.exp(-r)

    @nb.njit(cache=True)
    def cov_spatial_gaussian_nb(r):
        return np.exp(-(r * r))

    @nb.njit(cache=True)
    def cov_spatial_matern32_nb(r):
        # ν = 1.5: (1 + r) exp(-r)
        return (1.0 + r) * np.exp(-r)

    @nb.njit(cache=True)
    def cov_spatial_matern52_nb(r):
        # ν = 2.5: (1 + r + r^2/3) exp(-r)
        return (1.0 + r + (r * r) / 3.0) * np.exp(-r)

    @nb.njit(cache=True)
    def _ns_cov_cached_core_nb(range_vec, sigsq_vec, d1_sq, d2_sq, d12):
        N = range_vec.shape[0]
        arg11 = range_vec
        arg22 = range_vec
        arg12 = np.zeros(N)
        ones = np.ones(N)
        det1  = arg11*arg22 - arg12*arg12

        mat11_1 = np.reshape(arg11, (N, 1)) * ones
        mat11_2 = np.reshape(ones, (N, 1)) * arg11
        mat22_1 = np.reshape(arg22, (N, 1)) * ones
        mat22_2 = np.reshape(ones, (N, 1)) * arg22
        mat12_1 = np.reshape(arg12, (N, 1)) * ones
        mat12_2 = np.reshape(ones, (N, 1)) * arg12

        mat11 = 0.5*(mat11_1 + mat11_2)
        mat22 = 0.5*(mat22_1 + mat22_2)
        mat12 = 0.5*(mat12_1 + mat12_2)

        det12 = mat11*mat22 - mat12*mat12

        Scale_mat = np.diag(det1**0.25) @ np.sqrt(1.0/det12) @ np.diag(det1**0.25)

        inv11 = mat22/det12
        inv22 = mat11/det12
        inv12 = -mat12/det12

        Dist_mat_sqd = inv11*d1_sq + 2.0*inv12*d12 + inv22*d2_sq
        Dist_mat = np.sqrt(np.maximum(Dist_mat_sqd, 0.0))
        return Scale_mat, Dist_mat

    @nb.njit(cache=True)
    def _ns_cov_cached_exp_nb_core(range_vec, sigsq_vec, d1_sq, d2_sq, d12):
        Scale_mat, Dist_mat = _ns_cov_cached_core_nb(range_vec, sigsq_vec, d1_sq, d2_sq, d12)
        Unscl_corr = cov_spatial_exp_nb(Dist_mat)
        NS_corr = Scale_mat * Unscl_corr
        return np.diag(sigsq_vec) @ NS_corr @ np.diag(sigsq_vec)

    @nb.njit(cache=True)
    def _ns_cov_cached_gaussian_nb_core(range_vec, sigsq_vec, d1_sq, d2_sq, d12):
        Scale_mat, Dist_mat = _ns_cov_cached_core_nb(range_vec, sigsq_vec, d1_sq, d2_sq, d12)
        Unscl_corr = cov_spatial_gaussian_nb(Dist_mat)
        NS_corr = Scale_mat * Unscl_corr
        return np.diag(sigsq_vec) @ NS_corr @ np.diag(sigsq_vec)

    @nb.njit(cache=True)
    def _ns_cov_cached_matern32_nb_core(range_vec, sigsq_vec, d1_sq, d2_sq, d12):
        Scale_mat, Dist_mat = _ns_cov_cached_core_nb(range_vec, sigsq_vec, d1_sq, d2_sq, d12)
        Unscl_corr = cov_spatial_matern32_nb(Dist_mat)
        NS_corr = Scale_mat * Unscl_corr
        return np.diag(sigsq_vec) @ NS_corr @ np.diag(sigsq_vec)

    @nb.njit(cache=True)
    def _ns_cov_cached_matern52_nb_core(range_vec, sigsq_vec, d1_sq, d2_sq, d12):
        Scale_mat, Dist_mat = _ns_cov_cached_core_nb(range_vec, sigsq_vec, d1_sq, d2_sq, d12)
        Unscl_corr = cov_spatial_matern52_nb(Dist_mat)
        NS_corr = Scale_mat * Unscl_corr
        return np.diag(sigsq_vec) @ NS_corr @ np.diag(sigsq_vec)

    def ns_cov_cached_exp_nb(range_vec, sigsq_vec, dist_cache):
        return _ns_cov_cached_exp_nb_core(range_vec, sigsq_vec,
                                          dist_cache['d1_sq'], dist_cache['d2_sq'], dist_cache['d12'])

    def ns_cov_cached_gaussian_nb(range_vec, sigsq_vec, dist_cache):
        return _ns_cov_cached_gaussian_nb_core(range_vec, sigsq_vec,
                                               dist_cache['d1_sq'], dist_cache['d2_sq'], dist_cache['d12'])

    def ns_cov_cached_matern32_nb(range_vec, sigsq_vec, dist_cache):
        return _ns_cov_cached_matern32_nb_core(range_vec, sigsq_vec,
                                               dist_cache['d1_sq'], dist_cache['d2_sq'], dist_cache['d12'])

    def ns_cov_cached_matern52_nb(range_vec, sigsq_vec, dist_cache):
        return _ns_cov_cached_matern52_nb_core(range_vec, sigsq_vec,
                                               dist_cache['d1_sq'], dist_cache['d2_sq'], dist_cache['d12'])
else:
    # Fallbacks when numba is unavailable
    def ns_cov_cached_exp_nb(range_vec, sigsq_vec, dist_cache):
        return ns_cov_cached(range_vec, sigsq_vec, dist_cache, kappa=0.5, cov_model="exponential")

    def ns_cov_cached_gaussian_nb(range_vec, sigsq_vec, dist_cache):
        return ns_cov_cached(range_vec, sigsq_vec, dist_cache, kappa=0.5, cov_model="gaussian")

    def ns_cov_cached_matern32_nb(range_vec, sigsq_vec, dist_cache):
        return ns_cov_cached(range_vec, sigsq_vec, dist_cache, kappa=1.5, cov_model="matern")

    def ns_cov_cached_matern52_nb(range_vec, sigsq_vec, dist_cache):
        return ns_cov_cached(range_vec, sigsq_vec, dist_cache, kappa=2.5, cov_model="matern")



## -------------------------------------------------------------------------- ##
##               Calculate a locally isotropic spatial covariance
## -------------------------------------------------------------------------- ##

def ns_cov(range_vec, sigsq_vec, coords, kappa = 0.5, cov_model = "matern"):
    ## Arguments:
    ##    range_vec = N-vector of range parameters (one for each location) 
    ##    sigsq_vec = N-vector of marginal variance parameters (one for each location)
    ##    coords = N x 2 matrix of coordinates
    ##    cov.model = "matern" --> underlying covariance model: "gaussian", "exponential", or "matern"
    ##    kappa = 0.5 --> Matern smoothness, scalar
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
        range_vec = np.array(range_vec)
        sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N: 
        sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) --- 
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones  
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones 
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = cov_spatial(Dist_mat, cov_model = cov_model, cov_pars = np.array([1,1]), kappa = kappa)
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec)) 
    return(Spatial_cov)
    

def ns_cov_interp(range_vec, sigsq_vec, coords, tck):
    # Using the grid of values to interpolate because sc.special.kv is computationally expensive
    # tck is the output function of sc.interpolate.pchip (Contains information about roughness kappa)
    # ** Has to be Matern model **
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
        range_vec = np.array(range_vec)
        sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N:
        sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) ---
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = np.ones(Dist_mat_sqd.shape)
    Unscl_corr[Dist_mat_sqd>0] = tck(Dist_mat[Dist_mat_sqd>0])
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec))
    return(Spatial_cov)
## -------------------------------------------------------------------------- ##

#########################################################################################
# Write my own covariance function ######################################################
#########################################################################################
# note: gives same result as Likun's
#       paremeterization is same, up to a constant in specifying the range

# def matern_correlation(d, range, nu):
#     # using wikipedia definition
#     part1 = 2**(1-nu)/scipy.special.gamma(nu)
#     part2 = (np.sqrt(2*nu) * d / range)**nu
#     part3 = scipy.special.kv(nu, np.sqrt(2*nu) * d / range)
#     return(part1*part2*part3)
# matern_correlation_vec = np.vectorize(matern_correlation, otypes=[float])

# # pairwise_distance = scipy.spatial.distance.pdist(sites_xy)
# # matern_correlation_vec(pairwise_distance, 1, nu) # gives same result as skMatern(sites_xy)
# # tri = np.zeros((4,4))
# # tri[np.triu_indices(4,1)] = matern_correlation_vec(pairwise_distance, 1, 1)
# # tri + tri.T + np.identity(4)

# # Note:
# #       K[i,j] (row i, col j) means the correlation between site_i and site_j
# #       np.mean(np.round(K,3) == np.round(K_current, 3)) # 1.0, meaning they are the same. 
# K = np.full(shape = (Ns, Ns), fill_value = 0.0)
# for i in range(Ns):
#     for j in range(i+1, Ns):
#         site_i = sites_xy[i,]
#         site_j = sites_xy[j,]
#         d = scipy.spatial.distance.pdist([site_i, site_j])
#         rho_i = range_vec[i]
#         rho_j = range_vec[j]
#         sigma_i = sigsq_vec[i]
#         sigma_j = sigsq_vec[j]
#         M = matern_correlation(d/np.sqrt((rho_i + rho_j)/2), 1, 0.5)
#         C = sigma_i * sigma_j * (np.sqrt(rho_i*rho_j)) * (1/((rho_i + rho_j)/2)) * M
#         K[i,j] = C[0]
# K = K + K.T + sigsq * np.identity(Ns)
