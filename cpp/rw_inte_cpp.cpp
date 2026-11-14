// g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas
// g++ -I$GSL_INC -I$BOOST_INC -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L$BOOST_LIB -L$GSL_LIB -o RW_inte_cpp.so -lgsl -lgslcblas
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

// a non-fatal, logging error handler (replaces the aborting default)
static void nonfatal_gsl_handler(const char* reason,
                                 const char* file,
                                 int line,
                                 int gsl_errno) {
    fprintf(stderr, "[GSL %s] %s:%d  code=%d  %s\n",
            gsl_strerror(gsl_errno), file, line, gsl_errno, reason);
}
__attribute__((constructor))
static void set_nonfatal_gsl_handler_on_load() {
    gsl_set_error_handler(&nonfatal_gsl_handler);
}

// double(*)[3] params_ptr         ---- treats params_ptr as a pointer to double[3] isntead of a pointer to void
// *(double(*)[3]) params_ptr      ---- derefernece the pointer
// (*(double(*)[3]) params_ptr)[2] ---- access the third element of the dereferenced double[3]
extern "C"
{

// Define incomplete gamma functions (use for standard Pareto)
// double upper_gamma_C(double a, double x){ // x is the integration lower bound
//     return gsl_sf_gamma_inc(a, x);
// }
// double lower_gamma_C(double a, double x){ // x is the integration uppder bound
//     return gsl_sf_gamma(a) - upper_gamma_C(a, x);
// }    

// New: safe wrappers that report status but don't abort
inline double _upper_gamma_gsl(double a, double x) {
    gsl_sf_result U;
    int st = gsl_sf_gamma_inc_e(a, x, &U);        // U = Γ(a, x)
    if (st == GSL_EUNDRFLW) return 0.0;           // true underflow ⇒ numerically 0
    // Optional: log other nonzero statuses
    if (st) fprintf(stderr, "[GSL] Γ(%g,%g) status=%s\n", a, x, gsl_strerror(st));
    return U.val;
}

// GSL's gsl_sf_gamma_inc is accurate for a > 0 but loses many significant digits
// for NEGATIVE-order upper incomplete gammas (e.g. Γ(1/2 - α0φ, λ) = Γ(-3.3, λ)
// deep in the RW tail, where 1/2 - α0φ < 0). That error is invisible in pRW (its
// offending term carries an x^{-α0} factor that suppresses it) but biases dRW by
// ~5-70% and dlogdRW_dx by ~15% -- enough to manufacture the -log(dRW) likelihood
// CLIFF that pins alpha0/beta. For a <= 0 we evaluate at the first positive base
// b = a + n (where GSL is accurate) and recurse DOWN via the exact identity
//   Γ(a, x) = ( Γ(a+1, x) - x^a e^{-x} ) / a,
// numerically stable here (the x^a term cleanly dominates for the small λ of the
// tail). Validated to machine precision vs an mpmath reference. The a > 0 path is
// byte-identical to the previous GSL wrapper.
inline double upper_gamma_C(double a, double x) {
    if (a > 0.0) return _upper_gamma_gsl(a, x);
    int n = 0;
    double b = a;
    while (b <= 0.0) { b += 1.0; ++n; }           // b = a + n, first order > 0
    double val = _upper_gamma_gsl(b, x);          // Γ(b, x), b > 0
    for (int k = 0; k < n; ++k) {
        b -= 1.0;                                 // divisor for Γ(b,x)=(Γ(b+1,x)-x^b e^-x)/b
        if (fabs(b) < 1e-6) return _upper_gamma_gsl(a, x);  // near recurrence pole: defer
        val = (val - pow(x, b) * exp(-x)) / b;
    }
    return val;
}

inline double lower_gamma_C(double a, double x) {
    // γ(a,x). Computing it as Γ(a) - Γ(a,x) CATASTROPHICALLY CANCELS for small x
    // (both ≈ Γ(a)): e.g. γ(1/2 + α0φ, λ) with the tiny λ of the RW tail lost all
    // significant digits, biasing dRW/dlog ~5% even after the negative-order upper
    // gamma was fixed. GSL's regularised lower incomplete gamma P(a,x)=γ(a,x)/Γ(a)
    // is instead evaluated by a stable series for small x, so γ(a,x)=P(a,x)·Γ(a) is
    // accurate. All callers use a > 0 (orders 1/2, 1/2 + α0φ), where P is defined.
    gsl_sf_result P, G;
    int stP = gsl_sf_gamma_inc_P_e(a, x, &P);     // P(a,x) = γ(a,x) / Γ(a)
    int stG = gsl_sf_gamma_e(a, &G);              // Γ(a)
    if (stP) fprintf(stderr, "[GSL] P(%g,%g) status=%s\n", a, x, gsl_strerror(stP));
    if (stG) fprintf(stderr, "[GSL] Γ(%g) status=%s\n", a, gsl_strerror(stG));
    return P.val * G.val;
}


// ---- incomplete-gamma ORDER derivatives (for the analytic d/dphi RW gradient) ----
// d/da Gamma(a,z) = int_z^inf t^{a-1} e^{-t} ln t dt. The integrand spikes at t=z for
// a<1 (small z); the substitution t = z e^s removes it, factoring the magnitude into
// a clean z^a prefactor and leaving a smooth analytic integrand on [0, Smax]:
//   d/da Gamma(a,z) = z^a int_0^Smax e^{ a s} e^{-z e^{ s}} (ln z + s) ds   (any a)
//   d/da gamma(a,z) = z^a int_0^Smax e^{-a s} e^{-z e^{-s}} (ln z - s) ds   (a > 0)
// Gauss-Legendre (fixed n) is spectral on the finite interval (n=64 -> ~1e-13 vs
// mpmath). See gamma_order_deriv.py and docs/phi_gradient_l2phi.md. Single-threaded
// per rank (the lazy table init is not thread-safe -- OK under MPI-over-time).
static gsl_integration_glfixed_table* _gl_tbl = nullptr;
static const int    _GL_N      = 64;
static const double _GL_CUTOFF = 50.0;   // integrand < e^{-cutoff} past Smax
inline gsl_integration_glfixed_table* _gl_table() {
    if (!_gl_tbl) _gl_tbl = gsl_integration_glfixed_table_alloc((size_t)_GL_N);
    return _gl_tbl;
}
double dupper_gamma_da_C(double a, double z) {
    gsl_integration_glfixed_table* t = _gl_table();
    double Smax = log(_GL_CUTOFF / z); if (Smax < 2.0) Smax = 2.0;
    double lnz = log(z), acc = 0.0, xi, wi;
    for (int i = 0; i < _GL_N; ++i) {
        gsl_integration_glfixed_point(0.0, Smax, (size_t)i, &xi, &wi, t);
        acc += wi * exp(a * xi - z * exp(xi)) * (lnz + xi);
    }
    return pow(z, a) * acc;
}
double dlower_gamma_da_C(double a, double z) {   // a > 0 (the RW lower orders always are)
    gsl_integration_glfixed_table* t = _gl_table();
    double Smax = _GL_CUTOFF / a;
    double s2 = (z < 1.0) ? log(_GL_CUTOFF / z) : 5.0; if (Smax < s2) Smax = s2;
    double lnz = log(z), acc = 0.0, xi, wi;
    for (int i = 0; i < _GL_N; ++i) {
        gsl_integration_glfixed_point(0.0, Smax, (size_t)i, &xi, &wi, t);
        acc += wi * exp(-a * xi - z * exp(-xi)) * (lnz - xi);
    }
    return pow(z, a) * acc;
}


// STANDARD PARETO ------------------------------------------------------------

// STANDARD PARETO NO NUGGET --------------------------------------------------

// CDF --------------------------------

double pRW_standard_Pareto_C(double x, double phi, double gamma){
    double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(x, 1/phi)));
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(x, 1/phi)));
    double survival = sqrt(1/M_PI) * lower_gamma + (1/x) * sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma;
    return 1.0 - survival;
}

// pdf --------------------------------

double dRW_standard_Pareto_C(double x, double phi, double gamma){
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(x, 1/phi)));
    return (1/pow(x,2)) * sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma;
}

// quantile function ------------------

double qRW_standard_Pareto_to_solve(double x, void * params_ptr){
    double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return pRW_standard_Pareto_C(x, phi, gamma) - p;
}

double qRW_standard_Pareto_C_brent(double p, double phi, double gamma){
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = 1e-2, x_hi = 2e16; // pRW_standard_Pareto_C(2e16, 1, 3) = 0.99999998
    gsl_function F;
    double params[3] = {p, phi, gamma};

    F.function = &qRW_standard_Pareto_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if(status != -2 && status != 0) 
        std::cout << "location 1 :" <<
        std::string( gsl_strerror (status) ) <<
        " p: " << p << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            if(status != -2 && status != 0) 
                std::cout << 
                "location 2 " << 
                "p:" << p << " " << 
                "phi:" << phi << " " << 
                "gamma:" << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            if(status != -2 && status != 0) 
                std::cout <<
                "location 3 " << 
                "p:" << p << " " <<
                "phi: " << phi << " " <<
                "gamma: " << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}


// STANDARD PARETO WITH NUGGET ------------------------------------------------
// ---------------------------------------------------------------------------
// Convolution with a Gaussian(0, var = tau^2) nugget for threshold exceedance
// ---------------------------------------------------------------------------

// CDF --------------------------------

double pRW_standard_Pareto_nugget_lower_gamma_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(t, 1/phi)));
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return lower_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_upper_gamma_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return (1/t) * upper_gamma * gaussian;
}

double pRW_standard_Pareto_nugget_C(double x, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    double lb = fmax(0.0, x - 38 * tau); // integration lowerbound for gaussian convolution
    double ub = x + 38 * tau; // integration upperbound for gaussian convolution

    // survival function of the Guassian
    double Phi_bar_x = gsl_cdf_gaussian_Q(x, tau);

    // convolution of the lower gamma piece
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
    double lower_gamma_convolution, error;
    double params[4] = {x, phi, gamma, tau};
    gsl_function F;
    F.function = &pRW_standard_Pareto_nugget_lower_gamma_integrand;
    F.params = &params;
    int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w, &lower_gamma_convolution, &error);
    // if (status) {
    //     fprintf (stderr, "failed, gsl_errno=%d\n", status);
    //     std::cout << "lower gamma convolution failed " <<
    //     std::string( gsl_strerror(status)) <<
    //     " x: "    << x     << " " <<
    //     "phi: "   << phi   << " " <<
    //     "gamma: " << gamma <<
    //     std::endl;
    // }
    gsl_integration_workspace_free(w);

    // convolution of the upper gamma piece
    gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1e8);
    double upper_gamma_convolution, error2;
    double params2[4] = {x, phi, gamma, tau};
    gsl_function F2;
    F2.function = &pRW_standard_Pareto_nugget_upper_gamma_integrand;
    F2.params = &params2;

    int status2 = gsl_integration_qag (&F2, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w2, &upper_gamma_convolution, &error2);
    // if (status2) {
    //     fprintf (stderr, "failed, gsl_errno=%d\n", status2);
    //     std::cout << "upper gamma convolution failed " <<
    //     std::string( gsl_strerror(status2)) <<
    //     " x: "    << x     << " " <<
    //     "phi: "   << phi   << " " <<
    //     "gamma: " << gamma <<
    //     std::endl;
    // }
    gsl_integration_workspace_free(w2);


    double survival = Phi_bar_x + 
                        sqrt(1/M_PI) * lower_gamma_convolution + 
                        sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma_convolution;
    return 1.0 - survival;
}


// transformaing to (0,1) bound worsen the reliability
// double pRW_standard_Pareto_nugget_lower_gamma_transform_integrand(double s, void * params_ptr){
//     double x     = (*(double(*)[4]) params_ptr)[0];
//     double phi   = (*(double(*)[4]) params_ptr)[1];
//     double gamma = (*(double(*)[4]) params_ptr)[2];
//     double tau   = (*(double(*)[4]) params_ptr)[3];
//     double jacobian = 1/pow(s,2);
//     double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow((1-s)/s, 1/phi)));
//     double gaussian = gsl_ran_gaussian_pdf(x - (1-s)/s, tau);
//     return jacobian * lower_gamma * gaussian;
// }

// double pRW_standard_Pareto_nugget_upper_gamma_transform_integrand(double s, void * params_ptr){
//     double x     = (*(double(*)[4]) params_ptr)[0];
//     double phi   = (*(double(*)[4]) params_ptr)[1];
//     double gamma = (*(double(*)[4]) params_ptr)[2];
//     double tau   = (*(double(*)[4]) params_ptr)[3];
//     double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow((1-s)/s, 1/phi)));;
//     double gaussian = gsl_ran_gaussian_pdf(x - (1-s)/s, tau);
//     return (1/s*(1-s)) * upper_gamma * gaussian;
// }

// double pRW_standard_Pareto_nugget_transform_C(double x, double phi, double gamma, double tau){
//     gsl_set_error_handler_off();
//     double lb = fmax(0.0, 1/(x + 38 * tau + 1)); // integration lowerbound for gaussian convolution
//     double ub = fmin(1.0, 1/(x - 38 * tau + 1)); // integration upperbound for gaussian convolution
//     // survival function of the Guassian
//     double Phi_bar_x = gsl_cdf_gaussian_Q(x, tau);
//     // convolution of the lower gamma piece
//     gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
//     double lower_gamma_convolution, error;
//     double params[4] = {x, phi, gamma, tau};
//     gsl_function F;
//     F.function = &pRW_standard_Pareto_nugget_lower_gamma_transform_integrand;
//     F.params = &params;
//     int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
//                                     1, w, &lower_gamma_convolution, &error);
//     if (status) {
//         fprintf (stderr, "failed, gsl_errno=%d\n", status);
//         std::cout << "lower gamma convolution failed " <<
//         std::string( gsl_strerror(status)) <<
//         " x: "    << x     << " " <<
//         "phi: "   << phi   << " " <<
//         "gamma: " << gamma <<
//         std::endl;
//     }
//     gsl_integration_workspace_free(w);
//     // convolution of the upper gamma piece
//     gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (1e8);
//     double upper_gamma_convolution, error2;
//     double params2[4] = {x, phi, gamma, tau};
//     gsl_function F2;
//     F2.function = &pRW_standard_Pareto_nugget_upper_gamma_transform_integrand;
//     F2.params = &params2;
//     int status2 = gsl_integration_qag (&F2, lb, ub, 1e-8, 1e-8, 1e8,
//                                     1, w2, &upper_gamma_convolution, &error2);
//     if (status2) {
//         fprintf (stderr, "failed, gsl_errno=%d\n", status2);
//         std::cout << "upper gamma convolution failed " <<
//         std::string( gsl_strerror(status2)) <<
//         " x: "    << x     << " " <<
//         "phi: "   << phi   << " " <<
//         "gamma: " << gamma <<
//         std::endl;
//     }
//     gsl_integration_workspace_free(w2);
//     double survival = Phi_bar_x + 
//                         sqrt(1/M_PI) * lower_gamma_convolution + 
//                         sqrt(1/M_PI) * pow(gamma/2, phi) * upper_gamma_convolution;
//     return 1.0 - survival;
// }

// explicit parameters for plotting
// double pRW_standard_Pareto_nugget_lower_gamma_integrand_forplot(double t, double x, double phi, double gamma, double tau){
//     double lower_gamma = lower_gamma_C(0.5, gamma / (2 * pow(t, 1/phi)));
//     double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
//     return lower_gamma * gaussian;
// }

// double pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot(double t, double x, double phi, double gamma, double tau){
//     double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
//     double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
//     return (1/t) * upper_gamma * gaussian;
// }


// pdf --------------------------------

double dRW_standard_Pareto_nugget_integrand(double t, void * params_ptr){
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double upper_gamma = upper_gamma_C(0.5 - phi, gamma / (2 * pow(t, 1/phi)));;
    double gaussian = gsl_ran_gaussian_pdf(x - t, tau);
    return (1/pow(t,2)) * upper_gamma * gaussian;        
}

double dRW_standard_Pareto_nugget_C(double x, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    double lb = fmax(0.0, x - 38 * tau); // integration lowerbound for gaussian convolution
    double ub = x + 38 * tau; // integration upperbound for gaussian convolution

    // convolution of the lower gamma piece
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1e8);
    double result, error;
    double params[4] = {x, phi, gamma, tau};
    gsl_function F;
    F.function = &dRW_standard_Pareto_nugget_integrand;
    F.params = &params;
    int status = gsl_integration_qag (&F, lb, ub, 1e-8, 1e-8, 1e8,
                                    1, w, &result, &error);
    // if (status) {
    //     fprintf (stderr, "failed, gsl_errno=%d\n", status);
    //     std::cout << "dRW integration failed " <<
    //     std::string( gsl_strerror(status)) <<
    //     " x: "    << x     << " " <<
    //     "phi: "   << phi   << " " <<
    //     "gamma: " << gamma <<
    //     std::endl;
    // }
    gsl_integration_workspace_free(w);

    return sqrt(1/M_PI) * pow(gamma/2, phi) * result;
}

// quantile function ------------------

double qRW_standard_Pareto_nugget_to_solve(double x, void * params_ptr){
    double p     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    return pRW_standard_Pareto_nugget_C(x, phi, gamma, tau) - p;
}

double qRW_standard_Pareto_nugget_C_brent(double p, double phi, double gamma, double tau){
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = -37.0 * tau, x_hi = 1e16; // pRW_standard_Pareto_nugget_C(1e16, 1, 2, 1e3) = 1
    gsl_function F;
    double params[4] = {p, phi, gamma, tau};

    F.function = &qRW_standard_Pareto_nugget_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    // if(status != -2 && status != 0) 
    //     std::cout << "location 1 :" <<
    //     std::string( gsl_strerror (status) ) <<
    //     " p: "    << p     << " " <<
    //     "phi: "   << phi   << " " <<
    //     "gamma: " << gamma << " " <<
    //     "tau: "   << tau   << std::endl;
    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            // if(status != -2 && status != 0) 
            //     std::cout << 
            //     "location 2 " << 
            //     "p:" << p << " " << 
            //     "phi:" << phi << " " << 
            //     "gamma:" << gamma <<
            //     std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            // if(status != -2 && status != 0) 
            //     std::cout <<
            //     "location 3 " << 
            //     "p:" << p << " " <<
            //     "phi: " << phi << " " <<
            //     "gamma: " << gamma <<
            //     std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}

// SHIFTED PARETO -------------------------------------------------------------

// SHIFTED PARETO NO NUGGET ---------------------------------------------------

// CDF --------------------------------

double pRW_transformed_integrand (double t, void * params_ptr) {
    double x     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    double jacobian = 1/(t*t);
    double r = (1-t)/t;
    double integrand = jacobian * pow(r, phi-1.5) * exp(-gamma/(2*r)) / (x + pow(r, phi));
    return integrand;
}

double pRW_transformed (double x, double phi, double gamma){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[3] = {x, phi, gamma};

    gsl_function F;
    F.function = &pRW_transformed_integrand;
    F.params = &params;

    int status = gsl_integration_qag (&F, 0, 1, 1e-12, 1e-12, 10000,
                                    1, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
        std::cout << "pRW_transformed failed " <<
        std::string( gsl_strerror(status)) <<
        " x: " << x << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return 1 - sqrt(gamma/(2*M_PI)) * result;
}

// no gain in accuracy, transformation is sufficient
// double pRW_transformed_2piece (double x, double phi, double gamma){
//     double result1, error1, result2, error2;
//     double params[3] = {x, phi, gamma};
//     gsl_integration_workspace * w1 = gsl_integration_workspace_alloc (10000);
//     gsl_function F1;
//     F1.function = &pRW_transformed_integrand;
//     F1.params = &params;
//     int status1 = gsl_integration_qag (&F1, 0, 0.8, 1e-12, 1e-12, 10000,
//                                     1, w1, &result1, &error1);
//     gsl_integration_workspace_free(w1);
//     if (status1) {
//         fprintf (stderr, "failed, gsl_errno1=%d\n", status1);
//     }
//     printf ("result1          = % .18f\n", result1);
//     printf ("estimated error1 = % .18f\n", error1);
//     printf ("intervals       = %zu\n", w1->size);
//     gsl_integration_workspace * w2 = gsl_integration_workspace_alloc (10000);
//     gsl_function F2;
//     F2.function = &pRW_transformed_integrand;
//     F2.params = &params;
//     int status2 = gsl_integration_qag (&F2, 0.8, 1, 1e-12, 1e-12, 10000,
//                                     1, w2, &result2, &error2);
//     gsl_integration_workspace_free(w2);
//     if (status2) {
//         fprintf (stderr, "failed, gsl_errno2=%d\n", status2);
//     }
//     printf ("result2          = % .18f\n", result2);
//     printf ("estimated error2 = % .18f\n", error2);
//     printf ("intervals       = %zu\n", w2->size);
//     return 1 - result1 - result2;
// }

// pdf --------------------------------

double dRW_transformed_integrand (double t, void * params_ptr) {
    double x     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    double jacobian = 1/(t*t);
    double integrand = jacobian * sqrt(gamma/(2*M_PI)) * 
                                    pow((1-t)/t, phi-1.5) * exp(-gamma/(2*(1-t)/t)) / pow((x + pow((1-t)/t, phi)),2);
    return integrand;
}

double dRW_transformed (double x, double phi, double gamma){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[3] = {x, phi, gamma};

    gsl_function F;
    F.function = &dRW_transformed_integrand;
    F.params = &params;

    int status = gsl_integration_qag (&F, 0, 1, 1e-12, 1e-12, 10000,
                                    1, w, &result, &error);
    if (status) {
        fprintf (stderr, "failed, gsl_errno=%d\n", status);
        std::cout << "dRW_transformed failed " <<
        std::string( gsl_strerror(status)) <<
        " x: " << x << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;
    }
    gsl_integration_workspace_free(w);

    // printf ("result          = % .18f\n", result);
    // printf ("estimated error = % .18f\n", error);
    // printf ("intervals       = %zu\n", w->size);

    return result;
}

// quantile function ------------------

double qRW_to_solve (double x, void * params_ptr) {
    double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return pRW_transformed(x, phi, gamma) - p;
}

double qRW_to_solve_df (double x, void * params_ptr){
    // double p     = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    return dRW_transformed(x, phi, gamma);
}

void qRW_to_solve_fdf (double x, void * params_ptr, double * f, double * df){
    *f = qRW_to_solve(x, params_ptr);
    *df = qRW_to_solve_df(x, params_ptr);
}

double qRW_transformed_brent (double p, double phi, double gamma){
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 10;
    double x_lo = 0, x_hi = 2e14;
    gsl_function F;
    double params[3] = {p, phi, gamma};

    F.function = &qRW_to_solve;
    F.params = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if(status != -2 && status != 0) 
        std::cout << "location 1 :" <<
        std::string( gsl_strerror (status) ) <<
        " p: " << p << " " <<
        "phi: " << phi << " " <<
        "gamma: " << gamma <<
        std::endl;

    // printf ("using %s method\n",
    //       gsl_root_fsolver_name (s));
    // printf ("%5s [%9s, %9s] %9s %9s\n",
    //       "iter", "lower", "upper", "root", "err(est)");

    do
        {
            iter++;
            status = gsl_root_fsolver_iterate (s);
            if(status != -2 && status != 0) 
                std::cout << 
                "location 2 " << 
                "p:" << p << " " << 
                "phi:" << phi << " " << 
                "gamma:" << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi, 1e-12, 1e-12);
            if(status != -2 && status != 0) 
                std::cout <<
                "location 3 " << 
                "p:" << p << " " <<
                "phi: " << phi << " " <<
                "gamma: " << gamma <<
                std::string( gsl_strerror (status) ) << std::endl;
            // if (status == GSL_SUCCESS) printf ("Converged:\n");

            // printf ("%5d [%.7f, %.7f] %.7f %.7f\n",
            //     iter, x_lo, x_hi, r, x_hi - x_lo);
        }
    while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        printf("after the loop\n");
        printf("%d\n", status);
        printf("p: %.5f, phi: %.5f, gamma: %.5f\n", p, phi, gamma);
    }

    gsl_root_fsolver_free (s);

    return r;
}


// SHIFTED PARETO WITH NUGGET -------------------------------------------------
// ---------------------------------------------------------------------------
// Convolution with a Gaussian(0, var = tau^2) nugget for threshold exceedance
// ---------------------------------------------------------------------------
// implementation with cubature in python


// CDF --------------------------------

// pdf --------------------------------

// quantile function ------------------






// STANDARD PARETO WITH MULTIPLICATIVE NUGGET -------------------------------------------------
// ---------------------------------------------------------------------------
// Convolution with a log-laplace(0, \alpha_0) nugget for threshold exceedance
// X = epsilon * X*,   epsilon ~ Log-Laplace(0, 1/alpha0)
// 
// Laplace Nugget requires: 
//   alpha0  > 0, 
// marginal of X requires:  
//   alpha0 != 1,
// incomplete gamma requires:
//   0 < phi < 0.5,  0 < phi*alpha0 < 0.5,
// ---------------------------------------------------------------------------

// CDF ----------------------------------------------------

double pRW_standard_Pareto_multiplicative_nugget_C (double x, double phi, double gamma, double alpha0){
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    const double lambda      = gamma / (2.0 * pow(x, 1.0 / phi));
    const double g_over_2    = gamma / 2.0;

    // lower_gamma_C(a, z) = γ(a, z);  upper_gamma_C(a, z) = Γ(a, z)
    const double L_half     = lower_gamma_C(0.5, lambda);                     // γ(1/2, λ)
    const double U_half_mp  = upper_gamma_C(0.5 - phi, lambda);               // Γ(1/2 - φ, λ)
    const double L_half_pap = lower_gamma_C(0.5 + alpha0 * phi, lambda);      // γ(1/2 + α0 φ, λ)
    const double U_half_map = upper_gamma_C(0.5 - alpha0 * phi, lambda);      // Γ(1/2 - α0 φ, λ)

    // Powers that recur
    const double g2_pow_phi        = pow(g_over_2,  phi);          // (γ/2)^φ
    const double g2_pow_malpha_phi = pow(g_over_2, -alpha0 * phi); // (γ/2)^(-α0φ)
    const double g2_pow_alpha_phi  = pow(g_over_2,  alpha0 * phi); // (γ/2)^(α0φ)

    // Coefficients
    const double cB = (alpha0 * alpha0) / (alpha0 * alpha0 - 1.0); // for x^{-1} Γ(1/2 - φ, λ)
    const double cC = -1.0 / (2.0 * (alpha0 + 1.0));               // for x^{+α0} γ(1/2 + α0 φ, λ)
    const double cD = -1.0 / (2.0 * (alpha0 - 1.0));               // for x^{-α0} Γ(1/2 - α0 φ, λ)

    // Assemble survival S(x) = 1 - F_X(x)
    const double termA = L_half;
    const double termB = cB * g2_pow_phi * (1.0 / x) * U_half_mp;
    const double termC = cC * pow(x,  alpha0) * g2_pow_malpha_phi * L_half_pap;
    const double termD = cD * pow(x, -alpha0) * g2_pow_alpha_phi  * U_half_map;

    const double survival = inv_sqrt_pi * (termA + termB + termC + termD);
    double Fx = 1.0 - survival;

    return Fx;
}


// pdf ----------------------------------------------------

double dRW_standard_Pareto_multiplicative_nugget_C(double x, double phi, double gamma, double alpha0){
    // Domain / parameter checks following the same assumptions as the CDF:
    //   x > 0; alpha0 > 0 and alpha0 != 1;

    if (!(alpha0 > 0.0) || fabs(alpha0 - 1.0) <= 0.0) {
        fprintf(stderr, "dRW_mult_nugget: alpha0 must be > 0 and != 1. Got alpha0=%.17g\n", alpha0);
        // continue anyway to allow exploratory regimes, but result may be NaN/Inf.
    }
    // Precompute common quantities
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    const double g_over_2    = gamma / 2.0;
    const double inv_phi     = 1.0 / phi;

    // λ(x) = γ / (2 x^{1/φ})
    const double lambda  = gamma / (2.0 * pow(x, inv_phi));
    // λ'(x) = -(γ)/(2 φ) * x^{-1/φ - 1}
    const double lambda_p = - (gamma) / (2.0 * phi) * pow(x, -inv_phi - 1.0);

    // Incomplete gamma values needed
    // lower_gamma_C(a,z) = γ(a,z);  upper_gamma_C(a,z) = Γ(a,z)
    const double L_half        = lower_gamma_C(0.5,                    lambda);  // γ(1/2, λ)
    const double U_half_mphi   = upper_gamma_C(0.5 - phi,              lambda);  // Γ(1/2 - φ, λ)
    const double L_half_paphi  = lower_gamma_C(0.5 + alpha0 * phi,     lambda);  // γ(1/2 + α0 φ, λ)
    const double U_half_maphi  = upper_gamma_C(0.5 - alpha0 * phi,     lambda);  // Γ(1/2 - α0 φ, λ)

    // Powers that recur
    const double g2_pow_phi        = pow(g_over_2,  phi);              // (γ/2)^φ
    const double g2_pow_malpha_phi = pow(g_over_2, -alpha0 * phi);     // (γ/2)^(-α0φ)
    const double g2_pow_alpha_phi  = pow(g_over_2,  alpha0 * phi);     // (γ/2)^(α0φ)

    // Coefficients from the survival decomposition
    const double cB = (alpha0 * alpha0) / (alpha0 * alpha0 - 1.0);     // for x^{-1} Γ(1/2 - φ, λ)
    const double cC =  1.0 / (2.0 * (alpha0 + 1.0));                   // for x^{+α0} γ(1/2 + α0 φ, λ)
    const double cD =  1.0 / (2.0 * (alpha0 - 1.0));                   // for x^{-α0} Γ(1/2 - α0 φ, λ)

    // Derivatives of each survival term:
    // A(x) = γ(1/2, λ(x))  ⇒  A'(x) = λ'(x) λ^{-1/2} e^{-λ}
    const double A_p = lambda_p * pow(lambda, -0.5) * exp(-lambda);

    // B(x) = cB * (γ/2)^φ * x^{-1} * Γ(1/2 - φ, λ(x))
    // B'(x) = cB * (γ/2)^φ * [ -x^{-2} Γ(1/2 - φ, λ)  - x^{-1} λ'(x) λ^{-1/2 - φ} e^{-λ} ]
    const double B_p = cB * g2_pow_phi * ( - (1.0/(x*x)) * U_half_mphi
                                           - (1.0/x) * lambda_p * pow(lambda, -0.5 - phi) * exp(-lambda) );

    // C(x) = cC * (γ/2)^{-α0 φ} * x^{α0} * γ(1/2 + α0 φ, λ(x))
    // C'(x) = cC * (γ/2)^{-α0 φ} * [ α0 x^{α0-1} γ(1/2 + α0 φ, λ)
    //                                + x^{α0} λ'(x) λ^{-1/2 + α0 φ} e^{-λ} ]
    const double C_p = cC * g2_pow_malpha_phi * ( alpha0 * pow(x, alpha0 - 1.0) * L_half_paphi
                                                 + pow(x, alpha0) * lambda_p * pow(lambda, -0.5 + alpha0 * phi) * exp(-lambda) );

    // D(x) = cD * (γ/2)^{+α0 φ} * x^{-α0} * Γ(1/2 - α0 φ, λ(x))
    // D'(x) = cD * (γ/2)^{+α0 φ} * [ -α0 x^{-α0-1} Γ(1/2 - α0 φ, λ)
    //                                - x^{-α0} λ'(x) λ^{-1/2 - α0 φ} e^{-λ} ]
    const double D_p = cD * g2_pow_alpha_phi * ( - alpha0 * pow(x, -alpha0 - 1.0) * U_half_maphi
                                                 - pow(x, -alpha0) * lambda_p * pow(lambda, -0.5 - alpha0 * phi) * exp(-lambda) );

    // f_X(x) = sqrt(1/pi) * ( -A'(x) - B'(x) + C'(x) + D'(x) )
    const double fx = inv_sqrt_pi * ( - A_p - B_p + C_p + D_p );

    // Basic nonnegativity guard for extreme numeric regimes
    if (!(fx >= 0.0) && !isnan(fx)) {
        // Clamp very small negative values caused by roundoff
        if (fx > -1e-18) return 0.0;
    }
    return fx;
}


// d/dx log dRW(x) -- EXACT closed-form derivative of the (smooth) analytic density
// above. The analytic density is biased in magnitude in the upper tail / large
// alpha0, but its LOG-DERIVATIVE is smooth and accurate (the bias is largely
// multiplicative and cancels in f'/f), so it is the right ingredient for the
// gradient of the beta/phi NUTS blocks (validated vs finite differences). Returns
// f'(x)/f(x), reusing the SAME four incomplete gammas as the density plus one more
// derivative level. lambda(x)=g2*x^{-1/phi}; every survival term T has the form
// [poly(x)]*[incomplete gamma] + [poly(x)]*lambda'*lambda^{a-1}*e^{-lambda}, so
// T'' reuses Q(a)=lambda^{a-1} e^{-lambda}, P1(a)=lambda'*Q(a), R1(a)=d/dx P1(a).
double dlogdRW_dx_standard_Pareto_multiplicative_nugget_C(double x, double phi, double gamma, double alpha0){
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    const double g_over_2    = gamma / 2.0;
    const double inv_phi     = 1.0 / phi;
    const double a0          = alpha0;

    const double lambda = g_over_2 * pow(x, -inv_phi);
    const double lam_p  = -inv_phi * lambda / x;                        // lambda'
    const double lam_pp = inv_phi * (inv_phi + 1.0) * lambda / (x * x); // lambda''
    const double e_ml   = exp(-lambda);

    // the four incomplete-gamma orders (identical to the density)
    const double bA = 0.5, bB = 0.5 - phi, bC = 0.5 + a0 * phi, bD = 0.5 - a0 * phi;
    const double GA = lower_gamma_C(bA, lambda);   // gamma(1/2, lambda)
    const double GB = upper_gamma_C(bB, lambda);   // Gamma(1/2 - phi, lambda)
    const double GC = lower_gamma_C(bC, lambda);   // gamma(1/2 + a0 phi, lambda)
    const double GD = upper_gamma_C(bD, lambda);   // Gamma(1/2 - a0 phi, lambda)
    (void) GA;                                     // A term uses only Q/P1/R1

    // Q(a) = lambda^{a-1} e^{-lambda}; P1(a) = lambda' Q(a);
    // R1(a) = d/dx P1(a) = Q(a)[ lambda'' + lambda'^2 ((a-1)/lambda - 1) ]
    const double QA = pow(lambda, bA - 1.0) * e_ml;
    const double QB = pow(lambda, bB - 1.0) * e_ml;
    const double QC = pow(lambda, bC - 1.0) * e_ml;
    const double QD = pow(lambda, bD - 1.0) * e_ml;
    const double P1A = lam_p * QA, P1B = lam_p * QB, P1C = lam_p * QC, P1D = lam_p * QD;
    const double R1A = QA * (lam_pp + lam_p * lam_p * ((bA - 1.0) / lambda - 1.0));
    const double R1B = QB * (lam_pp + lam_p * lam_p * ((bB - 1.0) / lambda - 1.0));
    const double R1C = QC * (lam_pp + lam_p * lam_p * ((bC - 1.0) / lambda - 1.0));
    const double R1D = QD * (lam_pp + lam_p * lam_p * ((bD - 1.0) / lambda - 1.0));

    const double kB = ((a0 * a0) / (a0 * a0 - 1.0)) * pow(g_over_2, phi);
    const double kC = (1.0 / (2.0 * (a0 + 1.0))) * pow(g_over_2, -a0 * phi);
    const double kD = (1.0 / (2.0 * (a0 - 1.0))) * pow(g_over_2,  a0 * phi);
    const double xm1 = 1.0 / x, xm2 = xm1 * xm1, xm3 = xm2 * xm1;

    // f(x) = inv_sqrt_pi ( -A' - B' + C' + D' )   (matches the density above)
    const double A1 = P1A;
    const double B1 = kB * (-xm2 * GB - xm1 * P1B);
    const double C1 = kC * (a0 * pow(x, a0 - 1.0) * GC + pow(x, a0) * P1C);
    const double D1 = kD * (-a0 * pow(x, -a0 - 1.0) * GD - pow(x, -a0) * P1D);
    const double f  = inv_sqrt_pi * (-A1 - B1 + C1 + D1);

    // f'(x) = inv_sqrt_pi ( -A'' - B'' + C'' + D'' )
    const double A2 = R1A;
    const double B2 = kB * (2.0 * xm3 * GB + 2.0 * xm2 * P1B - xm1 * R1B);
    const double C2 = kC * (a0 * (a0 - 1.0) * pow(x, a0 - 2.0) * GC
                            + 2.0 * a0 * pow(x, a0 - 1.0) * P1C + pow(x, a0) * R1C);
    const double D2 = kD * (a0 * (a0 + 1.0) * pow(x, -a0 - 2.0) * GD
                            + 2.0 * a0 * pow(x, -a0 - 1.0) * P1D - pow(x, -a0) * R1D);
    const double fp = inv_sqrt_pi * (-A2 - B2 + C2 + D2);

    if (!(f > 0.0) || !isfinite(f) || !isfinite(fp))
        return 0.0;                                // deep-tail floor (negligible mass)
    return fp / f;
}


// d/dphi pRW(x; phi, gamma, alpha0) -- analytic, reusing the density's incomplete
// gammas (upper/lower_gamma_C) plus their ORDER-derivatives (dupper/dlower_gamma_da_C).
// This is the fused kernel the phi gradient needs for dX/dphi = -(dpRW/dphi)/dRW,
// which avoids re-inverting qRW. Survival S = pi^{-1/2}(A+B+C+D), pRW = 1 - S, with
// lam = (gamma/2) x^{-1/phi}, dlam/dphi = lam ln(x)/phi^2, Q(a)=lam^{a-1}e^{-lam},
// L=ln(gamma/2). Matches the appendix of docs/phi_gradient_l2phi.md; validated vs the
// Python dpRW_dphi (itself validated vs an mpmath central difference of pRW).
double dpRW_dphi_standard_Pareto_multiplicative_nugget_C(double x, double phi,
                                                         double gamma, double alpha0) {
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    const double a0   = alpha0;
    const double g2   = gamma / 2.0;
    const double lam  = g2 * pow(x, -1.0 / phi);
    const double dlam = lam * log(x) / (phi * phi);     // dlam/dphi
    const double e_ml = exp(-lam);
    const double L    = log(g2);

    const double aB = 0.5 - phi, aC = 0.5 + a0 * phi, aD = 0.5 - a0 * phi;
    const double QA = pow(lam, 0.5 - 1.0) * e_ml;        // Q(1/2)
    const double QB = pow(lam, aB - 1.0) * e_ml;
    const double QC = pow(lam, aC - 1.0) * e_ml;
    const double QD = pow(lam, aD - 1.0) * e_ml;

    const double GB = upper_gamma_C(aB, lam);            // Gamma(1/2 - phi, lam)
    const double GC = lower_gamma_C(aC, lam);            // gamma(1/2 + a0 phi, lam)
    const double GD = upper_gamma_C(aD, lam);            // Gamma(1/2 - a0 phi, lam)
    const double dUB = dupper_gamma_da_C(aB, lam);
    const double dLC = dlower_gamma_da_C(aC, lam);
    const double dUD = dupper_gamma_da_C(aD, lam);

    const double cB = (a0 * a0) / (a0 * a0 - 1.0);
    const double cC = -1.0 / (2.0 * (a0 + 1.0));
    const double cD = -1.0 / (2.0 * (a0 - 1.0));
    const double g2pf  = pow(g2,  phi);
    const double g2pmf = pow(g2, -a0 * phi);
    const double g2ppf = pow(g2,  a0 * phi);

    const double dA = QA * dlam;
    const double dB = cB * (1.0 / x) * (L * g2pf * GB + g2pf * (-dUB - QB * dlam));
    const double dC = cC * pow(x, a0) * (-a0 * L * g2pmf * GC + g2pmf * (a0 * dLC + QC * dlam));
    const double dD = cD * pow(x, -a0) * (a0 * L * g2ppf * GD + g2ppf * (-a0 * dUD - QD * dlam));
    return -inv_sqrt_pi * (dA + dB + dC + dD);          // pRW = 1 - S
}


// d/dphi log dRW(x; phi, gamma, alpha0)  (the phi analogue of dlogdRW_dx_C). The
// density is f = pi^{-1/2}(-A'-B'+C'+D') with each term T' = coef*(g2)^{c phi}*
// [poly(x)*incgamma(a,lam) + poly(x)*lambda'*lam^{p}*e^{-lam}], a=1/2 +/- {1, a0}*phi,
// p=a-1, lam=(g2)x^{-1/phi}, lambda'=-lam/(phi x). d/dphi hits: the power (g2)^{c phi}
// (-> c*ln(g2)); the gamma via ORDER (dupper/dlower_gamma_da_C) and via lam; the
// lam^{p} whose exponent itself carries phi; e^{-lam}; and lambda'. Returns df/dphi / f.
// Coefficients are the DENSITY's (cC,cD > 0), NOT pRW's. Validated vs a central
// difference of log(dRW) (see tests/test_dpRW_dphi.py).
double dlogdRW_dphi_standard_Pareto_multiplicative_nugget_C(double x, double phi,
                                                            double gamma, double alpha0) {
    const double inv_sqrt_pi = 1.0 / sqrt(M_PI);
    const double a0  = alpha0;
    const double g2  = gamma / 2.0;
    const double lam = g2 * pow(x, -1.0 / phi);
    const double lnlam = log(lam);
    const double e_ml  = exp(-lam);
    const double L     = log(g2);
    const double dlam  = lam * log(x) / (phi * phi);        // dlam/dphi
    const double de_ml = -e_ml * dlam;
    const double lp    = -lam / (phi * x);                  // lambda' = dlam/dx
    const double dlp   = -dlam / (phi * x) + lam / (phi * phi * x);  // d lambda'/dphi

    const double cB = (a0 * a0) / (a0 * a0 - 1.0);
    const double cC =  1.0 / (2.0 * (a0 + 1.0));            // DENSITY coefficients (>0)
    const double cD =  1.0 / (2.0 * (a0 - 1.0));
    const double aB = 0.5 - phi, aC = 0.5 + a0 * phi, aD = 0.5 - a0 * phi;
    double p, powp, dpowp, S, dS, In, dIn;

    // A' = lp * lam^{-1/2} * e_ml   (orders constant)
    const double lam_mh = pow(lam, -0.5);
    const double Ap  = lp * lam_mh * e_ml;
    const double dAp = dlp * lam_mh * e_ml + lp * (-0.5 * pow(lam, -1.5) * dlam) * e_ml
                     + lp * lam_mh * de_ml;

    // B' = cB (g2)^phi [ -x^{-2} Gamma(aB,lam) - x^{-1} lp lam^{-1/2-phi} e_ml ]
    const double PB = pow(g2, phi), dPB = L * PB;
    const double GB = upper_gamma_C(aB, lam);
    const double dGB = -dupper_gamma_da_C(aB, lam) - pow(lam, aB - 1.0) * e_ml * dlam;
    p = -0.5 - phi; powp = pow(lam, p); dpowp = powp * (-1.0 * lnlam + p * dlam / lam);
    S = lp * powp * e_ml; dS = dlp * powp * e_ml + lp * dpowp * e_ml + lp * powp * de_ml;
    In = -1.0 / (x * x) * GB - 1.0 / x * S;  dIn = -1.0 / (x * x) * dGB - 1.0 / x * dS;
    const double Bp = cB * PB * In, dBp = cB * (dPB * In + PB * dIn);

    // C' = cC (g2)^{-a0 phi} [ a0 x^{a0-1} gamma(aC,lam) + x^{a0} lp lam^{-1/2+a0 phi} e_ml ]
    const double PC = pow(g2, -a0 * phi), dPC = -a0 * L * PC;
    const double GC = lower_gamma_C(aC, lam);
    const double dGC = a0 * dlower_gamma_da_C(aC, lam) + pow(lam, aC - 1.0) * e_ml * dlam;
    p = -0.5 + a0 * phi; powp = pow(lam, p); dpowp = powp * (a0 * lnlam + p * dlam / lam);
    S = lp * powp * e_ml; dS = dlp * powp * e_ml + lp * dpowp * e_ml + lp * powp * de_ml;
    In = a0 * pow(x, a0 - 1.0) * GC + pow(x, a0) * S;
    dIn = a0 * pow(x, a0 - 1.0) * dGC + pow(x, a0) * dS;
    const double Cp = cC * PC * In, dCp = cC * (dPC * In + PC * dIn);

    // D' = cD (g2)^{a0 phi} [ -a0 x^{-a0-1} Gamma(aD,lam) - x^{-a0} lp lam^{-1/2-a0 phi} e_ml ]
    const double PD = pow(g2, a0 * phi), dPD = a0 * L * PD;
    const double GD = upper_gamma_C(aD, lam);
    const double dGD = -a0 * dupper_gamma_da_C(aD, lam) - pow(lam, aD - 1.0) * e_ml * dlam;
    p = -0.5 - a0 * phi; powp = pow(lam, p); dpowp = powp * (-a0 * lnlam + p * dlam / lam);
    S = lp * powp * e_ml; dS = dlp * powp * e_ml + lp * dpowp * e_ml + lp * powp * de_ml;
    In = -a0 * pow(x, -a0 - 1.0) * GD - pow(x, -a0) * S;
    dIn = -a0 * pow(x, -a0 - 1.0) * dGD - pow(x, -a0) * dS;
    const double Dp = cD * PD * In, dDp = cD * (dPD * In + PD * dIn);

    const double f  = inv_sqrt_pi * (-Ap  - Bp  + Cp  + Dp);    // = dRW
    const double df = inv_sqrt_pi * (-dAp - dBp + dCp + dDp);   // = d dRW/dphi
    if (!(f > 0.0) || !isfinite(f) || !isfinite(df)) return 0.0;
    return df / f;
}



// quantile function --------------------------------------

// Validates p∈(0,1)
// Brackets the root by expanding the upper bound geometrically until F(x_hi) ≥ p.
// Uses gsl_root_fsolver_brent on f(x)=F(x)-p with tight tolerances.

double qRW_standard_Pareto_multiplicative_nugget_to_solve(double x, void * params_ptr){
    // params: [p, phi, gamma, alpha0]
    double p      = (*(double(*)[4]) params_ptr)[0];
    double phi    = (*(double(*)[4]) params_ptr)[1];
    double gamma  = (*(double(*)[4]) params_ptr)[2];
    double alpha0 = (*(double(*)[4]) params_ptr)[3];
    return pRW_standard_Pareto_multiplicative_nugget_C(x, phi, gamma, alpha0) - p;
}

// Adaptive-bracketed Brent solver
double qRW_standard_Pareto_multiplicative_nugget_C_brent(double p, double phi, double gamma, double alpha0){
    // Basic input checks
    if (!(p > 0.0 && p < 1.0)) {
        fprintf(stderr, "qRW_mult_nugget: p must be in (0,1). Got p=%.17g\n", p);
        return NAN;
    }

    gsl_set_error_handler_off();

    // Step 1: bracket the root so that f(x_lo) <= 0 and f(x_hi) >= 0
    // Start at 0; if F(0) > p, grow into the negative side until F(x_lo) <= p.
    double x_lo = 0.01;
    double Fx_lo = pRW_standard_Pareto_multiplicative_nugget_C(x_lo, phi, gamma, alpha0);
    double f_lo  = Fx_lo - p;

    // If F(0) > p (e.g., for p < 0.5 when the distribution has mass on x<0),
    // expand the lower bound geometrically into the negative side until f(x_lo) <= 0.
    if (f_lo > 0.0) {
        double x_lo_min = -1e300;        // generous floor
        const double growth_lo = 2.0;    // geometric expansion factor (more negative)
        const int    shrink_iter_max = 1024;
        int          shrink_iter = 0;

        // start from a modest negative value
        x_lo = -1.0;
        Fx_lo = pRW_standard_Pareto_multiplicative_nugget_C(x_lo, phi, gamma, alpha0);
        f_lo  = Fx_lo - p;

        while (f_lo > 0.0 && x_lo > x_lo_min && shrink_iter < shrink_iter_max) {
            x_lo *= growth_lo; // -1, -2, -4, ...
            Fx_lo = pRW_standard_Pareto_multiplicative_nugget_C(x_lo, phi, gamma, alpha0);
            f_lo  = Fx_lo - p;
            ++shrink_iter;
        }

        if (f_lo > 0.0) {
            fprintf(stderr,
                    "qRW_mult_nugget: could not bracket from below even at x_lo≈%.3e. p=%.17g, F(x_lo)=%.17g\n",
                    x_lo, p, Fx_lo);
            return NAN;
        }
    }

    // Grow the upper bound until F(x_hi) >= p (or we hit a ceiling).
    // growth=10 (not 2): a deep-tail exceedance root sits at ~1e6-1e15, so doubling
    // from 1 spends ~20-50 pRW evals JUST bracketing -- and profiling showed that
    // bracketing (not Brent refinement) is essentially all of qRW's cost (loosening
    // the Brent tolerance 1e-12->1e-4 gave ZERO speedup). x10 collapses bracketing to
    // ~6-15 steps for a ~2x qRW speedup; the wider initial bracket costs only a few of
    // Brent's (cheap, superlinear) iterations and the root is still solved to 1e-12
    // (values bit-accurate to ~5e-13 vs the x2 bracket). Plateaus for growth>=10.
    double x_hi = 1.0;
    const double growth = 10.0;
    const double x_hi_max = 1e300;  // generous ceiling
    double Fx_hi = pRW_standard_Pareto_multiplicative_nugget_C(x_hi, phi, gamma, alpha0);
    int grow_iter = 0;
    const int grow_iter_max = 1024;

    while (Fx_hi < p && x_hi < x_hi_max && grow_iter < grow_iter_max) {
        x_hi *= growth;
        Fx_hi = pRW_standard_Pareto_multiplicative_nugget_C(x_hi, phi, gamma, alpha0);
        ++grow_iter;
    }
    if (Fx_hi < p) {
        fprintf(stderr, "qRW_mult_nugget: failed to bracket upper bound (F(x_hi) < p even at x_hi≈%.3e). p=%.17g, F=%.17g\n",
                x_hi, p, Fx_hi);
        return NAN;
    }

    // Step 2: run Brent
    int status;
    int iter = 0, max_iter = 10000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;

    double params[4] = {p, phi, gamma, alpha0};
    F.function = &qRW_standard_Pareto_multiplicative_nugget_to_solve;
    F.params   = &params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);

    status = gsl_root_fsolver_set(s, &F, x_lo, x_hi);
    if (status != GSL_SUCCESS) {
        // std::cout << "qRW_mult_nugget: bracketing set failed: "
        //           << std::string(gsl_strerror(status))
        //           << "  p:" << p << " phi:" << phi << " gamma:" << gamma << " alpha0:" << alpha0
        //           << " x_lo:" << x_lo << " x_hi:" << x_hi
        //           << std::endl;
        gsl_root_fsolver_free(s);
        return NAN;
    }

    double r = NAN;
    do {
        ++iter;
        status = gsl_root_fsolver_iterate(s);
        if (status != GSL_SUCCESS && status != GSL_CONTINUE) {
            std::cout << "qRW_mult_nugget: iterate failed: "
                      << std::string(gsl_strerror(status))
                      << "  p:" << p << " phi:" << phi << " gamma:" << gamma << " alpha0:" << alpha0
                      << std::endl;
            break;
        }
        double xl = gsl_root_fsolver_x_lower(s);
        double xu = gsl_root_fsolver_x_upper(s);
        r = gsl_root_fsolver_root(s);

        status = gsl_root_test_interval(xl, xu, 1e-12, 1e-12);
    } while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS) {
        std::cout << "qRW_mult_nugget: did not converge. status="
                  << std::string(gsl_strerror(status))
                  << "  p:" << p << " phi:" << phi << " gamma:" << gamma << " alpha0:" << alpha0
                  << "  iter:" << iter
                  << std::endl;
    }

    gsl_root_fsolver_free(s);
    return r;
}


// Brent solve on a KNOWN-GOOD bracket [x_lo, x_hi] (f(x_lo) <= 0 <= f(x_hi)), to the
// same 1e-12 interval tolerance as the cold path. Used by the warm-start entry point.
static double brent_solve_mult_nugget(double p, double phi, double gamma, double alpha0,
                                      double x_lo, double x_hi){
    double params[4] = {p, phi, gamma, alpha0};
    gsl_function F;
    F.function = &qRW_standard_Pareto_multiplicative_nugget_to_solve;
    F.params   = &params;
    gsl_root_fsolver *s = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    if (gsl_root_fsolver_set(s, &F, x_lo, x_hi) != GSL_SUCCESS) {
        gsl_root_fsolver_free(s);
        return NAN;
    }
    int status, iter = 0, max_iter = 10000;
    double r = NAN;
    do {
        ++iter;
        status = gsl_root_fsolver_iterate(s);
        if (status != GSL_SUCCESS && status != GSL_CONTINUE) break;
        double xl = gsl_root_fsolver_x_lower(s), xu = gsl_root_fsolver_x_upper(s);
        r = gsl_root_fsolver_root(s);
        status = gsl_root_test_interval(xl, xu, 1e-12, 1e-12);
    } while (status == GSL_CONTINUE && iter < max_iter);
    gsl_root_fsolver_free(s);
    return r;
}

// WARM-START quantile: seed the Brent bracket from x_guess (a previous accepted qRW for
// the same cell; the MCMC inputs p/phi/gamma move slowly, so the root barely shifts).
// Bracketing the deep-tail root is the whole cost of the cold solve (~all of qRW's wall);
// a good guess collapses it from ~6-15 pRW evals to ~2. SAFETY: a stale/bad guess is never
// wrong -- if the guess bracket cannot be made to straddle the root we fall back to the
// exact cold solver, so this only ever affects SPEED. x_guess <= 0 / non-finite => cold.
double qRW_standard_Pareto_multiplicative_nugget_C_brent_warm(
        double p, double phi, double gamma, double alpha0, double x_guess){
    if (!(p > 0.0 && p < 1.0)) {
        fprintf(stderr, "qRW_mult_nugget_warm: p must be in (0,1). Got p=%.17g\n", p);
        return NAN;
    }
    if (x_guess > 0.0 && isfinite(x_guess)) {
        gsl_set_error_handler_off();
        const double F = 4.0;                 // initial half-width factor around the guess
        double x_lo = x_guess / F, x_hi = x_guess * F;
        double Flo = pRW_standard_Pareto_multiplicative_nugget_C(x_lo, phi, gamma, alpha0);
        double Fhi = pRW_standard_Pareto_multiplicative_nugget_C(x_hi, phi, gamma, alpha0);
        int it;
        // widen upward if the guess bracket sits below the root (few steps: seeded from guess)
        for (it = 0; Fhi < p && x_hi < 1e300 && it < 64; ++it) {
            x_hi *= 10.0; Fhi = pRW_standard_Pareto_multiplicative_nugget_C(x_hi, phi, gamma, alpha0);
        }
        // widen downward if the guess bracket sits above the root
        for (it = 0; Flo > p && x_lo > 1e-300 && it < 64; ++it) {
            x_lo /= 10.0; Flo = pRW_standard_Pareto_multiplicative_nugget_C(x_lo, phi, gamma, alpha0);
        }
        if (isfinite(Flo) && isfinite(Fhi) && Flo <= p && Fhi >= p) {
            double r = brent_solve_mult_nugget(p, phi, gamma, alpha0, x_lo, x_hi);
            if (isfinite(r)) return r;
        }
        // guess could not bracket the root (e.g. it moved to the negative side) -> cold
    }
    return qRW_standard_Pareto_multiplicative_nugget_C_brent(p, phi, gamma, alpha0);
}


}
int main(){
    std::cout << pRW_standard_Pareto_nugget_C(1e16, 1, 2,1000) << std::endl;
    return 0;
}
// g++ -I/opt/homebrew/include RW_inte_cpp.cpp -L/opt/homebrew/lib -o RW_inte_cpp -lgsl -lgslcblas
// g++ -I/opt/homebrew/include -std=c++11 -Wall -pedantic RW_inte_cpp.cpp -shared -fPIC -L/opt/homebrew/lib -o RW_inte_cpp.so -lgsl -lgslcblas