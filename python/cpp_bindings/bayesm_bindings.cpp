#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>
#include "bayesm.h"

namespace py = pybind11;

// Forward declarations from C++ files
double lndMvn(vec const& x, vec const& mu, mat const& rooti);
vec ghkvec(mat const& L, vec const& trunpt, vec const& above, int r, bool HALTON, vec pn);
mat rwishart(double nu, mat const& V);
vec rdirichlet(vec const& alpha);
double rtrun(double mu, double sigma, double a, double b);
double lndIWishart(double nu, mat const& V, mat const& IW);
double lndMvst(vec const& x, double nu, vec const& mu, mat const& rooti, bool NORMC);
vec rmvst(double nu, vec const& mu, mat const& root);
vec breg(vec const& y, mat const& X, vec const& betabar, mat const& A);
double llmnl(vec const& beta, vec const& y, mat const& X);
double lndIChisq(double nu, double ssq, double X);
mat rmultireg(mat const& Y, mat const& X, mat const& Bbar, mat const& A, double nu, mat const& V);
vec callroot(vec const& c1, vec const& c2, double tol, int iterlim);
std::tuple<mat, vec> runireg_rcpp_loop(vec const& y, mat const& X, vec const& betabar, mat const& A, double nu, double ssq, int R, int keep);
std::tuple<mat, vec> runiregGibbs_rcpp_loop(vec const& y, mat const& X, vec const& betabar, mat const& A, double nu, double ssq, double sigmasq, int R, int keep);

struct moments { vec y; mat X; mat XpX; vec Xpy; };
std::tuple<arma::cube, mat, mat, mat> rhierLinearModel_rcpp_loop(
    std::vector<moments> const& regdata_vector, mat const& Z, 
    mat const& Deltabar, mat const& A, double nu, mat const& V, 
    double nu_e, vec const& ssq, vec tau, mat Delta, mat Vbeta, int R, int keep);
std::tuple<mat, vec, int> rmnlIndepMetrop_rcpp_loop(
    int R, int keep, double nu, vec const& betastar, mat const& root,
    vec const& y, mat const& X, vec const& betabar, mat const& rootpi,
    mat const& rooti, double oldlimp, double oldlpost);
mat rbprobitGibbs_rcpp_loop(vec const& y, mat const& X, vec const& Abetabar, mat const& root,
                            vec beta, vec const& sigma, vec const& trunpt, vec const& above,
                            int R, int keep);
std::tuple<mat, mat, mat, double> rordprobitGibbs_rcpp_loop(
    vec const& y, mat const& X, int k, mat const& A, vec const& betabar, mat const& Ad,
    double s, mat const& inc_root, vec const& dstarbar, vec const& betahat, int R, int keep);
std::tuple<mat, vec, mat, mat> rivGibbs_rcpp_loop(
    vec const& y, vec const& x, mat const& z, mat const& w,
    vec const& mbg, mat const& Abg, vec const& md, mat const& Ad,
    mat const& V, double nu, int R, int keep);
std::tuple<mat, vec, int, int> rnegbinRw_rcpp_loop(
    vec const& y, mat const& X, vec const& betabar, mat const& rootA,
    double a, double b, vec beta, double alpha, bool fixalpha,
    mat const& betaroot, double alphacroot, int R, int keep);

struct sur_moments { vec y; mat X; };
std::tuple<mat, mat> rsurGibbs_rcpp_loop(
    std::vector<sur_moments> const& regdata_vector, vec const& indreg, vec const& cumnk,
    vec const& nk, mat const& XspXs, mat Sigmainv, mat const& A, vec const& Abetabar,
    double nu, mat const& V, int nvar, mat E, mat const& Y, int R, int keep);

std::tuple<mat, mat> rmvpGibbs_rcpp_loop(int R, int keep, int p,
                                          ivec const& y, mat const& X, vec const& beta0, mat const& sigma0,
                                          mat const& V, double nu, vec const& betabar, mat const& A);

std::tuple<mat, mat> rmnpGibbs_rcpp_loop(int R, int keep, int pm1,
                                          ivec const& y, mat const& X, vec const& beta0, mat const& sigma0,
                                          mat const& V, double nu, vec const& betabar, mat const& A);

struct MixComp { vec mu; mat rooti; };
std::tuple<mat, mat, std::vector<std::vector<MixComp>>> rnmixGibbs_rcpp_loop(
    mat const& y, mat const& Mubar, mat const& A, double nu, mat const& V,
    vec const& a, vec p, vec z, int R, int keep);

struct hier_moments { vec y; mat X; mat XpX; vec Xpy; };
struct MixComp_hier { vec mu; mat rooti; };
std::tuple<mat, mat, arma::cube, mat, std::vector<std::vector<MixComp_hier>>>
rhierLinearMixture_rcpp_loop(
    std::vector<hier_moments> const& regdata_vector, mat const& Z,
    vec const& deltabar, mat const& Ad, mat const& mubar, mat const& Amu,
    double nu, mat const& V, double nu_e, vec const& ssq,
    int R, int keep, bool drawdelta,
    vec olddelta, vec const& a, vec oldprob, vec ind, vec tau);

struct mnl_moments { vec y; mat X; mat hess; };
struct MixComp_mnl { vec mu; mat rooti; };
std::tuple<mat, arma::cube, mat, vec, std::vector<std::vector<MixComp_mnl>>>
rhierMnlRwMixture_rcpp_loop(
    std::vector<mnl_moments> const& lgtdata_vector, mat const& Z,
    vec const& deltabar, mat const& Ad, mat const& mubar, mat const& Amu,
    double nu, mat const& V, double s,
    int R, int keep, bool drawdelta,
    vec olddelta, vec const& a, vec oldprob, mat oldbetas, vec ind, vec const& SignRes);

struct murooti_dp { vec mu; mat rooti; };
std::tuple<vec, vec, vec, vec, vec, imat, std::vector<std::vector<murooti_dp>>>
rDPGibbs_rcpp_loop(int R, int keep, mat y, vec const& alim, vec const& nulim, vec const& vlim,
                   bool SCALE, int maxuniq, double power, double alphamin, double alphamax, int n,
                   int gridsize, double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha);

struct murooti_iv { vec mu; mat rooti; };
std::tuple<mat, vec, vec, vec, mat, vec, vec, vec, std::vector<std::vector<murooti_iv>>>
rivDP_rcpp_loop(int R, int keep, int dimd, vec const& mbg, mat const& Abg, vec const& md, mat const& Ad,
                vec const& y, bool isgamma, mat const& z, vec const& x, mat const& w, vec delta,
                double power, double alphamin, double alphamax, int n_prior, int gridsize,
                bool SCALE, int maxuniq, double scalex, double scaley,
                vec const& alim, vec const& nulim, vec const& vlim,
                double BayesmConstantA, int BayesmConstantnu);

struct mnlDP_moments { vec y; mat X; mat hess; };
struct murooti_mnldp { vec mu; mat rooti; };
std::tuple<mat, arma::cube, vec, vec, vec, vec, vec, vec, vec, std::vector<std::vector<murooti_mnldp>>>
rhierMnlDP_rcpp_loop(int R, int keep, std::vector<mnlDP_moments> const& lgtdata_vector, mat const& Z,
                      vec const& deltabar, mat const& Ad, double power, double alphamin, double alphamax, int n_prior,
                      vec const& alim, vec const& nulim, vec const& vlim, bool drawdelta, int nvar, mat oldbetas, double s,
                      int maxuniq, int gridsize, double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha);

std::tuple<vec, mat, mat, mat, mat, mat, vec, double>
rbayesBLP_rcpp_loop(bool IV, mat const& X, mat const& Z, vec const& share,
                    int J, int T, mat const& v, int R,
                    vec const& sigmasqR, mat const& A, vec const& theta_hat,
                    vec const& deltabar, mat const& Ad,
                    double nu0, double s0_sq, mat const& VOmega,
                    double ssq, mat const& cand_cov,
                    vec const& theta_bar_initial, vec const& r_initial,
                    double tau_sq_initial, mat const& Omega_initial, vec const& delta_initial,
                    double tol, int keep);

struct moments_negbin { vec y; mat X; mat hess; };
std::tuple<arma::cube, vec, mat, mat, vec, double, double>
rhierNegbinRw_rcpp_loop(std::vector<moments_negbin> const& regdata_vector, mat const& Z, mat Beta, mat Delta,
                        mat const& Deltabar, mat const& Adelta, double nu, mat const& V,
                        double a, double b, int R, int keep, double sbeta, double alphacroot,
                        mat rootA, double alpha, bool fixalpha);

std::tuple<mat, mat, mat, mat, mat, vec>
rscaleUsage_rcpp_loop(int k, mat const& x, int p, int n,
                      int R, int keep, int ndghk,
                      mat y, vec mu, mat Sigma, vec tau, vec sigma, mat Lambda, double e,
                      bool domu, bool doSigma, bool dosigma, bool dotau, bool doLambda, bool doe,
                      double nu, mat const& V, mat const& mubar, mat const& Am,
                      vec const& gsigma, vec const& gl11, vec const& gl22, vec const& gl12,
                      int nuL, mat const& VL, vec const& ge);

// Conversion helpers
vec numpy_to_vec(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    return vec((double*)buf.ptr, buf.size, false, true);
}

ivec numpy_to_ivec(py::array_t<int> arr) {
    py::buffer_info buf = arr.request();
    int* ptr = (int*)buf.ptr;
    ivec result(buf.size);
    for (size_t i = 0; i < buf.size; i++) {
        result[i] = ptr[i];
    }
    return result;
}

mat numpy_to_mat(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
    // numpy is row-major, armadillo is column-major - must copy and transpose
    mat temp((double*)buf.ptr, buf.shape[1], buf.shape[0], false, true);
    return temp.t();
}

py::array_t<double> vec_to_numpy(const vec& v) {
    auto result = py::array_t<double>(v.n_elem);
    auto buf = result.request();
    double* ptr = (double*)buf.ptr;
    for (size_t i = 0; i < v.n_elem; i++) ptr[i] = v(i);
    return result;
}

py::array_t<double> mat_to_numpy(const mat& m) {
    auto result = py::array_t<double>({m.n_rows, m.n_cols});
    auto buf = result.request();
    double* ptr = (double*)buf.ptr;
    for (size_t i = 0; i < m.n_rows; i++)
        for (size_t j = 0; j < m.n_cols; j++)
            ptr[i * m.n_cols + j] = m(i, j);
    return result;
}

py::array_t<double> cube_to_numpy(const arma::cube& c) {
    auto result = py::array_t<double>({c.n_rows, c.n_cols, c.n_slices});
    auto buf = result.request();
    double* ptr = (double*)buf.ptr;
    for (size_t k = 0; k < c.n_slices; k++)
        for (size_t i = 0; i < c.n_rows; i++)
            for (size_t j = 0; j < c.n_cols; j++)
                ptr[i * c.n_cols * c.n_slices + j * c.n_slices + k] = c(i, j, k);
    return result;
}

PYBIND11_MODULE(_bayesm_cpp, m) {
    m.doc() = "bayesm C++ functions";
    
    m.def("lndMvn", [](py::array_t<double> x, py::array_t<double> mu, py::array_t<double> rooti) {
        return lndMvn(numpy_to_vec(x), numpy_to_vec(mu), numpy_to_mat(rooti));
    }, "Log density multivariate normal");
    
    m.def("ghkvec", [](py::array_t<double> L, py::array_t<double> trunpt, py::array_t<double> above, 
                       int r, bool HALTON, py::array_t<double> pn) {
        return vec_to_numpy(ghkvec(numpy_to_mat(L), numpy_to_vec(trunpt), 
                                   numpy_to_vec(above), r, HALTON, numpy_to_vec(pn)));
    }, "GHK simulator");
    
    m.def("rwishart", [](double nu, py::array_t<double> V) {
        return mat_to_numpy(rwishart(nu, numpy_to_mat(V)));
    }, "Random Wishart");
    
    m.def("rdirichlet", [](py::array_t<double> alpha) {
        return vec_to_numpy(rdirichlet(numpy_to_vec(alpha)));
    }, "Random Dirichlet");
    
    m.def("rtrun", &rtrun, "Random truncated normal");
    
    m.def("lndIWishart", [](double nu, py::array_t<double> V, py::array_t<double> IW) {
        return lndIWishart(nu, numpy_to_mat(V), numpy_to_mat(IW));
    }, "Log density inverse Wishart");
    
    m.def("lndMvst", [](py::array_t<double> x, double nu, py::array_t<double> mu, 
                        py::array_t<double> rooti, bool NORMC) {
        return lndMvst(numpy_to_vec(x), nu, numpy_to_vec(mu), numpy_to_mat(rooti), NORMC);
    }, "Log density multivariate Student-t");
    
    m.def("rmvst", [](double nu, py::array_t<double> mu, py::array_t<double> root) {
        return vec_to_numpy(rmvst(nu, numpy_to_vec(mu), numpy_to_mat(root)));
    }, "Random multivariate Student-t");
    
    m.def("breg", [](py::array_t<double> y, py::array_t<double> X, py::array_t<double> betabar, 
                     py::array_t<double> A) {
        return vec_to_numpy(breg(numpy_to_vec(y), numpy_to_mat(X), 
                                numpy_to_vec(betabar), numpy_to_mat(A)));
    }, "Bayesian regression");
    
    m.def("llmnl", [](py::array_t<double> beta, py::array_t<double> y, py::array_t<double> X) {
        return llmnl(numpy_to_vec(beta), numpy_to_vec(y), numpy_to_mat(X));
    }, "Log-likelihood multinomial logit");
    
    m.def("lndIChisq", &lndIChisq, "Log density inverse chi-square");
    
    m.def("rmultireg", [](py::array_t<double> Y, py::array_t<double> X, py::array_t<double> Bbar,
                          py::array_t<double> A, double nu, py::array_t<double> V) {
        return mat_to_numpy(rmultireg(numpy_to_mat(Y), numpy_to_mat(X), numpy_to_mat(Bbar),
                                      numpy_to_mat(A), nu, numpy_to_mat(V)));
    }, "Multivariate regression");
    
    m.def("callroot", [](py::array_t<double> c1, py::array_t<double> c2, double tol, int iterlim) {
        return vec_to_numpy(callroot(numpy_to_vec(c1), numpy_to_vec(c2), tol, iterlim));
    }, "Find roots for non-homothetic logit");
    
    m.def("runireg_rcpp_loop", [](py::array_t<double> y, py::array_t<double> X, 
                                   py::array_t<double> betabar, py::array_t<double> A,
                                   double nu, double ssq, int R, int keep) {
        auto result = runireg_rcpp_loop(
            numpy_to_vec(y), numpy_to_mat(X), numpy_to_vec(betabar), numpy_to_mat(A),
            nu, ssq, R, keep);
        return py::make_tuple(mat_to_numpy(std::get<0>(result)), vec_to_numpy(std::get<1>(result)));
    }, "Univariate regression MCMC loop");
    
    m.def("runiregGibbs_rcpp_loop", [](py::array_t<double> y, py::array_t<double> X, 
                                        py::array_t<double> betabar, py::array_t<double> A,
                                        double nu, double ssq, double sigmasq, int R, int keep) {
        auto result = runiregGibbs_rcpp_loop(
            numpy_to_vec(y), numpy_to_mat(X), numpy_to_vec(betabar), numpy_to_mat(A),
            nu, ssq, sigmasq, R, keep);
        return py::make_tuple(mat_to_numpy(std::get<0>(result)), vec_to_numpy(std::get<1>(result)));
    }, "Univariate regression Gibbs MCMC loop");
    
    m.def("rhierLinearModel_rcpp_loop", [](py::list regdata_list, py::array_t<double> Z,
                                            py::array_t<double> Deltabar, py::array_t<double> A,
                                            double nu, py::array_t<double> V, double nu_e,
                                            py::array_t<double> ssq, py::array_t<double> tau,
                                            py::array_t<double> Delta, py::array_t<double> Vbeta,
                                            int R, int keep) {
        std::vector<moments> regdata_vector;
        for (size_t i = 0; i < regdata_list.size(); i++) {
            py::dict d = regdata_list[i].cast<py::dict>();
            moments m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            m.XpX = numpy_to_mat(d["XpX"].cast<py::array_t<double>>());
            m.Xpy = numpy_to_vec(d["Xpy"].cast<py::array_t<double>>());
            regdata_vector.push_back(m);
        }
        auto result = rhierLinearModel_rcpp_loop(
            regdata_vector, numpy_to_mat(Z), numpy_to_mat(Deltabar), numpy_to_mat(A),
            nu, numpy_to_mat(V), nu_e, numpy_to_vec(ssq), numpy_to_vec(tau),
            numpy_to_mat(Delta), numpy_to_mat(Vbeta), R, keep);
        return py::make_tuple(
            cube_to_numpy(std::get<0>(result)),
            mat_to_numpy(std::get<1>(result)),
            mat_to_numpy(std::get<2>(result)),
            mat_to_numpy(std::get<3>(result)));
    }, "Hierarchical linear model MCMC loop");
    
    m.def("rmnlIndepMetrop_rcpp_loop", [](int R, int keep, double nu,
                                          py::array_t<double> betastar, py::array_t<double> root,
                                          py::array_t<double> y, py::array_t<double> X,
                                          py::array_t<double> betabar, py::array_t<double> rootpi,
                                          py::array_t<double> rooti, double oldlimp, double oldlpost) {
        auto result = rmnlIndepMetrop_rcpp_loop(
            R, keep, nu, numpy_to_vec(betastar), numpy_to_mat(root),
            numpy_to_vec(y), numpy_to_mat(X), numpy_to_vec(betabar),
            numpy_to_mat(rootpi), numpy_to_mat(rooti), oldlimp, oldlpost);
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),
            vec_to_numpy(std::get<1>(result)),
            std::get<2>(result));
    }, "MNL Independence Metropolis MCMC loop");
    
    m.def("rbprobitGibbs_rcpp_loop", [](py::array_t<double> y, py::array_t<double> X,
                                        py::array_t<double> Abetabar, py::array_t<double> root,
                                        py::array_t<double> beta, py::array_t<double> sigma,
                                        py::array_t<double> trunpt, py::array_t<double> above,
                                        int R, int keep) {
        return mat_to_numpy(rbprobitGibbs_rcpp_loop(
            numpy_to_vec(y), numpy_to_mat(X), numpy_to_vec(Abetabar), numpy_to_mat(root),
            numpy_to_vec(beta), numpy_to_vec(sigma), numpy_to_vec(trunpt), numpy_to_vec(above),
            R, keep));
    }, "Binary probit Gibbs MCMC loop");
    
    m.def("rordprobitGibbs_rcpp_loop", [](py::array_t<double> y, py::array_t<double> X, int k,
                                          py::array_t<double> A, py::array_t<double> betabar,
                                          py::array_t<double> Ad, double s, py::array_t<double> inc_root,
                                          py::array_t<double> dstarbar, py::array_t<double> betahat,
                                          int R, int keep) {
        auto result = rordprobitGibbs_rcpp_loop(
            numpy_to_vec(y), numpy_to_mat(X), k, numpy_to_mat(A), numpy_to_vec(betabar),
            numpy_to_mat(Ad), s, numpy_to_mat(inc_root), numpy_to_vec(dstarbar),
            numpy_to_vec(betahat), R, keep);
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),
            mat_to_numpy(std::get<1>(result)),
            mat_to_numpy(std::get<2>(result)),
            std::get<3>(result));
    }, "Ordered probit Gibbs MCMC loop");
    
    m.def("rivGibbs_rcpp_loop", [](py::array_t<double> y, py::array_t<double> x,
                                   py::array_t<double> z, py::array_t<double> w,
                                   py::array_t<double> mbg, py::array_t<double> Abg,
                                   py::array_t<double> md, py::array_t<double> Ad,
                                   py::array_t<double> V, double nu, int R, int keep) {
        auto result = rivGibbs_rcpp_loop(
            numpy_to_vec(y), numpy_to_vec(x), numpy_to_mat(z), numpy_to_mat(w),
            numpy_to_vec(mbg), numpy_to_mat(Abg), numpy_to_vec(md), numpy_to_mat(Ad),
            numpy_to_mat(V), nu, R, keep);
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),
            vec_to_numpy(std::get<1>(result)),
            mat_to_numpy(std::get<2>(result)),
            mat_to_numpy(std::get<3>(result)));
    }, "IV regression Gibbs MCMC loop");

    m.def("rnegbinRw_rcpp_loop", [](py::array_t<double> y, py::array_t<double> X,
                                    py::array_t<double> betabar, py::array_t<double> rootA,
                                    double a, double b, py::array_t<double> beta, double alpha,
                                    bool fixalpha, py::array_t<double> betaroot, double alphacroot,
                                    int R, int keep) {
        auto result = rnegbinRw_rcpp_loop(
            numpy_to_vec(y), numpy_to_mat(X), numpy_to_vec(betabar), numpy_to_mat(rootA),
            a, b, numpy_to_vec(beta), alpha, fixalpha, numpy_to_mat(betaroot), alphacroot, R, keep);
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)), vec_to_numpy(std::get<1>(result)),
            std::get<2>(result), std::get<3>(result));
    }, "Negative binomial RW MCMC loop");

    m.def("rsurGibbs_rcpp_loop", [](py::list regdata_list, py::array_t<double> indreg,
                                    py::array_t<double> cumnk, py::array_t<double> nk,
                                    py::array_t<double> XspXs, py::array_t<double> Sigmainv,
                                    py::array_t<double> A, py::array_t<double> Abetabar,
                                    double nu, py::array_t<double> V, int nvar,
                                    py::array_t<double> E, py::array_t<double> Y, int R, int keep) {
        std::vector<sur_moments> regdata_vector;
        for (size_t i = 0; i < regdata_list.size(); i++) {
            py::dict d = regdata_list[i].cast<py::dict>();
            sur_moments m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            regdata_vector.push_back(m);
        }
        auto result = rsurGibbs_rcpp_loop(
            regdata_vector, numpy_to_vec(indreg), numpy_to_vec(cumnk), numpy_to_vec(nk),
            numpy_to_mat(XspXs), numpy_to_mat(Sigmainv), numpy_to_mat(A), numpy_to_vec(Abetabar),
            nu, numpy_to_mat(V), nvar, numpy_to_mat(E), numpy_to_mat(Y), R, keep);
        return py::make_tuple(mat_to_numpy(std::get<0>(result)), mat_to_numpy(std::get<1>(result)));
    }, "SUR Gibbs MCMC loop");

    m.def("rmvpGibbs_rcpp_loop", [](int R, int keep, int p,
                                    py::array_t<int> y, py::array_t<double> X,
                                    py::array_t<double> beta0, py::array_t<double> sigma0,
                                    py::array_t<double> V, double nu,
                                    py::array_t<double> betabar, py::array_t<double> A) {
        auto result = rmvpGibbs_rcpp_loop(R, keep, p,
            numpy_to_ivec(y), numpy_to_mat(X), numpy_to_vec(beta0), numpy_to_mat(sigma0),
            numpy_to_mat(V), nu, numpy_to_vec(betabar), numpy_to_mat(A));
        return py::make_tuple(mat_to_numpy(std::get<0>(result)), mat_to_numpy(std::get<1>(result)));
    }, "Multivariate probit Gibbs MCMC loop");

    m.def("rmnpGibbs_rcpp_loop", [](int R, int keep, int pm1,
                                    py::array_t<int> y, py::array_t<double> X,
                                    py::array_t<double> beta0, py::array_t<double> sigma0,
                                    py::array_t<double> V, double nu,
                                    py::array_t<double> betabar, py::array_t<double> A) {
        auto result = rmnpGibbs_rcpp_loop(R, keep, pm1,
            numpy_to_ivec(y), numpy_to_mat(X), numpy_to_vec(beta0), numpy_to_mat(sigma0),
            numpy_to_mat(V), nu, numpy_to_vec(betabar), numpy_to_mat(A));
        return py::make_tuple(mat_to_numpy(std::get<0>(result)), mat_to_numpy(std::get<1>(result)));
    }, "Multinomial probit Gibbs MCMC loop");

    m.def("rnmixGibbs_rcpp_loop", [](py::array_t<double> y, py::array_t<double> Mubar,
                                     py::array_t<double> A, double nu, py::array_t<double> V,
                                     py::array_t<double> a, py::array_t<double> p,
                                     py::array_t<double> z, int R, int keep) {
        auto result = rnmixGibbs_rcpp_loop(
            numpy_to_mat(y), numpy_to_mat(Mubar), numpy_to_mat(A), nu, numpy_to_mat(V),
            numpy_to_vec(a), numpy_to_vec(p), numpy_to_vec(z), R, keep);
        
        // Convert compdraw to Python list of lists of dicts
        py::list compdraw_py;
        for (const auto& draw : std::get<2>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            compdraw_py.append(comp_list);
        }
        
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),
            mat_to_numpy(std::get<1>(result)),
            compdraw_py);
    }, "Normal mixture Gibbs MCMC loop");

    m.def("rhierLinearMixture_rcpp_loop", [](py::list regdata_list, py::array_t<double> Z,
                                              py::array_t<double> deltabar, py::array_t<double> Ad,
                                              py::array_t<double> mubar, py::array_t<double> Amu,
                                              double nu, py::array_t<double> V, double nu_e,
                                              py::array_t<double> ssq, int R, int keep, bool drawdelta,
                                              py::array_t<double> olddelta, py::array_t<double> a,
                                              py::array_t<double> oldprob, py::array_t<double> ind,
                                              py::array_t<double> tau) {
        std::vector<hier_moments> regdata_vector;
        for (size_t i = 0; i < regdata_list.size(); i++) {
            py::dict d = regdata_list[i].cast<py::dict>();
            hier_moments m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            m.XpX = numpy_to_mat(d["XpX"].cast<py::array_t<double>>());
            m.Xpy = numpy_to_vec(d["Xpy"].cast<py::array_t<double>>());
            regdata_vector.push_back(m);
        }
        auto result = rhierLinearMixture_rcpp_loop(
            regdata_vector, numpy_to_mat(Z), numpy_to_vec(deltabar), numpy_to_mat(Ad),
            numpy_to_mat(mubar), numpy_to_mat(Amu), nu, numpy_to_mat(V), nu_e,
            numpy_to_vec(ssq), R, keep, drawdelta, numpy_to_vec(olddelta),
            numpy_to_vec(a), numpy_to_vec(oldprob), numpy_to_vec(ind), numpy_to_vec(tau));
        
        py::list compdraw_py;
        for (const auto& draw : std::get<4>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            compdraw_py.append(comp_list);
        }
        
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),    // taudraw
            mat_to_numpy(std::get<1>(result)),    // Deltadraw
            cube_to_numpy(std::get<2>(result)),   // betadraw
            mat_to_numpy(std::get<3>(result)),    // probdraw
            compdraw_py);                         // compdraw
    }, "Hierarchical linear mixture model MCMC loop");

    m.def("rhierMnlRwMixture_rcpp_loop", [](py::list lgtdata_list, py::array_t<double> Z,
                                            py::array_t<double> deltabar, py::array_t<double> Ad,
                                            py::array_t<double> mubar, py::array_t<double> Amu,
                                            double nu, py::array_t<double> V, double s,
                                            int R, int keep, bool drawdelta,
                                            py::array_t<double> olddelta, py::array_t<double> a,
                                            py::array_t<double> oldprob, py::array_t<double> oldbetas,
                                            py::array_t<double> ind, py::array_t<double> SignRes) {
        std::vector<mnl_moments> lgtdata_vector;
        for (size_t i = 0; i < lgtdata_list.size(); i++) {
            py::dict d = lgtdata_list[i].cast<py::dict>();
            mnl_moments m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            m.hess = numpy_to_mat(d["hess"].cast<py::array_t<double>>());
            lgtdata_vector.push_back(m);
        }
        auto result = rhierMnlRwMixture_rcpp_loop(
            lgtdata_vector, numpy_to_mat(Z), numpy_to_vec(deltabar), numpy_to_mat(Ad),
            numpy_to_mat(mubar), numpy_to_mat(Amu), nu, numpy_to_mat(V), s,
            R, keep, drawdelta, numpy_to_vec(olddelta), numpy_to_vec(a),
            numpy_to_vec(oldprob), numpy_to_mat(oldbetas), numpy_to_vec(ind), numpy_to_vec(SignRes));
        
        py::list compdraw_py;
        for (const auto& draw : std::get<4>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            compdraw_py.append(comp_list);
        }
        
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),    // Deltadraw
            cube_to_numpy(std::get<1>(result)),   // betadraw
            mat_to_numpy(std::get<2>(result)),    // probdraw
            vec_to_numpy(std::get<3>(result)),    // loglike
            compdraw_py);                         // compdraw
    }, "Hierarchical MNL RW mixture model MCMC loop");

    m.def("rDPGibbs_rcpp_loop", [](int R, int keep, py::array_t<double> y,
                                   py::array_t<double> alim, py::array_t<double> nulim, py::array_t<double> vlim,
                                   bool SCALE, int maxuniq, double power, double alphamin, double alphamax, int n,
                                   int gridsize, double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha) {
        auto result = rDPGibbs_rcpp_loop(R, keep, numpy_to_mat(y),
                                          numpy_to_vec(alim), numpy_to_vec(nulim), numpy_to_vec(vlim),
                                          SCALE, maxuniq, power, alphamin, alphamax, n,
                                          gridsize, BayesmConstantA, BayesmConstantnuInc, BayesmConstantDPalpha);
        
        py::list thetaNp1draw_py;
        for (const auto& draw : std::get<6>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            thetaNp1draw_py.append(comp_list);
        }
        
        // Convert imat to numpy
        imat& inddraw = std::get<5>(result);
        auto inddraw_np = py::array_t<int>({(int)inddraw.n_rows, (int)inddraw.n_cols});
        auto buf = inddraw_np.request();
        int* ptr = (int*)buf.ptr;
        for (size_t i = 0; i < inddraw.n_rows; i++)
            for (size_t j = 0; j < inddraw.n_cols; j++)
                ptr[i * inddraw.n_cols + j] = inddraw(i, j);
        
        return py::make_tuple(
            vec_to_numpy(std::get<0>(result)),  // alphadraw
            vec_to_numpy(std::get<1>(result)),  // Istardraw
            vec_to_numpy(std::get<2>(result)),  // adraw
            vec_to_numpy(std::get<3>(result)),  // nudraw
            vec_to_numpy(std::get<4>(result)),  // vdraw
            inddraw_np,                         // inddraw
            thetaNp1draw_py);                   // thetaNp1draw
    }, "Dirichlet Process Gibbs sampler loop");

    m.def("rivDP_rcpp_loop", [](int R, int keep, int dimd,
                                py::array_t<double> mbg, py::array_t<double> Abg,
                                py::array_t<double> md, py::array_t<double> Ad,
                                py::array_t<double> y, bool isgamma, py::array_t<double> z,
                                py::array_t<double> x, py::array_t<double> w, py::array_t<double> delta,
                                double power, double alphamin, double alphamax, int n_prior, int gridsize,
                                bool SCALE, int maxuniq, double scalex, double scaley,
                                py::array_t<double> alim, py::array_t<double> nulim, py::array_t<double> vlim,
                                double BayesmConstantA, int BayesmConstantnu) {
        auto result = rivDP_rcpp_loop(R, keep, dimd,
                                       numpy_to_vec(mbg), numpy_to_mat(Abg),
                                       numpy_to_vec(md), numpy_to_mat(Ad),
                                       numpy_to_vec(y), isgamma, numpy_to_mat(z),
                                       numpy_to_vec(x), numpy_to_mat(w), numpy_to_vec(delta),
                                       power, alphamin, alphamax, n_prior, gridsize,
                                       SCALE, maxuniq, scalex, scaley,
                                       numpy_to_vec(alim), numpy_to_vec(nulim), numpy_to_vec(vlim),
                                       BayesmConstantA, BayesmConstantnu);
        
        py::list thetaNp1draw_py;
        for (const auto& draw : std::get<8>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            thetaNp1draw_py.append(comp_list);
        }
        
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),  // deltadraw
            vec_to_numpy(std::get<1>(result)),  // betadraw
            vec_to_numpy(std::get<2>(result)),  // alphadraw
            vec_to_numpy(std::get<3>(result)),  // Istardraw
            mat_to_numpy(std::get<4>(result)),  // gammadraw
            vec_to_numpy(std::get<5>(result)),  // adraw
            vec_to_numpy(std::get<6>(result)),  // nudraw
            vec_to_numpy(std::get<7>(result)),  // vdraw
            thetaNp1draw_py);                   // thetaNp1draw
    }, "IV regression with Dirichlet Process prior loop");

    m.def("rhierMnlDP_rcpp_loop", [](int R, int keep, py::list lgtdata_list, py::array_t<double> Z,
                                     py::array_t<double> deltabar, py::array_t<double> Ad,
                                     double power, double alphamin, double alphamax, int n_prior,
                                     py::array_t<double> alim, py::array_t<double> nulim, py::array_t<double> vlim,
                                     bool drawdelta, int nvar, py::array_t<double> oldbetas, double s,
                                     int maxuniq, int gridsize,
                                     double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha) {
        std::vector<mnlDP_moments> lgtdata_vector;
        for (size_t i = 0; i < lgtdata_list.size(); i++) {
            py::dict d = lgtdata_list[i].cast<py::dict>();
            mnlDP_moments m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            m.hess = numpy_to_mat(d["hess"].cast<py::array_t<double>>());
            lgtdata_vector.push_back(m);
        }
        auto result = rhierMnlDP_rcpp_loop(R, keep, lgtdata_vector, numpy_to_mat(Z),
                                            numpy_to_vec(deltabar), numpy_to_mat(Ad),
                                            power, alphamin, alphamax, n_prior,
                                            numpy_to_vec(alim), numpy_to_vec(nulim), numpy_to_vec(vlim),
                                            drawdelta, nvar, numpy_to_mat(oldbetas), s,
                                            maxuniq, gridsize,
                                            BayesmConstantA, BayesmConstantnuInc, BayesmConstantDPalpha);
        
        py::list compdraw_py;
        for (const auto& draw : std::get<9>(result)) {
            py::list comp_list;
            for (const auto& comp : draw) {
                py::dict comp_dict;
                comp_dict["mu"] = vec_to_numpy(comp.mu);
                comp_dict["rooti"] = mat_to_numpy(comp.rooti);
                comp_list.append(comp_dict);
            }
            compdraw_py.append(comp_list);
        }
        
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),    // Deltadraw
            cube_to_numpy(std::get<1>(result)),   // betadraw
            vec_to_numpy(std::get<2>(result)),    // probdraw
            vec_to_numpy(std::get<3>(result)),    // loglike
            vec_to_numpy(std::get<4>(result)),    // alphadraw
            vec_to_numpy(std::get<5>(result)),    // Istardraw
            vec_to_numpy(std::get<6>(result)),    // adraw
            vec_to_numpy(std::get<7>(result)),    // nudraw
            vec_to_numpy(std::get<8>(result)),    // vdraw
            compdraw_py);                         // compdraw
    }, "Hierarchical MNL with Dirichlet Process prior loop");

    m.def("rbayesBLP_rcpp_loop", [](bool IV, py::array_t<double> X, py::array_t<double> Z, py::array_t<double> share,
                                    int J, int T, py::array_t<double> v, int R,
                                    py::array_t<double> sigmasqR, py::array_t<double> A, py::array_t<double> theta_hat,
                                    py::array_t<double> deltabar, py::array_t<double> Ad,
                                    double nu0, double s0_sq, py::array_t<double> VOmega,
                                    double ssq, py::array_t<double> cand_cov,
                                    py::array_t<double> theta_bar_initial, py::array_t<double> r_initial,
                                    double tau_sq_initial, py::array_t<double> Omega_initial, py::array_t<double> delta_initial,
                                    double tol, int keep) {
        auto result = rbayesBLP_rcpp_loop(IV, numpy_to_mat(X), numpy_to_mat(Z), numpy_to_vec(share),
                                           J, T, numpy_to_mat(v), R,
                                           numpy_to_vec(sigmasqR), numpy_to_mat(A), numpy_to_vec(theta_hat),
                                           numpy_to_vec(deltabar), numpy_to_mat(Ad),
                                           nu0, s0_sq, numpy_to_mat(VOmega),
                                           ssq, numpy_to_mat(cand_cov),
                                           numpy_to_vec(theta_bar_initial), numpy_to_vec(r_initial),
                                           tau_sq_initial, numpy_to_mat(Omega_initial), numpy_to_vec(delta_initial),
                                           tol, keep);
        return py::make_tuple(
            vec_to_numpy(std::get<0>(result)),    // tau_sq_all (or empty if IV)
            mat_to_numpy(std::get<1>(result)),    // Omega_all (or small if not IV)
            mat_to_numpy(std::get<2>(result)),    // delta_all (or small if not IV)
            mat_to_numpy(std::get<3>(result)),    // theta_bar_all
            mat_to_numpy(std::get<4>(result)),    // r_all
            mat_to_numpy(std::get<5>(result)),    // Sigma_all
            vec_to_numpy(std::get<6>(result)),    // ll_all
            std::get<7>(result));                 // acceptrate
    }, "BLP demand estimation MCMC loop");

    m.def("rhierNegbinRw_rcpp_loop", [](py::list regdata_list, py::array_t<double> Z,
                                        py::array_t<double> Beta, py::array_t<double> Delta,
                                        py::array_t<double> Deltabar, py::array_t<double> Adelta,
                                        double nu, py::array_t<double> V, double a, double b,
                                        int R, int keep, double sbeta, double alphacroot,
                                        py::array_t<double> rootA, double alpha, bool fixalpha) {
        std::vector<moments_negbin> regdata_vector;
        for (size_t i = 0; i < regdata_list.size(); i++) {
            py::dict d = regdata_list[i].cast<py::dict>();
            moments_negbin m;
            m.y = numpy_to_vec(d["y"].cast<py::array_t<double>>());
            m.X = numpy_to_mat(d["X"].cast<py::array_t<double>>());
            m.hess = numpy_to_mat(d["hess"].cast<py::array_t<double>>());
            regdata_vector.push_back(m);
        }
        auto result = rhierNegbinRw_rcpp_loop(
            regdata_vector, numpy_to_mat(Z), numpy_to_mat(Beta), numpy_to_mat(Delta),
            numpy_to_mat(Deltabar), numpy_to_mat(Adelta), nu, numpy_to_mat(V), a, b,
            R, keep, sbeta, alphacroot, numpy_to_mat(rootA), alpha, fixalpha);
        return py::make_tuple(
            cube_to_numpy(std::get<0>(result)),   // Betadraw
            vec_to_numpy(std::get<1>(result)),    // alphadraw
            mat_to_numpy(std::get<2>(result)),    // Vbetadraw
            mat_to_numpy(std::get<3>(result)),    // Deltadraw
            vec_to_numpy(std::get<4>(result)),    // llike
            std::get<5>(result),                  // acceptrbeta
            std::get<6>(result));                 // acceptralpha
    }, "Hierarchical negative binomial RW Metropolis loop");

    m.def("rscaleUsage_rcpp_loop", [](int k, py::array_t<double> x, int p, int n,
                                      int R, int keep, int ndghk,
                                      py::array_t<double> y, py::array_t<double> mu,
                                      py::array_t<double> Sigma, py::array_t<double> tau,
                                      py::array_t<double> sigma, py::array_t<double> Lambda, double e,
                                      bool domu, bool doSigma, bool dosigma, bool dotau, bool doLambda, bool doe,
                                      double nu, py::array_t<double> V, py::array_t<double> mubar, py::array_t<double> Am,
                                      py::array_t<double> gsigma, py::array_t<double> gl11,
                                      py::array_t<double> gl22, py::array_t<double> gl12,
                                      int nuL, py::array_t<double> VL, py::array_t<double> ge) {
        auto result = rscaleUsage_rcpp_loop(k, numpy_to_mat(x), p, n,
                                             R, keep, ndghk,
                                             numpy_to_mat(y), numpy_to_vec(mu),
                                             numpy_to_mat(Sigma), numpy_to_vec(tau),
                                             numpy_to_vec(sigma), numpy_to_mat(Lambda), e,
                                             domu, doSigma, dosigma, dotau, doLambda, doe,
                                             nu, numpy_to_mat(V), numpy_to_mat(mubar), numpy_to_mat(Am),
                                             numpy_to_vec(gsigma), numpy_to_vec(gl11),
                                             numpy_to_vec(gl22), numpy_to_vec(gl12),
                                             nuL, numpy_to_mat(VL), numpy_to_vec(ge));
        return py::make_tuple(
            mat_to_numpy(std::get<0>(result)),   // drSigma
            mat_to_numpy(std::get<1>(result)),   // drmu
            mat_to_numpy(std::get<2>(result)),   // drtau
            mat_to_numpy(std::get<3>(result)),   // drsigma
            mat_to_numpy(std::get<4>(result)),   // drLambda
            vec_to_numpy(std::get<5>(result)));  // dre
    }, "Scale usage model MCMC loop");
}
