// rhierNegbinRw - Hierarchical Negative Binomial Random Walk Metropolis
#include "bayesm.h"
#include <tuple>

struct moments_negbin { vec y; mat X; mat hess; };

namespace {

double llnegbin_local(vec const& y, vec const& lambda, double alpha, bool constant) {
    int nobs = y.size();
    vec prob = alpha / (alpha + lambda);
    vec logp(nobs);
    
    if (constant) {
        for (int i = 0; i < nobs; i++) {
            logp[i] = R::dnbinom(y[i], alpha, prob[i], 1);
        }
    } else {
        logp = alpha * log(prob) + y % log(1 - prob);
    }
    return sum(logp);
}

double lpostbeta_local(double alpha, vec const& beta, mat const& X, vec const& y, 
                       vec const& betabar, mat const& rootA) {
    vec lambda = exp(X * beta);
    double ll = llnegbin_local(y, lambda, alpha, false);
    vec z = rootA * (beta - betabar);
    double lprior = -0.5 * sum(z % z);
    return ll + lprior;
}

double llnegbinpooled_local(std::vector<moments_negbin> const& regdata_vector, mat const& Beta, double alpha) {
    int nreg = regdata_vector.size();
    double ll = 0.0;
    for (int reg = 0; reg < nreg; reg++) {
        vec lambda = exp(regdata_vector[reg].X * Beta.row(reg).t());
        ll = ll + llnegbin_local(regdata_vector[reg].y, lambda, alpha, true);
    }
    return ll;
}

std::tuple<mat, mat, mat, mat> rwishart_negbin(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    for (int i = 0; i < m; i++) T(i, i) = std::sqrt(R::rchisq(nu - i));
    for (int i = 1; i < m; i++)
        for (int j = 0; j < i; j++) T(i, j) = R::rnorm(0, 1);
    
    mat Vsym = 0.5 * (V + V.t());
    vec eigval_V;
    eig_sym(eigval_V, Vsym);
    double minEig_V = eigval_V.min();
    if (minEig_V < 1e-8) {
        Vsym = Vsym + (std::abs(minEig_V) + 1e-6) * eye(m, m);
    }
    mat C = chol(Vsym, "lower");
    mat CT = C * T;
    mat W = CT * CT.t();
    W = 0.5 * (W + W.t());
    vec eigval_W;
    eig_sym(eigval_W, W);
    double minEig_W = eigval_W.min();
    if (minEig_W < 1e-8) {
        W = W + (std::abs(minEig_W) + 1e-6) * eye(m, m);
    }
    mat IW = solve(W, eye<mat>(m, m));
    mat Cupper = chol(W);
    mat CI = solve(trimatu(Cupper), eye<mat>(m, m));
    return std::make_tuple(W, IW, Cupper, CI);
}

mat rmultireg_negbin(mat const& Y, mat const& X, mat const& Bbar, mat const& A, double nu, mat const& V) {
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    
    mat RA = chol(A);
    mat W = join_cols(X, RA);
    mat Z = join_cols(Y, RA * Bbar);
    mat IR = solve(trimatu(chol(W.t() * W)), eye(k, k));
    mat Btilde = (IR * IR.t()) * (W.t() * Z);
    mat E = Z - W * Btilde;
    mat S = E.t() * E;
    
    mat ucholinv = solve(trimatu(chol(V + S)), eye(m, m));
    mat VSinv = ucholinv * ucholinv.t();
    auto [Wout, IW, Cupper, CI] = rwishart_negbin(nu + n, VSinv);
    mat Sigma = IW;
    
    mat C_sigma = chol(Sigma, "lower");
    mat btilde_vec = vectorise(Btilde);
    mat Sigma_kron_IR = kron(C_sigma, IR);
    vec rnorm_vec(k * m);
    for (int i = 0; i < k * m; i++) rnorm_vec[i] = R::rnorm(0, 1);
    mat B = reshape(btilde_vec + Sigma_kron_IR * rnorm_vec, k, m);
    
    return B;
}

} // anonymous namespace

std::tuple<arma::cube, vec, mat, mat, vec, double, double>
rhierNegbinRw_rcpp_loop(std::vector<moments_negbin> const& regdata_vector, mat const& Z, mat Beta, mat Delta,
                        mat const& Deltabar, mat const& Adelta, double nu, mat const& V,
                        double a, double b, int R, int keep, double sbeta, double alphacroot,
                        mat rootA, double alpha, bool fixalpha) {
    
    int nreg = regdata_vector.size();
    int nz = Z.n_cols;
    int nvar = rootA.n_cols;
    int nacceptbeta = 0;
    int nacceptalpha = 0;
    
    mat Vbetainv = rootA.t() * rootA;
    
    // Allocate storage
    vec oldlpostbeta = zeros<vec>(nreg);
    vec clpostbeta = zeros<vec>(nreg);
    arma::cube Betadraw = zeros<arma::cube>(nreg, nvar, R / keep);
    vec alphadraw = zeros<vec>(R / keep);
    vec llike = zeros<vec>(R / keep);
    mat Vbetadraw = zeros<mat>(R / keep, nvar * nvar);
    mat Deltadraw = zeros<mat>(R / keep, nvar * nz);
    
    int mkeep;
    
    for (int rep = 0; rep < R; rep++) {
        mat betabar = Z * Delta;
        
        // Draw betai
        for (int reg = 0; reg < nreg; reg++) {
            vec betabari = betabar.row(reg).t();
            mat hessinv = regdata_vector[reg].hess + Vbetainv;
            hessinv = 0.5 * (hessinv + hessinv.t());
            vec eigval_h;
            eig_sym(eigval_h, hessinv);
            double minEig_h = eigval_h.min();
            if (minEig_h < 1e-8) {
                hessinv = hessinv + (std::abs(minEig_h) + 1e-6) * eye(nvar, nvar);
            }
            mat betacvar = sbeta * solve(hessinv, eye(nvar, nvar));
            betacvar = 0.5 * (betacvar + betacvar.t());
            vec eigval_bc;
            eig_sym(eigval_bc, betacvar);
            double minEig_bc = eigval_bc.min();
            if (minEig_bc < 1e-8) {
                betacvar = betacvar + (std::abs(minEig_bc) + 1e-6) * eye(nvar, nvar);
            }
            mat betaroot = chol(betacvar, "lower");
            
            vec rnorm_vec(nvar);
            for (int i = 0; i < nvar; i++) rnorm_vec[i] = R::rnorm(0, 1);
            vec betac = Beta.row(reg).t() + betaroot * rnorm_vec;
            
            oldlpostbeta[reg] = lpostbeta_local(alpha, Beta.row(reg).t(), regdata_vector[reg].X, 
                                                regdata_vector[reg].y, betabari, rootA);
            clpostbeta[reg] = lpostbeta_local(alpha, betac, regdata_vector[reg].X, 
                                              regdata_vector[reg].y, betabari, rootA);
            double ldiff = clpostbeta[reg] - oldlpostbeta[reg];
            double acc = exp(ldiff);
            if (acc > 1) acc = 1;
            double unif = (acc < 1) ? R::runif(0, 1) : 0;
            if (unif <= acc) {
                Beta.row(reg) = betac.t();
                nacceptbeta++;
            }
        }
        
        // Draw alpha
        if (!fixalpha) {
            double logalphac = log(alpha) + alphacroot * R::rnorm(0, 1);
            double oldlpostalpha = llnegbinpooled_local(regdata_vector, Beta, alpha) + 
                                   (a - 1) * log(alpha) - b * alpha;
            double clpostalpha = llnegbinpooled_local(regdata_vector, Beta, exp(logalphac)) + 
                                 (a - 1) * logalphac - b * exp(logalphac);
            double ldiff = clpostalpha - oldlpostalpha;
            double acc = exp(ldiff);
            if (acc > 1) acc = 1;
            double unif = (acc < 1) ? R::runif(0, 1) : 0;
            if (unif <= acc) {
                alpha = exp(logalphac);
                nacceptalpha++;
            }
        }
        
        // Draw Vbeta and Delta using rmultireg
        mat B = rmultireg_negbin(Beta, Z, Deltabar, Adelta, nu, V);
        Delta = B;
        
        mat E = Beta - Z * Delta;
        mat S = E.t() * E;
        mat ucholinv = solve(trimatu(chol(V + S)), eye(nvar, nvar));
        mat VSinv = ucholinv * ucholinv.t();
        auto [W, IW, Cupper, CI] = rwishart_negbin(nu + nreg, VSinv);
        mat Vbeta = IW;
        
        Vbetainv = solve(Vbeta, eye(nvar, nvar));
        Vbetainv = 0.5 * (Vbetainv + Vbetainv.t());
        vec eigval_vbi;
        eig_sym(eigval_vbi, Vbetainv);
        double minEig_vbi = eigval_vbi.min();
        if (minEig_vbi < 1e-8) {
            Vbetainv = Vbetainv + (std::abs(minEig_vbi) + 1e-6) * eye(nvar, nvar);
        }
        rootA = chol(Vbetainv);
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            Betadraw.slice(mkeep - 1) = Beta;
            alphadraw[mkeep - 1] = alpha;
            Vbetadraw.row(mkeep - 1) = vectorise(Vbeta).t();
            Deltadraw.row(mkeep - 1) = vectorise(Delta).t();
            llike[mkeep - 1] = llnegbinpooled_local(regdata_vector, Beta, alpha);
        }
    }
    
    double acceptrbeta = nacceptbeta / (R * nreg * 1.0) * 100;
    double acceptralpha = nacceptalpha / (R * 1.0) * 100;
    
    return std::make_tuple(Betadraw, alphadraw, Vbetadraw, Deltadraw, llike, acceptrbeta, acceptralpha);
}
