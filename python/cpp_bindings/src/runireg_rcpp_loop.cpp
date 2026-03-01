#include "bayesm.h"
#include <tuple>

std::tuple<mat, vec> runireg_rcpp_loop(
    vec const& y, mat const& X, vec const& betabar, mat const& A, 
    double nu, double ssq, int R, int keep) {

    int mkeep;
    double s, sigmasq;
    mat RA, W, IR;
    vec z, btilde, res, beta;
    
    int nvar = X.n_cols;
    int nobs = y.size();
    
    vec sigmasqdraw(R/keep);
    mat betadraw(R/keep, nvar);

    for (int rep = 0; rep < R; rep++) {
        RA = chol(A);
        W = join_cols(X, RA);
        z = join_cols(y, RA * betabar);
        IR = solve(trimatu(chol(trans(W) * W)), eye(nvar, nvar));
        btilde = (IR * trans(IR)) * (trans(W) * z);
        res = z - W * btilde;
        s = as_scalar(trans(res) * res);
        
        // Draw sigmasq from inverse chi-square
        double chi_draw = 0.0;
        std::chi_squared_distribution<double> chisq_dist(nu + nobs);
        chi_draw = chisq_dist(R::rng());
        sigmasq = (nu * ssq + s) / chi_draw;
        
        // Draw beta given sigmasq
        vec znorm(nvar);
        for (int i = 0; i < nvar; i++) {
            znorm(i) = R::rnorm(0, 1);
        }
        beta = btilde + sqrt(sigmasq) * (IR * znorm);
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = trans(beta);
            sigmasqdraw(mkeep - 1) = sigmasq;
        }
    }
    
    return std::make_tuple(betadraw, sigmasqdraw);
}
