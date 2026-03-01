#include "bayesm.h"
#include <tuple>

std::tuple<mat, vec> runiregGibbs_rcpp_loop(
    vec const& y, mat const& X, vec const& betabar, mat const& A, 
    double nu, double ssq, double sigmasq, int R, int keep) {

    int mkeep;
    double s;
    mat IR;
    vec btilde, beta;
    
    int nvar = X.n_cols;
    int nobs = y.size();
    
    vec sigmasqdraw(R/keep);
    mat betadraw(R/keep, nvar);
    
    mat XpX = trans(X) * X;
    vec Xpy = trans(X) * y;
    vec Abetabar = A * betabar;
    
    std::chi_squared_distribution<double> chisq_dist(nu + nobs);

    for (int rep = 0; rep < R; rep++) {
        // Draw beta | sigmasq
        IR = solve(trimatu(chol(XpX / sigmasq + A)), eye(nvar, nvar));
        btilde = (IR * trans(IR)) * (Xpy / sigmasq + Abetabar);
        
        vec znorm(nvar);
        for (int i = 0; i < nvar; i++) {
            znorm(i) = R::rnorm(0, 1);
        }
        beta = btilde + IR * znorm;
        
        // Draw sigmasq | beta
        s = sum(square(y - X * beta));
        sigmasq = (nu * ssq + s) / chisq_dist(R::rng());
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = trans(beta);
            sigmasqdraw(mkeep - 1) = sigmasq;
        }
    }
    
    return std::make_tuple(betadraw, sigmasqdraw);
}
