#include "bayesm.h"
#include <tuple>

double llmnl(vec const& beta, vec const& y, mat const& X);
double lndMvn(vec const& x, vec const& mu, mat const& rooti);
double lndMvst(vec const& x, double nu, vec const& mu, mat const& rooti, bool NORMC);
vec rmvst(double nu, vec const& mu, mat const& root);

std::tuple<mat, vec, int> rmnlIndepMetrop_rcpp_loop(
    int R, int keep, double nu, vec const& betastar, mat const& root,
    vec const& y, mat const& X, vec const& betabar, mat const& rootpi,
    mat const& rooti, double oldlimp, double oldlpost) {

    int mkeep = 0;
    int naccept = 0;
    int ncolX = X.n_cols;
    
    mat betadraw(R/keep, ncolX);
    vec loglike(R/keep);
    vec betac = zeros<vec>(ncolX);
    rowvec beta = zeros<rowvec>(ncolX);
    double cloglike, clpost, climp, ldiff, alpha, unif, oldloglike = 0.0;

    for (int rep = 0; rep < R; rep++) {
        betac = rmvst(nu, betastar, root);
        cloglike = llmnl(betac, y, X);
        clpost = cloglike + lndMvn(betac, betabar, rootpi);
        climp = lndMvst(betac, nu, betastar, rooti, false);
        ldiff = clpost + oldlimp - oldlpost - climp;
        alpha = std::min(1.0, exp(ldiff));
        
        if (alpha < 1.0) {
            unif = R::runif(0, 1);
        } else {
            unif = 0.0;
        }
        
        if (unif <= alpha) {
            beta = trans(betac);
            oldloglike = cloglike;
            oldlpost = clpost;
            oldlimp = climp;
            naccept++;
        }
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = beta;
            loglike(mkeep - 1) = oldloglike;
        }
    }
    
    return std::make_tuple(betadraw, loglike, naccept);
}
