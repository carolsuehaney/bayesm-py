// rnegbinRw - Negative Binomial Random Walk Metropolis Sampler
#include "bayesm.h"
#include <tuple>
#include <cmath>

namespace {

double llnegbin_local(vec const& y, vec const& lambda, double alpha, bool constant) {
    int nobs = y.size();
    vec prob = alpha / (alpha + lambda);
    vec logp(nobs);
    
    if (constant) {
        // Normalized log-likelihood using dnbinom formula
        // dnbinom(y, size=alpha, prob=p) = choose(y+alpha-1, y) * p^alpha * (1-p)^y
        // log form: lgamma(y+alpha) - lgamma(y+1) - lgamma(alpha) + alpha*log(p) + y*log(1-p)
        for (int i = 0; i < nobs; i++) {
            double yi = y[i];
            double pi = prob[i];
            logp[i] = std::lgamma(yi + alpha) - std::lgamma(yi + 1) - std::lgamma(alpha)
                    + alpha * std::log(pi) + yi * std::log(1 - pi);
        }
    } else {
        // Unnormalized log-likelihood
        logp = alpha * arma::log(prob) + y % arma::log(1 - prob);
    }
    return arma::sum(logp);
}

double lpostbeta_local(double alpha, vec const& beta, mat const& X, vec const& y,
                        vec const& betabar, mat const& rootA) {
    vec lambda = arma::exp(X * beta);
    double ll = llnegbin_local(y, lambda, alpha, false);
    
    // Unnormalized prior: beta ~ N(betabar, A^-1)
    vec z = rootA * (beta - betabar);
    double lprior = -0.5 * arma::sum(z % z);
    
    return ll + lprior;
}

double lpostalpha_local(double alpha, vec const& beta, mat const& X, vec const& y,
                         double a, double b) {
    vec lambda = arma::exp(X * beta);
    double ll = llnegbin_local(y, lambda, alpha, true);
    
    // Unnormalized prior: alpha ~ Gamma(a,b)
    double lprior = (a - 1) * std::log(alpha) - b * alpha;
    
    return ll + lprior;
}

} // anonymous namespace

std::tuple<mat, vec, int, int> rnegbinRw_rcpp_loop(
    vec const& y, mat const& X, vec const& betabar, mat const& rootA,
    double a, double b, vec beta, double alpha, bool fixalpha,
    mat const& betaroot, double alphacroot, int R, int keep) {

    int nvar = X.n_cols;
    int nacceptbeta = 0;
    int nacceptalpha = 0;

    vec alphadraw(R / keep);
    mat betadraw(R / keep, nvar);

    for (int rep = 0; rep < R; rep++) {
        // Draw beta
        vec rnorm_vec(nvar);
        for (int i = 0; i < nvar; i++) rnorm_vec[i] = R::rnorm(0, 1);
        vec betac = beta + betaroot * rnorm_vec;
        
        double oldlpostbeta = lpostbeta_local(alpha, beta, X, y, betabar, rootA);
        double clpostbeta = lpostbeta_local(alpha, betac, X, y, betabar, rootA);
        double ldiff = clpostbeta - oldlpostbeta;
        double acc = std::exp(ldiff);
        if (acc > 1) acc = 1;
        
        double unif = (acc < 1) ? R::runif(0, 1) : 0;
        if (unif <= acc) {
            beta = betac;
            nacceptbeta++;
        }

        // Draw alpha
        if (!fixalpha) {
            double logalphac = std::log(alpha) + alphacroot * R::rnorm(0, 1);
            double oldlpostalpha = lpostalpha_local(alpha, beta, X, y, a, b);
            double clpostalpha = lpostalpha_local(std::exp(logalphac), beta, X, y, a, b);
            ldiff = clpostalpha - oldlpostalpha;
            acc = std::exp(ldiff);
            if (acc > 1) acc = 1;
            
            unif = (acc < 1) ? R::runif(0, 1) : 0;
            if (unif <= acc) {
                alpha = std::exp(logalphac);
                nacceptalpha++;
            }
        }

        // Store draws
        if ((rep + 1) % keep == 0) {
            int mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = beta.t();
            alphadraw[mkeep - 1] = alpha;
        }
    }

    return std::make_tuple(betadraw, alphadraw, nacceptbeta, nacceptalpha);
}
