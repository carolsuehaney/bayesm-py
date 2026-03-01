#include "bayesm.h"
#include <tuple>

double trunNormBelow(double a) {
    double z;
    if (a > 4) {
        int success = 0;
        while (success == 0) {
            double e = -log(R::runif(0, 1));
            double e1 = -log(R::runif(0, 1));
            if (pow(e, 2) <= 2.0 * e1 * pow(a, 2)) {
                z = a + e / a;
                success = 1;
            }
        }
    } else if (a <= -4) {
        int success = 0;
        while (success == 0) {
            double candz = R::rnorm(0, 1);
            if (candz >= a) {
                z = candz;
                success = 1;
            }
        }
    } else {
        double Phia = R::pnorm(a, 0.0, 1.0, 1, 0);
        double unif = R::runif(0, 1);
        double arg = unif * (1.0 - Phia) + Phia;
        z = R::qnorm(arg, 0.0, 1.0, 1, 0);
    }
    return z;
}

double trunNorm(double mu, double sig, double trunpt, int above) {
    double a, z, draw;
    if (above == 0) {
        a = (trunpt - mu) / sig;
        z = trunNormBelow(a);
        draw = sig * z + mu;
    } else {
        a = (mu - trunpt) / sig;
        z = trunNormBelow(a);
        draw = -sig * z + mu;
    }
    return draw;
}

vec trunNorm_vec(vec const& mu, vec const& sig, vec const& trunpt, vec const& above) {
    int nd = mu.size();
    vec rtn_vec(nd);
    for (int i = 0; i < nd; i++) {
        rtn_vec[i] = trunNorm(mu[i], sig[i], trunpt[i], (int)above[i]);
    }
    return rtn_vec;
}

vec breg1(mat const& root, mat const& X, vec const& y, vec const& Abetabar) {
    mat cov = trans(root) * root;
    int k = root.n_cols;
    vec rnorms(k);
    for (int i = 0; i < k; i++) rnorms[i] = R::rnorm(0, 1);
    return cov * (trans(X) * y + Abetabar) + trans(root) * rnorms;
}

mat rbprobitGibbs_rcpp_loop(vec const& y, mat const& X, vec const& Abetabar, mat const& root,
                            vec beta, vec const& sigma, vec const& trunpt, vec const& above,
                            int R, int keep) {
    int mkeep;
    vec mu, z;
    int nvar = X.n_cols;
    
    mat betadraw(R / keep, nvar);
    
    for (int rep = 0; rep < R; rep++) {
        mu = X * beta;
        z = trunNorm_vec(mu, sigma, trunpt, above);
        beta = breg1(root, X, z, Abetabar);
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = trans(beta);
        }
    }
    
    return betadraw;
}
