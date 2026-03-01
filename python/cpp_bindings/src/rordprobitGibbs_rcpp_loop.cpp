#include "bayesm.h"
#include <tuple>

double lndMvn(vec const& x, vec const& mu, mat const& rooti);

double trunNormBelow_ord(double a) {
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

vec rtrunVec_ord(vec const& mu, vec const& sigma, vec const& a, vec const& b) {
    int n = mu.size();
    vec out(n);
    for (int i = 0; i < n; i++) {
        double trunpt_a = (a[i] - mu[i]) / sigma[i];
        double trunpt_b = (b[i] - mu[i]) / sigma[i];
        double Phi_a = R::pnorm(trunpt_a, 0.0, 1.0, 1, 0);
        double Phi_b = R::pnorm(trunpt_b, 0.0, 1.0, 1, 0);
        double u = R::runif(0, 1);
        double arg = Phi_a + u * (Phi_b - Phi_a);
        double z = R::qnorm(arg, 0.0, 1.0, 1, 0);
        out[i] = mu[i] + sigma[i] * z;
    }
    return out;
}

vec breg1_ord(mat const& root, mat const& X, vec const& y, vec const& Abetabar) {
    mat cov = trans(root) * root;
    int k = root.n_cols;
    vec rnorms(k);
    for (int i = 0; i < k; i++) rnorms[i] = R::rnorm(0, 1);
    return cov * (trans(X) * y + Abetabar) + trans(root) * rnorms;
}

vec dstartoc_ord(vec const& dstar) {
    int ndstar = dstar.size();
    vec c(ndstar + 3);
    c[0] = -100;
    c[1] = 0;
    vec exp_dstar = exp(dstar);
    double cumsum_val = 0;
    for (int i = 0; i < ndstar; i++) {
        cumsum_val += exp_dstar[i];
        c[i + 2] = cumsum_val;
    }
    c[ndstar + 2] = 100;
    return c;
}

double lldstar_ord(vec const& dstar, vec const& y, vec const& mu) {
    vec gamma = dstartoc_ord(dstar);
    int ny = y.size();
    vec arg(ny);
    for (int i = 0; i < ny; i++) {
        int yi = (int)y[i];
        double g1 = gamma[yi] - mu[i];
        double g2 = gamma[yi - 1] - mu[i];
        arg[i] = R::pnorm(g1, 0.0, 1.0, 1, 0) - R::pnorm(g2, 0.0, 1.0, 1, 0);
    }
    double epsilon = 1e-50;
    for (int j = 0; j < ny; j++) {
        if (arg[j] < epsilon) arg[j] = epsilon;
    }
    return sum(log(arg));
}

struct dstarMetropOut { vec dstardraw; double oldll; int stay; };

dstarMetropOut dstarRwMetrop_ord(vec const& y, vec const& mu, vec const& olddstar, double s, 
                                  mat const& inc_root, vec const& dstarbar, double oldll, 
                                  mat const& rootdi, int ncut) {
    dstarMetropOut out;
    int stay = 0;
    
    vec rnorms(ncut);
    for (int i = 0; i < ncut; i++) rnorms[i] = R::rnorm(0, 1);
    
    vec dstarc = olddstar + s * trans(inc_root) * rnorms;
    double cll = lldstar_ord(dstarc, y, mu);
    double clpost = cll + lndMvn(dstarc, dstarbar, rootdi);
    double ldiff = clpost - oldll - lndMvn(olddstar, dstarbar, rootdi);
    double alpha = std::min(1.0, exp(ldiff));
    
    double unif = (alpha < 1.0) ? R::runif(0, 1) : 0.0;
    
    if (unif <= alpha) {
        out.dstardraw = dstarc;
        out.oldll = cll;
    } else {
        out.dstardraw = olddstar;
        out.oldll = oldll;
        stay = 1;
    }
    out.stay = stay;
    return out;
}

std::tuple<mat, mat, mat, double> rordprobitGibbs_rcpp_loop(
    vec const& y, mat const& X, int k, mat const& A, vec const& betabar, mat const& Ad,
    double s, mat const& inc_root, vec const& dstarbar, vec const& betahat, int R, int keep) {
    
    int nvar = X.n_cols;
    int ncuts = k + 1;
    int ncut = ncuts - 3;
    int ndstar = k - 2;
    int ny = y.size();
    
    mat betadraw(R / keep, nvar);
    mat cutdraw(R / keep, ncuts);
    mat dstardraw(R / keep, ndstar);
    vec staydraw(R / keep);
    vec cutoff1(ny);
    vec cutoff2(ny);
    vec sigma = ones<vec>(X.n_rows);
    
    mat ucholinv = solve(trimatu(chol(trans(X) * X + A)), eye(nvar, nvar));
    mat XXAinv = ucholinv * trans(ucholinv);
    mat root = chol(XXAinv);
    vec Abetabar = trans(A) * betabar;
    
    ucholinv = solve(trimatu(chol(Ad)), eye(ndstar, ndstar));
    mat Adinv = ucholinv * trans(ucholinv);
    mat rootdi = chol(Adinv);
    
    vec olddstar = zeros<vec>(ndstar);
    vec beta = betahat;
    vec cutoffs = dstartoc_ord(olddstar);
    double oldll = lldstar_ord(olddstar, y, X * betahat);
    
    for (int rep = 0; rep < R; rep++) {
        for (int i = 0; i < ny; i++) {
            int yi = (int)y[i];
            cutoff1[i] = cutoffs[yi - 1];
            cutoff2[i] = cutoffs[yi];
        }
        vec z = rtrunVec_ord(X * beta, sigma, cutoff1, cutoff2);
        
        beta = breg1_ord(root, X, z, Abetabar);
        
        dstarMetropOut metropout = dstarRwMetrop_ord(y, X * beta, olddstar, s, inc_root, dstarbar, oldll, rootdi, ncut);
        olddstar = metropout.dstardraw;
        oldll = metropout.oldll;
        cutoffs = dstartoc_ord(olddstar);
        int stay = metropout.stay;
        
        if ((rep + 1) % keep == 0) {
            int mkeep = (rep + 1) / keep;
            cutdraw.row(mkeep - 1) = trans(cutoffs);
            dstardraw.row(mkeep - 1) = trans(olddstar);
            betadraw.row(mkeep - 1) = trans(beta);
            staydraw[mkeep - 1] = stay;
        }
    }
    
    double accept = 1.0 - sum(staydraw) / (R / keep);
    
    return std::make_tuple(betadraw, cutdraw, dstardraw, accept);
}
