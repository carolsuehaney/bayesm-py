// rmvpGibbs - Multivariate Probit Gibbs Sampler
#include "bayesm.h"
#include <tuple>

namespace {

vec condmom_mvp(vec const& x, vec const& mu, mat const& sigmai, int p, int j) {
    vec out(2);
    int jm1 = j - 1;
    int ind = p * jm1;
    
    double csigsq = 1.0 / sigmai(ind + jm1);
    double m = 0.0;
    
    for (int i = 0; i < p; i++) {
        if (i != jm1) m += -csigsq * sigmai(ind + i) * (x[i] - mu[i]);
    }
    
    out[0] = mu[jm1] + m;
    out[1] = std::sqrt(csigsq);
    return out;
}

double trunNorm_mvp(double mu, double sigma, double trunpt, int above) {
    double FA, FB, rnd, arg;
    if (above) {
        FA = 0.0;
        FB = R::pnorm((trunpt - mu) / sigma, 0, 1, 1, 0);
    } else {
        FB = 1.0;
        FA = R::pnorm((trunpt - mu) / sigma, 0, 1, 1, 0);
    }
    
    rnd = R::runif(0, 1);
    arg = rnd * (FB - FA) + FA;
    if (arg > 0.999999999) arg = 0.999999999;
    if (arg < 0.0000000001) arg = 0.0000000001;
    
    return mu + sigma * R::qnorm(arg, 0, 1, 1, 0);
}

vec drawwi_mvp(vec const& w, vec const& mu, mat const& sigmai, int p, ivec const& y) {
    vec outwi = w;
    for (int i = 0; i < p; i++) {
        int above = y[i] ? 0 : 1;
        vec CMout = condmom_mvp(outwi, mu, sigmai, p, i + 1);
        outwi[i] = trunNorm_mvp(CMout[0], CMout[1], 0.0, above);
    }
    return outwi;
}

vec draww_mvp(vec const& w, vec const& mu, mat const& sigmai, ivec const& y) {
    int p = sigmai.n_cols;
    int n = w.size() / p;
    vec outw = zeros<vec>(w.size());
    
    for (int i = 0; i < n; i++) {
        int ind = p * i;
        outw.subvec(ind, ind + p - 1) = drawwi_mvp(
            w.subvec(ind, ind + p - 1),
            mu.subvec(ind, ind + p - 1),
            sigmai, p,
            y.subvec(ind, ind + p - 1));
    }
    return outw;
}

vec breg_mvp(vec const& y, mat const& X, vec const& betabar, mat const& A) {
    int k = X.n_cols;
    mat XpX = X.t() * X;
    vec Xpy = X.t() * y;
    
    mat IR = solve(trimatu(chol(XpX + A)), eye<mat>(k, k));
    vec btilde = (IR * IR.t()) * (Xpy + A * betabar);
    
    vec rnorm_vec(k);
    for (int i = 0; i < k; i++) rnorm_vec[i] = R::rnorm(0, 1);
    
    return btilde + IR * rnorm_vec;
}

std::tuple<mat, mat, mat> rwishart_mvp(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    
    for (int i = 0; i < m; i++) {
        T(i, i) = std::sqrt(R::rchisq(nu - i));
    }
    for (int i = 1; i < m; i++) {
        for (int j = 0; j < i; j++) {
            T(i, j) = R::rnorm(0, 1);
        }
    }
    
    mat C = chol(V, "lower");
    mat CT = C * T;
    mat W = CT * CT.t();
    mat IW = solve(W, eye<mat>(m, m));
    mat Cupper = chol(W);  // Upper triangular root of W
    
    return std::make_tuple(W, IW, Cupper);
}

} // anonymous namespace

std::tuple<mat, mat> rmvpGibbs_rcpp_loop(int R, int keep, int p,
                                          ivec const& y, mat const& X, vec const& beta0, mat const& sigma0,
                                          mat const& V, double nu, vec const& betabar, mat const& A) {

    int n = y.size() / p;
    int k = X.n_cols;

    mat sigmadraw = zeros<mat>(R / keep, p * p);
    mat betadraw = zeros<mat>(R / keep, k);
    vec wnew = zeros<vec>(X.n_rows);
    vec wold = wnew;
    vec betaold = beta0;
    mat C = chol(solve(trimatu(sigma0), eye<mat>(sigma0.n_cols, sigma0.n_cols)));

    int mkeep = 0;

    for (int rep = 0; rep < R; rep++) {
        // Draw w given beta, sigma
        mat sigmai = C.t() * C;
        wnew = draww_mvp(wold, X * betaold, sigmai, y);

        // Draw beta given w and sigma
        mat zmat = join_rows(wnew, X);
        zmat.reshape(p, n * (k + 1));
        zmat = C * zmat;
        zmat.reshape(X.n_rows, k + 1);

        vec betanew = breg_mvp(zmat.col(0), zmat.cols(1, k), betabar, A);

        // Draw sigmai given w and beta
        vec epsilon_vec = wnew - X * betanew;
        mat eps_mat = reshape(epsilon_vec, p, n);
        mat S = eps_mat * eps_mat.t();

        mat ucholinv = solve(trimatu(chol(V + S)), eye<mat>(p, p));
        mat VSinv = ucholinv * ucholinv.t();

        auto [W, IW, Cnew] = rwishart_mvp(nu + n, VSinv);
        C = Cnew;

        // Save every keep-th draw
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = betanew.t();
            sigmadraw.row(mkeep - 1) = vectorise(IW).t();
        }

        wold = wnew;
        betaold = betanew;
    }

    return std::make_tuple(betadraw, sigmadraw);
}
