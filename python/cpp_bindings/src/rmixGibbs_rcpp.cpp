// rmixGibbs - Mixture of Normals Gibbs Sampler (core iteration)
#include "bayesm.h"
#include <tuple>
#include <vector>

namespace {

// Forward declarations for rmultireg and rwishart from existing bindings
extern std::tuple<mat, mat> rmultireg_full(mat const& Y, mat const& X, mat const& Bbar,
                                            mat const& A, double nu, mat const& V);

std::tuple<mat, mat, mat, mat> rwishart_full(double nu, mat const& V) {
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
    mat Cupper = chol(W);
    mat CI = solve(trimatu(Cupper), eye<mat>(m, m));
    
    return std::make_tuple(W, IW, Cupper, CI);
}

vec rdirichlet_local(vec const& alpha) {
    int n = alpha.size();
    vec y(n);
    double sum_y = 0.0;
    
    for (int i = 0; i < n; i++) {
        y[i] = R::rgamma(alpha[i], 1.0);
        sum_y += y[i];
    }
    
    return y / sum_y;
}

std::tuple<mat, mat> rmultireg_local(mat const& Y, mat const& X, mat const& Bbar,
                                      mat const& A, double nu, mat const& V) {
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    
    mat XpX = X.t() * X;
    mat XpY = X.t() * Y;
    
    mat IR = solve(trimatu(chol(XpX + A)), eye<mat>(k, k));
    mat Btilde = (IR * IR.t()) * (XpY + A * Bbar);
    
    mat S = Y.t() * Y - XpY.t() * (IR * IR.t()) * XpY
            - Bbar.t() * A * Bbar + Btilde.t() * (XpX + A) * Btilde;
    
    mat ucholinv = solve(trimatu(chol(S + V)), eye<mat>(m, m));
    mat Vinv = ucholinv * ucholinv.t();
    
    auto [W, IW, C, CI] = rwishart_full(nu + n, Vinv);
    mat Sigma = IW;
    
    mat root_Sigma = chol(Sigma, "lower");
    mat rnorm_mat(k, m);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            rnorm_mat(i, j) = R::rnorm(0, 1);
        }
    }
    
    mat B = Btilde + IR * rnorm_mat * root_Sigma.t();
    
    return std::make_tuple(B, Sigma);
}

} // anonymous namespace

struct MixComp {
    vec mu;
    mat rooti;
};

std::tuple<vec, vec, std::vector<MixComp>> rmixGibbs(
    mat const& y, mat const& Bbar, mat const& A, double nu, mat const& V,
    vec const& a, vec const& p, vec const& z) {

    int n = z.n_rows;
    int ncomp = a.size();
    int dimy = y.n_cols;
    
    vec nobincomp = zeros<vec>(ncomp);
    for (int i = 0; i < n; i++) {
        nobincomp[(int)z[i] - 1]++;
    }
    
    // Draw components from labels
    std::vector<MixComp> comps(ncomp);
    
    for (int k = 0; k < ncomp; k++) {
        if (nobincomp[k] > 0) {
            // Get observations in this component
            uvec idx = find(z == (k + 1));
            mat yk = y.rows(idx);
            mat Xk = ones<mat>(nobincomp[k], 1);
            
            auto [B, Sigma] = rmultireg_local(yk, Xk, Bbar, A, nu, V);
            
            mat rooti = solve(trimatu(chol(Sigma)), eye<mat>(dimy, dimy));
            
            comps[k].mu = B.as_col();
            comps[k].rooti = rooti;
        } else {
            // Draw from prior
            mat S = solve(trimatu(chol(V)), eye<mat>(dimy, dimy));
            S = S * S.t();
            
            auto [W, IW, C, CI] = rwishart_full(nu, S);
            
            mat rooti = solve(trimatu(chol(IW)), eye<mat>(dimy, dimy));
            vec b = vectorise(Bbar);
            vec r(b.n_rows);
            for (size_t i = 0; i < b.n_rows; i++) r[i] = R::rnorm(0, 1);
            
            vec mu = b + (CI * r) / std::sqrt(A(0, 0));
            
            comps[k].mu = mu;
            comps[k].rooti = rooti;
        }
    }
    
    // Draw labels from components
    mat prob(n, ncomp);
    
    for (int k = 0; k < ncomp; k++) {
        vec mu = comps[k].mu;
        mat rooti = comps[k].rooti;
        
        double logprod = std::log(prod(diagvec(rooti)));
        mat zmat = y;
        zmat.each_row() -= mu.t();
        zmat = rooti.t() * zmat.t();
        
        vec logdens = -(dimy / 2.0) * std::log(2 * M_PI) + logprod 
                      - 0.5 * sum(zmat % zmat, 0).t();
        
        prob.col(k) = logdens;
    }
    
    prob = exp(prob);
    prob.each_row() %= p.t();
    
    // Cumulative sum and draw
    prob = cumsum(prob, 1);
    vec u(n);
    for (int i = 0; i < n; i++) {
        u[i] = R::runif(0, 1) * prob(i, ncomp - 1);
    }
    
    vec z2 = zeros<vec>(n);
    for (int i = 0; i < n; i++) {
        int k = 0;
        while (u[i] > prob(i, k)) k++;
        z2[i] = k + 1;
    }
    
    // Draw probabilities from labels
    vec a2 = a;
    for (int i = 0; i < n; i++) {
        a2[(int)z2[i] - 1]++;
    }
    vec p2 = rdirichlet_local(a2);
    
    return std::make_tuple(p2, z2, comps);
}

std::tuple<mat, mat, std::vector<std::vector<MixComp>>> rnmixGibbs_rcpp_loop(
    mat const& y, mat const& Mubar, mat const& A, double nu, mat const& V,
    vec const& a, vec p, vec z, int R, int keep) {

    int ncomp = a.size();
    int nobs = z.size();
    
    mat pdraw(R / keep, ncomp);
    mat zdraw(R / keep, nobs);
    std::vector<std::vector<MixComp>> compdraw(R / keep);
    
    int mkeep = 0;
    
    for (int rep = 0; rep < R; rep++) {
        auto [p_new, z_new, comps] = rmixGibbs(y, Mubar, A, nu, V, a, p, z);
        p = p_new;
        z = z_new;
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            pdraw.row(mkeep - 1) = p.t();
            zdraw.row(mkeep - 1) = z.t();
            compdraw[mkeep - 1] = comps;
        }
    }
    
    return std::make_tuple(pdraw, zdraw, compdraw);
}
