// rhierMnlRwMixture - Hierarchical MNL with Mixture of Normals
#include "bayesm.h"
#include <tuple>
#include <vector>

struct mnl_moments { vec y; mat X; mat hess; };
struct MixComp_mnl { vec mu; mat rooti; };

namespace {

double llmnl_con_local(vec const& betastar, vec const& y, mat const& X, vec const& SignRes) {
    vec beta = betastar;
    uvec signInd = find(SignRes != 0);
    if (signInd.n_elem > 0) {
        beta.elem(signInd) = SignRes.elem(signInd) % exp(beta.elem(signInd));
    }
    
    int n = y.size();
    int j = X.n_rows / n;
    vec Xbeta = X * beta;
    
    vec xby = zeros<vec>(n);
    vec denom = zeros<vec>(n);
    
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < j; p++) denom[i] += exp(Xbeta[i * j + p]);
        xby[i] = Xbeta[i * j + (int)y[i] - 1];
    }
    
    return sum(xby - log(denom));
}

double lndMvn_local(vec const& x, vec const& mu, mat const& rooti) {
    vec z = rooti * (x - mu);
    double val = -(x.n_elem / 2.0) * std::log(2 * M_PI) + sum(log(diagvec(rooti))) - 0.5 * dot(z, z);
    return val;
}

struct mnlMetropOnceOut { vec betadraw; int stay; double oldll; };

mnlMetropOnceOut mnlMetropOnce_con_local(vec const& y, mat const& X, vec const& oldbeta,
                                          double oldll, double s, mat const& incroot,
                                          vec const& betabar, mat const& rootpi, vec const& SignRes) {
    mnlMetropOnceOut out;
    int k = X.n_cols;
    
    vec rnorm_vec(k);
    for (int i = 0; i < k; i++) rnorm_vec[i] = R::rnorm(0, 1);
    vec betac = oldbeta + s * incroot.t() * rnorm_vec;
    
    double cll = llmnl_con_local(betac, y, X, SignRes);
    double clpost = cll + lndMvn_local(betac, betabar, rootpi);
    double ldiff = clpost - oldll - lndMvn_local(oldbeta, betabar, rootpi);
    double alpha = std::min(1.0, exp(ldiff));
    
    double unif = (alpha < 1) ? R::runif(0, 1) : 0.0;
    
    if (unif <= alpha) {
        out.betadraw = betac;
        out.oldll = cll;
        out.stay = 0;
    } else {
        out.betadraw = oldbeta;
        out.oldll = oldll;
        out.stay = 1;
    }
    
    return out;
}

std::tuple<mat, mat, mat, mat> rwishart_full(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    for (int i = 0; i < m; i++) T(i, i) = std::sqrt(R::rchisq(nu - i));
    for (int i = 1; i < m; i++)
        for (int j = 0; j < i; j++) T(i, j) = R::rnorm(0, 1);
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
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++) rnorm_mat(i, j) = R::rnorm(0, 1);
    mat B = Btilde + IR * rnorm_mat * root_Sigma.t();
    return std::make_tuple(B, Sigma);
}

std::tuple<vec, vec, std::vector<MixComp_mnl>> rmixGibbs_local(
    mat const& y, mat const& Bbar, mat const& A, double nu, mat const& V,
    vec const& a, vec const& p, vec const& z) {

    int n = z.n_rows;
    int ncomp = a.size();
    int dimy = y.n_cols;
    
    vec nobincomp = zeros<vec>(ncomp);
    for (int i = 0; i < n; i++) nobincomp[(int)z[i] - 1]++;
    
    std::vector<MixComp_mnl> comps(ncomp);
    for (int k = 0; k < ncomp; k++) {
        if (nobincomp[k] > 0) {
            uvec idx = find(z == (k + 1));
            mat yk = y.rows(idx);
            mat Xk = ones<mat>(nobincomp[k], 1);
            auto [B, Sigma] = rmultireg_local(yk, Xk, Bbar, A, nu, V);
            mat rooti = solve(trimatu(chol(Sigma)), eye<mat>(dimy, dimy));
            comps[k].mu = B.as_col();
            comps[k].rooti = rooti;
        } else {
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
    prob = cumsum(prob, 1);
    
    vec u(n);
    for (int i = 0; i < n; i++) u[i] = R::runif(0, 1) * prob(i, ncomp - 1);
    vec z2 = zeros<vec>(n);
    for (int i = 0; i < n; i++) {
        int k = 0;
        while (u[i] > prob(i, k)) k++;
        z2[i] = k + 1;
    }
    
    vec a2 = a;
    for (int i = 0; i < n; i++) a2[(int)z2[i] - 1]++;
    vec p2 = rdirichlet_local(a2);
    
    return std::make_tuple(p2, z2, comps);
}

mat drawDelta_local(mat const& x, mat const& y, vec const& z,
                    std::vector<MixComp_mnl> const& comps, vec const& deltabar, mat const& Ad) {
    int p = y.n_cols;
    int k = x.n_cols;
    int ncomp = comps.size();
    mat xtx = zeros<mat>(k * p, k * p);
    mat xty = zeros<mat>(p, k);
    
    for (int compi = 0; compi < ncomp; compi++) {
        uvec ind = find(z == (compi + 1));
        if (ind.size() > 0) {
            mat yi = y.rows(ind);
            mat xi = x.rows(ind);
            rowvec mui = comps[compi].mu.t();
            mat rootii = trimatu(comps[compi].rooti);
            yi.each_row() -= mui;
            mat sigi = rootii * rootii.t();
            xtx = xtx + kron(xi.t() * xi, sigi);
            xty = xty + (sigi * (yi.t() * xi));
        }
    }
    xty.reshape(xty.n_rows * xty.n_cols, 1);
    
    mat ucholinv = solve(trimatu(chol(xtx + Ad)), eye(k * p, k * p));
    mat Vinv = ucholinv * ucholinv.t();
    
    vec rnorm_vec(deltabar.size());
    for (size_t i = 0; i < deltabar.size(); i++) rnorm_vec[i] = R::rnorm(0, 1);
    
    return Vinv * (xty + Ad * deltabar) + chol(Vinv).t() * rnorm_vec;
}

} // anonymous namespace

std::tuple<mat, arma::cube, mat, vec, std::vector<std::vector<MixComp_mnl>>>
rhierMnlRwMixture_rcpp_loop(
    std::vector<mnl_moments> const& lgtdata_vector, mat const& Z,
    vec const& deltabar, mat const& Ad, mat const& mubar, mat const& Amu,
    double nu, mat const& V, double s,
    int R, int keep, bool drawdelta,
    vec olddelta, vec const& a, vec oldprob, mat oldbetas, vec ind, vec const& SignRes) {

    int nlgt = lgtdata_vector.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;
    int ncomp = a.size();
    
    vec oldll = zeros<vec>(nlgt);
    arma::cube betadraw(nlgt, nvar, R / keep);
    mat probdraw(R / keep, ncomp);
    vec loglike(R / keep);
    mat Deltadraw;
    if (drawdelta) Deltadraw = zeros<mat>(R / keep, nz * nvar);
    std::vector<std::vector<MixComp_mnl>> compdraw(R / keep);
    
    mat olddelta_mat;
    
    int mkeep = 0;
    for (int rep = 0; rep < R; rep++) {
        mat y_mix;
        if (drawdelta) {
            olddelta_mat = reshape(olddelta, nvar, nz);
            y_mix = oldbetas - Z * olddelta_mat.t();
        } else {
            y_mix = oldbetas;
        }
        
        auto [p_new, z_new, oldcomp] = rmixGibbs_local(y_mix, mubar, Amu, nu, V, a, oldprob, ind);
        oldprob = p_new;
        ind = z_new;
        
        if (drawdelta) {
            olddelta = drawDelta_local(Z, oldbetas, ind, oldcomp, deltabar, Ad);
            olddelta_mat = reshape(olddelta, nvar, nz);
        }
        
        for (int lgt = 0; lgt < nlgt; lgt++) {
            int comp_idx = (int)ind[lgt] - 1;
            mat rootpi = oldcomp[comp_idx].rooti;
            vec betabar;
            
            if (drawdelta) {
                betabar = oldcomp[comp_idx].mu + olddelta_mat * Z.row(lgt).t();
            } else {
                betabar = oldcomp[comp_idx].mu;
            }
            
            if (rep == 0) {
                oldll[lgt] = llmnl_con_local(oldbetas.row(lgt).t(), lgtdata_vector[lgt].y,
                                              lgtdata_vector[lgt].X, SignRes);
            }
            
            mat ucholinv = solve(trimatu(chol(lgtdata_vector[lgt].hess + rootpi * rootpi.t())),
                                 eye(nvar, nvar));
            mat incroot = chol(ucholinv * ucholinv.t());
            
            auto metropout = mnlMetropOnce_con_local(lgtdata_vector[lgt].y, lgtdata_vector[lgt].X,
                                                      oldbetas.row(lgt).t(), oldll[lgt], s,
                                                      incroot, betabar, rootpi, SignRes);
            
            oldbetas.row(lgt) = metropout.betadraw.t();
            oldll[lgt] = metropout.oldll;
        }
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.slice(mkeep - 1) = oldbetas;
            probdraw.row(mkeep - 1) = oldprob.t();
            loglike[mkeep - 1] = sum(oldll);
            if (drawdelta) Deltadraw.row(mkeep - 1) = olddelta.t();
            compdraw[mkeep - 1] = oldcomp;
        }
    }
    
    // Apply sign constraints to betadraw
    bool conStatus = any(SignRes != 0);
    if (conStatus) {
        for (size_t i = 0; i < SignRes.n_elem; i++) {
            if (SignRes[i] != 0) {
                for (int s = 0; s < R / keep; s++) {
                    for (int lgt = 0; lgt < nlgt; lgt++) {
                        betadraw(lgt, i, s) = SignRes[i] * exp(betadraw(lgt, i, s));
                    }
                }
            }
        }
    }
    
    return std::make_tuple(Deltadraw, betadraw, probdraw, loglike, compdraw);
}
