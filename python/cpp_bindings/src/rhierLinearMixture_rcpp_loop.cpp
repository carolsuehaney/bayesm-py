// rhierLinearMixture - Hierarchical Linear Model with Mixture of Normals
#include "bayesm.h"
#include <tuple>
#include <vector>

struct hier_moments { vec y; mat X; mat XpX; vec Xpy; };
struct MixComp_hier { vec mu; mat rooti; };

namespace {

struct unireg_result { vec beta; double sigmasq; };

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

std::tuple<vec, vec, std::vector<MixComp_hier>> rmixGibbs_local(
    mat const& y, mat const& Bbar, mat const& A, double nu, mat const& V,
    vec const& a, vec const& p, vec const& z) {

    int n = z.n_rows;
    int ncomp = a.size();
    int dimy = y.n_cols;
    
    vec nobincomp = zeros<vec>(ncomp);
    for (int i = 0; i < n; i++) nobincomp[(int)z[i] - 1]++;
    
    std::vector<MixComp_hier> comps(ncomp);
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
                    std::vector<MixComp_hier> const& comps, vec const& deltabar, mat const& Ad) {
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

unireg_result runiregG_local(vec const& y, mat const& X, mat const& XpX, vec const& Xpy,
                              double sigmasq, mat const& A, vec const& Abetabar, double nu, double ssq) {
    int n = y.size();
    int k = XpX.n_cols;
    
    mat IR = solve(trimatu(chol(XpX / sigmasq + A)), eye(k, k));
    vec btilde = (IR * IR.t()) * (Xpy / sigmasq + Abetabar);
    vec rnorm_vec(k);
    for (int i = 0; i < k; i++) rnorm_vec[i] = R::rnorm(0, 1);
    vec beta = btilde + IR * rnorm_vec;
    
    double s = sum(square(y - X * beta));
    sigmasq = (s + nu * ssq) / R::rchisq(nu + n);
    
    unireg_result out;
    out.beta = beta;
    out.sigmasq = sigmasq;
    return out;
}

} // anonymous namespace

std::tuple<mat, mat, arma::cube, mat, std::vector<std::vector<MixComp_hier>>>
rhierLinearMixture_rcpp_loop(
    std::vector<hier_moments> const& regdata_vector, mat const& Z,
    vec const& deltabar, mat const& Ad, mat const& mubar, mat const& Amu,
    double nu, mat const& V, double nu_e, vec const& ssq,
    int R, int keep, bool drawdelta,
    vec olddelta, vec const& a, vec oldprob, vec ind, vec tau) {

    int nreg = regdata_vector.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;
    int ncomp = a.size();
    
    mat oldbetas = zeros<mat>(nreg, nvar);
    mat taudraw(R / keep, nreg);
    arma::cube betadraw(nreg, nvar, R / keep);
    mat probdraw(R / keep, ncomp);
    mat Deltadraw;
    if (drawdelta) Deltadraw = zeros<mat>(R / keep, nz * nvar);
    std::vector<std::vector<MixComp_hier>> compdraw(R / keep);
    
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
        
        for (int reg = 0; reg < nreg; reg++) {
            int comp_idx = (int)ind[reg] - 1;
            mat rootpi = oldcomp[comp_idx].rooti;
            vec betabar;
            
            if (drawdelta) {
                betabar = oldcomp[comp_idx].mu + olddelta_mat * Z.row(reg).t();
            } else {
                betabar = oldcomp[comp_idx].mu;
            }
            
            mat Abeta = rootpi.t() * rootpi;
            vec Abetabar = Abeta * betabar;
            
            auto res = runiregG_local(regdata_vector[reg].y, regdata_vector[reg].X,
                                       regdata_vector[reg].XpX, regdata_vector[reg].Xpy,
                                       tau[reg], Abeta, Abetabar, nu_e, ssq[reg]);
            
            oldbetas.row(reg) = res.beta.t();
            tau[reg] = res.sigmasq;
        }
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            taudraw.row(mkeep - 1) = tau.t();
            betadraw.slice(mkeep - 1) = oldbetas;
            probdraw.row(mkeep - 1) = oldprob.t();
            if (drawdelta) Deltadraw.row(mkeep - 1) = olddelta.t();
            compdraw[mkeep - 1] = oldcomp;
        }
    }
    
    return std::make_tuple(taudraw, Deltadraw, betadraw, probdraw, compdraw);
}
