#include "bayesm.h"
#include <tuple>

vec breg(vec const& y, mat const& X, vec const& betabar, mat const& A);

struct rwishart_iv_result { mat W; mat IW; mat C; mat CI; };

rwishart_iv_result rwishart_iv(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    
    for (int i = 0; i < m; i++) {
        T(i, i) = sqrt(R::rchisq(nu - i));
    }
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            T(i, j) = R::rnorm(0, 1);
        }
    }
    
    mat C = chol(V);
    mat CI = solve(trimatu(C), eye(m, m));
    mat W = T * trans(T);
    W = trans(C) * W * C;
    mat IW = CI * trans(CI);
    IW = trans(T) * IW * T;
    IW = solve(trimatu(IW), eye(m, m));
    IW = IW * trans(IW);
    
    rwishart_iv_result out;
    out.W = W;
    out.IW = IW;
    out.C = C;
    out.CI = CI;
    return out;
}

std::tuple<mat, vec, mat, mat> rivGibbs_rcpp_loop(
    vec const& y, vec const& x, mat const& z, mat const& w,
    vec const& mbg, mat const& Abg, vec const& md, mat const& Ad,
    mat const& V, double nu, int R, int keep) {
    
    int n = y.size();
    int dimd = z.n_cols;
    int dimg = w.n_cols;
    
    mat deltadraw(R / keep, dimd);
    vec betadraw(R / keep);
    mat gammadraw(R / keep, dimg);
    mat Sigmadraw(R / keep, 4);
    mat C = eye(2, 2);
    
    mat Sigma = eye(2, 2);
    vec delta = 0.1 * ones<vec>(dimd);
    
    mat xtd(2 * n, dimd);
    vec zvec = vectorise(trans(z));
    
    for (int rep = 0; rep < R; rep++) {
        vec e1 = x - z * delta;
        vec ee2 = (Sigma(0, 1) / Sigma(0, 0)) * e1;
        double sig = sqrt(Sigma(1, 1) - (Sigma(0, 1) * Sigma(0, 1)) / Sigma(0, 0));
        vec yt = (y - ee2) / sig;
        mat xt = join_rows(x, w) / sig;
        vec bg = breg(yt, xt, mbg, Abg);
        double beta = bg[0];
        vec gamma = bg(span(1, bg.size() - 1));
        
        C(1, 0) = beta;
        mat B = C * Sigma * trans(C);
        mat L = trans(chol(B));
        mat Li = solve(trimatl(L), eye(2, 2));
        vec u = y - w * gamma;
        yt = vectorise(Li * trans(join_rows(x, u)));
        mat z2 = trans(join_rows(zvec, beta * zvec));
        z2 = Li * z2;
        rowvec zt1 = z2.row(0);
        rowvec zt2 = z2.row(1);
        mat zt1_mat = reshape(zt1, dimd, n).t();
        mat zt2_mat = reshape(zt2, dimd, n).t();
        for (int i = 0; i < n; i++) {
            xtd.row(2 * i) = zt1_mat.row(i);
            xtd.row(2 * i + 1) = zt2_mat.row(i);
        }
        delta = breg(yt, xtd, md, Ad);
        
        mat Res = join_rows(x - z * delta, y - beta * x - w * gamma);
        mat S = trans(Res) * Res;
        
        mat ucholinv = solve(trimatu(chol(V + S)), eye(2, 2));
        mat VSinv = ucholinv * trans(ucholinv);
        
        rwishart_iv_result rwout = rwishart_iv(nu + n, VSinv);
        Sigma = rwout.IW;
        
        if ((rep + 1) % keep == 0) {
            int mkeep = (rep + 1) / keep;
            deltadraw.row(mkeep - 1) = trans(delta);
            betadraw[mkeep - 1] = beta;
            gammadraw.row(mkeep - 1) = trans(gamma);
            Sigmadraw.row(mkeep - 1) = trans(vectorise(Sigma));
        }
    }
    
    return std::make_tuple(deltadraw, betadraw, gammadraw, Sigmadraw);
}
