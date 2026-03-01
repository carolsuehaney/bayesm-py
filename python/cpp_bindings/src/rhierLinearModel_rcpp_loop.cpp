#include "bayesm.h"
#include <tuple>
#include <vector>

struct unireg {
    vec beta;
    double sigmasq;
};

struct moments {
    vec y;
    mat X;
    mat XpX;
    vec Xpy;
};

struct rwishart_out {
    mat W;
    mat IW;
    mat C;
    mat CI;
};

rwishart_out rwishart_full(double nu, mat const& V) {
    int m = V.n_cols;
    mat C = chol(V);
    mat CI = solve(trimatu(C), eye(m, m));
    
    vec z(m);
    mat W = zeros(m, m);
    
    for (int i = 0; i < m; i++) {
        std::chi_squared_distribution<double> chisq_dist(nu - i);
        W(i, i) = sqrt(chisq_dist(R::rng()));
        for (int j = 0; j < i; j++) {
            W(j, i) = R::rnorm(0, 1);
        }
    }
    
    mat Wmat = trans(W) * W;
    mat IW = CI * Wmat * trans(CI);
    Wmat = C * Wmat * trans(C);
    
    rwishart_out out;
    out.W = Wmat;
    out.IW = IW;
    out.C = C;
    out.CI = CI;
    return out;
}

std::tuple<mat, mat> rmultireg_full(mat const& Y, mat const& X, mat const& Bbar, 
                                     mat const& A, double nu, mat const& V) {
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    
    mat RA = chol(A);
    mat W = join_cols(X, RA);
    mat Z = join_cols(Y, RA * Bbar);
    mat IR = solve(trimatu(chol(trans(W) * W)), eye(k, k));
    mat Btilde = (IR * trans(IR)) * (trans(W) * Z);
    mat E = Z - W * Btilde;
    mat S = trans(E) * E;
    
    mat ucholinv = solve(trimatu(chol(V + S)), eye(m, m));
    mat VSinv = ucholinv * trans(ucholinv);
    
    rwishart_out rwout = rwishart_full(nu + n, VSinv);
    
    mat draw(k, m);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            draw(i, j) = R::rnorm(0, 1);
        }
    }
    mat B = Btilde + IR * draw * trans(rwout.CI);
    
    return std::make_tuple(B, rwout.IW);
}

unireg runiregG(vec const& y, mat const& X, mat const& XpX, vec const& Xpy, 
                double sigmasq, mat const& Abeta, vec const& Abetabar, 
                double nu, double ssq) {
    int n = y.size();
    int k = XpX.n_cols;
    
    mat IR = solve(trimatu(chol(XpX / sigmasq + Abeta)), eye(k, k));
    vec btilde = (IR * trans(IR)) * (Xpy / sigmasq + Abetabar);
    
    vec znorm(k);
    for (int i = 0; i < k; i++) {
        znorm(i) = R::rnorm(0, 1);
    }
    vec beta = btilde + IR * znorm;
    
    double s = sum(square(y - X * beta));
    std::chi_squared_distribution<double> chisq_dist(nu + n);
    sigmasq = (s + nu * ssq) / chisq_dist(R::rng());
    
    unireg out;
    out.beta = beta;
    out.sigmasq = sigmasq;
    return out;
}

std::tuple<cube, mat, mat, mat> rhierLinearModel_rcpp_loop(
    std::vector<moments> const& regdata_vector, mat const& Z, 
    mat const& Deltabar, mat const& A, double nu, mat const& V, 
    double nu_e, vec const& ssq, vec tau, mat Delta, mat Vbeta, 
    int R, int keep) {

    int reg, mkeep;
    mat Abeta, betabar, ucholinv, Abetabar;
    unireg regout_struct;
    
    int nreg = regdata_vector.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;
    
    mat betas(nreg, nvar);
    mat Vbetadraw(R/keep, nvar * nvar);
    mat Deltadraw(R/keep, nz * nvar);
    mat taudraw(R/keep, nreg);
    cube betadraw(nreg, nvar, R/keep);

    for (int rep = 0; rep < R; rep++) {
        ucholinv = solve(trimatu(chol(Vbeta)), eye(nvar, nvar));
        Abeta = ucholinv * trans(ucholinv);
        
        betabar = Z * Delta;
        Abetabar = Abeta * trans(betabar);
        
        for (reg = 0; reg < nreg; reg++) {
            regout_struct = runiregG(
                regdata_vector[reg].y, regdata_vector[reg].X,
                regdata_vector[reg].XpX, regdata_vector[reg].Xpy,
                tau[reg], Abeta, Abetabar.col(reg), nu_e, ssq[reg]);
            betas.row(reg) = trans(regout_struct.beta);
            tau[reg] = regout_struct.sigmasq;
        }
        
        auto rmreg_result = rmultireg_full(betas, Z, Deltabar, A, nu, V);
        Delta = std::get<0>(rmreg_result);
        Vbeta = std::get<1>(rmreg_result);
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            Vbetadraw.row(mkeep - 1) = trans(vectorise(Vbeta));
            Deltadraw.row(mkeep - 1) = trans(vectorise(Delta));
            taudraw.row(mkeep - 1) = trans(tau);
            betadraw.slice(mkeep - 1) = betas;
        }
    }
    
    return std::make_tuple(betadraw, taudraw, Deltadraw, Vbetadraw);
}
