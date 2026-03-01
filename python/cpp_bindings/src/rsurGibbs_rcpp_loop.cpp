// rsurGibbs - Seemingly Unrelated Regression Gibbs Sampler
#include "bayesm.h"
#include <tuple>

namespace {

std::tuple<mat, mat> rwishart_sur(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    
    // Fill diagonal with sqrt of chi-squared draws
    for (int i = 0; i < m; i++) {
        T(i, i) = std::sqrt(R::rchisq(nu - i));
    }
    
    // Fill lower triangle with standard normals
    for (int i = 1; i < m; i++) {
        for (int j = 0; j < i; j++) {
            T(i, j) = R::rnorm(0, 1);
        }
    }
    
    // Compute C = chol(V)
    mat C = chol(V, "lower");
    mat CT = C * T;
    
    mat W = CT * CT.t();
    mat IW = solve(W, eye<mat>(m, m));
    
    return std::make_tuple(W, IW);
}

} // anonymous namespace

struct sur_moments {
    vec y;
    mat X;
};

std::tuple<mat, mat> rsurGibbs_rcpp_loop(
    std::vector<sur_moments> const& regdata_vector, vec const& indreg, vec const& cumnk,
    vec const& nk, mat const& XspXs, mat Sigmainv, mat const& A, vec const& Abetabar,
    double nu, mat const& V, int nvar, mat E, mat const& Y, int R, int keep) {

    int nreg = regdata_vector.size();
    int nobs = regdata_vector[0].y.size();
    int sum_nk = arma::sum(nk);

    mat XtipXti = zeros<mat>(sum_nk, sum_nk);
    mat Sigmadraw(R / keep, nreg * nreg);
    mat betadraw(R / keep, nvar);

    for (int rep = 0; rep < R; rep++) {
        // Draw beta | Sigma
        for (int i = 0; i < nreg; i++) {
            for (int j = 0; j < nreg; j++) {
                int row_start = (int)cumnk[i] - (int)nk[i];
                int row_end = (int)cumnk[i] - 1;
                int col_start = (int)cumnk[j] - (int)nk[j];
                int col_end = (int)cumnk[j] - 1;
                
                XtipXti(span(row_start, row_end), span(col_start, col_end)) =
                    Sigmainv(i, j) * XspXs(span(row_start, row_end), span(col_start, col_end));
            }
        }

        // Compute Xtilde'ytilde
        mat Ydti = Y * Sigmainv;
        vec Xtipyti = regdata_vector[0].X.t() * Ydti.col(0);
        for (int reg = 1; reg < nreg; reg++) {
            Xtipyti = join_cols(Xtipyti, regdata_vector[reg].X.t() * Ydti.col(reg));
        }

        mat IR = solve(trimatu(chol(XtipXti + A)), eye<mat>(nvar, nvar));
        vec btilde = (IR * IR.t()) * (Xtipyti + Abetabar);
        vec rnorm_vec(nvar);
        for (int i = 0; i < nvar; i++) rnorm_vec[i] = R::rnorm(0, 1);
        vec beta = btilde + IR * rnorm_vec;

        // Draw Sigma | beta
        for (int reg = 0; reg < nreg; reg++) {
            int start_idx = (int)indreg[reg] - 1;
            int end_idx = (int)indreg[reg + 1] - 2;
            E.col(reg) = regdata_vector[reg].y - regdata_vector[reg].X * beta(span(start_idx, end_idx));
        }

        mat ucholinv = solve(trimatu(chol(E.t() * E + V)), eye<mat>(nreg, nreg));
        mat EEVinv = ucholinv * ucholinv.t();

        auto [W, IW] = rwishart_sur(nu + nobs, EEVinv);
        mat Sigma = IW;
        Sigmainv = W;

        // Store draws
        if ((rep + 1) % keep == 0) {
            int mkeep = (rep + 1) / keep;
            betadraw.row(mkeep - 1) = beta.t();
            Sigmadraw.row(mkeep - 1) = vectorise(Sigma).t();
        }
    }

    return std::make_tuple(betadraw, Sigmadraw);
}
