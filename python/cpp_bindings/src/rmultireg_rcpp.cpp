#include "bayesm.h"

mat rmultireg(mat const& Y, mat const& X, mat const& Bbar, mat const& A, double nu, mat const& V){
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    
    mat RA = chol(A);
    mat W = join_cols(X, RA);
    mat Z = join_cols(Y, RA * Bbar);
    mat IR = inv(trans(W) * W);
    mat Btilde = IR * (trans(W) * Z);
    
    mat S = trans(Z - W * Btilde) * (Z - W * Btilde);
    mat IW = inv(S + V);
    
    // Draw from Wishart
    mat C = chol(IW);
    mat T = zeros<mat>(m, m);
    for(int i = 0; i < m; i++){
        std::chi_squared_distribution<double> d(nu + n - i);
        T(i,i) = sqrt(d(R::rng()));
    }
    for(int i = 1; i < m; i++){
        for(int j = 0; j < i; j++){
            T(i,j) = R::rnorm(0, 1);
        }
    }
    mat CI = T * C;
    mat Sigma = inv(trans(CI) * CI);
    
    // Draw beta
    mat C2 = chol(IR);
    mat CSig = chol(Sigma);
    mat Z2(k, m);
    for(int i = 0; i < k; i++){
        for(int j = 0; j < m; j++){
            Z2(i,j) = R::rnorm(0, 1);
        }
    }
    
    return Btilde + trans(C2) * Z2 * CSig;
}
