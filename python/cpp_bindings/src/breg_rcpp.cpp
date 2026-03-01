#include "bayesm.h"

vec breg(vec const& y, mat const& X, vec const& betabar, mat const& A){
    mat XpX = trans(X) * X;
    vec Xpy = trans(X) * y;
    mat IR = inv(XpX + A);
    vec btilde = IR * (Xpy + A * betabar);
    
    mat C = chol(IR);
    int k = btilde.n_elem;
    vec z(k);
    for(int i = 0; i < k; i++){
        z(i) = R::rnorm(0, 1);
    }
    
    return btilde + trans(C) * z;
}
