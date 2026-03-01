#include "bayesm.h"

double lndIWishart(double nu, mat const& V, mat const& IW){
    int p = V.n_rows;
    mat cholV = chol(V);
    mat cholIW = chol(IW);
    
    double lndetV = 2.0 * sum(log(diagvec(cholV)));
    double lndetIW = 2.0 * sum(log(diagvec(cholIW)));
    
    double ldenom = (nu * p / 2.0) * log(2.0) + (p * (p - 1.0) / 4.0) * log(datum::pi);
    for(int i = 0; i < p; i++){
        ldenom += lgamma((nu - i) / 2.0);
    }
    
    double lnum = (nu / 2.0) * lndetV - ((nu + p + 1.0) / 2.0) * lndetIW - 0.5 * trace(V * inv(IW));
    
    return lnum - ldenom;
}
