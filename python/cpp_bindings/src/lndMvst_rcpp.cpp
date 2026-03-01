#include "bayesm.h"

double lndMvst(vec const& x, double nu, vec const& mu, mat const& rooti, bool NORMC){
    int p = x.n_elem;
    vec z = trans(rooti) * (x - mu);
    double q = as_scalar(trans(z) * z);
    
    double lnorm = 0.0;
    if(NORMC){
        lnorm = lgamma((nu + p) / 2.0) - lgamma(nu / 2.0) - (p / 2.0) * log(nu * datum::pi) + sum(log(diagvec(rooti)));
    }
    
    return lnorm - ((nu + p) / 2.0) * log(1.0 + q / nu);
}
