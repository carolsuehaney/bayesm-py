#include "bayesm.h"

vec rmvst(double nu, vec const& mu, mat const& root){
    int p = mu.n_elem;
    vec z(p);
    for(int i = 0; i < p; i++){
        z(i) = R::rnorm(0, 1);
    }
    
    std::chi_squared_distribution<double> d(nu);
    double chi = d(R::rng());
    
    return mu + sqrt(nu / chi) * (root * z);
}
