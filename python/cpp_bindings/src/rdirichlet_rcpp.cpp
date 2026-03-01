#include "bayesm.h"

vec rdirichlet(vec const& alpha){
    int k = alpha.n_elem;
    vec y(k);
    
    for(int i = 0; i < k; i++){
        std::gamma_distribution<double> d(alpha(i), 1.0);
        y(i) = d(R::rng());
    }
    
    return y / sum(y);
}
