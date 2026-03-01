#include "bayesm.h"

// Simplified rwishart - returns just the W matrix
mat rwishart(double nu, mat const& V){
    int p = V.n_rows;
    mat C = chol(V);
    mat T = zeros<mat>(p, p);
    
    // Fill diagonal with sqrt of chi-squared draws
    for(int i = 0; i < p; i++){
        std::chi_squared_distribution<double> d(nu - i);
        T(i,i) = sqrt(d(R::rng()));
    }
    
    // Fill lower triangle with standard normals
    for(int i = 1; i < p; i++){
        for(int j = 0; j < i; j++){
            T(i,j) = R::rnorm(0, 1);
        }
    }
    
    C = T * C;
    return trans(C) * C;
}
