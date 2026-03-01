#include "bayesm.h"

double llmnl(vec const& beta, vec const& y, mat const& X){
    int n = y.n_elem;
    int j = X.n_rows / n;
    int k = X.n_cols;
    
    vec Xbeta = X * beta;
    double ll = 0.0;
    
    for(int i = 0; i < n; i++){
        double maxv = Xbeta(i * j);
        for(int jj = 1; jj < j; jj++){
            if(Xbeta(i * j + jj) > maxv) maxv = Xbeta(i * j + jj);
        }
        
        double denom = 0.0;
        for(int jj = 0; jj < j; jj++){
            denom += exp(Xbeta(i * j + jj) - maxv);
        }
        
        int yi = (int)y(i);
        ll += Xbeta(i * j + yi) - maxv - log(denom);
    }
    
    return ll;
}
