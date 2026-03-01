#include "bayesm.h"

double root(double c1, double c2, double tol, int iterlim) {
    int iter = 0;
    double uold = 0.1;
    double unew = 0.00001;
    
    while (iter <= iterlim && fabs(uold - unew) > tol) {
        uold = unew;
        unew = uold + (uold * (c1 - c2 * uold - log(uold))) / (1.0 + c2 * uold);
        if (unew < 1.0e-50) unew = 1.0e-50;
        iter++;
    }
    
    return unew;
}

vec callroot(vec const& c1, vec const& c2, double tol, int iterlim) {
    int n = c1.size();
    vec u = zeros<vec>(n);
    
    for (int i = 0; i < n; i++) {
        u[i] = root(c1[i], c2[i], tol, iterlim);
    }
    
    return u;
}
