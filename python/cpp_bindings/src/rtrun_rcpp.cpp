#include "bayesm.h"

double rtrun(double mu, double sigma, double a, double b){
    double pa = R::pnorm(a, mu, sigma, 1, 0);
    double pb = R::pnorm(b, mu, sigma, 1, 0);
    double u = R::runif(pa, pb);
    return R::qnorm(u, mu, sigma, 1, 0);
}
