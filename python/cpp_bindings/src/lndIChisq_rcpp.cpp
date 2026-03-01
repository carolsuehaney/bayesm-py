#include "bayesm.h"

double lndIChisq(double nu, double ssq, double X){
    double alpha = nu / 2.0;
    double beta = nu * ssq / 2.0;
    return alpha * log(beta) - lgamma(alpha) - (alpha + 1.0) * log(X) - beta / X;
}
