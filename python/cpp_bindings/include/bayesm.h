#ifndef __BAYESM_H__
#define __BAYESM_H__

#include <armadillo>
#include <cmath>
#include <random>

using namespace arma;

// R compatibility
#ifndef TRUE
#define TRUE true
#endif
#ifndef FALSE
#define FALSE false
#endif

// Stub for IntegerVector::create
struct IntegerVector {
    static vec create(int val) { return vec({(double)val}); }
};

// Replace Rcpp random functions with standard C++
namespace R {
    inline std::mt19937& rng() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }
    
    inline double rnorm(double mu, double sigma) {
        std::normal_distribution<double> d(mu, sigma);
        return d(rng());
    }
    
    inline double runif(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(rng());
    }
    
    inline double rgamma(double shape, double scale) {
        std::gamma_distribution<double> d(shape, scale);
        return d(rng());
    }
    
    inline double rchisq(double df) {
        std::chi_squared_distribution<double> d(df);
        return d(rng());
    }
    
    inline double pnorm(double x, double mu, double sigma, int lower, int log_p) {
        double z = (x - mu) / sigma;
        double p = 0.5 * (1 + std::erf(z / std::sqrt(2.0)));
        if (!lower) p = 1 - p;
        if (log_p) p = std::log(p);
        return p;
    }
    
    inline double qnorm(double p, double mu, double sigma, int lower, int log_p) {
        if (log_p) p = std::exp(p);
        if (!lower) p = 1 - p;
        // Approximation using inverse error function
        auto erfinv = [](double x) {
            double w = -std::log((1 - x) * (1 + x));
            double p;
            if (w < 5.0) {
                w = w - 2.5;
                p = 2.81022636e-08;
                p = 3.43273939e-07 + p * w;
                p = -3.5233877e-06 + p * w;
                p = -4.39150654e-06 + p * w;
                p = 0.00021858087 + p * w;
                p = -0.00125372503 + p * w;
                p = -0.00417768164 + p * w;
                p = 0.246640727 + p * w;
                p = 1.50140941 + p * w;
            } else {
                w = std::sqrt(w) - 3.0;
                p = -0.000200214257;
                p = 0.000100950558 + p * w;
                p = 0.00134934322 + p * w;
                p = -0.00367342844 + p * w;
                p = 0.00573950773 + p * w;
                p = -0.0076224613 + p * w;
                p = 0.00943887047 + p * w;
                p = 1.00167406 + p * w;
                p = 2.83297682 + p * w;
            }
            return p * x;
        };
        return mu + sigma * std::sqrt(2.0) * erfinv(2 * p - 1);
    }
    
    inline double lgamma(double x) {
        return std::lgamma(x);
    }
    
    inline double dnbinom(double x, double size, double prob, int log_p) {
        // Negative binomial PMF: P(X=x) = C(x+size-1, x) * prob^size * (1-prob)^x
        // log PMF = lgamma(x+size) - lgamma(x+1) - lgamma(size) + size*log(prob) + x*log(1-prob)
        double log_pmf = std::lgamma(x + size) - std::lgamma(x + 1) - std::lgamma(size) 
                        + size * std::log(prob) + x * std::log(1 - prob);
        return log_p ? log_pmf : std::exp(log_pmf);
    }
}

// Rcpp compatibility macros
#define Rcout std::cout
#define Rprintf printf

inline vec runif(int n) {
    vec result(n);
    for (int i = 0; i < n; i++) {
        result(i) = R::runif(0, 1);
    }
    return result;
}

#endif
