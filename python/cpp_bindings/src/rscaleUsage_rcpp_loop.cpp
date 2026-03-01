// rscaleUsage - Scale Usage Model MCMC
#include "bayesm.h"
#include <tuple>

namespace {

vec cgetC_local(double e, int k) {
    vec temp = zeros<vec>(k - 1);
    for (int i = 0; i < (k - 1); i++) temp[i] = i + 1.5;
    double m1 = sum(temp);
    temp = pow(temp, 2.0);
    double m2 = sum(temp);
    
    vec c = zeros<vec>(k + 1);
    
    double s0 = k - 1;
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    for (int i = 1; i < k; i++) {
        s1 += i;
        s2 += i * i;
        s3 += i * i * i;
        s4 += i * i * i * i;
    }
    
    double aq = s0 * s2 - s1 * s1;
    double bq = 2 * e * s0 * s3 - 2 * e * s1 * s2;
    double cq = m1 * m1 - m2 * s0 + e * e * s0 * s4 - e * e * s2 * s2;
    
    double det = bq * bq - 4 * aq * cq;
    if (det < 0) det = 0;
    double b_coef = (-bq + sqrt(det)) / (2.0 * aq);
    double a_coef = (m1 - b_coef * s1 - e * s2) / s0;
    
    c[0] = -1000.0;
    c[k] = 1000.0;
    for (int i = 1; i < k; i++) c[i] = a_coef + b_coef * i + e * i * i;
    
    return sort(c);
}

double ghk_local(mat const& L, vec const& a, vec const& b, int n, int dim) {
    double res = 0.0;
    
    for (int i = 0; i < n; i++) {
        double prod = 1.0;
        vec z(dim);
        
        for (int j = 0; j < dim; j++) {
            double mu = 0.0;
            if (j > 0) mu = as_scalar(L(j, span(0, j - 1)) * z(span(0, j - 1)));
            
            double aa = (a[j] - mu) / L(j, j);
            double bb = (b[j] - mu) / L(j, j);
            
            double pa = R::pnorm(aa, 0, 1, 1, 0);
            double pb = R::pnorm(bb, 0, 1, 1, 0);
            
            prod *= pb - pa;
            
            double u = R::runif(0, 1);
            double arg = u * pb + (1.0 - u) * pa;
            
            if (arg > 0.999999999) arg = 0.999999999;
            if (arg < 0.0000000001) arg = 0.0000000001;
            
            z[j] = R::qnorm(arg, 0, 1, 1, 0);
        }
        res += prod;
    }
    
    return res / n;
}

std::tuple<mat, vec> condd_local(mat const& Sigma) {
    int p = Sigma.n_rows;
    mat Si = solve(Sigma, eye(p, p));
    int cbetarows = p - 1;
    mat cbeta = zeros<mat>(cbetarows, p);
    
    for (int i = 0; i < p; i++) {
        uvec ind(p - 1);
        int counter = 0;
        for (int j = 0; j < p - 1; j++) {
            if (j == i) counter++;
            ind[j] = counter;
            counter++;
        }
        for (int j = 0; j < p - 1; j++) {
            cbeta(j, i) = -Si(ind[j], i) / Si(i, i);
        }
    }
    
    vec s = sqrt(1.0 / Si.diag());
    return std::make_tuple(cbeta, s);
}

mat dy_local(mat y, mat const& x, vec const& c, vec const& mu, vec const& beta,
             vec const& s, vec const& tau, vec const& sigma) {
    int p = y.n_cols;
    int nobs = y.n_rows;
    
    for (int n = 0; n < nobs; n++) {
        double sigman = sigma[n];
        double taun = tau[n];
        rowvec yn = y.row(n);
        vec xn = x.row(n).t();
        
        for (int i = 0; i < p; i++) {
            double cs = s[i] * sigman;
            double cm = mu[i] + taun;
            
            for (int j = 0; j < i; j++) cm += beta[i * (p - 1) + j] * (yn[j] - mu[j] - taun);
            for (int j = i + 1; j < p; j++) cm += beta[i * (p - 1) + j - 1] * (yn[j] - mu[j] - taun);
            
            int xi = (int)xn[i];
            if (xi < 1) xi = 1;
            if (xi > (int)c.n_elem - 1) xi = c.n_elem - 1;
            
            double a = (c[xi - 1] - cm) / cs;
            double b = (c[xi] - cm) / cs;
            
            double pa = R::pnorm(a, 0, 1, 1, 0);
            double pb = R::pnorm(b, 0, 1, 1, 0);
            
            if (pb - pa < 1e-10) pb = pa + 1e-10;
            
            double u = R::runif(0, 1);
            double arg = u * pb + (1 - u) * pa;
            if (arg > 0.999999999) arg = 0.999999999;
            if (arg < 0.0000000001) arg = 0.0000000001;
            double qout = R::qnorm(arg, 0, 1, 1, 0);
            yn[i] = cm + cs * qout;
        }
        y.row(n) = yn;
    }
    return y;
}

double rlpx_local(mat const& x, double e, int k, vec const& mu, vec const& tau,
                  mat const& Sigma, vec const& sigma, int nd) {
    int n = x.n_rows;
    int p = x.n_cols;
    vec cc = cgetC_local(e, k);
    
    mat Sigma_safe = 0.5 * (Sigma + Sigma.t());
    vec eigval_S;
    eig_sym(eigval_S, Sigma_safe);
    if (eigval_S.min() < 1e-8) {
        Sigma_safe = Sigma_safe + (std::abs(eigval_S.min()) + 1e-6) * eye(p, p);
    }
    mat L = chol(Sigma_safe, "lower");
    
    vec lpv = zeros<vec>(n);
    double offset = p * log((double)k);
    
    for (int i = 0; i < n; i++) {
        mat Li = sigma[i] * L;
        
        uvec xia(p), xib(p);
        for (int u = 0; u < p; u++) {
            xia[u] = (int)x(i, u) - 1;
            xib[u] = (int)x(i, u);
        }
        
        vec a = cc.elem(xia) - mu - tau[i];
        vec b = cc.elem(xib) - mu - tau[i];
        
        double ghkres = ghk_local(Li, a, b, nd, L.n_rows);
        double lghkres = (ghkres > 1e-300) ? log(ghkres) : log(1e-300);
        lpv[i] = lghkres + offset;
    }
    
    return sum(lpv);
}

mat getS_local(mat const& Lam, int n, vec const& moms) {
    mat S = zeros<mat>(2, 2);
    S(0, 0) = (n - 1) * moms[2] + n * pow(moms[0], 2.0);
    S(0, 1) = (n - 1) * moms[3] + n * moms[0] * (moms[1] - Lam(1, 1));
    S(1, 0) = S(0, 1);
    S(1, 1) = (n - 1) * moms[4] + n * pow(moms[1] - Lam(1, 1), 2.0);
    return S;
}

double llL_local(mat const& Lam, int n, mat const& S, mat const& V, double nu) {
    int d = Lam.n_cols;
    double dlam = Lam(0, 0) * Lam(1, 1) - pow(Lam(0, 1), 2.0);
    mat M = (S + V) * solve(Lam, eye(d, d));
    double ll = -0.5 * (n + nu + 3) * log(dlam) - 0.5 * sum(M.diag());
    return ll;
}

std::tuple<mat, mat, mat, mat> rwishart_scale(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    for (int i = 0; i < m; i++) T(i, i) = std::sqrt(R::rchisq(nu - i));
    for (int i = 1; i < m; i++)
        for (int j = 0; j < i; j++) T(i, j) = R::rnorm(0, 1);
    
    mat Vsym = 0.5 * (V + V.t());
    vec eigval_V;
    eig_sym(eigval_V, Vsym);
    if (eigval_V.min() < 1e-8) {
        Vsym = Vsym + (std::abs(eigval_V.min()) + 1e-6) * eye(m, m);
    }
    mat C = chol(Vsym, "lower");
    mat CT = C * T;
    mat W = CT * CT.t();
    W = 0.5 * (W + W.t());
    vec eigval_W;
    eig_sym(eigval_W, W);
    if (eigval_W.min() < 1e-8) {
        W = W + (std::abs(eigval_W.min()) + 1e-6) * eye(m, m);
    }
    mat IW = solve(W, eye<mat>(m, m));
    mat Cupper = chol(W);
    mat CI = solve(trimatu(Cupper), eye<mat>(m, m));
    return std::make_tuple(W, IW, Cupper, CI);
}

int sample_discrete(vec const& probs) {
    double u = R::runif(0, 1);
    double cumsum = 0;
    for (uword i = 0; i < probs.n_elem; i++) {
        cumsum += probs[i];
        if (u <= cumsum) return i;
    }
    return probs.n_elem - 1;
}

} // anonymous namespace

std::tuple<mat, mat, mat, mat, mat, vec>
rscaleUsage_rcpp_loop(int k, mat const& x, int p, int n,
                      int R, int keep, int ndghk,
                      mat y, vec mu, mat Sigma, vec tau, vec sigma, mat Lambda, double e,
                      bool domu, bool doSigma, bool dosigma, bool dotau, bool doLambda, bool doe,
                      double nu, mat const& V, mat const& mubar, mat const& Am,
                      vec const& gsigma, vec const& gl11, vec const& gl22, vec const& gl12,
                      int nuL, mat const& VL, vec const& ge) {
    
    int nk = R / keep;
    int ndpost = nk * keep;
    
    mat drSigma = zeros<mat>(nk, p * p);
    mat drmu = zeros<mat>(nk, p);
    mat drtau = zeros<mat>(nk, n);
    mat drsigma = zeros<mat>(nk, n);
    mat drLambda = zeros<mat>(nk, 4);
    vec dre = zeros<vec>(nk);
    
    rowvec onesp = ones<rowvec>(p);
    int mkeep;
    
    for (int rep = 0; rep < ndpost; rep++) {
        vec cc = cgetC_local(e, k);
        auto [beta, s] = condd_local(Sigma);
        y = dy_local(y, x, cc, mu, vectorise(beta), s, tau, sigma);
        
        // Draw Sigma
        if (doSigma) {
            mat Res = y;
            Res.each_row() -= mu.t();
            Res.each_col() -= tau;
            Res.each_col() /= sigma;
            
            mat S = Res.t() * Res;
            mat Vinv = solve(V + S, eye(p, p));
            auto [W, IW, Cupper, CI] = rwishart_scale(nu + n, Vinv);
            Sigma = IW;
        }
        
        // Draw mu
        if (domu) {
            mat yd = y;
            yd.each_col() -= tau;
            mat Si = solve(Sigma, eye(p, p));
            mat Vmi = as_scalar(sum(1.0 / pow(sigma, 2.0))) * Si + Am;
            Vmi = 0.5 * (Vmi + Vmi.t());
            vec eigval_Vmi;
            eig_sym(eigval_Vmi, Vmi);
            if (eigval_Vmi.min() < 1e-8) {
                Vmi = Vmi + (std::abs(eigval_Vmi.min()) + 1e-6) * eye(p, p);
            }
            mat Rm = chol(Vmi);
            mat Ri = solve(trimatu(Rm), eye(p, p));
            mat Vm = solve(Vmi, eye(p, p));
            mat mm = Vm * (Si * (yd.t() * (1.0 / pow(sigma, 2.0))) + Am * mubar);
            vec rnorm_vec(p);
            for (int i = 0; i < p; i++) rnorm_vec[i] = R::rnorm(0, 1);
            mu = vectorise(mm + Ri * rnorm_vec);
        }
        
        // Draw tau
        if (dotau) {
            double Ai = Lambda(0, 0) - pow(Lambda(0, 1), 2.0) / Lambda(1, 1);
            double A = 1.0 / Ai;
            mat onev = ones<mat>(p, 1);
            
            mat Sigma_safe = 0.5 * (Sigma + Sigma.t());
            vec eigval_Sig;
            eig_sym(eigval_Sig, Sigma_safe);
            if (eigval_Sig.min() < 1e-8) {
                Sigma_safe = Sigma_safe + (std::abs(eigval_Sig.min()) + 1e-6) * eye(p, p);
            }
            mat Rm = chol(Sigma_safe);
            mat xx = solve(Rm.t(), onev).t();
            mat ytemp = y.t();
            ytemp.each_col() -= mu;
            mat yy = solve(Rm.t(), ytemp).t();
            double xtx = accu(pow(xx, 2.0));
            vec xty = vectorise(xx * yy.t());
            double beta_coef = A * Lambda(0, 1) / Lambda(1, 1);
            
            for (int j = 0; j < n; j++) {
                double s2 = xtx / pow(sigma[j], 2.0) + A;
                s2 = 1.0 / s2;
                double m = s2 * ((xty[j] / pow(sigma[j], 2.0)) + beta_coef * (log(sigma[j]) - Lambda(1, 1)));
                tau[j] = m + sqrt(s2) * R::rnorm(0, 1);
            }
        }
        
        // Draw sigma
        if (dosigma) {
            mat Sigma_safe = 0.5 * (Sigma + Sigma.t());
            vec eigval_Sig;
            eig_sym(eigval_Sig, Sigma_safe);
            if (eigval_Sig.min() < 1e-8) {
                Sigma_safe = Sigma_safe + (std::abs(eigval_Sig.min()) + 1e-6) * eye(p, p);
            }
            mat Rm = chol(Sigma_safe);
            mat ytemp = y;
            ytemp.each_col() -= tau;
            ytemp = ytemp.t();
            ytemp.each_col() -= mu;
            mat eps = solve(Rm.t(), ytemp);
            vec ete = vectorise(onesp * pow(eps, 2.0));
            
            double a_lam = Lambda(1, 1);
            double b_lam = Lambda(0, 1) / Lambda(0, 0);
            double s_lam = sqrt(Lambda(1, 1) - pow(Lambda(0, 1), 2.0) / Lambda(0, 0));
            
            int ng = gsigma.n_elem;
            for (int j = 0; j < n; j++) {
                vec pv(ng);
                for (int g = 0; g < ng; g++) {
                    pv[g] = -(p + 1) * log(gsigma[g]) - 0.5 * ete[j] / pow(gsigma[g], 2.0)
                            - 0.5 * pow((log(gsigma[g]) - (a_lam + b_lam * tau[j])) / s_lam, 2.0);
                }
                pv = exp(pv - max(pv));
                pv = pv / sum(pv);
                int idx = sample_discrete(pv);
                sigma[j] = gsigma[idx];
            }
        }
        
        // Draw Lambda
        if (doLambda) {
            vec h = log(sigma);
            mat dat = join_rows(tau, h);
            mat temp = cov(dat);
            vec moms = {mean(tau), mean(h), temp(0, 0), temp(0, 1), temp(1, 1)};
            
            mat SS = getS_local(Lambda, n, moms);
            
            // Draw Lambda(0,0)
            uvec rgl11_idx = find(gl11 > pow(Lambda(0, 1), 2.0) / Lambda(1, 1));
            vec rgl11 = gl11.elem(rgl11_idx);
            int ng = rgl11.n_elem;
            if (ng > 0) {
                vec pv(ng);
                for (int j = 0; j < ng; j++) {
                    Lambda(0, 0) = rgl11[j];
                    pv[j] = llL_local(Lambda, n, SS, VL, nuL);
                }
                pv = exp(pv - max(pv));
                pv = pv / sum(pv);
                int idx = sample_discrete(pv);
                Lambda(0, 0) = rgl11[idx];
            }
            
            // Draw Lambda(0,1)
            double bound = sqrt(Lambda(0, 0) * Lambda(1, 1));
            uvec rgl12a_idx = find(gl12 < bound);
            vec rgl12a = gl12.elem(rgl12a_idx);
            uvec rgl12_idx = find(rgl12a > -bound);
            vec rgl12 = rgl12a.elem(rgl12_idx);
            ng = rgl12.n_elem;
            if (ng > 0) {
                vec pv(ng);
                for (int j = 0; j < ng; j++) {
                    Lambda(0, 1) = rgl12[j];
                    Lambda(1, 0) = Lambda(0, 1);
                    pv[j] = llL_local(Lambda, n, SS, VL, nuL);
                }
                pv = exp(pv - max(pv));
                pv = pv / sum(pv);
                int idx = sample_discrete(pv);
                Lambda(0, 1) = rgl12[idx];
                Lambda(1, 0) = Lambda(0, 1);
            }
            
            // Draw Lambda(1,1)
            uvec rgl22_idx = find(gl22 > pow(Lambda(0, 1), 2.0) / Lambda(0, 0));
            vec rgl22 = gl22.elem(rgl22_idx);
            ng = rgl22.n_elem;
            if (ng > 0) {
                vec pv(ng);
                for (int j = 0; j < ng; j++) {
                    Lambda(1, 1) = rgl22[j];
                    SS = getS_local(Lambda, n, moms);
                    pv[j] = llL_local(Lambda, n, SS, VL, nuL);
                }
                pv = exp(pv - max(pv));
                pv = pv / sum(pv);
                int idx = sample_discrete(pv);
                Lambda(1, 1) = rgl22[idx];
            }
        }
        
        // Draw e
        if (doe) {
            int ng = ge.n_elem;
            vec absege = abs(e - ge);
            uword ei = absege.index_min();
            
            int pi;
            double qr;
            if (ei == 0) {
                pi = 1;
                qr = 0.5;
            } else if (ei == (uword)(ng - 1)) {
                pi = ng - 2;
                qr = 0.5;
            } else {
                int coin = (R::runif(0, 1) < 0.5) ? 0 : 1;
                pi = ei + coin * 2 - 1;
                qr = 1.0;
            }
            
            double eold = ge[ei];
            double eprop = ge[pi];
            
            double llold = rlpx_local(x, eold, k, mu, tau, Sigma, sigma, ndghk);
            double llprop = rlpx_local(x, eprop, k, mu, tau, Sigma, sigma, ndghk);
            double lrat = llprop - llold + log(qr);
            
            double paccept = std::min(1.0, exp(lrat));
            if (R::runif(0, 1) < paccept) {
                e = eprop;
            } else {
                e = eold;
            }
        }
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            drSigma.row(mkeep - 1) = vectorise(Sigma).t();
            drmu.row(mkeep - 1) = mu.t();
            drtau.row(mkeep - 1) = tau.t();
            drsigma.row(mkeep - 1) = sigma.t();
            drLambda.row(mkeep - 1) = vectorise(Lambda).t();
            dre[mkeep - 1] = e;
        }
    }
    
    return std::make_tuple(drSigma, drmu, drtau, drsigma, drLambda, dre);
}
