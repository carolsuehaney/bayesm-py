// rDPGibbs - Dirichlet Process Gibbs Sampler for Density Estimation
#include "bayesm.h"
#include <tuple>
#include <vector>

struct murooti_dp { vec mu; mat rooti; };
struct lambda_dp { vec mubar; double Amu; double nu; mat V; };
struct priorAlpha_dp { double power; double alphamin; double alphamax; int n; };
struct thetaStarIndex_dp { ivec indic; std::vector<murooti_dp> thetaStar_vector; };

namespace {

int rmultinomF_local(vec const& p) {
    vec csp = cumsum(p);
    double rnd = R::runif(0, 1);
    int res = 0;
    int psize = p.size();
    for (int i = 0; i < psize; i++) {
        if (rnd > csp[i]) res++;
    }
    return res + 1;
}

mat yden_local(std::vector<murooti_dp> const& thetaStar_vector, mat const& y) {
    int nunique = thetaStar_vector.size();
    int n = y.n_rows;
    int k = y.n_cols;
    mat ydenmat = zeros<mat>(nunique, n);
    
    for (int i = 0; i < nunique; i++) {
        vec mu = thetaStar_vector[i].mu;
        mat rooti = thetaStar_vector[i].rooti;
        mat transy = y.t();
        transy.each_col() -= mu;
        mat quads_mat = sum(square(rooti.t() * transy), 0);
        ydenmat.row(i) = exp(-(k / 2.0) * log(2 * M_PI) + sum(log(rooti.diag())) - 0.5 * quads_mat);
    }
    return ydenmat;
}

ivec numcomp_local(ivec const& indic, int k) {
    ivec ncomp(k);
    for (int comp = 0; comp < k; comp++) {
        ncomp[comp] = sum(indic == (comp + 1));
    }
    return ncomp;
}

std::tuple<mat, mat, mat, mat> rwishart_dp(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    for (int i = 0; i < m; i++) T(i, i) = std::sqrt(R::rchisq(nu - i));
    for (int i = 1; i < m; i++)
        for (int j = 0; j < i; j++) T(i, j) = R::rnorm(0, 1);
    mat C = chol(V, "lower");
    mat CT = C * T;
    mat W = CT * CT.t();
    mat IW = solve(W, eye<mat>(m, m));
    mat Cupper = chol(W);
    mat CI = solve(trimatu(Cupper), eye<mat>(m, m));
    return std::make_tuple(W, IW, Cupper, CI);
}

std::tuple<mat, mat> rmultireg_dp(mat const& Y, mat const& X, mat const& Bbar,
                                   mat const& A, double nu, mat const& V) {
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    
    mat XpX = X.t() * X;
    mat XpY = X.t() * Y;
    mat IR = solve(trimatu(chol(XpX + A)), eye<mat>(k, k));
    mat Btilde = (IR * IR.t()) * (XpY + A * Bbar);
    mat S = Y.t() * Y - XpY.t() * (IR * IR.t()) * XpY
            - Bbar.t() * A * Bbar + Btilde.t() * (XpX + A) * Btilde;
    mat ucholinv = solve(trimatu(chol(S + V)), eye<mat>(m, m));
    mat Vinv = ucholinv * ucholinv.t();
    auto [W, IW, C, CI] = rwishart_dp(nu + n, Vinv);
    mat Sigma = IW;
    mat root_Sigma = chol(Sigma, "lower");
    mat rnorm_mat(k, m);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++) rnorm_mat(i, j) = R::rnorm(0, 1);
    mat B = Btilde + IR * rnorm_mat * root_Sigma.t();
    return std::make_tuple(B, Sigma);
}

murooti_dp thetaD_local(mat const& y, lambda_dp const& lam) {
    mat X = ones<mat>(y.n_rows, 1);
    mat A(1, 1); A.fill(lam.Amu);
    auto [B, Sigma] = rmultireg_dp(y, X, lam.mubar.t(), A, lam.nu, lam.V);
    murooti_dp out;
    out.mu = B.as_col();
    out.rooti = solve(trimatu(chol(Sigma)), eye(y.n_cols, y.n_cols));
    return out;
}

vec q0_local(mat const& y, lambda_dp const& lam) {
    int k = y.n_cols;
    mat R = chol(lam.V);
    double logdetR = sum(log(R.diag()));
    double lnk1k2;
    
    if (k > 1) {
        double sum_lgamma = 0;
        for (int i = 1; i < k; i++) sum_lgamma += lgamma(lam.nu / 2.0 - i / 2.0);
        lnk1k2 = (k / 2.0) * log(2.0) + log((lam.nu - k) / 2.0) + lgamma((lam.nu - k) / 2.0) - lgamma(lam.nu / 2.0) + sum_lgamma;
    } else {
        lnk1k2 = (k / 2.0) * log(2.0) + log((lam.nu - k) / 2.0) + lgamma((lam.nu - k) / 2.0) - lgamma(lam.nu / 2.0);
    }
    
    double constant = -(k / 2.0) * log(2 * M_PI) + (k / 2.0) * log(lam.Amu / (1 + lam.Amu)) + lnk1k2 + lam.nu * logdetR;
    
    mat transy = y.t();
    transy.each_col() -= lam.mubar;
    mat Rinv = solve(trimatu(R), eye(y.n_cols, y.n_cols));
    mat m = sqrt(lam.Amu / (1 + lam.Amu)) * Rinv.t() * transy;
    rowvec vivi = sum(square(m), 0);
    rowvec lnq0v = constant - ((lam.nu + 1) / 2.0) * (2 * logdetR + log(1 + vivi));
    return exp(lnq0v.t());
}

thetaStarIndex_dp thetaStarDraw_local(ivec indic, std::vector<murooti_dp> thetaStar_vector,
                                       mat const& y, mat ydenmat, vec const& q0v, double alpha,
                                       lambda_dp const& lam, int maxuniq) {
    int n = indic.size();
    
    for (int i = 0; i < n; i++) {
        int k = thetaStar_vector.size();
        vec probs(k + 1);
        probs[k] = q0v[i] * (alpha / (alpha + (n - 1)));
        
        ivec indicmi = zeros<ivec>(n - 1);
        int inc = 0;
        for (int j = 0; j < (n - 1); j++) {
            if (j == i) inc++;
            indicmi[j] = indic[inc];
            inc++;
        }
        
        ivec ncomp = numcomp_local(indicmi, k);
        for (int comp = 0; comp < k; comp++) {
            probs[comp] = ydenmat(comp, i) * ncomp[comp] / (alpha + (n - 1));
        }
        
        probs = probs / sum(probs);
        indic[i] = rmultinomF_local(probs);
        
        if (indic[i] == (k + 1)) {
            if ((k + 1) > maxuniq) {
                throw std::runtime_error("max number of comps exceeded");
            }
            murooti_dp newtheta = thetaD_local(y.row(i), lam);
            thetaStar_vector.push_back(newtheta);
            std::vector<murooti_dp> listofone(1);
            listofone[0] = newtheta;
            ydenmat.row(k) = yden_local(listofone, y).row(0);
        }
    }
    
    int k = thetaStar_vector.size();
    ivec indicC = zeros<ivec>(n);
    ivec ncomp = numcomp_local(indic, k);
    
    std::vector<murooti_dp> thetaStarC_vector;
    int cntNonzero = 0;
    for (int comp = 0; comp < k; comp++) {
        if (ncomp[comp] != 0) {
            thetaStarC_vector.push_back(thetaStar_vector[comp]);
            cntNonzero++;
            for (int i = 0; i < n; i++) {
                if (indic[i] == (comp + 1)) indicC[i] = cntNonzero;
            }
        }
    }
    
    thetaStarIndex_dp out;
    out.indic = indicC;
    out.thetaStar_vector = thetaStarC_vector;
    return out;
}

vec seq_local(double from, double to, int len) {
    vec res(len);
    res[len - 1] = to;
    res[0] = from;
    double increment = (res[len - 1] - res[0]) / (len - 1);
    for (int i = 1; i < (len - 1); i++) res[i] = res[i - 1] + increment;
    return res;
}

double alphaD_local(priorAlpha_dp const& pa, int Istar, int gridsize) {
    vec alpha = seq_local(pa.alphamin, pa.alphamax - 0.000001, gridsize);
    vec lnprob(gridsize);
    for (int i = 0; i < gridsize; i++) {
        lnprob[i] = Istar * log(alpha[i]) + lgamma(alpha[i]) - lgamma(pa.n + alpha[i])
                    + pa.power * log(1 - (alpha[i] - pa.alphamin) / (pa.alphamax - pa.alphamin));
    }
    lnprob = lnprob - median(lnprob);
    vec probs = exp(lnprob);
    probs = probs / sum(probs);
    return alpha(rmultinomF_local(probs) - 1);
}

murooti_dp GD_local(lambda_dp const& lam) {
    int k = lam.mubar.size();
    mat Vinv = solve(trimatu(lam.V), eye(k, k));
    auto [W, IW, C, CI] = rwishart_dp(lam.nu, Vinv);
    mat Sigma = IW;
    mat root = chol(Sigma);
    vec draws(k);
    for (int i = 0; i < k; i++) draws[i] = R::rnorm(0, 1);
    vec mu = lam.mubar + (1 / sqrt(lam.Amu)) * root.t() * draws;
    
    murooti_dp out;
    out.mu = mu;
    out.rooti = solve(trimatu(root), eye(k, k));
    return out;
}

lambda_dp lambdaD_local(lambda_dp const& lam, std::vector<murooti_dp> const& thetaStar_vector,
                        vec const& alim, vec const& nulim, vec const& vlim, int gridsize) {
    int d = thetaStar_vector[0].mu.size();
    int Istar = thetaStar_vector.size();
    
    vec aseq = seq_local(alim[0], alim[1], gridsize);
    vec nuseq = d - 1 + exp(seq_local(nulim[0], nulim[1], gridsize));
    vec vseq = seq_local(vlim[0], vlim[1], gridsize);
    
    mat mout = zeros<mat>(d, Istar * d);
    for (int i = 0; i < Istar; i++) {
        int ind = i * d;
        mout.submat(0, ind, d - 1, ind + d - 1) = thetaStar_vector[i].rooti.t();
    }
    double sumdiagriri = accu(square(mout));
    
    double sumlogdiag = 0.0;
    for (int i = 0; i < Istar; i++) {
        int ind = i * d;
        for (int j = 0; j < d; j++) {
            sumlogdiag += log(mout(j, ind + j));
        }
    }
    
    mat rimu = zeros<mat>(d, Istar);
    for (int i = 0; i < Istar; i++) {
        rimu.col(i) = thetaStar_vector[i].rooti.t() * thetaStar_vector[i].mu;
    }
    double sumquads = accu(square(rimu));
    
    vec lnprob = Istar * (-(d / 2.0) * log(2 * M_PI)) - 0.5 * aseq * sumquads
                 + Istar * d * log(sqrt(aseq)) + sumlogdiag;
    lnprob = lnprob - max(lnprob) + 200;
    vec probs = exp(lnprob);
    probs = probs / sum(probs);
    double adraw = aseq[rmultinomF_local(probs) - 1];
    
    mat arg = zeros<mat>(gridsize, d);
    for (int i = 0; i < d; i++) {
        arg.col(i) = nuseq - i;
    }
    arg = arg / 2.0;
    
    mat lgammaarg = zeros<mat>(gridsize, d);
    for (int i = 0; i < gridsize; i++) {
        for (int j = 0; j < d; j++) {
            lgammaarg(i, j) = lgamma(arg(i, j));
        }
    }
    vec rowSumslgammaarg = sum(lgammaarg, 1);
    
    lnprob = zeros<vec>(gridsize);
    for (int i = 0; i < gridsize; i++) {
        lnprob[i] = -Istar * log(2.0) * d / 2.0 * nuseq[i] - Istar * rowSumslgammaarg[i]
                    + Istar * d * log(sqrt(lam.V(0, 0))) * nuseq[i] + sumlogdiag * nuseq[i];
    }
    lnprob = lnprob - max(lnprob) + 200;
    probs = exp(lnprob);
    probs = probs / sum(probs);
    double nudraw = nuseq[rmultinomF_local(probs) - 1];
    
    lnprob = Istar * nudraw * d * log(sqrt(vseq * nudraw)) - 0.5 * sumdiagriri * vseq * nudraw;
    lnprob = lnprob - max(lnprob) + 200;
    probs = exp(lnprob);
    probs = probs / sum(probs);
    double vdraw = vseq[rmultinomF_local(probs) - 1];
    
    lambda_dp out;
    out.mubar = zeros<vec>(d);
    out.Amu = adraw;
    out.nu = nudraw;
    out.V = nudraw * vdraw * eye(d, d);
    return out;
}

} // anonymous namespace

std::tuple<vec, vec, vec, vec, vec, imat, std::vector<std::vector<murooti_dp>>>
rDPGibbs_rcpp_loop(int R, int keep, mat y, vec const& alim, vec const& nulim, vec const& vlim,
                   bool SCALE, int maxuniq, double power, double alphamin, double alphamax, int n,
                   int gridsize, double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha) {

    int dimy = y.n_cols;
    int nobs = y.n_rows;
    
    ivec indic = ones<ivec>(nobs);
    
    std::vector<murooti_dp> thetaStar_vector(1);
    thetaStar_vector[0].mu = zeros<vec>(dimy);
    thetaStar_vector[0].rooti = eye(dimy, dimy);
    
    priorAlpha_dp priorAlpha_struct;
    priorAlpha_struct.power = power;
    priorAlpha_struct.alphamin = alphamin;
    priorAlpha_struct.alphamax = alphamax;
    priorAlpha_struct.n = n;
    
    lambda_dp lambda_struct;
    lambda_struct.mubar = zeros<vec>(dimy);
    lambda_struct.Amu = BayesmConstantA;
    lambda_struct.nu = dimy + BayesmConstantnuInc;
    lambda_struct.V = lambda_struct.nu * eye(dimy, dimy);
    
    double alpha = BayesmConstantDPalpha;
    
    vec alphadraw = zeros<vec>(R / keep);
    vec Istardraw = zeros<vec>(R / keep);
    vec adraw = zeros<vec>(R / keep);
    vec nudraw = zeros<vec>(R / keep);
    vec vdraw = zeros<vec>(R / keep);
    imat inddraw = zeros<imat>(R / keep, nobs);
    std::vector<std::vector<murooti_dp>> thetaNp1draw(R / keep);
    
    rowvec dvec, ybar;
    if (SCALE) {
        dvec = 1 / sqrt(var(y, 0, 0));
        ybar = mean(y, 0);
        y.each_row() -= ybar;
        y.each_row() %= dvec;
    }
    
    int mkeep = 0;
    for (int rep = 0; rep < R; rep++) {
        vec q0v = q0_local(y, lambda_struct);
        int nunique = thetaStar_vector.size();
        
        if (nunique > maxuniq) throw std::runtime_error("max number of unique thetas exceeded");
        
        mat ydenmat = zeros<mat>(maxuniq, nobs);
        ydenmat.rows(0, nunique - 1) = yden_local(thetaStar_vector, y);
        
        auto thetaStarOut = thetaStarDraw_local(indic, thetaStar_vector, y, ydenmat, q0v, alpha, lambda_struct, maxuniq);
        thetaStar_vector = thetaStarOut.thetaStar_vector;
        indic = thetaStarOut.indic;
        nunique = thetaStar_vector.size();
        
        vec probs = zeros<vec>(nunique + 1);
        uvec spanall(dimy);
        for (int i = 0; i < dimy; i++) spanall[i] = i;
        
        for (int j = 0; j < nunique; j++) {
            uvec ind_j = find(indic == (j + 1));
            int indsize = ind_j.size();
            probs[j] = indsize / (alpha + nobs + 0.0);
            thetaStar_vector[j] = thetaD_local(y.rows(ind_j), lambda_struct);
        }
        
        probs[nunique] = alpha / (alpha + nobs + 0.0);
        int ind = rmultinomF_local(probs);
        
        murooti_dp thetaNp1;
        if (ind == (int)probs.size()) {
            thetaNp1 = GD_local(lambda_struct);
        } else {
            thetaNp1 = thetaStar_vector[ind - 1];
        }
        
        alpha = alphaD_local(priorAlpha_struct, nunique, gridsize);
        lambda_struct = lambdaD_local(lambda_struct, thetaStar_vector, alim, nulim, vlim, gridsize);
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            alphadraw[mkeep - 1] = alpha;
            Istardraw[mkeep - 1] = nunique;
            adraw[mkeep - 1] = lambda_struct.Amu;
            double nu = lambda_struct.nu;
            nudraw[mkeep - 1] = nu;
            vdraw[mkeep - 1] = lambda_struct.V(0, 0) / nu;
            inddraw.row(mkeep - 1) = indic.t();
            
            murooti_dp thetaNp1_scaled = thetaNp1;
            if (SCALE) {
                thetaNp1_scaled.mu = thetaNp1.mu / dvec.t() + ybar.t();
                thetaNp1_scaled.rooti = diagmat(dvec) * thetaNp1.rooti;
            }
            std::vector<murooti_dp> thetaNp1_vec(1);
            thetaNp1_vec[0] = thetaNp1_scaled;
            thetaNp1draw[mkeep - 1] = thetaNp1_vec;
        }
    }
    
    return std::make_tuple(alphadraw, Istardraw, adraw, nudraw, vdraw, inddraw, thetaNp1draw);
}
