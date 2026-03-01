// rhierMnlDP - Hierarchical MNL with Dirichlet Process prior
#include "bayesm.h"
#include <tuple>
#include <vector>

struct mnlDP_moments { vec y; mat X; mat hess; };
struct murooti_mnldp { vec mu; mat rooti; };
struct lambda_mnldp { vec mubar; double Amu; double nu; mat V; };
struct priorAlpha_mnldp { double power; double alphamin; double alphamax; int n; };
struct thetaStarIndex_mnldp { ivec indic; std::vector<murooti_mnldp> thetaStar_vector; };
struct DPOut_mnldp { ivec indic; std::vector<murooti_mnldp> thetaStar_vector; std::vector<murooti_mnldp> thetaNp1_vector; double alpha; lambda_mnldp lambda_struct; };
struct mnlMetropOnceOut_mnldp { vec betadraw; int stay; double oldll; };

namespace {

int rmultinomF_mnldp(vec const& p) {
    vec csp = cumsum(p);
    double rnd = R::runif(0, 1);
    int res = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (rnd > csp[i]) res++;
    }
    return res + 1;
}

double llmnl_local(vec const& beta, vec const& y, mat const& X) {
    int n = y.size();
    int j = X.n_rows / n;
    vec Xbeta = X * beta;
    
    vec xby = zeros<vec>(n);
    vec denom = zeros<vec>(n);
    
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < j; p++) denom[i] += exp(Xbeta[i * j + p]);
        xby[i] = Xbeta[i * j + (int)y[i] - 1];
    }
    return sum(xby - log(denom));
}

double lndMvn_mnldp(vec const& x, vec const& mu, mat const& rooti) {
    vec z = rooti.t() * (x - mu);
    return -(x.size() / 2.0) * log(2 * M_PI) + sum(log(rooti.diag())) - 0.5 * dot(z, z);
}

mnlMetropOnceOut_mnldp mnlMetropOnce_mnldp(vec const& y, mat const& X, vec const& oldbeta, double oldll,
                                            double s, mat const& incroot, vec const& betabar, mat const& rootpi) {
    int stay = 0;
    vec betac = oldbeta + s * incroot * arma::randn<vec>(oldbeta.size());
    double cll = llmnl_local(betac, y, X);
    double clpost = cll + lndMvn_mnldp(betac, betabar, rootpi);
    double ldiff = clpost - oldll - lndMvn_mnldp(oldbeta, betabar, rootpi);
    double alpha = std::min(1.0, exp(ldiff));
    if (R::runif(0, 1) <= alpha) {
        mnlMetropOnceOut_mnldp out;
        out.betadraw = betac;
        out.stay = stay;
        out.oldll = cll;
        return out;
    } else {
        stay = 1;
        mnlMetropOnceOut_mnldp out;
        out.betadraw = oldbeta;
        out.stay = stay;
        out.oldll = oldll;
        return out;
    }
}

mat yden_mnldp(std::vector<murooti_mnldp> const& thetaStar_vector, mat const& y) {
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

ivec numcomp_mnldp(ivec const& indic, int k) {
    ivec ncomp(k);
    for (int comp = 0; comp < k; comp++) {
        ncomp[comp] = sum(indic == (comp + 1));
    }
    return ncomp;
}

std::tuple<mat, mat, mat, mat> rwishart_mnldp(double nu, mat const& V) {
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

std::tuple<mat, mat> rmultireg_mnldp(mat const& Y, mat const& X, mat const& Bbar, mat const& A, double nu, mat const& V) {
    int n = Y.n_rows;
    int m = Y.n_cols;
    int k = X.n_cols;
    mat XpX = X.t() * X;
    mat XpY = X.t() * Y;
    mat IR = solve(trimatu(chol(XpX + A)), eye<mat>(k, k));
    mat Btilde = (IR * IR.t()) * (XpY + A * Bbar);
    mat S = Y.t() * Y - XpY.t() * (IR * IR.t()) * XpY - Bbar.t() * A * Bbar + Btilde.t() * (XpX + A) * Btilde;
    mat ucholinv = solve(trimatu(chol(S + V)), eye<mat>(m, m));
    mat Vinv = ucholinv * ucholinv.t();
    auto [W, IW, C, CI] = rwishart_mnldp(nu + n, Vinv);
    mat Sigma = IW;
    mat root_Sigma = chol(Sigma, "lower");
    mat rnorm_mat(k, m);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++) rnorm_mat(i, j) = R::rnorm(0, 1);
    mat B = Btilde + IR * rnorm_mat * root_Sigma.t();
    return std::make_tuple(B, Sigma);
}

murooti_mnldp thetaD_mnldp(mat const& y, lambda_mnldp const& lam) {
    mat X = ones<mat>(y.n_rows, 1);
    mat A(1, 1); A.fill(lam.Amu);
    auto [B, Sigma] = rmultireg_mnldp(y, X, lam.mubar.t(), A, lam.nu, lam.V);
    murooti_mnldp out;
    out.mu = B.as_col();
    out.rooti = solve(trimatu(chol(Sigma)), eye(y.n_cols, y.n_cols));
    return out;
}

vec q0_mnldp(mat const& y, lambda_mnldp const& lam) {
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

thetaStarIndex_mnldp thetaStarDraw_mnldp(ivec indic, std::vector<murooti_mnldp> thetaStar_vector,
                                          mat const& y, mat ydenmat, vec const& q0v, double alpha,
                                          lambda_mnldp const& lam, int maxuniq) {
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
        ivec ncomp = numcomp_mnldp(indicmi, k);
        for (int comp = 0; comp < k; comp++) {
            probs[comp] = ydenmat(comp, i) * ncomp[comp] / (alpha + (n - 1));
        }
        probs = probs / sum(probs);
        indic[i] = rmultinomF_mnldp(probs);
        if (indic[i] == (k + 1)) {
            if ((k + 1) > maxuniq) throw std::runtime_error("max number of comps exceeded");
            murooti_mnldp newtheta = thetaD_mnldp(y.row(i), lam);
            thetaStar_vector.push_back(newtheta);
            std::vector<murooti_mnldp> listofone(1);
            listofone[0] = newtheta;
            ydenmat.row(k) = yden_mnldp(listofone, y).row(0);
        }
    }
    int k = thetaStar_vector.size();
    ivec indicC = zeros<ivec>(n);
    ivec ncomp = numcomp_mnldp(indic, k);
    std::vector<murooti_mnldp> thetaStarC_vector;
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
    thetaStarIndex_mnldp out;
    out.indic = indicC;
    out.thetaStar_vector = thetaStarC_vector;
    return out;
}

vec seq_mnldp(double from, double to, int len) {
    vec res(len);
    res[len - 1] = to;
    res[0] = from;
    double increment = (res[len - 1] - res[0]) / (len - 1);
    for (int i = 1; i < (len - 1); i++) res[i] = res[i - 1] + increment;
    return res;
}

double alphaD_mnldp(priorAlpha_mnldp const& pa, int Istar, int gridsize) {
    vec alpha = seq_mnldp(pa.alphamin, pa.alphamax - 0.000001, gridsize);
    vec lnprob(gridsize);
    for (int i = 0; i < gridsize; i++) {
        lnprob[i] = Istar * log(alpha[i]) + lgamma(alpha[i]) - lgamma(pa.n + alpha[i])
                    + pa.power * log(1 - (alpha[i] - pa.alphamin) / (pa.alphamax - pa.alphamin));
    }
    lnprob = lnprob - median(lnprob);
    vec probs = exp(lnprob);
    probs = probs / sum(probs);
    return alpha(rmultinomF_mnldp(probs) - 1);
}

murooti_mnldp GD_mnldp(lambda_mnldp const& lam) {
    int k = lam.mubar.size();
    mat Vinv = solve(trimatu(lam.V), eye(k, k));
    auto [W, IW, C, CI] = rwishart_mnldp(lam.nu, Vinv);
    mat Sigma = IW;
    mat root = chol(Sigma);
    vec draws(k);
    for (int i = 0; i < k; i++) draws[i] = R::rnorm(0, 1);
    vec mu = lam.mubar + (1 / sqrt(lam.Amu)) * root.t() * draws;
    murooti_mnldp out;
    out.mu = mu;
    out.rooti = solve(trimatu(root), eye(k, k));
    return out;
}

lambda_mnldp lambdaD_mnldp(lambda_mnldp const& lam, std::vector<murooti_mnldp> const& thetaStar_vector,
                           vec const& alim, vec const& nulim, vec const& vlim, int gridsize) {
    int d = thetaStar_vector[0].mu.size();
    int Istar = thetaStar_vector.size();
    vec aseq = seq_mnldp(alim[0], alim[1], gridsize);
    vec nuseq = d - 1 + exp(seq_mnldp(nulim[0], nulim[1], gridsize));
    vec vseq = seq_mnldp(vlim[0], vlim[1], gridsize);
    
    mat mout = zeros<mat>(d, Istar * d);
    for (int i = 0; i < Istar; i++) {
        int ind = i * d;
        mout.submat(0, ind, d - 1, ind + d - 1) = thetaStar_vector[i].rooti.t();
    }
    double sumdiagriri = accu(square(mout));
    double sumlogdiag = 0.0;
    for (int i = 0; i < Istar; i++) {
        int ind = i * d;
        for (int j = 0; j < d; j++) sumlogdiag += log(mout(j, ind + j));
    }
    mat rimu = zeros<mat>(d, Istar);
    for (int i = 0; i < Istar; i++) rimu.col(i) = thetaStar_vector[i].rooti.t() * thetaStar_vector[i].mu;
    double sumquads = accu(square(rimu));
    
    vec lnprob = Istar * (-(d / 2.0) * log(2 * M_PI)) - 0.5 * aseq * sumquads + Istar * d * log(sqrt(aseq)) + sumlogdiag;
    lnprob = lnprob - max(lnprob) + 200;
    vec probs = exp(lnprob);
    probs = probs / sum(probs);
    double adraw = aseq[rmultinomF_mnldp(probs) - 1];
    
    mat arg = zeros<mat>(gridsize, d);
    for (int i = 0; i < d; i++) arg.col(i) = nuseq - i;
    arg = arg / 2.0;
    mat lgammaarg = zeros<mat>(gridsize, d);
    for (int i = 0; i < gridsize; i++) for (int j = 0; j < d; j++) lgammaarg(i, j) = lgamma(arg(i, j));
    vec rowSumslgammaarg = sum(lgammaarg, 1);
    lnprob = zeros<vec>(gridsize);
    for (int i = 0; i < gridsize; i++) {
        lnprob[i] = -Istar * log(2.0) * d / 2.0 * nuseq[i] - Istar * rowSumslgammaarg[i]
                    + Istar * d * log(sqrt(lam.V(0, 0))) * nuseq[i] + sumlogdiag * nuseq[i];
    }
    lnprob = lnprob - max(lnprob) + 200;
    probs = exp(lnprob);
    probs = probs / sum(probs);
    double nudraw = nuseq[rmultinomF_mnldp(probs) - 1];
    
    lnprob = Istar * nudraw * d * log(sqrt(vseq * nudraw)) - 0.5 * sumdiagriri * vseq * nudraw;
    lnprob = lnprob - max(lnprob) + 200;
    probs = exp(lnprob);
    probs = probs / sum(probs);
    double vdraw = vseq[rmultinomF_mnldp(probs) - 1];
    
    lambda_mnldp out;
    out.mubar = zeros<vec>(d);
    out.Amu = adraw;
    out.nu = nudraw;
    out.V = nudraw * vdraw * eye(d, d);
    return out;
}

mat drawDelta_mnldp(mat const& x, mat const& y, ivec const& z, std::vector<murooti_mnldp> const& comps,
                    vec const& deltabar, mat const& Ad) {
    int p = y.n_cols;
    int k = x.n_cols;
    int ncomp = comps.size();
    mat xtx = zeros<mat>(k * p, k * p);
    mat xty = zeros<mat>(p, k);
    
    for (int compi = 0; compi < ncomp; compi++) {
        uvec ind = find(z == (compi + 1));
        if (ind.size() > 0) {
            mat yi = y.rows(ind);
            mat xi = x.rows(ind);
            yi.each_row() -= comps[compi].mu.t();
            mat sigi = comps[compi].rooti * comps[compi].rooti.t();
            xtx = xtx + kron(xi.t() * xi, sigi);
            xty = xty + sigi * yi.t() * xi;
        }
    }
    xty.reshape(xty.n_rows * xty.n_cols, 1);
    mat ucholinv = solve(trimatu(chol(xtx + Ad)), eye(k * p, k * p));
    mat Vinv = ucholinv * ucholinv.t();
    vec rnorm_vec(deltabar.size());
    for (size_t i = 0; i < deltabar.size(); i++) rnorm_vec[i] = R::rnorm(0, 1);
    return Vinv * (xty + Ad * deltabar) + chol(Vinv).t() * rnorm_vec;
}

DPOut_mnldp rDPGibbs1_local(mat y, lambda_mnldp lam, std::vector<murooti_mnldp> thetaStar_vector, int maxuniq,
                             ivec indic, double alpha, priorAlpha_mnldp const& pa, int gridsize,
                             vec const& alim, vec const& nulim, vec const& vlim) {
    int n = y.n_rows;
    int dimy = y.n_cols;
    vec q0v = q0_mnldp(y, lam);
    int nunique = thetaStar_vector.size();
    if (nunique > maxuniq) throw std::runtime_error("max number of unique thetas exceeded");
    
    mat ydenmat = zeros<mat>(maxuniq, n);
    ydenmat.rows(0, nunique - 1) = yden_mnldp(thetaStar_vector, y);
    
    auto thetaStarOut = thetaStarDraw_mnldp(indic, thetaStar_vector, y, ydenmat, q0v, alpha, lam, maxuniq);
    thetaStar_vector = thetaStarOut.thetaStar_vector;
    indic = thetaStarOut.indic;
    nunique = thetaStar_vector.size();
    
    vec probs = zeros<vec>(nunique + 1);
    uvec spanall(dimy);
    for (int i = 0; i < dimy; i++) spanall[i] = i;
    for (int j = 0; j < nunique; j++) {
        uvec ind_j = find(indic == (j + 1));
        probs[j] = ind_j.size() / (alpha + n + 0.0);
        thetaStar_vector[j] = thetaD_mnldp(y.rows(ind_j), lam);
    }
    probs[nunique] = alpha / (alpha + n + 0.0);
    int ind = rmultinomF_mnldp(probs);
    
    std::vector<murooti_mnldp> thetaNp1_vector(1);
    if (ind == (int)probs.size()) {
        thetaNp1_vector[0] = GD_mnldp(lam);
    } else {
        thetaNp1_vector[0] = thetaStar_vector[ind - 1];
    }
    
    alpha = alphaD_mnldp(pa, nunique, gridsize);
    lam = lambdaD_mnldp(lam, thetaStar_vector, alim, nulim, vlim, gridsize);
    
    DPOut_mnldp out;
    out.thetaStar_vector = thetaStar_vector;
    out.thetaNp1_vector = thetaNp1_vector;
    out.alpha = alpha;
    out.lambda_struct = lam;
    out.indic = indic;
    return out;
}

} // anonymous namespace

std::tuple<mat, arma::cube, vec, vec, vec, vec, vec, vec, vec, std::vector<std::vector<murooti_mnldp>>>
rhierMnlDP_rcpp_loop(int R, int keep, std::vector<mnlDP_moments> const& lgtdata_vector, mat const& Z,
                      vec const& deltabar, mat const& Ad, double power, double alphamin, double alphamax, int n_prior,
                      vec const& alim, vec const& nulim, vec const& vlim, bool drawdelta, int nvar, mat oldbetas, double s,
                      int maxuniq, int gridsize, double BayesmConstantA, int BayesmConstantnuInc, double BayesmConstantDPalpha) {

    int nz = Z.n_cols;
    int nlgt = lgtdata_vector.size();
    
    ivec indic = ones<ivec>(nlgt);
    mat olddelta;
    if (drawdelta) olddelta = zeros<vec>(nz * nvar);
    
    std::vector<murooti_mnldp> thetaStar_vector(1);
    thetaStar_vector[0].mu = zeros<vec>(nvar);
    thetaStar_vector[0].rooti = eye(nvar, nvar);
    
    double alpha = BayesmConstantDPalpha;
    
    priorAlpha_mnldp priorAlpha_struct;
    priorAlpha_struct.power = power;
    priorAlpha_struct.alphamin = alphamin;
    priorAlpha_struct.alphamax = alphamax;
    priorAlpha_struct.n = n_prior;
    
    lambda_mnldp lambda_struct;
    lambda_struct.mubar = zeros<vec>(nvar);
    lambda_struct.Amu = BayesmConstantA;
    lambda_struct.nu = nvar + BayesmConstantnuInc;
    lambda_struct.V = lambda_struct.nu * eye(nvar, nvar);
    
    mat Deltadraw;
    if (drawdelta) Deltadraw = zeros<mat>(R / keep, nz * nvar);
    else Deltadraw = zeros<mat>(1, 1);
    arma::cube betadraw(nlgt, nvar, R / keep);
    vec probdraw = zeros<vec>(R / keep);
    vec oldll = zeros<vec>(nlgt);
    vec loglike = zeros<vec>(R / keep);
    vec Istardraw = zeros<vec>(R / keep);
    vec alphadraw = zeros<vec>(R / keep);
    vec nudraw = zeros<vec>(R / keep);
    vec vdraw = zeros<vec>(R / keep);
    vec adraw = zeros<vec>(R / keep);
    std::vector<std::vector<murooti_mnldp>> compdraw(R / keep);
    
    int mkeep = 0;
    for (int rep = 0; rep < R; rep++) {
        mat y_dp;
        if (drawdelta) {
            olddelta.reshape(nvar, nz);
            y_dp = oldbetas - Z * olddelta.t();
        } else {
            y_dp = oldbetas;
        }
        
        auto mgout = rDPGibbs1_local(y_dp, lambda_struct, thetaStar_vector, maxuniq, indic, alpha, priorAlpha_struct, gridsize, alim, nulim, vlim);
        indic = mgout.indic;
        lambda_struct = mgout.lambda_struct;
        alpha = mgout.alpha;
        thetaStar_vector = mgout.thetaStar_vector;
        int Istar = thetaStar_vector.size();
        
        if (drawdelta) {
            olddelta = drawDelta_mnldp(Z, oldbetas, indic, thetaStar_vector, deltabar, Ad);
        }
        
        for (int lgt = 0; lgt < nlgt; lgt++) {
            auto& theta = thetaStar_vector[indic[lgt] - 1];
            mat rootpi = theta.rooti;
            vec betabar;
            if (drawdelta) {
                olddelta.reshape(nvar, nz);
                betabar = theta.mu + olddelta * Z.row(lgt).t();
            } else {
                betabar = theta.mu;
            }
            
            if (rep == 0) {
                oldll[lgt] = llmnl_local(oldbetas.row(lgt).t(), lgtdata_vector[lgt].y, lgtdata_vector[lgt].X);
            }
            
            mat hessSym = 0.5 * lgtdata_vector[lgt].hess + 0.5 * lgtdata_vector[lgt].hess.t();
            mat combo = hessSym + rootpi * rootpi.t();
            // Ensure positive definiteness by adding small diagonal if needed
            double minEig = eig_sym(combo).min();
            if (minEig < 1e-6) {
                combo = combo + (std::abs(minEig) + 0.01) * eye(nvar, nvar);
            }
            mat ucholinv = solve(trimatu(chol(combo)), eye(nvar, nvar));
            mat incroot = chol(ucholinv * ucholinv.t());
            
            auto metropout = mnlMetropOnce_mnldp(lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, oldbetas.row(lgt).t(),
                                                  oldll[lgt], s, incroot, betabar, rootpi);
            oldbetas.row(lgt) = metropout.betadraw.t();
            oldll[lgt] = metropout.oldll;
        }
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            betadraw.slice(mkeep - 1) = oldbetas;
            probdraw[mkeep - 1] = 1.0;
            alphadraw[mkeep - 1] = alpha;
            Istardraw[mkeep - 1] = Istar;
            adraw[mkeep - 1] = lambda_struct.Amu;
            nudraw[mkeep - 1] = lambda_struct.nu;
            vdraw[mkeep - 1] = lambda_struct.V(0, 0) / lambda_struct.nu;
            loglike[mkeep - 1] = sum(oldll);
            if (drawdelta) {
                olddelta.reshape(nz * nvar, 1);
                Deltadraw.row(mkeep - 1) = olddelta.t();
            }
            compdraw[mkeep - 1] = mgout.thetaNp1_vector;
        }
    }
    
    return std::make_tuple(Deltadraw, betadraw, probdraw, loglike, alphadraw, Istardraw, adraw, nudraw, vdraw, compdraw);
}
