// rivDP - Linear IV Model with Dirichlet Process prior
#include "bayesm.h"
#include <tuple>
#include <vector>

struct murooti_iv { vec mu; mat rooti; };
struct lambda_iv { vec mubar; double Amu; double nu; mat V; };
struct priorAlpha_iv { double power; double alphamin; double alphamax; int n; };
struct thetaStarIndex_iv { ivec indic; std::vector<murooti_iv> thetaStar_vector; };
struct DPOut_iv { ivec indic; std::vector<murooti_iv> thetaStar_vector; std::vector<murooti_iv> thetaNp1_vector; double alpha; int Istar; lambda_iv lambda_struct; };
struct ytxtxtd { vec yt; mat xt; mat xtd; };

namespace {

int rmultinomF_iv(vec const& p) {
    vec csp = cumsum(p);
    double rnd = R::runif(0, 1);
    int res = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (rnd > csp[i]) res++;
    }
    return res + 1;
}

vec breg_local(vec const& y, mat const& X, vec const& betabar, mat const& A) {
    int k = X.n_cols;
    mat XpX = X.t() * X;
    vec Xpy = X.t() * y;
    mat IR = solve(trimatu(chol(XpX + A)), eye(k, k));
    vec btilde = (IR * IR.t()) * (Xpy + A * betabar);
    vec rnorm_vec(k);
    for (int i = 0; i < k; i++) rnorm_vec[i] = R::rnorm(0, 1);
    return btilde + IR * rnorm_vec;
}

mat yden_iv(std::vector<murooti_iv> const& thetaStar_vector, mat const& y) {
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

ivec numcomp_iv(ivec const& indic, int k) {
    ivec ncomp(k);
    for (int comp = 0; comp < k; comp++) {
        ncomp[comp] = sum(indic == (comp + 1));
    }
    return ncomp;
}

std::tuple<mat, mat, mat, mat> rwishart_iv(double nu, mat const& V) {
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

std::tuple<mat, mat> rmultireg_iv(mat const& Y, mat const& X, mat const& Bbar, mat const& A, double nu, mat const& V) {
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
    auto [W, IW, C, CI] = rwishart_iv(nu + n, Vinv);
    mat Sigma = IW;
    mat root_Sigma = chol(Sigma, "lower");
    mat rnorm_mat(k, m);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++) rnorm_mat(i, j) = R::rnorm(0, 1);
    mat B = Btilde + IR * rnorm_mat * root_Sigma.t();
    return std::make_tuple(B, Sigma);
}

murooti_iv thetaD_iv(mat const& y, lambda_iv const& lam) {
    mat X = ones<mat>(y.n_rows, 1);
    mat A(1, 1); A.fill(lam.Amu);
    auto [B, Sigma] = rmultireg_iv(y, X, lam.mubar.t(), A, lam.nu, lam.V);
    murooti_iv out;
    out.mu = B.as_col();
    out.rooti = solve(trimatu(chol(Sigma)), eye(y.n_cols, y.n_cols));
    return out;
}

vec q0_iv(mat const& y, lambda_iv const& lam) {
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

thetaStarIndex_iv thetaStarDraw_iv(ivec indic, std::vector<murooti_iv> thetaStar_vector,
                                    mat const& y, mat ydenmat, vec const& q0v, double alpha,
                                    lambda_iv const& lam, int maxuniq) {
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
        ivec ncomp = numcomp_iv(indicmi, k);
        for (int comp = 0; comp < k; comp++) {
            probs[comp] = ydenmat(comp, i) * ncomp[comp] / (alpha + (n - 1));
        }
        probs = probs / sum(probs);
        indic[i] = rmultinomF_iv(probs);
        if (indic[i] == (k + 1)) {
            if ((k + 1) > maxuniq) throw std::runtime_error("max number of comps exceeded");
            murooti_iv newtheta = thetaD_iv(y.row(i), lam);
            thetaStar_vector.push_back(newtheta);
            std::vector<murooti_iv> listofone(1);
            listofone[0] = newtheta;
            ydenmat.row(k) = yden_iv(listofone, y).row(0);
        }
    }
    int k = thetaStar_vector.size();
    ivec indicC = zeros<ivec>(n);
    ivec ncomp = numcomp_iv(indic, k);
    std::vector<murooti_iv> thetaStarC_vector;
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
    thetaStarIndex_iv out;
    out.indic = indicC;
    out.thetaStar_vector = thetaStarC_vector;
    return out;
}

vec seq_iv(double from, double to, int len) {
    vec res(len);
    res[len - 1] = to;
    res[0] = from;
    double increment = (res[len - 1] - res[0]) / (len - 1);
    for (int i = 1; i < (len - 1); i++) res[i] = res[i - 1] + increment;
    return res;
}

double alphaD_iv(priorAlpha_iv const& pa, int Istar, int gridsize) {
    vec alpha = seq_iv(pa.alphamin, pa.alphamax - 0.000001, gridsize);
    vec lnprob(gridsize);
    for (int i = 0; i < gridsize; i++) {
        lnprob[i] = Istar * log(alpha[i]) + lgamma(alpha[i]) - lgamma(pa.n + alpha[i])
                    + pa.power * log(1 - (alpha[i] - pa.alphamin) / (pa.alphamax - pa.alphamin));
    }
    lnprob = lnprob - median(lnprob);
    vec probs = exp(lnprob);
    probs = probs / sum(probs);
    return alpha(rmultinomF_iv(probs) - 1);
}

murooti_iv GD_iv(lambda_iv const& lam) {
    int k = lam.mubar.size();
    mat Vinv = solve(trimatu(lam.V), eye(k, k));
    auto [W, IW, C, CI] = rwishart_iv(lam.nu, Vinv);
    mat Sigma = IW;
    mat root = chol(Sigma);
    vec draws(k);
    for (int i = 0; i < k; i++) draws[i] = R::rnorm(0, 1);
    vec mu = lam.mubar + (1 / sqrt(lam.Amu)) * root.t() * draws;
    murooti_iv out;
    out.mu = mu;
    out.rooti = solve(trimatu(root), eye(k, k));
    return out;
}

lambda_iv lambdaD_iv(lambda_iv const& lam, std::vector<murooti_iv> const& thetaStar_vector,
                     vec const& alim, vec const& nulim, vec const& vlim, int gridsize) {
    int d = thetaStar_vector[0].mu.size();
    int Istar = thetaStar_vector.size();
    vec aseq = seq_iv(alim[0], alim[1], gridsize);
    vec nuseq = d - 1 + exp(seq_iv(nulim[0], nulim[1], gridsize));
    vec vseq = seq_iv(vlim[0], vlim[1], gridsize);
    
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
    double adraw = aseq[rmultinomF_iv(probs) - 1];
    
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
    double nudraw = nuseq[rmultinomF_iv(probs) - 1];
    
    lnprob = Istar * nudraw * d * log(sqrt(vseq * nudraw)) - 0.5 * sumdiagriri * vseq * nudraw;
    lnprob = lnprob - max(lnprob) + 200;
    probs = exp(lnprob);
    probs = probs / sum(probs);
    double vdraw = vseq[rmultinomF_iv(probs) - 1];
    
    lambda_iv out;
    out.mubar = zeros<vec>(d);
    out.Amu = adraw;
    out.nu = nudraw;
    out.V = nudraw * vdraw * eye(d, d);
    return out;
}

ytxtxtd get_ytxt_local(vec const& y, mat const& z, mat const& delta, vec const& x, mat const& w,
                        int ncomp, ivec const& indic, std::vector<murooti_iv> const& thetaStar_vector, bool isgamma) {
    int dimz = z.n_cols;
    vec yt;
    mat xt;
    for (int k = 0; k < ncomp; k++) {
        uvec ind = find(indic == (k + 1));
        if (ind.size() > 0) {
            mat zk = z.rows(ind);
            vec yk = y(ind);
            vec xk = x(ind);
            murooti_iv theta = thetaStar_vector[k];
            vec mu = theta.mu;
            mat rooti = theta.rooti;
            mat Sigma = solve(rooti, eye(2, 2));
            Sigma = Sigma.t() * Sigma;
            vec e1 = xk - zk * delta;
            vec ee2 = mu[1] + (Sigma(0, 1) / Sigma(0, 0)) * (e1 - mu[0]);
            double sig = sqrt(Sigma(1, 1) - pow(Sigma(0, 1), 2.0) / Sigma(0, 0));
            yt = join_cols(yt, (yk - ee2) / sig);
            if (isgamma) {
                mat wk = w.rows(ind);
                xt = join_cols(xt, join_rows(xk, wk) / sig);
            } else {
                xt = join_cols(xt, xk / sig);
            }
        }
    }
    ytxtxtd out;
    out.yt = yt;
    out.xt = xt;
    return out;
}

ytxtxtd get_ytxtd_local(vec const& y, mat const& z, double beta, vec const& gamma, vec const& x, mat const& w,
                         int ncomp, ivec const& indic, std::vector<murooti_iv> const& thetaStar_vector, int dimd, bool isgamma) {
    vec yt;
    mat xtd;
    mat C = eye(2, 2);
    C(1, 0) = beta;
    for (int k = 0; k < ncomp; k++) {
        uvec ind = find(indic == (k + 1));
        int indsize = ind.size();
        if (indsize > 0) {
            mat zk = z.rows(ind);
            vec yk = y(ind);
            vec xk = x(ind);
            murooti_iv theta = thetaStar_vector[k];
            vec mu = theta.mu;
            mat rooti = theta.rooti;
            mat Sigma = solve(rooti, eye(2, 2));
            Sigma = Sigma.t() * Sigma;
            mat B = C * Sigma * C.t();
            mat L = chol(B, "lower");
            mat Li = solve(trimatl(L), eye(2, 2));
            vec u;
            if (isgamma) {
                mat wk = w.rows(ind);
                u = yk - wk * gamma - mu[1] - beta * mu[0];
            } else {
                u = yk - mu[1] - beta * mu[0];
            }
            mat temp = join_cols((xk - mu[0]).t(), u.t());
            vec ytk = vectorise(Li * temp);
            vec zveck = vectorise(zk.t());
            mat z2 = join_rows(zveck, beta * zveck).t();
            z2 = Li * z2;
            rowvec zt1 = z2.row(0);
            rowvec zt2 = z2.row(1);
            mat zt1m = reshape(zt1, dimd, indsize).t();
            mat zt2m = reshape(zt2, dimd, indsize).t();
            mat xtdk(2 * indsize, dimd);
            for (int i = 0; i < indsize; i++) {
                xtdk.row(2 * i) = zt1m.row(i);
                xtdk.row(2 * i + 1) = zt2m.row(i);
            }
            yt = join_cols(yt, ytk);
            xtd = join_cols(xtd, xtdk);
        }
    }
    ytxtxtd out;
    out.yt = yt;
    out.xtd = xtd;
    return out;
}

DPOut_iv rthetaDP_local(int maxuniq, double alpha, lambda_iv lam, priorAlpha_iv const& pa,
                         std::vector<murooti_iv> thetaStar_vector, ivec indic, vec const& q0v, mat const& y,
                         int gridsize, vec const& alim, vec const& nulim, vec const& vlim) {
    int n = y.n_rows;
    int dimy = y.n_cols;
    int nunique = thetaStar_vector.size();
    if (nunique > maxuniq) throw std::runtime_error("max number of unique thetas exceeded");
    mat ydenmat = zeros<mat>(maxuniq, n);
    ydenmat.rows(0, nunique - 1) = yden_iv(thetaStar_vector, y);
    auto thetaStarOut = thetaStarDraw_iv(indic, thetaStar_vector, y, ydenmat, q0v, alpha, lam, maxuniq);
    thetaStar_vector = thetaStarOut.thetaStar_vector;
    indic = thetaStarOut.indic;
    nunique = thetaStar_vector.size();
    
    vec probs = zeros<vec>(nunique + 1);
    uvec spanall(dimy);
    for (int i = 0; i < dimy; i++) spanall[i] = i;
    for (int j = 0; j < nunique; j++) {
        uvec ind_j = find(indic == (j + 1));
        probs[j] = ind_j.size() / (alpha + n + 0.0);
        thetaStar_vector[j] = thetaD_iv(y.rows(ind_j), lam);
    }
    probs[nunique] = alpha / (alpha + n + 0.0);
    int ind = rmultinomF_iv(probs);
    
    std::vector<murooti_iv> thetaNp1_vector(1);
    if (ind == (int)probs.size()) {
        thetaNp1_vector[0] = GD_iv(lam);
    } else {
        thetaNp1_vector[0] = thetaStar_vector[ind - 1];
    }
    
    alpha = alphaD_iv(pa, nunique, gridsize);
    lam = lambdaD_iv(lam, thetaStar_vector, alim, nulim, vlim, gridsize);
    
    DPOut_iv out;
    out.indic = indic;
    out.thetaStar_vector = thetaStar_vector;
    out.thetaNp1_vector = thetaNp1_vector;
    out.alpha = alpha;
    out.Istar = nunique;
    out.lambda_struct = lam;
    return out;
}

} // anonymous namespace

std::tuple<mat, vec, vec, vec, mat, vec, vec, vec, std::vector<std::vector<murooti_iv>>>
rivDP_rcpp_loop(int R, int keep, int dimd, vec const& mbg, mat const& Abg, vec const& md, mat const& Ad,
                vec const& y, bool isgamma, mat const& z, vec const& x, mat const& w, vec delta,
                double power, double alphamin, double alphamax, int n_prior, int gridsize,
                bool SCALE, int maxuniq, double scalex, double scaley,
                vec const& alim, vec const& nulim, vec const& vlim,
                double BayesmConstantA, int BayesmConstantnu) {

    int n = y.size();
    int dimg = isgamma ? w.n_cols : 1;
    
    ivec indic = ones<ivec>(n);
    std::vector<murooti_iv> thetaStar_vector(1);
    thetaStar_vector[0].mu = zeros<vec>(2);
    thetaStar_vector[0].rooti = eye(2, 2);
    
    lambda_iv lambda_struct;
    lambda_struct.mubar = zeros<vec>(2);
    lambda_struct.Amu = BayesmConstantA;
    lambda_struct.nu = BayesmConstantnu;
    lambda_struct.V = lambda_struct.nu * eye(2, 2);
    
    priorAlpha_iv priorAlpha_struct;
    priorAlpha_struct.power = power;
    priorAlpha_struct.alphamin = alphamin;
    priorAlpha_struct.alphamax = alphamax;
    priorAlpha_struct.n = n_prior;
    
    int ncomp = 1;
    double alpha = 1.0;
    
    mat deltadraw = zeros<mat>(R / keep, dimd);
    vec betadraw = zeros<vec>(R / keep);
    vec alphadraw = zeros<vec>(R / keep);
    vec Istardraw = zeros<vec>(R / keep);
    mat gammadraw = zeros<mat>(R / keep, dimg);
    vec nudraw = zeros<vec>(R / keep);
    vec vdraw = zeros<vec>(R / keep);
    vec adraw = zeros<vec>(R / keep);
    std::vector<std::vector<murooti_iv>> thetaNp1draw(R / keep);
    
    vec gammaVec = zeros<vec>(dimg);
    
    int mkeep = 0;
    for (int rep = 0; rep < R; rep++) {
        auto out1 = get_ytxt_local(y, z, delta, x, w, ncomp, indic, thetaStar_vector, isgamma);
        vec bg = breg_local(out1.yt, out1.xt, mbg, Abg);
        double beta = bg[0];
        if (isgamma) gammaVec = bg.subvec(1, bg.size() - 1);
        
        auto out2 = get_ytxtd_local(y, z, beta, gammaVec, x, w, ncomp, indic, thetaStar_vector, dimd, isgamma);
        delta = breg_local(out2.yt, out2.xtd, md, Ad);
        
        mat errMat;
        if (isgamma) {
            errMat = join_rows(x - z * delta, y - beta * x - w * gammaVec);
        } else {
            errMat = join_rows(x - z * delta, y - beta * x);
        }
        
        vec q0v = q0_iv(errMat, lambda_struct);
        auto DPout = rthetaDP_local(maxuniq, alpha, lambda_struct, priorAlpha_struct,
                                     thetaStar_vector, indic, q0v, errMat, gridsize, alim, nulim, vlim);
        
        indic = DPout.indic;
        thetaStar_vector = DPout.thetaStar_vector;
        alpha = DPout.alpha;
        int Istar = DPout.Istar;
        ncomp = thetaStar_vector.size();
        lambda_struct = DPout.lambda_struct;
        
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            deltadraw.row(mkeep - 1) = delta.t();
            betadraw[mkeep - 1] = beta;
            alphadraw[mkeep - 1] = alpha;
            Istardraw[mkeep - 1] = Istar;
            if (isgamma) gammadraw.row(mkeep - 1) = gammaVec.t();
            adraw[mkeep - 1] = lambda_struct.Amu;
            nudraw[mkeep - 1] = lambda_struct.nu;
            vdraw[mkeep - 1] = lambda_struct.V(0, 0) / lambda_struct.nu;
            thetaNp1draw[mkeep - 1] = DPout.thetaNp1_vector;
        }
    }
    
    if (SCALE) {
        deltadraw = deltadraw * scalex;
        betadraw = betadraw * scaley / scalex;
        if (isgamma) gammadraw = gammadraw * scaley;
    }
    
    return std::make_tuple(deltadraw, betadraw, alphadraw, Istardraw, gammadraw, adraw, nudraw, vdraw, thetaNp1draw);
}
