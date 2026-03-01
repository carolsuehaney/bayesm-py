// rbayesBLP - BLP Demand Estimation with Random Coefficients
#include "bayesm.h"
#include <tuple>

namespace {

mat r2Sigma_local(vec const& r, int K) {
    mat L = zeros<mat>(K, K);
    L.diag() = exp(r.subvec(0, K - 1));
    int k = 0;
    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            L(j, i) = r[K + k];
            k++;
        }
    }
    return L * L.t();
}

double logJacob_local(mat const& choiceProb, int J) {
    int H = choiceProb.n_cols;
    int T = choiceProb.n_rows / J;
    
    mat onesJJ = ones<mat>(J, J);
    mat struc = zeros<mat>(J * T, J * T);
    for (int t = 0; t < T; t++) {
        struc.submat(t * J, t * J, t * J + J - 1, t * J + J - 1) = onesJJ;
    }
    struc = struc - eye<mat>(T * J, T * J);
    mat offDiag = -choiceProb * choiceProb.t() / H;
    mat Jac = struc % offDiag;
    Jac.diag() = sum(choiceProb % (1 - choiceProb), 1) / H;
    
    double sumlogJacob = 0;
    for (int t = 0; t < T; t++) {
        mat blockMat = Jac.submat(t * J, t * J, (t + 1) * J - 1, (t + 1) * J - 1);
        double detblockMat = det(blockMat);
        sumlogJacob += log(sqrt(detblockMat * detblockMat));
    }
    return -sumlogJacob;
}

mat share2mu_local(mat const& Sigma, mat const& X, mat const& v, vec const& share, int J, double tol) {
    int H = v.n_cols;
    int T = X.n_rows / J;
    int K = X.n_cols;
    
    mat Sigma_safe = 0.5 * (Sigma + Sigma.t());
    vec eigval_Sigma;
    eig_sym(eigval_Sigma, Sigma_safe);
    double minEig_Sigma = eigval_Sigma.min();
    if (minEig_Sigma < 1e-8) {
        Sigma_safe = Sigma_safe + (std::abs(minEig_Sigma) + 1e-6) * eye(K, K);
    }
    mat u = X * (chol(Sigma_safe).t() * v);
    vec mu0 = ones<vec>(J * T);
    vec mu1 = mu0 / 2;
    
    vec rel = (mu1 - mu0) / mu0;
    double max_rel = max(abs(rel));
    
    mat temp2(T * J, H);
    mat choiceProb;
    
    while (max_rel > tol) {
        mu0 = mu1;
        mat expU = exp(u.each_col() + mu0);
        
        mat temp1 = reshape(expU, J, T * H);
        rowvec expSum_row = 1 + sum(temp1, 0);
        mat expSum_mat = reshape(expSum_row, T, H);
        
        for (int t = 0; t < T; t++) {
            temp2.rows(t * J, t * J + J - 1) = ones<vec>(J) * expSum_mat.row(t);
        }
        
        choiceProb = expU / temp2;
        vec share_hat = sum(choiceProb, 1) / H;
        
        mu1 = mu0 + log(share / share_hat);
        rel = (mu0 - mu1) / mu0;
        max_rel = max(abs(rel));
    }
    
    mat rtn = zeros(J * T, H + 1);
    rtn.col(0) = mu1;
    rtn.cols(1, H) = choiceProb;
    return rtn;
}

vec breg_local(vec const& y, mat const& X, vec const& betabar, mat const& A) {
    int k = X.n_cols;
    mat XpX = X.t() * X;
    vec Xpy = X.t() * y;
    mat combo = XpX + A;
    combo = 0.5 * (combo + combo.t());
    vec eigval_combo;
    eig_sym(eigval_combo, combo);
    double minEig_combo = eigval_combo.min();
    if (minEig_combo < 1e-8) {
        combo = combo + (std::abs(minEig_combo) + 1e-6) * eye(k, k);
    }
    mat IR = solve(trimatu(chol(combo)), eye(k, k));
    vec btilde = (IR * IR.t()) * (Xpy + A * betabar);
    vec rnorm_vec(k);
    for (int i = 0; i < k; i++) rnorm_vec[i] = R::rnorm(0, 1);
    return btilde + IR * rnorm_vec;
}

std::tuple<mat, mat, mat, mat> rwishart_local(double nu, mat const& V) {
    int m = V.n_rows;
    mat T = zeros<mat>(m, m);
    for (int i = 0; i < m; i++) T(i, i) = std::sqrt(R::rchisq(nu - i));
    for (int i = 1; i < m; i++)
        for (int j = 0; j < i; j++) T(i, j) = R::rnorm(0, 1);
    
    mat Vsym = 0.5 * (V + V.t());
    vec eigval_V;
    eig_sym(eigval_V, Vsym);
    double minEig_V = eigval_V.min();
    if (minEig_V < 1e-8) {
        Vsym = Vsym + (std::abs(minEig_V) + 1e-6) * eye(m, m);
    }
    mat C = chol(Vsym, "lower");
    mat CT = C * T;
    mat W = CT * CT.t();
    W = 0.5 * (W + W.t());
    vec eigval_W;
    eig_sym(eigval_W, W);
    double minEig_W = eigval_W.min();
    if (minEig_W < 1e-8) {
        W = W + (std::abs(minEig_W) + 1e-6) * eye(m, m);
    }
    mat IW = solve(W, eye<mat>(m, m));
    mat Cupper = chol(W);
    mat CI = solve(trimatu(Cupper), eye<mat>(m, m));
    return std::make_tuple(W, IW, Cupper, CI);
}

double lndMvn_local(vec const& x, vec const& mu, mat const& rooti) {
    vec z = rooti.t() * (x - mu);
    return -(x.size() / 2.0) * log(2 * M_PI) + sum(log(rooti.diag())) - 0.5 * dot(z, z);
}

std::tuple<vec, vec, mat> rivDraw_local(vec const& mu, vec const& Xend, mat const& z, mat const& Xexo,
                                         vec const& theta_hat, mat const& A, vec const& deltabar, mat const& Ad,
                                         mat const& V, double nu, vec const& delta_old, mat const& Omega_old) {
    int n = mu.size();
    int dimd = z.n_cols;
    int dimg = Xexo.n_cols;
    
    mat C = eye(2, 2);
    mat Omega = Omega_old;
    vec delta = delta_old;
    
    // Draw beta, gamma
    vec e1 = Xend - z * delta;
    vec ee2 = (Omega(0, 1) / Omega(0, 0)) * e1;
    double sig = sqrt(Omega(1, 1) - (Omega(0, 1) * Omega(0, 1)) / Omega(0, 0));
    vec mut = (mu - ee2) / sig;
    mat xt = join_rows(Xend, Xexo) / sig;
    vec bg = breg_local(mut, xt, theta_hat, A);
    double theta1 = bg[0];
    vec theta2 = bg.subvec(1, bg.size() - 1);
    
    // Draw delta
    C(1, 0) = theta1;
    mat B = C * Omega * C.t();
    B = 0.5 * (B + B.t());  // ensure symmetric
    vec eigval_B;
    eig_sym(eigval_B, B);
    double minEig_B = eigval_B.min();
    if (minEig_B < 1e-8) {
        B = B + (std::abs(minEig_B) + 1e-6) * eye(2, 2);
    }
    mat L = chol(B, "lower");
    mat Li = solve(trimatl(L), eye(2, 2));
    vec u = mu - Xexo * theta2;
    mat temp = join_rows(Xend, u);
    mut = vectorise(Li * temp.t());
    
    vec zvec = vectorise(z.t());
    mat z2 = join_rows(zvec, theta1 * zvec).t();
    z2 = Li * z2;
    rowvec zt1 = z2.row(0);
    rowvec zt2 = z2.row(1);
    mat zt1m = reshape(zt1, dimd, n).t();
    mat zt2m = reshape(zt2, dimd, n).t();
    
    mat xtd(2 * n, dimd);
    for (int i = 0; i < n; i++) {
        xtd.row(2 * i) = zt1m.row(i);
        xtd.row(2 * i + 1) = zt2m.row(i);
    }
    delta = breg_local(mut, xtd, deltabar, Ad);
    
    // Draw Sigma (Omega)
    mat Res = join_rows(Xend - z * delta, mu - theta1 * Xend - Xexo * theta2);
    mat S = Res.t() * Res;
    mat VS = V + S;
    VS = 0.5 * (VS + VS.t());
    vec eigval_VS;
    eig_sym(eigval_VS, VS);
    double minEig_VS = eigval_VS.min();
    if (minEig_VS < 1e-8) {
        VS = VS + (std::abs(minEig_VS) + 1e-6) * eye(2, 2);
    }
    mat ucholinv = solve(trimatu(chol(VS)), eye(2, 2));
    mat VSinv = ucholinv * ucholinv.t();
    auto [W, IW, Cupper, CI] = rwishart_local(nu + n, VSinv);
    Omega = IW;
    
    vec thetabar(dimg + 1);
    thetabar.subvec(0, dimg - 1) = theta2;
    thetabar[dimg] = theta1;
    
    return std::make_tuple(delta, thetabar, Omega);
}

} // anonymous namespace

std::tuple<vec, mat, mat, mat, mat, mat, vec, double>
rbayesBLP_rcpp_loop(bool IV, mat const& X, mat const& Z, vec const& share,
                    int J, int T, mat const& v, int R,
                    vec const& sigmasqR, mat const& A, vec const& theta_hat,
                    vec const& deltabar, mat const& Ad,
                    double nu0, double s0_sq, mat const& VOmega,
                    double ssq, mat const& cand_cov,
                    vec const& theta_bar_initial, vec const& r_initial,
                    double tau_sq_initial, mat const& Omega_initial, vec const& delta_initial,
                    double tol, int keep) {

    int K = theta_hat.size();
    int H = v.n_cols;
    int I = IV ? Z.n_cols : 1;
    
    mat Xexo, Xend_mat;
    vec Xend;
    if (IV) {
        Xexo = X.cols(0, K - 2);
        Xend = X.col(K - 1);
    }
    
    // Allocate storage
    vec tau_sq_all = IV ? zeros<vec>(1) : zeros<vec>(R / keep);
    mat Omega_all = IV ? zeros<mat>(4, R / keep) : zeros<mat>(1, 1);
    mat delta_all = IV ? zeros<mat>(I, R / keep) : zeros<mat>(1, 1);
    mat theta_bar_all = zeros<mat>(K, R / keep);
    mat r_all = zeros<mat>(K * (K + 1) / 2, R / keep);
    mat Sigma_all = zeros<mat>(K * K, R / keep);
    vec ll_all = zeros<vec>(R / keep);
    
    // Initial values
    vec theta_bar = theta_bar_initial;
    mat Omega = Omega_initial;
    vec delta = delta_initial;
    vec r_old = r_initial;
    double tau_sq = tau_sq_initial;
    mat Sigma_old = r2Sigma_local(r_old, K);
    
    // Initial mu and Jacobian via contraction mapping
    mat out_cont = share2mu_local(Sigma_old, X, v, share, J, tol);
    vec mu_old = out_cont.col(0);
    mat choiceProb = out_cont.cols(1, H);
    double sumLogJaco_old = logJacob_local(choiceProb, J);
    
    double n_accept = 0.0;
    int mkeep = 0;
    
    // Precompute chol of cand_cov with safeguard
    mat cand_cov_safe = ssq * cand_cov;
    cand_cov_safe = 0.5 * (cand_cov_safe + cand_cov_safe.t());
    vec eigval_cand;
    eig_sym(eigval_cand, cand_cov_safe);
    double minEig_cand = eigval_cand.min();
    if (minEig_cand < 1e-8) {
        cand_cov_safe = cand_cov_safe + (std::abs(minEig_cand) + 1e-6) * eye(K * (K + 1) / 2, K * (K + 1) / 2);
    }
    mat cand_chol = chol(cand_cov_safe);
    
    for (int rep = 0; rep < R; rep++) {
        // STEP 1: Draw r (for Sigma) via Metropolis-Hastings
        vec rnorm_vec(K * (K + 1) / 2);
        for (int i = 0; i < K * (K + 1) / 2; i++) rnorm_vec[i] = R::rnorm(0, 1);
        vec r_new = r_old + cand_chol.t() * rnorm_vec;
        mat Sigma_new = r2Sigma_local(r_new, K);
        
        out_cont = share2mu_local(Sigma_new, X, v, share, J, tol);
        vec mu_new = out_cont.col(0);
        choiceProb = out_cont.cols(1, H);
        
        vec eta_new = mu_new - X * theta_bar;
        vec eta_old = mu_old - X * theta_bar;
        
        double ll_old, ll_new;
        double sumLogJaco_new = logJacob_local(choiceProb, J);
        
        if (IV) {
            vec zeta = Xend - Z * delta;
            mat zetaeta_old = join_rows(zeta, eta_old);
            mat Omega_safe = 0.5 * (Omega + Omega.t());
            vec eigval_Omega;
            eig_sym(eigval_Omega, Omega_safe);
            double minEig_Omega = eigval_Omega.min();
            if (minEig_Omega < 1e-8) {
                Omega_safe = Omega_safe + (std::abs(minEig_Omega) + 1e-6) * eye(2, 2);
            }
            mat rootiOmega = solve(trimatu(chol(Omega_safe)), eye(2, 2));
            
            ll_old = 0;
            for (int jt = 0; jt < J * T; jt++) {
                ll_old += lndMvn_local(zetaeta_old.row(jt).t(), zeros<vec>(2), rootiOmega);
            }
            ll_old += sumLogJaco_old;
            
            mat zetaeta_new = join_rows(zeta, eta_new);
            ll_new = 0;
            for (int jt = 0; jt < J * T; jt++) {
                ll_new += lndMvn_local(zetaeta_new.row(jt).t(), zeros<vec>(2), rootiOmega);
            }
            ll_new += sumLogJaco_new;
        } else {
            ll_old = sum(log((1 / sqrt(2 * M_PI * tau_sq)) * exp(-(eta_old % eta_old) / (2 * tau_sq)))) + sumLogJaco_old;
            ll_new = sum(log((1 / sqrt(2 * M_PI * tau_sq)) * exp(-(eta_new % eta_new) / (2 * tau_sq)))) + sumLogJaco_new;
        }
        
        double prior_new = sum(log((1 / sqrt(2 * M_PI * sigmasqR)) % exp(-(r_new % r_new) / (2 * sigmasqR))));
        double prior_old = sum(log((1 / sqrt(2 * M_PI * sigmasqR)) % exp(-(r_old % r_old) / (2 * sigmasqR))));
        
        double alpha = std::min(1.0, exp(ll_new + prior_new - ll_old - prior_old));
        
        if (R::runif(0, 1) <= alpha) {
            r_old = r_new;
            Sigma_old = Sigma_new;
            mu_old = mu_new;
            sumLogJaco_old = sumLogJaco_new;
            n_accept++;
        }
        
        // STEP 2: Draw theta_bar & tau_sq (or Omega & delta)
        if (IV) {
            auto [delta_new, thetabar_new, Omega_new] = rivDraw_local(mu_old, Xend, Z, Xexo, theta_hat, A,
                                                                       deltabar, Ad, VOmega, nu0, delta, Omega);
            delta = delta_new;
            theta_bar = thetabar_new;
            Omega = Omega_new;
        } else {
            mat XXA = (X.t() * X) / tau_sq + A;
            XXA = 0.5 * (XXA + XXA.t());
            vec eigval_XXA;
            eig_sym(eigval_XXA, XXA);
            double minEig_XXA = eigval_XXA.min();
            if (minEig_XXA < 1e-8) {
                XXA = XXA + (std::abs(minEig_XXA) + 1e-6) * eye(K, K);
            }
            mat ucholinv = solve(trimatu(chol(XXA)), eye(K, K));
            mat XXAinv = ucholinv * ucholinv.t();
            vec theta_tilde = XXAinv * (X.t() * mu_old / tau_sq + A * theta_hat);
            vec rnorm_k(K);
            for (int i = 0; i < K; i++) rnorm_k[i] = R::rnorm(0, 1);
            theta_bar = theta_tilde + ucholinv * rnorm_k;
            
            double nu1 = nu0 + J * T;
            vec err = mu_old - X * theta_bar;
            double s1_sq = (nu0 * s0_sq + sum(err % err)) / nu1;
            vec z_vec(nu1);
            for (int i = 0; i < nu1; i++) z_vec[i] = R::rnorm(0, 1);
            tau_sq = nu1 * s1_sq / sum(z_vec % z_vec);
        }
        
        // STEP 3: Store draws
        if ((rep + 1) % keep == 0) {
            mkeep = (rep + 1) / keep;
            if (IV) {
                Omega_all.col(mkeep - 1) = vectorise(Omega);
                delta_all.col(mkeep - 1) = delta;
            } else {
                tau_sq_all[mkeep - 1] = tau_sq;
            }
            theta_bar_all.col(mkeep - 1) = theta_bar;
            r_all.col(mkeep - 1) = r_old;
            Sigma_all.col(mkeep - 1) = vectorise(r2Sigma_local(r_old, K));
            ll_all[mkeep - 1] = ll_old;
        }
    }
    
    double acceptrate = n_accept / R;
    
    return std::make_tuple(tau_sq_all, Omega_all, delta_all, theta_bar_all, r_all, Sigma_all, ll_all, acceptrate);
}
