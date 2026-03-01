"""Summary functions for bayesm MCMC output."""

import numpy as np
from .num_eff import num_eff
from .utilities import nmat
from .mom_mix import mom_mix


def summary_mat(X, names=None, burnin=None, tvalues=None, quantiles=True, trailer=True):
    """
    Compute and print posterior summaries for MCMC draws.
    
    Parameters
    ----------
    X : array (R, k)
        Matrix of MCMC draws
    names : list of str, optional
        Names for each variable
    burnin : int, optional
        Burn-in period (default: 10% of draws)
    tvalues : array (k,), optional
        True values to include in output
    quantiles : bool
        Whether to print quantiles
    trailer : bool
        Whether to print sample size info
    
    Returns
    -------
    mat : array
        Summary statistics matrix (transposed)
    """
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    
    R, k = X.shape
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if names is None:
        names = [str(i+1) for i in range(k)]
    
    if R < 100:
        print("fewer than 100 draws submitted")
        return None
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return None
    
    X_post = X[burnin:, :]
    n_post = X_post.shape[0]
    
    means = np.mean(X_post, axis=0)
    stds = np.std(X_post, axis=0)
    
    num_se = np.zeros(k)
    rel_eff = np.zeros(k)
    eff_s_size = np.zeros(k)
    
    for i in range(k):
        out = num_eff(X_post[:, i])
        if out['stderr'] is None or np.isnan(out['stderr']):
            num_se[i] = -9999
            rel_eff[i] = -9999
            eff_s_size[i] = -9999
        else:
            num_se[i] = out['stderr']
            rel_eff[i] = out['f']
            eff_s_size[i] = n_post / np.ceil(out['f'])
    
    mat = np.vstack([means, stds, num_se, rel_eff, eff_s_size])
    row_names = ['mean', 'std dev', 'num se', 'rel eff', 'sam size']
    
    if tvalues is not None:
        mat = np.vstack([tvalues, mat])
        row_names = ['true'] + row_names
    
    print("Summary of Posterior Marginal Distributions")
    print("\nMoments")
    
    header = "".ljust(12) + "".join([n[:10].rjust(12) for n in names])
    print(header)
    for i, rn in enumerate(row_names):
        row = rn.ljust(12) + "".join([f"{mat[i, j]:12.2f}" for j in range(k)])
        print(row)
    
    if quantiles:
        qmat = np.quantile(X_post, [0.025, 0.05, 0.5, 0.95, 0.975], axis=0)
        q_names = ['2.5%', '5%', '50%', '95%', '97.5%']
        
        if tvalues is not None:
            qmat = np.vstack([tvalues, qmat])
            q_names = ['true'] + q_names
        
        print("\nQuantiles")
        header = "".ljust(12) + "".join([n[:10].rjust(12) for n in names])
        print(header)
        for i, qn in enumerate(q_names):
            row = qn.ljust(12) + "".join([f"{qmat[i, j]:12.2f}" for j in range(k)])
            print(row)
    
    if trailer:
        print(f"\n   based on {n_post} valid draws (burn-in={burnin})")
    
    return mat.T


def summary_var(Vard, names=None, burnin=None, tvalues=None, quantiles=False):
    """
    Summarize draws of variance-covariance matrix.
    
    Parameters
    ----------
    Vard : array (R, d^2)
        Draws of var-cov matrix stored as flattened vectors
    names : list of str, optional
        Variable names
    burnin : int, optional
        Burn-in (default: 10% of draws)
    tvalues : array, optional
        True values
    quantiles : bool
        Whether to print quantiles
    """
    if Vard.ndim != 2:
        print("Requires matrix argument")
        return
    
    R, k = Vard.shape
    d = int(np.sqrt(k))
    
    if d * d != k:
        print("Argument cannot be draws from a square matrix")
        return
    
    if R < 100:
        print("fewer than 100 draws submitted")
        return
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return
    
    if names is None:
        names = [str(i+1) for i in range(d)]
    
    Vard_post = Vard[burnin:, :]
    
    corrd = np.apply_along_axis(nmat, 1, Vard_post)
    pmeancorr = np.mean(corrd, axis=0).reshape(d, d)
    
    diag_idx = np.arange(d) * d + np.arange(d)
    var_draws = Vard_post[:, diag_idx]
    sd_draws = np.sqrt(var_draws)
    pmeansd = np.mean(sd_draws, axis=0)
    
    mat = np.column_stack([pmeansd, pmeancorr])
    
    print("Posterior Means of Std Deviations and Correlation Matrix")
    header = "".ljust(12) + "Std Dev".rjust(12) + "".join([n[:10].rjust(12) for n in names])
    print(header)
    for i, n in enumerate(names):
        row = n.ljust(12) + f"{mat[i, 0]:12.2f}" + "".join([f"{mat[i, j+1]:12.2f}" for j in range(d)])
        print(row)
    
    print("\nUpper Triangle of Var-Cov Matrix")
    upper_idx = np.triu_indices(d)
    labels = [f"{i+1},{j+1}" for i, j in zip(upper_idx[0], upper_idx[1])]
    upper_flat_idx = upper_idx[0] * d + upper_idx[1]
    uppertri = Vard[:, upper_flat_idx]
    
    summary_mat(uppertri, names=labels, burnin=burnin, tvalues=tvalues, quantiles=quantiles)


def summary_nmix(nmixlist, names=None, burnin=None):
    """
    Summarize draws from normal mixture model.
    
    Parameters
    ----------
    nmixlist : dict or list
        Output from rnmixGibbs with 'probdraw' and 'compdraw'
    names : list of str, optional
        Variable names
    burnin : int, optional
        Burn-in (default: 10% of draws)
    """
    if isinstance(nmixlist, dict):
        probdraw = nmixlist['probdraw']
        compdraw = nmixlist['compdraw']
    else:
        probdraw = nmixlist[0]
        compdraw = nmixlist[2]
    
    R = probdraw.shape[0]
    ncomp = len(compdraw[0])
    
    if probdraw.shape[1] != ncomp:
        print("Dim of probdraw not compatible with compdraw")
        return
    
    if R < 100:
        print("fewer than 100 draws submitted")
        return
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return
    
    datad = len(compdraw[0][0]['mu'])
    
    if names is None:
        names = [str(i+1) for i in range(datad)]
    
    mumat = np.zeros((R, datad))
    sigmat = np.zeros((R, datad * datad))
    
    for i in range(burnin, R):
        if i % 500 == 0:
            print(f"processing draw {i}")
        out = mom_mix(probdraw[i:i+1, :], [compdraw[i]])
        mumat[i, :] = out['mu']
        sigmat[i, :] = out['sigma'].flatten()
    
    print("\nNormal Mixture Moments")
    print("Mean")
    summary_mat(mumat, names, burnin=burnin, quantiles=False, trailer=False)
    
    print()
    summary_var(sigmat, burnin=burnin)
    
    print("note: 1st and 2nd Moments for a Normal Mixture")
    print("      may not be interpretable, consider plots")
