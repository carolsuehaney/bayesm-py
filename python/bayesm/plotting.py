"""Plotting functions for bayesm MCMC output."""

import numpy as np
import matplotlib.pyplot as plt
from .num_eff import num_eff
from .mom_mix import mom_mix
from .mix_den import mix_den_bi
from .e_mix_marg_den import e_mix_marg_den


def plot_mat(X, names=None, burnin=None, tvalues=None, traceplot=True, 
             density=True, intervals=True, check_ndraws=True):
    """
    Plot histograms and traceplots for MCMC draws.
    
    Parameters
    ----------
    X : array (R, k)
        Matrix of MCMC draws
    names : list of str, optional
        Names for each variable
    burnin : int, optional
        Burn-in period (default: 10% of draws)
    tvalues : array (k,), optional
        True values to mark on plots
    traceplot : bool
        Whether to show traceplots and ACF
    density : bool
        Whether to plot density (vs frequency)
    intervals : bool
        Whether to show credible intervals
    check_ndraws : bool
        Whether to check for minimum draws
    """
    X = np.atleast_2d(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    R, k = X.shape
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if check_ndraws and R < 100:
        print("fewer than 100 draws submitted")
        return
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return
    
    if names is None:
        names = [str(i+1) for i in range(k)]
    
    X_post = X[burnin:, :]
    
    ncols = min(k, 2)
    nrows = (k + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(k):
        ax = axes[i]
        ax.hist(X_post[:, i], bins=30, density=density, color='magenta', alpha=0.7)
        ax.set_title(names[i])
        ax.set_ylabel('density' if density else 'freq')
        
        if tvalues is not None:
            ax.axvline(tvalues[i], color='blue', linewidth=2)
        
        if intervals:
            q025, q975 = np.quantile(X_post[:, i], [0.025, 0.975])
            mean = np.mean(X_post[:, i])
            neff = num_eff(X_post[:, i])
            se = neff['stderr']
            
            ax.axvline(q025, color='green', linewidth=2, linestyle='--')
            ax.axvline(q975, color='green', linewidth=2, linestyle='--')
            ax.axvline(mean, color='red', linewidth=2)
            if se is not None and not np.isnan(se):
                ax.axvline(mean - 2*se, color='yellow', linewidth=1.5)
                ax.axvline(mean + 2*se, color='yellow', linewidth=1.5)
    
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    if traceplot:
        fig, axes = plt.subplots(k, 2, figsize=(10, 3*k))
        if k == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(k):
            axes[i, 0].plot(X[:, i], color='red', linewidth=0.5)
            axes[i, 0].set_title(names[i])
            if tvalues is not None:
                axes[i, 0].axhline(tvalues[i], color='blue', linewidth=2)
            
            if np.var(X[:, i]) > 1e-20:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(X[:, i], ax=axes[i, 1], lags=40)
                axes[i, 1].set_title('')
            else:
                axes[i, 1].text(0.5, 0.5, 'No ACF (constant)', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()


def plot_nmix(nmixlist, names=None, burnin=None, grid=None, bi_sel=None, 
              nstd=2, marg=True, data=None, ngrid=50, ndraw=200):
    """
    Plot normal mixture marginal and bivariate densities.
    
    Parameters
    ----------
    nmixlist : dict or list
        Output from rnmixGibbs with 'probdraw' and 'compdraw'
    names : list of str, optional
        Variable names
    burnin : int, optional
        Burn-in (default: 10% of draws)
    grid : array (ngrid, d), optional
        Grid points for evaluation
    bi_sel : list of tuples, optional
        Pairs of variables for bivariate plots (default: [(0,1)])
    nstd : float
        Number of std devs for grid range
    marg : bool
        Whether to plot marginals
    data : array (n, d), optional
        Data for histogram overlay
    ngrid : int
        Number of grid points
    ndraw : int
        Number of draws to use for averaging
    """
    if isinstance(nmixlist, dict):
        probdraw = nmixlist['probdraw']
        compdraw = nmixlist['compdraw']
    else:
        probdraw = nmixlist[0]
        compdraw = nmixlist[2]
    
    R = probdraw.shape[0]
    
    if R < 100:
        print("fewer than 100 draws submitted")
        return
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return
    
    datad = len(compdraw[0][0]['mu'])
    
    if bi_sel is None:
        bi_sel = [(0, 1)] if datad > 1 else []
    
    ind = np.linspace(burnin, R-1, min(ndraw, int(0.05*R))).astype(int)
    
    if names is None:
        names = [str(i+1) for i in range(datad)]
    
    if grid is None:
        grid = np.zeros((ngrid, datad))
        if data is not None:
            for i in range(datad):
                grid[:, i] = np.linspace(data[:, i].min(), data[:, i].max(), ngrid)
        else:
            out = mom_mix(probdraw[ind, :], [compdraw[i] for i in ind])
            mu = out['mu']
            sd = out['sd']
            for i in range(datad):
                grid[:, i] = np.linspace(mu[i] - nstd*sd[i], mu[i] + nstd*sd[i], ngrid)
    
    if marg:
        mden = e_mix_marg_den(grid, probdraw[ind, :], [compdraw[i] for i in ind])
        
        ncols = min(datad, 2)
        nrows = (datad + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows))
        axes = np.atleast_1d(axes).flatten()
        
        for i in range(datad):
            ax = axes[i]
            if data is not None:
                deltax = (grid[-1, i] - grid[0, i]) / ngrid
                ax.hist(data[:, i], bins=max(20, int(0.1*len(data))), 
                        density=True, color='yellow', alpha=0.7)
                ax.plot(grid[:, i], mden[:, i] / (np.sum(mden[:, i]) * deltax), 
                        color='red', linewidth=2)
            else:
                ax.fill_between(grid[:, i], 0, mden[:, i], color='magenta', alpha=0.7)
                ax.plot(grid[:, i], mden[:, i], color='black', linewidth=2)
            ax.set_title(names[i])
            ax.set_ylabel('density')
        
        for i in range(datad, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    if datad > 1 and bi_sel:
        for sel in bi_sel:
            i, j = sel
            xi = grid[:, i]
            xj = grid[:, j]
            den = np.zeros((ngrid, ngrid))
            
            for elt in ind:
                den += mix_den_bi(i, j, xi, xj, probdraw[elt, :], compdraw[elt])
            den /= np.sum(den)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(den.T, origin='lower', aspect='auto',
                          extent=[xi.min(), xi.max(), xj.min(), xj.max()],
                          cmap='terrain')
            ax.contour(xi, xj, den.T, colors='black', linewidths=0.5)
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()


def plot_hcoef(X, names=None, burnin=None, nsample=30):
    """
    Plot hierarchical coefficient draws.
    
    Parameters
    ----------
    X : array (nunits, nvar, R)
        3D array of coefficient draws
    names : list of str, optional
        Variable names
    burnin : int, optional
        Burn-in (default: 10% of draws)
    nsample : int
        Number of units to sample for boxplots
    """
    if X.ndim != 3:
        print("Requires 3-dim array")
        return
    
    nunits, nvar, R = X.shape
    
    if burnin is None:
        burnin = int(0.1 * R)
    
    if R < 100:
        print("fewer than 100 draws submitted")
        return
    
    if burnin > R:
        print("burnin set larger than number of draws submitted")
        return
    
    if names is None:
        names = [str(i+1) for i in range(nvar)]
    
    rsam = np.sort(np.random.choice(nunits, min(nsample, nunits), replace=False))
    
    for var in range(nvar):
        fig, ax = plt.subplots(figsize=(12, 4))
        data = X[rsam, var, burnin:].T
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('magenta')
        ax.set_xticklabels([str(r) for r in rsam], rotation=90)
        ax.set_xlabel('Cross-sectional Unit')
        ax.set_title(f'Var {names[var]} Coefficient')
        plt.tight_layout()
        plt.show()
    
    pmeans = np.mean(X[:, :, burnin:], axis=2)
    plot_mat(pmeans, names=[f'Posterior Means of Coef {n}' for n in names],
             traceplot=False, intervals=False, density=False, check_ndraws=False)
