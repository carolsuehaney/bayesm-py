"""
Create design matrix for MNL and MNP models

Converted from createX.R
"""

import numpy as np
from .utilities import pandterm


def create_x(p, na, nd, Xa, Xd, INT=True, DIFF=False, base=None):
    """
    Create X array in format needed for MNL and MNP routines
    
    Parameters
    ----------
    p : int
        Number of choices
    na : int or None
        Number of choice attribute variables (choice-specific characteristics)
        Use None if no attributes
    nd : int or None
        Number of "demo" variables or characteristics of choosers
        Use None if no demos
    Xa : array_like or None
        n x (na*p) matrix of choice attributes. First p cols are values of 
        attribute #1 for each of p choices, second p for attribute #2, etc.
        Use None if no attributes
    Xd : array_like or None
        n x nd matrix of values of "demo" variables
        Use None if no demos
    INT : bool, default=True
        Include intercepts
    DIFF : bool, default=False
        Difference with respect to base alternative (required for MNP)
    base : int or None
        Base alternative (default is p)
    
    Returns
    -------
    X : ndarray
        Modified X matrix with n*p rows and INT*(p-1) + nd*(p-1) + na cols
    
    Notes
    -----
    Converted from createX.R
    """
    # Check arguments
    if p is None:
        pandterm("requires p (# choice alternatives)")
    if na is None:
        pandterm("requires na arg (use na=None if none)")
    if nd is None:
        pandterm("requires nd arg (use nd=None if none)")
    if Xa is None:
        pandterm("requires Xa arg (use Xa=None if none)")
    if Xd is None:
        pandterm("requires Xd arg (use Xd=None if none)")
    if Xa is None and Xd is None:
        pandterm("both Xa and Xd None -- requires one non-null")
    
    if base is None:
        base = p
    
    # Convert to arrays
    if Xa is not None:
        Xa = np.asarray(Xa)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
    if Xd is not None:
        Xd = np.asarray(Xd)
        if Xd.ndim == 1:
            Xd = Xd.reshape(-1, 1)
    
    # Validate dimensions
    if na is not None and Xa is not None:
        if Xa.shape[1] != p * na:
            pandterm(f"bad Xa dim, dim={Xa.shape}")
    if nd is not None and Xd is not None:
        if Xd.shape[1] != nd:
            pandterm(f"ncol(Xd) ne nd, ncol(Xd)={Xd.shape[1]}")
    if Xa is not None and Xd is not None:
        if Xa.shape[0] != Xd.shape[0]:
            pandterm(f"nrow(Xa) ne nrow(Xd), nrow(Xa)={Xa.shape[0]} nrow(Xd)={Xd.shape[0]}")
    
    n = Xa.shape[0] if Xa is not None else Xd.shape[0]
    
    # Add intercept to Xd if requested
    if INT and Xd is not None:
        Xd = np.column_stack([np.ones(n), Xd])
    elif INT and Xd is None:
        Xd = np.ones((n, 1))
    
    # Create Imod matrix
    if DIFF:
        Imod = np.eye(p - 1)
    else:
        Imod = np.zeros((p, p - 1))
        mask = np.arange(p) != (base - 1)
        Imod[mask, :] = np.eye(p - 1)
    
    # Compute Xone (Kronecker product)
    if Xd is not None:
        Xone = np.kron(Xd, Imod)
    else:
        Xone = None
    
    # Compute Xtwo
    Xtwo = None
    if Xa is not None:
        if DIFF:
            tXa = Xa.T.reshape(p, -1, order='F')
            Idiff = np.eye(p)
            Idiff[:, base - 1] = -1
            Idiff = Idiff[np.arange(p) != (base - 1), :]
            tXa = Idiff @ tXa
            Xa_diff = tXa.T.reshape(n, -1, order='F')
            
            Xtwo_list = []
            for i in range(na):
                Xext = Xa_diff[:, i*(p-1):(i+1)*(p-1)]
                Xtwo_list.append(Xext.T.ravel(order='F'))
            Xtwo = np.column_stack(Xtwo_list)
        else:
            Xtwo_list = []
            for i in range(na):
                Xext = Xa[:, i*p:(i+1)*p]
                Xtwo_list.append(Xext.T.ravel(order='F'))
            Xtwo = np.column_stack(Xtwo_list)
    
    # Combine Xone and Xtwo
    if Xone is not None and Xtwo is not None:
        return np.column_stack([Xone, Xtwo])
    elif Xone is not None:
        return Xone
    else:
        return Xtwo
