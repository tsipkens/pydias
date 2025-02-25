
import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.linalg import cholesky, solve, lstsq
from scipy.optimize import lsq_linear, nnls

import cvxpy as cp

from autils.autils import textdone

from bidias.Grid import PartialGrid

def lsq(A, b, method=None):
    if method == None:
        method = 'osqp'

    if method == 'nnls':
        A = A.todense()
        x = nnls(A, b)[0]
        
    elif method == 'solve':
        A = A.todense()
        x = solve(A.T @ A, (A.T @ b).T)

    elif method in ['spsolve', 'algebraic']:
        x = sp.linalg.spsolve(A.T @ A, (A.T @ b))

    elif method == 'lsqr':
        x = lsqr(A, b)[0]

    elif method == 'lsq_linear':
        res = lsq_linear(A, b, bounds=(0, np.inf))[0]
        x = res['x']

    # elif method == 'lstsq':
    #     res = lsq_linear(A, b, bounds=(0, np.inf))[0]
    #     x = res

    elif method in ['cp', 'osqp']:
        xc = cp.Variable(np.size(A, 1))
        objective = cp.Minimize(cp.sum_squares(A @ xc - b))
        constraints = [0 <= xc, xc <= np.inf]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='OSQP', eps_abs=1e-9)
        x = xc.value

    return x


def tikhonov_lpr(order=1, n=None, x_length=None, bc=None, grid=None):
    """
    Generates Tikhonov smoothing operators/matrix, L.

    Parameters
    ----------
    order : int
        The order of the Tikhonov regularization. 
        Default is 1 (first derivative).
    
    n : int or Grid
        The number of grid points in the first dimension.
        Alternatively, use the grid input. 
    
    x_length : int
        Total number of grid elements.
    
    bc : int, optional
        Boundary condition, default is to match the order.
        
    grid : Grid
        A Grid object, it is used to extract adjacent cell information.

    Returns
    -------
    Lpr0 : sparse matrix
        The Tikhonov smoothing matrix for the specified order.
    
    Lpr1 : sparse matrix, optional
        Subcomponent of the Tikhonov matrix in the first dimension.
    
    Lpr2 : sparse matrix, optional
        Subcomponent of the Tikhonov matrix in the second dimension.
    """
    
    if order is None:
        order = 2  # Default order

    if type(order) == list:
        if len(order) == 1:
            var = [0]
            order = order[0]
        else:
            var = order[1]
            order = order[0]
    else:
        var = [0]
    
    if bc is None:
        bc = int(np.floor(order))  # By default, match boundary condition to order
    
    if not grid == None:
        n = grid.ne[0]
        x_length = np.prod(grid.ne)
    else:
        if x_length % n != 0 and order != 0:
            raise ValueError("x_length must be an integer multiple of n.")

    # Initialize Lpr0 (and Lpr1, Lpr2 if applicable)
    Lpr0 = None
    Lpr1 = None
    Lpr2 = None

    # Generate Tikhonov smoothing matrix based on the order
    if order == 0:
        # 0th order Tikhonov
        if not grid == None:
            Lpr0 = -sp.eye(grid.Ne, format='csr')
        else:
            Lpr0 = -sp.eye(x_length, format='csr')

    elif order == 1:  # 1st order Tikhonov
        if type(grid) == PartialGrid:
            Lpr0 = grid.l1()
            Lpr0 = sp.csr_matrix(Lpr0)

        else:
            I1 = 0.5 * sp.eye(n)
            E1 = sp.csr_matrix((np.ones(n-1), (np.arange(n-1), np.arange(1, n))), shape=(n, n))
            D1 = E1 - I1

            m = x_length // n
            I2 = 0.5 * sp.eye(m)
            E2 = sp.csr_matrix((np.ones(m-1), (np.arange(m-1), np.arange(1, m))), shape=(m, m))
            D2 = E2 - I2

            Lpr0 = sp.kron(I2, D1) + sp.kron(D2, I1)
            Lpr0 = Lpr0 - sp.diags(Lpr0.sum(axis=1).A1, 0)
            Lpr0 = Lpr0[:-1, :]
            

    elif order == 2:
        # 2nd order Tikhonov (with variants)

        # Case 0: standard Laplacian
        if var[0] == 0:
            if len(var) == 1:
                var.append(1)
                
            I1 = 0.25 * sp.eye(n, n)
            E1 = sp.csr_matrix((np.ones(n-1), (np.arange(n-1), np.arange(1, n))), shape=(n, n))
            D1 = E1 + E1.T - I1

            m = x_length // n
            I2 = 0.25 * sp.eye(m, m)
            E2 = sp.csr_matrix((np.ones(m-1), (np.arange(m-1), np.arange(1, m))), shape=(m, m))
            D2 = E2 + E2.T - I2

            Lpr1 = var[1] * sp.kron(I2, D1)
            Lpr2 = sp.kron(D2, I1)
            Lpr0 = Lpr1 + Lpr2
            Lpr0 = Lpr0 - sp.diags(Lpr0.sum(axis=1).A1, 0)

        # Case 1: difference in both dimensions
        elif var[0] == 1:
            if len(var) == 1:
                var.append(1)

            m = x_length // n

            I1 = sp.eye(n, n)
            D1 = -2 * sp.eye(m, m)
            D1 = sp.diags([np.ones(m - 1), np.ones(m - 1)], [-1, 1]) + D1

            I2 = sp.eye(m, m)
            D2 = -2 * sp.eye(n, n)
            D2 = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1]) + D2

            Lpr1 = var[1] * sp.kron(I2, D2)
            Lpr2 = sp.kron(D1, I1)
            Lpr0 = Lpr1 - Lpr2

            # if isinstance(grid, PartialGrid):
            #     Lpr0 = Lpr0.tolil()
            #     Lpr0[grid.missing, :] = 0
            #     Lpr0[:, grid.missing] = 0
            #     Lpr0 = Lpr0.tocsr()

        else:
            print('Variant not available.')

    elif order == 3:
        # 3rd order derivative
        I1 = sp.eye(n)
        m = x_length // n
        
        D1 = sp.csr_matrix((np.ones(m-2), (np.arange(m-2), np.arange(2, m))), shape=(m, m))
        D1 = sp.diags([-0.5, 1, -1, 0.5], [-2, -1, 1, 2], shape=(m, m))

        I2 = sp.eye(m)
        D2 = sp.csr_matrix((np.ones(n-2), (np.arange(n-2), np.arange(2, n))), shape=(n, n))
        D2 = sp.diags([-0.5, 1, -1, 0.5], [-2, -1, 1, 2], shape=(n, n))

        Lpr1 = sp.kron(I2, D2)
        Lpr2 = sp.kron(D1, I1)
        Lpr0 = Lpr1 + Lpr2

        # if hasattr(grid, 'missing'):
        #     Lpr0 = Lpr0.tocsr()
        #     Lpr0[grid.missing, :] = 0
        #     Lpr0[:, grid.missing] = 0

    else:
        if 1 < order < 2:
            slope = (order - 1) * 10
            if hasattr(grid, 'l1'):
                Lpr0 = grid.l1(slope, bc)
            else:
                I1 = 0.5 * sp.eye(n)
                E1 = sp.csr_matrix((np.ones(n-1), (np.arange(n-1), np.arange(1, n))), shape=(n, n))
                D1 = E1 - I1

                m = x_length // n
                I2 = slope / 2 * sp.eye(m)
                E2 = sp.csr_matrix((np.ones(m-1), (np.arange(m-1), np.arange(1, m))), shape=(m, m))
                D2 = E2 - I2

                Lpr0 = sp.kron(I2, D1) + sp.kron(D2, I1)
                Lpr0 = Lpr0 - sp.diags(Lpr0.sum(axis=1).A1, 0)
                Lpr0 = Lpr0[:-1, :]

        else:
            raise ValueError("The specified order of Tikhonov is not available.")

    return Lpr0, Lpr1, Lpr2


def tikhonov(A, b, lam, order=None, n=None, bc=None, xi=None, grid=None, Lpr0=None, method=None):
    """
    Performs inversion using various order Tikhonov regularization.
    Regularization takes place in 2D. The type of regularization or prior
    information added depends on the order of Tikhonov applied.

    Parameters:
    -----------
    A : np.ndarray or scipy.sparse matrix
        Model matrix or kernel.
    b : np.ndarray
        Data vector.
    lam : float
        Regularization parameter.
    order : int, can be replaced by L
        Specifies the order of Tikhonov regularization.
    n : int or Grid, optional
        Length of the first dimension of the solution (otherwise, specify grid).
    bc : int, optional
        Boundary conditions.
    xi : np.ndarray, optional
        Initial guess. Defaults to zero if not provided.
    grid : Grid, optional
        A instance of the Grid class to be used to build the Tikhonov matrix.
    Lpr0 : np.ndarray, optional
        A precomputed Tikhonov matrix.

    Returns:
    --------
    x : np.ndarray
        Inverted solution.
    D : np.ndarray
        Explicit inverse operator.
    Lpr0 : np.ndarray, optional
        Unscaled Tikhonov matrix.
    Gpo_inv : np.ndarray, optional
        Inverse posterior covariance.
    """

    print('Performing Tikhonov inversion ...')

    x_length = A.shape[1]

    # Parse inputs
    if order is None:
        order = 1  # Default to 1st order if not specified

    if bc is None:
        bc = 0

    if xi is None:
        xi = np.zeros(x_length)

    # Get Tikhonov smoothing matrix (if not given)
    if Lpr0 == None:
        Lpr0, _, _ = tikhonov_lpr(order, n, x_length, bc, grid=grid)

    Lpr = lam * Lpr0.todense()
    
    # Choose and execute solver
    pr_length = Lpr0.shape[0]
    
    A_aug = np.asarray(np.vstack((A, Lpr)))
    b_aug = np.concatenate([b, np.zeros(pr_length)])
    
    A_aug2 = sp.csr_matrix(A_aug)

    x = lsq(A_aug2, b_aug, method=method)

    D = None # np.linalg.pinv(A_aug.toarray())  # Calculate explicit inverse operator

    # Uncertainty quantification
    Gpo_inv = None

    textdone()

    return x, D, Lpr0, Gpo_inv


def exp_dist_lpr(Gd, vec2, vec1, grid=None):

    if hasattr(grid, 'elements'):
        vec1 = grid.elements[:,0]
        vec2 = grid.elements[:,1]
    
    #-- Compute distances between elements -----------------------------------#
    vec2_a, vec2_b = np.meshgrid(vec2, vec2)  # for differences in 2nd dim
    vec1_a, vec1_b = np.meshgrid(vec1, vec1)  # for differences in 1st dim

    Gd_inv = np.linalg.inv(Gd)
    dr1 = np.log10(vec1_a) - np.log10(vec1_b)
    dr2 = vec2_a - vec2_b

    D = np.sqrt(
        dr1**2 * Gd_inv[0, 0] +
        2 * dr2 * dr1 * Gd_inv[0, 1] +
        dr2**2 * Gd_inv[1, 1]
    )  # distance

    #-- Compute prior covariance matrix --------------------------------------#
    Gpr = np.exp(-D)

    Gpr_inv = np.linalg.pinv(Gpr)
    Lpr = cholesky(Gpr_inv, lower=False)

    return Lpr, D, Gpr

def exp_dist(A, b, lam, Gd=np.eye(2), vec2=None, vec1=None, xi=None, solver=None, grid=None):

    if vec1 is None:
        vec1 = []

    x_length = A.shape[1]

    #-- Parse inputs ---------------------------------------------#
    if Gd[0, 1] / np.sqrt(Gd[0, 0] * Gd[1, 1]) >= 1:
        raise ValueError('Correlation greater than 1.')

    if xi is None:
        xi = None  # if no initial x is given
    if solver is None:
        solver = None  # if computation method not specified
    #-------------------------------------------------------------#

    # Use external function to evaluate prior covariance
    Lpr0, _, _ = exp_dist_lpr(Gd, vec2, vec1, grid)
    Lpr = lam * Lpr0

    #-- Choose and execute solver --------------------------------#
    A_aug = sp.vstack([A, Lpr]).todense()
    b_aug = np.hstack([b, np.zeros(x_length)])

    x = nnls(A_aug, b_aug)[0]
    D = None  # inverse operator placeholder (modify based on the logic)

    #-- Uncertainty quantification -------------------------------#
    if Gd is not None:
        Gpo_inv = A.T @ A + Lpr.T @ Lpr
    else:
        Gpo_inv = None

    return x, D, Lpr0, Gpo_inv
