
import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.linalg import cholesky, solve, lstsq
from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import pdist, squareform

import time

from cmap import textdone

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
        import cvxpy as cp  # import package

        xc = cp.Variable(np.size(A, 1))
        objective = cp.Minimize(cp.sum_squares(A @ xc - b))
        constraints = [0 <= xc, xc <= np.inf]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='OSQP', eps_abs=1e-9)
        x = xc.value

    return x

def adjacency(nx, ny, w=1):
    """
    Compute the adjacency matrix using a four-point stencil.
    
    Parameters:
    w: Optional weight to apply to vertical pixels.
    
    Returns:
    adj: Adjacency matrix after processing.
    isedge: Boolean array indicating whether an element is adjacent to a new edge.
    """
    ind1 = []
    ind2 = []
    vec = []

    for jj in range(nx * ny):
        if (jj + 1) % nx != 0:  # up pixels
            ind1.append(jj)
            ind2.append(jj + 1)
            vec.append(w)
        if jj % nx != 0:  # down pixels
            ind1.append(jj)
            ind2.append(jj - 1)
            vec.append(w)
        if jj >= nx:  # left pixels
            ind1.append(jj)
            ind2.append(jj - nx)
            vec.append(1)
        if jj < (nx * ny - nx):  # right pixels
            ind1.append(jj)
            ind2.append(jj + nx)
            vec.append(1)

    adj = sp.coo_matrix((vec, (ind1, ind2)), shape=(nx * ny, nx * ny))
    return adj

def tikhonov_lpr(order=1, nx=None, x_length=None, bc=None, grid=None, variant=0, anisotropy=1):
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

    # Interpret boundary condition. 
    # Convert to integers correspond to which order derivative is zero.
    if bc == 'dirichlet':
        bc = 0
    elif bc == 'nuemann':
        bc = 1
    elif bc is None:
        bc = int(np.floor(order))  # By default, match boundary condition to order
    
    # Get dimensions. 
    # If Grid or PartialGrid, derive from grid. 
    # Otherwise, use n and x_length. 
    if not grid == None:
        nx = grid.ne[0]
        ny = grid.ne[1]
        x_length = np.prod(grid.ne)

        A = grid.adjacency()  # adjacency matrix
        isedge = grid.isedge()  # elements adjacent to an edge (used for Dirichlet BCs)

    else:
        if x_length % nx != 0 and order != 0:
            raise ValueError("x_length must be an integer multiple of n.")
        
        # Now build adjacency matrix. 
        ny = np.int32(x_length / nx)
        A = adjacency(nx, ny)
        isedge = np.where(np.sum(A, axis=1).A1 != 4)
        
    # Handle aniostropy by calculating weightings.
    # wx = anisotropy / ((1 + anisotropy))
    # wy = 1 / (1 + anisotropy)

    Lpr1 = None
    Lpr2 = None # default if not overwritten

    # Build base matrix.
    if order == 0:
        Lpr0 = -sp.eye(x_length, format='csr')

    elif order == 1:  # 1st order Tikhonov

        Lpr0 = sp.triu(A, k=1)  # forward difference corresponds to upper triangle
        if bc == 1:
            D = sp.diags(Lpr0.sum(axis=1).A1)  # adjust diagonal
        elif bc == 0:
            D = sp.diags(2 * np.ones(np.shape(Lpr0)[0]))

        Lpr0 = D - Lpr0
        Lpr0 = Lpr0[:-1, :]  # remove trailing zeros

        # Lpr0 = sp.vstack((sp.csr_matrix(np.zeros((1,np.shape(Lpr0)[0]))), Lpr0))  # would append zeros at top for bc handling

    elif order == 2:  # 2nd order Tikhonov (with variants)

        # Case 0: standard Laplacian
        if variant == 0:
            if bc == 1:
                D = sp.diags(A.sum(axis=1).A1)  # adjust diagonal
            elif bc == 0:
                D = sp.diags(4 * np.ones(np.shape(A)[0]))

            Lpr0 = D - A

        # Case 1: difference in both dimensions
        elif variant == 1:
            Ix = sp.eye(nx, nx)
            Dx = -2 * sp.eye(ny, ny)
            Dx = sp.diags([np.ones(ny - 1), np.ones(ny - 1)], [-1, 1]) + Dx

            Iy = sp.eye(ny, ny)
            Dy = -2 * sp.eye(nx, nx)
            Dy = sp.diags([np.ones(nx - 1), np.ones(nx - 1)], [-1, 1]) + Dy

            Lpr1 = var[1] * sp.kron(Iy, Dy)
            Lpr2 = sp.kron(Dx, Ix)
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
        Ix = sp.eye(nx)
        ny = x_length // nx
        
        Dx = sp.csr_matrix((np.ones(ny-2), (np.arange(ny-2), np.arange(2, ny))), shape=(ny, ny))
        Dx = sp.diags([-0.5, 1, -1, 0.5], [-2, -1, 1, 2], shape=(ny, ny))

        Iy = sp.eye(ny)
        Dy = sp.csr_matrix((np.ones(nx-2), (np.arange(nx-2), np.arange(2, nx))), shape=(nx, nx))
        Dy = sp.diags([-0.5, 1, -1, 0.5], [-2, -1, 1, 2], shape=(nx, nx))

        Lpr1 = sp.kron(Iy, Dy)
        Lpr2 = sp.kron(Dx, Ix)
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
                Ix = 0.5 * sp.eye(nx)
                Ex = sp.csr_matrix((np.ones(nx-1), (np.arange(nx-1), np.arange(1, nx))), shape=(nx, nx))
                Dx = Ex - Ix

                ny = x_length // nx
                Iy = slope / 2 * sp.eye(ny)
                Ey = sp.csr_matrix((np.ones(ny-1), (np.arange(ny-1), np.arange(1, ny))), shape=(ny, ny))
                Dy = Ey - Iy

                Lpr0 = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)
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
    print('\r' + '\033[36m' + '[ TIKHONOV INVERSION ]' + '\033[0m')
    print('Running ...', end="", flush=True)

    start_time = time.time() # enables timing

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

    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    return x, D, Lpr0, Gpo_inv


def exp_dist_lpr(Gd, vec2, vec1, grid=None):

    if hasattr(grid, 'elements'):
        el = grid.elements
    else:
        el = np.hstack((vec1, vec2))
    
    #-- Compute Mahalanobis distances between elements -----------------------#
    Gd_inv = np.linalg.inv(Gd)
    D = squareform(pdist(np.log10(el), metric='mahalanobis', VI=Gd_inv))

    #-- Compute prior covariance matrix --------------------------------------#
    Gpr = np.exp(-D)

    Gpr_inv = np.linalg.pinv(Gpr)
    # Gpr_inv[Gpr_inv < 0.01 * np.max(Gpr_inv)] = 0  # zero very small values

    Lpr = cholesky(Gpr_inv, lower=False)

    Lpr[D > 1.75] = 0  # zero values where distances are large

    return Lpr, D, Gpr

def get_Gd(gsd1, gsd2, R=0.95):
    """
    Build covariance matrix from two correlation lengths (given as GSDs) and a correlation.
    """
    lengths = np.array([np.log10(gsd1), np.log10(gsd2)])  # correlation lengths, i.e., standard deviations
    Gd = np.diag(lengths) ** 2
    Gd[1,0] = R * np.prod(lengths)
    Gd[0,1] = Gd[1,0]
    return Gd

def exp_dist(A, b, lam, Gd=np.eye(2), vec2=None, vec1=None, xi=None, solver=None, grid=None):

    print('\r' + '\033[36m' + '[ EXPONENTIAL DISTANCE INVERSION ]' + '\033[0m')

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
    print('Building Lpr ...', end="", flush=True)
    start_time = time.time()  # time the contribution
    Lpr0, _, _ = exp_dist_lpr(Gd, vec2, vec1, grid)
    Lpr = lam * Lpr0
    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')

    # Augment data with prior matrix.
    A_aug = sp.vstack([A, Lpr]).todense()
    b_aug = np.hstack([b, np.zeros(x_length)])

    #-- Choose and execute solver --------------------------------#
    print('Inverting system ...', end="", flush=True)
    start_time = time.time()  # time the contribution
    x = nnls(A_aug, b_aug)[0]
    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')

    D = None  # inverse operator placeholder (modify based on the logic)

    #-- Uncertainty quantification -------------------------------#
    if Gd is not None:
        Gpo_inv = A.T @ A + Lpr.T @ Lpr
    else:
        Gpo_inv = None

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    return x, D, Lpr0, Gpo_inv
