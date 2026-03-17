
import numpy as np

from bidias import Grid

import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.linalg import cholesky, solve, lstsq
from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian

import time

from cmap import textdone

from bidias.Grid import PartialGrid

# NOTE: To be added.
def reducer(A, b):
    """
    Reduce the size of the problem by removing 
    empty columns and empty data points
    (but then need to force towards zero).

    Generalizable, but current only remove all zero columns (i.e., data does not impact x).
    """
    # x_keep = np.sum(A, axis=0) > -999 # != 0

    # Used to zero rows where data has no impact on the reconstruction.
    # Can shrink inversion matrices in some instances.
    b_keep = np.sum(A, axis=1) != 0
    A = A[b_keep, :]
    b = b[b_keep]

    return A, b, b_keep

# def expander(x, x_keep):
#     """
#     Supplement x back up to the main size. 
#     """
#     xn = np.zeros(np.shape(x_keep))
#     xn[x_keep] = x

#     return xn


def lsq(A, b, method=None, C=None, d=None):
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

        if C is None:
            constraints = [0 <= xc, xc <= np.inf]
        else:
            
            constraints = [0 <= xc, xc <= np.inf, C @ xc == d]
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

def apply_dirichlet(L, isedge):
    """
    Apply Dirichlet boundary conditions given a list of edges.
    """
    L = L.tolil() # Use LIL for efficient row manipulation
    
    for idx in isedge:# Set the entire row to zeros first
        L[idx, :] = 0  # zero the entire row
        L[idx, idx] = 1.0  # set the diagonal to 1
    
    return L.tocsr() # convert back to CSR and return


def tikhonov_lpr(order=1, nx=None, x_length=None, bc=None, grid=None, variant=0, anisotropy=1.0):
    """
    Generates Tikhonov smoothing operators/matrix, L.

    Parameters
    ----------
    order : int
        The order of the Tikhonov regularization. 
        Default is 1 (first derivative).
    
    nx : int or Grid
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
    # Convert to integers corresponding to which order derivative is zero.
    if bc == 'dirichlet':
        bc = 0
    elif bc == 'nuemann':
        bc = 1
    elif bc is None:
        bc = int(np.floor(order))  # by default, match boundary condition to order
    
    # Get dimensions. 
    # If Grid or PartialGrid, derive from grid. 
    if not grid == None:
        nx = grid.ne[0]
        ny = grid.ne[1]
        x_length = np.prod(grid.ne)

        adj = grid.adjacency(anisotropy=anisotropy)  # adjacency matrix
        isedge = grid.isedge()  # elements adjacent to an edge (used for Dirichlet BCs)

    else: # otherwise build adjacency matrix for full grid for compatibility
        if x_length % nx != 0 and order != 0:
            raise ValueError("x_length must be an integer multiple of n.")
        
        # Now build adjacency matrix. 
        ny = np.int32(x_length / nx)
        adj = Grid.Grid.adjacency0([nx, ny], anisotropy=anisotropy)  # use existing Grid method
        isedge = np.where(np.sum(adj, axis=1).A1 != 4)
        
    # Handle aniostropy by calculating weightings.
    # wx = anisotropy / ((1 + anisotropy))
    # wy = 1 / (1 + anisotropy)

    Lpr1 = None
    Lpr2 = None # default if not overwritten

    # Build base matrix.
    if order == 0:
        Lpr0 = -sp.eye(x_length, format='csr')

    elif order == 1:  # 1st order Tikhonov
        # Produces a flattened Tikhonov matrix corresponding to Lx + Ly = 0. 

        if variant == 0:
            Lpr0 = sp.triu(adj, k=1)  # forward difference corresponds to upper triangle
            
            # Perform operation common to BCs. 
            D = sp.diags(Lpr0.sum(axis=1).A1)  # degree: used to adjust diagonal (each row now sums to zero)
            Lpr0 = D - Lpr0

            # Modify based on boundary conditions.
            if bc == 1:
                Lpr0 = Lpr0[:-1, :]  # remove trailing zeros
                
                # To remove possible bending as edges. Remove anisotropy for edge cells.
                target_rows = np.where(np.diff(Lpr0.indptr) == 2)[0]
                diags = Lpr0.diagonal()
                scale = np.ones(Lpr0.shape[0])  # scale parameter to be adjusted
                scale[target_rows] = 1.0 / diags[target_rows]  # adjust on only target rows
                Lpr0 = sp.diags(scale) @ Lpr0

            elif bc == 0: Lpr0 = apply_dirichlet(Lpr0, isedge)
            else: raise ValueError("Boundary condition must be 0 (dirichlet) or 1 (nuemann).")

        elif variant == 1:
            adj_coo = sp.triu(sp.coo_matrix(adj)) # Use upper triangle to avoid double-counting edges
            
            rows = adj_coo.row
            cols = adj_coo.col
            num_edges = len(rows)
            num_nodes = adj.shape[0]
            
            # Create the incidence matrix L
            # Each row represents an edge (i, j)
            # L[edge_k, i] = -1, L[edge_k, j] = 1
            edge_indices = np.arange(num_edges)
            
            data = np.concatenate([-np.ones(num_edges), np.ones(num_edges)])
            row_indices = np.concatenate([edge_indices, edge_indices])
            col_indices = np.concatenate([rows, cols])
            
            Lpr0 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, num_nodes))

        else:
            raise ValueError("Variant not available.")

    elif order == 2:  # 2nd order Tikhonov (with variants)

        # Case 0: standard Laplacian
        if variant == 0:
            Lpr0 = laplacian(adj)  # use existing Laplacian function

            # Modify based on boundary conditions.
            if bc == 0: Lpr0 = apply_dirichlet(Lpr0, isedge)

        # Case 1: difference in both dimensions
        elif variant == 1:
            Ix = sp.eye(nx, nx)
            Dx = -2 * sp.eye(ny, ny)
            Dx = sp.diags([np.ones(ny - 1), np.ones(ny - 1)], [-1, 1]) + Dx

            Iy = sp.eye(ny, ny)
            Dy = -2 * sp.eye(nx, nx)
            Dy = sp.diags([np.ones(nx - 1), np.ones(nx - 1)], [-1, 1]) + Dy

            Lpr1 = sp.kron(Iy, Dy)
            Lpr2 = sp.kron(Dx, Ix)
            Lpr0 = Lpr1 - Lpr2

        else:
            raise ValueError("Variant not available.")

        # Modify based on boundary conditions.
        if bc == 0: Lpr0 = apply_dirichlet(Lpr0, isedge)

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

        # Modify based on boundary conditions.
        if bc == 0: Lpr0 = apply_dirichlet(Lpr0, isedge)

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


def tikhonov(A, b, lam, order=None, n=None, bc=None, xi=None, grid=None, Lpr0=None, anisotropy=1.0, **kwargs):
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
    anisotropy : float
        Anisotropy parameter, allowing for different weighting in horiztonal direction.
    **kwargs
        Other keyword arguments that are passed to lsq.

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

    A, b, _ = reducer(A, b)  # reduce matrix depending on all zero cols

    start_time = time.time() # enables timing

    x_length = A.shape[1]  # get x_length from A matrix

    # Parse inputs
    if order is None:
        order = 1  # Default to 1st order if not specified

    if bc is None:
        bc = 0

    if xi is None:
        xi = np.zeros(x_length)

    # Get Tikhonov smoothing matrix (if not given)
    if Lpr0 == None:
        Lpr0, _, _ = tikhonov_lpr(order, n, x_length, bc, grid=grid, anisotropy=anisotropy)

    Lpr = lam * Lpr0.todense()

    # Lpr = Lpr[:, x_keep][x_keep[:-1], :]
    # if 'C' in kwargs:
    #     kwargs['C'] = kwargs['C'][:, x_keep]
    
    # Choose and execute solver
    pr_length = Lpr.shape[0]
    
    A_aug = np.asarray(np.vstack((A, Lpr)))
    b_aug = np.concatenate([b, np.zeros(pr_length)])
    
    A_aug2 = sp.csc_matrix(A_aug)
    x = lsq(A_aug2, b_aug, **kwargs)

    D = None # np.linalg.pinv(A_aug.toarray())  # Calculate explicit inverse operator

    # Uncertainty quantification
    Gpo_inv = None

    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    # x = expander(x, x_keep)

    return x, D, Lpr0, Gpo_inv


def exp_dist_lpr(Gd, vec2, vec1, grid=None):

    if hasattr(grid, 'elements'):
        el = grid.elements.copy()
        for ii in range(2):
            if grid.discrete[ii] == 'log':  # use information in grid
                el[:,ii] = np.log10(el[:,ii])
    else:
        el = np.hstack((vec1, vec2))
        el = np.log10(el)  # assume elements are log-spaced
    
    #-- Compute Mahalanobis distances between elements -----------------------#
    Gd_inv = np.linalg.inv(Gd)
    D = squareform(pdist(el, metric='mahalanobis', VI=Gd_inv))

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

def exp_dist(A, b, lam, Gd=np.eye(2), vec2=None, vec1=None, grid=None, **kwargs):

    print('\r' + '\033[36m' + '[ EXPONENTIAL DISTANCE INVERSION ]' + '\033[0m')

    A, b, _ = reducer(A, b)  # reduce matrix depending on all zero cols

    if vec1 is None:
        vec1 = []

    x_length = A.shape[1]

    #-- Parse inputs ---------------------------------------------#
    if Gd[0, 1] / np.sqrt(Gd[0, 0] * Gd[1, 1]) >= 1:
        raise ValueError('Correlation greater than 1.')
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

    A_aug2 = sp.csr_matrix(A_aug)
    x = lsq(A_aug2, b_aug, **kwargs)

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
