
import numpy as np

from bidias import Grid

# Import overlapping functions from odias.
import odias.invert as odias_invert

import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.linalg import cholesky, solve, solve_triangular
from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import laplacian

from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

import numpy as np

from autils.tools import tqdm2 as tqdm

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

    A = sp.coo_matrix(A)  # convert to sparse

    return A, b, b_keep

# def expander(x, x_keep):
#     """
#     Supplement x back up to the main size. 
#     """
#     xn = np.zeros(np.shape(x_keep))
#     xn[x_keep] = x

#     return xn


def apply_dirichlet(L):
    """
    Apply Dirichlet boundary conditions given a list of edges.
    """
    L.setdiag(L.diagonal().max(), k=0)
    return L


# =========== TIKHONOV REGULARIZATION =========== #
#             (i.e., ridge regression)
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
    if str(bc).lower() == 'dirichlet':
        bc = 0
    elif str(bc).lower() == 'nuemann':
        bc = 1
    elif bc is None:
        bc = int(np.floor(order))  # by default, match boundary condition to order
    
    # ---- Get dimensions and adjacency ----
    if grid == None:  # skip if Grid or PartialGrid is given
        if x_length % nx != 0 and order != 0:
            raise ValueError("x_length must be an integer multiple of n.")
        grid = Grid.Grid(span=[[0.1,1],[0.1,1]], ne=[nx, x_length/nx])  # create generic grid (span doesn't matter)
    
    nx = grid.ne[0]  # extract grid properties
    ny = grid.ne[1]
    x_length = np.prod(grid.ne)

    adj = grid.adjacency(anisotropy=anisotropy)  # adjacency matrix
    is_edge = grid.isedge()  # elements adjacent to an edge (used for Dirichlet BCs)
    # ---------------------------------------

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
                Lpr0 = Lpr0[:-1, :]  # only remove trailing zeros
                
                # To remove possible bending as edges. Remove anisotropy for edge cells.
                target_rows = np.where(np.diff(Lpr0.indptr) == 2)[0]
                diags = Lpr0.diagonal()
                scale = np.ones(Lpr0.shape[0])  # scale parameter to be adjusted
                scale[target_rows] = 1.0 / diags[target_rows]  # adjust on only target rows
                Lpr0 = sp.diags(scale) @ Lpr0

            elif bc == 0: Lpr0 = apply_dirichlet(Lpr0)
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
            if bc == 0: Lpr0 = apply_dirichlet(Lpr0)

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
        if bc == 0: Lpr0 = apply_dirichlet(Lpr0)

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
        if bc == 0: Lpr0 = apply_dirichlet(Lpr0)

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


def tikhonov(A, b, lam, order=None, nx=None, bc=None, xi=None, grid=None, Lpr0=None, variant=0, anisotropy=1.0, **kwargs):
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
    nx : int, optional (but then requires grid or Lpr0)
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
    print('\r' + '\033[36m' + f'[ TIKHONOV INVERSION (order={order}) ]' + '\033[0m')
    print('Running ...', end="", flush=True)

    A, b, _ = reducer(A, b)  # reduce matrix depending on all zero cols

    x_length = A.shape[1]  # get x_length from A matrix

    # Parse inputs
    if order is None:
        order = 1  # Default to 1st order if not specified

    if xi is None:
        xi = np.zeros(x_length)

    # Get Tikhonov smoothing matrix (if not given)
    if Lpr0 == None:
        Lpr0, _, _ = tikhonov_lpr(order, nx, x_length, bc, grid=grid, anisotropy=anisotropy, variant=variant)

    #-- Choose and execute solver --------------------------------#
    start_time = time.time() # enables timing

    x = odias_invert.regularization_engine(A, b, lam, Lpr0, **kwargs)

    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')
    #-------------------------------------------------------------#

    # Uncertainty quantification
    D = None # np.linalg.pinv(A_aug.toarray())  # Calculate explicit inverse operator
    Gpo_inv = None

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    # x = expander(x, x_keep)

    return x, D, Lpr0, Gpo_inv


# =========== EXP. DISTANCE REGULARIZATION =========== #
def exp_dist_lpr(Gd, vec2=None, vec1=None, grid=None, dist_cutoff=1.75, fast=False):
    # 1. Pre-process elements efficiently
    if hasattr(grid, 'elements'):
        el = grid.elements.copy()
        for ii in range(2):
            if grid.discrete[ii] == 'log':
                el[:, ii] = np.log10(el[:, ii])
    else:
        el = np.log10(np.column_stack((vec1, vec2)))
    
    #-- Compute Mahalanobis distances between elements -----------------------#
    # Linear transformation approach: (x-y).T @ Gd_inv @ (x-y) 
    # is equivalent to Euclidean distance of (L_inv @ el)
    # where L @ L.T = Gd.
    L_Gd = cholesky(Gd, lower=True)
    el_transformed = np.linalg.solve(L_Gd, el.T).T
    D = cdist(el_transformed, el_transformed, metric='euclidean')

    # #-- Compute prior covariance matrix --------------------------------------#
    # Gpr is a positive definite kernel (Squared Exponential/Exponential)
    Gpr = np.exp(-D)

    if fast:  # instead of pinv, get the Precision matrix directly via Cholesky
        Lpr = cholesky(Gpr + 1e-5 * np.eye(Gpr.shape[0]), lower=True)
        I = np.eye(Lpr.shape[0])
        Lpr = solve_triangular(Lpr, I, lower=True).T

    else:  # more accurate fallback to pinv, incl. if Gpr is singular
        Gpr_inv = np.linalg.pinv(Gpr)
        Lpr = cholesky(Gpr_inv, lower=False)

    Lpr[D > dist_cutoff] = 0  # zero values where distances are large

    return Lpr, D, Gpr

# def exp_dist_lpr_sparse(Gd, vec2, vec1, grid=None, dist_cutoff=1.75):
#     # 1. Pre-process elements
#     if hasattr(grid, 'elements'):
#         el = grid.elements.copy()
#         for ii in range(2):
#             if grid.discrete[ii] == 'log':
#                 el[:, ii] = np.log10(el[:, ii])
#     else:
#         el = np.log10(np.column_stack((vec1, vec2)))
    
#     # 2. Linear transformation (Whitening)
#     # Transforming coordinates so Euclidean distance = Mahalanobis distance
#     L_Gd = cholesky(Gd, lower=True)
#     el_transformed = np.linalg.solve(L_Gd, el.T).T
    
#     # 3. Sparse Distance Calculation using KDTree
#     # This replaces cdist and only finds neighbors within the cutoff
#     tree = cKDTree(el_transformed)
    
#     # sparse_distance_matrix returns a coordinate format (COO) sparse matrix
#     D_sparse = tree.sparse_distance_matrix(tree, max_distance=dist_cutoff, output_type='ndarray')
    
#     # D_sparse contains [row_indices, col_indices, distances]
#     rows = D_sparse['i']
#     cols = D_sparse['j']
#     dist_values = D_sparse['v']
    
#     # 4. Compute Sparse Gpr (Covariance)
#     # Gpr = exp(-D)
#     gpr_values = np.exp(-dist_values)
#     Gpr_sparse = sp.csr_matrix((gpr_values, (rows, cols)), shape=(len(el), len(el)))
    
#     # 5. Precision Matrix (Lpr)
#     # NOTE: Calculating the inverse or Cholesky of a sparse matrix 
#     # and keeping it sparse is mathematically complex. 
#     # Usually, we approximate the precision matrix directly for spatial processes.
    
#     # If you strictly need the Lpr logic from your original code:
#     Gpr_dense = Gpr_sparse.toarray()
#     Lpr = cholesky(np.linalg.pinv(Gpr_dense), lower=False)
#     Lpr[Gpr_dense == 0] = 0 # Enforce sparsity pattern
    
#     return sp.csr_matrix(Lpr), Gpr_sparse

def exp_dist_lpr_s21(Gd, sd=1.0, vec2=None, vec1=None, grid=None, dist_cutoff=1.75, fast=False):
    """
    Special exponential distance Lpr for charging problem that has varying s2|1.
    Corresponds to varying amount of correlation across the domain.
    """
    
    # Pre-process elements efficiently
    if hasattr(grid, 'elements'):
        el = grid.elements.copy()
        for ii in range(2):
            el[:, ii] = np.log10(el[:, ii])
    else:
        # Avoid hstack if possible, but keeping logic consistent
        el = np.log10(np.column_stack((vec1, vec2)))
    
    #-- Compute Mahalanobis distances between elements -----------------------#
    # Extract Gd components.
    g11, g22, g12 = Gd[0, 0], Gd[1, 1], Gd[0, 1]
    sgn = np.sign(g12) # Direction of correlation

    # Define s2|1 as a function. Wider at small charge states. 
    y = el[:, 1]  # for function of second dimension
    # s21 = np.sqrt(g22 - g12**2 / g11)  # default condition sd.
    s21_vec = np.log(1 / (1 - sd / 10**y))  # function for conditional width

    # g12 varies to satisfy the s2|1 requirement
    # Note: As long as s21_vec is constant, g12_vec will be constant and equal to g12
    # g12 = sgn * sqrt( g11 * (g22 - s2|1**2) )
    g12_vec = sgn * np.sqrt(np.maximum(g11 * (g22 - s21_vec**2), 0))

    # Pairwise differences for both dimensions.
    dx = el[:, 0][:, np.newaxis] - el[:, 0][np.newaxis, :]
    dy = el[:, 1][:, np.newaxis] - el[:, 1][np.newaxis, :]

    # Symmetric averaging of the varying component.
    g12_avg = 0.5 * (g12_vec[:, np.newaxis] + g12_vec[np.newaxis, :])

    # Calculate local determinant and Mahalanobis distance.
    det_Gd = g11 * g22 - g12_avg**2
    D2 = (g22 * dx**2 - 2 * g12_avg * dx * dy + g11 * dy**2) / det_Gd
    D = np.sqrt(np.maximum(D2, 0))
    # -------------------------------------------------

    # #-- Compute prior covariance matrix --------------------------------------#
    # Gpr is a positive definite kernel (Squared Exponential/Exponential)
    Gpr = np.exp(-D)

    if fast:  # instead of pinv, get the Precision matrix directly via Cholesky
        Lpr = cholesky(Gpr + 1e-5 * np.eye(Gpr.shape[0]), lower=True)
        I = np.eye(Lpr.shape[0])
        Lpr = solve_triangular(Lpr, I, lower=True).T

    else:  # more accurate fallback to pinv, incl. if Gpr is singular
        Gpr_inv = np.linalg.pinv(Gpr)
        Lpr = cholesky(Gpr_inv, lower=False)

    Lpr[D > dist_cutoff] = 0  # zero values where distances are large

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

def exp_dist(A, b, lam, Gd=np.eye(2), vec2=None, vec1=None, grid=None, Lpr0=None, fast=False, **kwargs):

    print('\r' + '\033[36m' + '[ EXPONENTIAL DISTANCE INVERSION ]' + '\033[0m')

    A, b, _ = reducer(A, b)  # reduce matrix depending on all zero cols

    x_length = A.shape[1]

    if vec1 is None:
        vec1 = []

    # Compute Lpr0 if not given.
    # Can be bypassed by pre-computing Lpr0.
    if Lpr0 is None:
        #-- Validate Gd input -------------------------------------#
        if Gd[0, 1] / np.sqrt(Gd[0, 0] * Gd[1, 1]) >= 1:
            raise ValueError('Correlation greater than 1.')
        #----------------------------------------------------------#

        # Use external function to evaluate prior covariance
        print('Building Lpr ...', end="", flush=True)
        start_time = time.time()  # time the contribution
        Lpr0, _, _ = exp_dist_lpr(Gd, vec2, vec1, grid, fast=fast)
        end_time = time.time()
        textdone(f' ({end_time - start_time:.2f} s)')

    #-- Choose and execute solver --------------------------------#
    print('Inverting system ...', end="", flush=True)
    start_time = time.time()  # time the contribution

    x = odias_invert.regularization_engine(A, b, lam, Lpr0, **kwargs)

    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')
    #-------------------------------------------------------------#

    #-- Uncertainty quantification -------------------------------#
    D = None  # inverse operator placeholder (modify based on the logic)
    if Gd is not None:
        Gpo_inv = A.T @ A + lam**2 * Lpr0.T @ Lpr0
    else:
        Gpo_inv = None

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    return x, D, Lpr0, Gpo_inv


# =========== TOTAL VARIATION REGULARIZATION =========== #
def total_variation(A, b, lam, nx=None, grid=None, max_iter=3, xi=None, delta=1e-5, **kwargs):
    """
    Total variation regularization based on that in Grauer et al. (2018). 
    """
    A, b, _ = reducer(A, b)
    x_length = A.shape[1]

    if xi is None:
        xi = np.ones(x_length)
    x = xi
    
    # Get gradient operators. 
    D1, _, _ = tikhonov_lpr(order=1, nx=nx, grid=grid, x_length=x_length)
    D2, _, _ = tikhonov_lpr(order=2, nx=nx, grid=grid, x_length=x_length)
    
    print('\r' + '\033[36m' + '[ TOTAL VARIATION INVERSION ]' + '\033[0m')
    print('Inverting system ...', end="", flush=True)
    start_time = time.time()  # time the contribution

    for ii in tqdm(range(max_iter)):
        # Build the prior matrix.
        w = np.concatenate((1 / np.sqrt(np.sqrt((D1 @ x)**2 + delta**2)), [0]))
        Lpr = lam * (sp.diags(w) @ D2)
        Lpr = sp.coo_matrix(Lpr)

        # Augment data with prior matrix.
        A_aug = sp.vstack([A, Lpr])
        b_aug = np.concatenate([b, np.zeros(Lpr.shape[0])])

        #-- Choose and execute solver --------------------------------
        A_aug2 = sp.csr_matrix(A_aug)
        x = odias_invert.lsq(A_aug2, b_aug, **kwargs)

    end_time = time.time()
    textdone(f' ({end_time - start_time:.2f} s)')

    print('\r' + '\033[36m' + '[ INVERSION COMPLETE! ]' + '\033[0m' + '\n\n')

    # -- Alternate code could be used if bypassing lsq and linearization --
    # # x is the grid quantity we want to find
    # x = cp.Variable(n_nodes)

    # # D is the difference operator from the adjacency matrix
    # objective = cp.Minimize(cp.sum_squares(A @ x - y) + lmbda * cp.norm(D @ x, 1))
    # prob = cp.Problem(objective)
    # prob.solve()
            
    return x


def twomey(*args, **kwargs):
    """
    Twomey regularization, refers to odias method.
    """
    return odias_invert.twomey(*args, **kwargs)

