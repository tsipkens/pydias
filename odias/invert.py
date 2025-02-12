
import numpy as np

from scipy.sparse import eye, diags, vstack, csr_matrix
from scipy.optimize import lsq_linear, nnls
from scipy.interpolate import interp1d

import cvxpy as cp

from tqdm import tqdm


def get_init(A, b, d, d_star):
    """
    Compute an estimate of x to initialize various algorithms.

    Parameters:
    A : array
        Model matrix.
    b : array
        Data vector.
    d : array
        Particle mobility diameters for interpolation.
    d_star : array
        Setpoints for interpolation.

    Returns:
    xi : array
        The initialized estimate.
    """
    
    # Compute the initial estimate by interpolating
    # Note: b should be converted to a 1D array if it's not already
    b = np.asarray(b).flatten()
    
    # Compute the initial estimate
    xi = interp1d(d_star, b / (A @ np.ones(A.shape[1])), bounds_error=False, fill_value=0)(d)
    
    # Filter out unphysical outputs
    xi[np.isnan(xi)] = 0
    xi[np.isinf(xi)] = 0
    xi = np.maximum(0, xi)
    
    return xi


def mean_sq_err(A, x, b):
    """
    Calculates the mean squared error of the non-zero entries of b.

    Parameters:
    A : array
        Model matrix.
    x : array
        The estimated vector.
    b : array
        Data vector.

    Returns:
    sig : float
        The mean squared error of the non-zero entries of b.
    """
    
    # Calculate squared errors
    sq_err = (A @ x - b) ** 2  # squared errors
    
    # Calculate mean squared error where b is not zero
    sig = np.mean(sq_err[b != 0])
    
    return sig


def lsq(A, b, xi=None, method=None):
    """
    Performs least-squares or equivalent optimization.

    Parameters:
    A       : array-like or sparse matrix
              Model matrix
    b       : array-like
              Data vector
    xi      : array-like, optional
              Initial guess for solver. Default is None.
    method  : str, optional
              Solver to use. Options: 'non-neg', 'interior-point', ...

    Returns:
    x       : array-like
              Regularized estimate
    D       : array-like or None
              Inverse operator (x = D @ [b;0]). May be None if not calculated.
    """
    
    # Parse inputs
    x_length = A.shape[1]
    x_lb = np.zeros(x_length)  # Non-negativity bound
    
    if xi is not None:
        xi = np.maximum(xi, x_lb)  # Enforce non-negativity constraint if xi is provided
    else:
        xi = np.zeros(x_length)

    if method == None:
        method = 'osqp'
    
    D = None  # Default value for D, unless computed by the solver
    
    # Choose solver and evaluate
    if method == 'non-neg':  # Constrained non-negative least squares
        if type(A) == csr_matrix:
            A = A.todense()
        
        x = nnls(A, b)[0]
        D = None

    elif method == 'lsq_linear':
        res = lsq_linear(A, b, bounds=(0, np.inf))
        x = res['x']
        
    elif method == 'algebraic':  # Unconstrained least squares using matrix multiplication
        D = np.linalg.pinv(A.T @ A) @ A.T  # Inverse operator
        x = D @ b
        
    elif method == 'algebraic-inv':  # Unconstrained least squares using matrix inverse (less stable)
        D = np.linalg.inv(A.T @ A) @ A.T  # Inverse operator
        x = D @ b

    elif method in ['cp', 'cvxpy', 'osqp']:
        xc = cp.Variable(np.size(A, 1))
        objective = cp.Minimize(cp.sum_squares(A @ xc - b))
        constraints = [0 <= xc, xc <= np.inf]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='OSQP', eps_abs=1e-9)
        x = xc.value
        
    else:
        raise ValueError(f"Solver '{method}' is not recognized.")
    
    return x, D


def tikhonov_lpr(order, x_length, bc=None):
    """
    Generates Tikhonov matrices/operators for the given order and boundary conditions.

    Parameters:
    order     : int
                Order of the Tikhonov regularization (0, 1, 2, 3, 4, or 34)
    x_length  : int
                Length of the vector x
    bc        : int (optional)
                Boundary conditions:
                None or []: No explicit boundary conditions
                0: Zero BCs
                1: No slope BCs
                2: No curvature BCs

    Returns:
    L         : scipy.sparse matrix
                The generated Tikhonov matrix
    """
    
    if bc is None:
        bc = order  # Default to the same as the order if boundary condition not provided

    # Choose between order of Tikhonov operator to generate.
    if order == 0:  # 0th order Tikhonov promotes small solutions
        L = eye(x_length, format='lil')

    elif order == 1:  # 1st order Tikhonov minimizes the first derivative
        L = -eye(x_length, format='lil')
        L = diags([np.ones(x_length)], [1], shape=(x_length, x_length), format='lil') + L
        
        if bc != 0:
            L = L[:-1, :]  # Remove the last row if not zero boundary condition

    elif order == 2:  # 2nd order Tikhonov minimizes the second derivative
        L = -2 * eye(x_length, format='lil')
        L = diags([np.ones(x_length), np.ones(x_length)], [-1, 1], shape=(x_length, x_length), format='lil') + L
        
        if bc == 0:  # Zero BCs
            L[0, 1] = 0
            L[-1, -2] = 0
        elif bc == 1:  # No slope BCs
            L[0, 1] = 1
            L[-1, -2] = 1
        else:
            L = L[1:-1, :]  # Remove the first and last rows for no explicit BC

    elif order == 3:  # 3rd order Tikhonov (Central difference)
        L = diags([-0.5 * np.ones(x_length), np.ones(x_length), -np.ones(x_length), 0.5 * np.ones(x_length)],
                  [-2, -1, 1, 2], shape=(x_length, x_length), format='lil')
        
        if bc == 0:  # Zero BCs
            L[0, :] = 0
            L[-1, :] = 0
            L[0, 0] = -1
            L[-1, -1] = -1
            L = L[1:-1, :]
        else:
            L = L[2:-2, :]  # Remove two rows at both ends for no explicit BC

    elif order == 4:  # 4th order Tikhonov
        L = 6 * eye(x_length, format='lil')
        L = diags([np.ones(x_length), -4 * np.ones(x_length), -4 * np.ones(x_length), np.ones(x_length)],
                  [-2, -1, 1, 2], shape=(x_length, x_length), format='lil') + L
        
        if bc == 0:  # Zero BCs
            L[0, :] = 0
            L[-1, :] = 0
            L[0, 0] = -1
            L[-1, -1] = -1
            L = L[1:-1, :]
        else:
            L = L[2:-2, :]  # Remove two rows at both ends for no explicit BC

    elif order == 34:  # Combined 3rd and 4th order Tikhonov
        L3 = tikhonov_lpr(3, x_length, bc)
        L4 = tikhonov_lpr(4, x_length, bc)
        L = vstack([L3, L4], format='lil')

    else:
        raise ValueError("The specified order of Tikhonov is not available.")

    L = csr_matrix(L)

    return L

def tikhonov(A, b, lam, order=1, bc=None, xi=None, method=None, Lpr0=None):
    """
    Performs inversion using various order Tikhonov regularization in 2D.

    Parameters:
    A       : ndarray
              Model matrix
    b       : ndarray
              Data
    lam     : scalar or list
              Regularization parameter(s)
    order   : scalar or list (optional)
              Order of regularization or pre-computed Tikhonov matrix structure
    bc      : scalar (optional)
              Boundary conditions
    xi      : ndarray (optional)
              Initial guess for the solver (default: zeros)
    solver  : String (optional)
              Type of least-squares solver (default: set in lsq(...))

    Returns:
    x       : ndarray
              Regularized estimate
    sys     : tuple
              System solved by least-squares.
    D       : ndarray
              Inverse operator (x = D @ [b;0])
    Lpr0    : ndarray
              Tikhonov matrix structure
    """

    n = A.shape[1]  # Length of x
    n_stages = 0
    
    if xi is None:
        xi = np.zeros(n)  # If initial guess is not provided, assume zeros

    # Skip this is Lpr0 is given
    if Lpr0 == None:
        if type(order) == int:
            order = [order]

        # Get Tikhonov smoothing matrix
        Lpr0 = tikhonov_lpr(order[0], n, bc)
    
    Lpr = lam * Lpr0  # Incorporate regularization parameter

    # Execute solver (default solver assumed to be least squares)
    pr_length = Lpr0.shape[0]
    A_aug = vstack([csr_matrix(A), csr_matrix(Lpr)])
    b_aug = np.hstack([np.squeeze(b), np.zeros(pr_length)])
    x, D = lsq(A_aug, b_aug, xi)

    # Multiple stage Tikhonov
    if n_stages > 1:
        for ii in range(1, n_stages):
            # Get Tikhonov smoothing matrix for the next stage
            if len(order) > 0:
                if len(bc) == 0:
                    Lpr0 = tikhonov_lpr(order[ii], n)
                else:
                    Lpr0 = tikhonov_lpr(order[ii], n, bc[ii])

            Lpr = lam[ii] * Lpr0  # Incorporate regularization parameter

            # Apply diagonal weight matrix D2 based on the solution from the previous stage
            D2 = csr_matrix(np.diag(1. / np.maximum(np.abs(x), 0.01)))

            # Solve again for the next stage
            A_aug = vstack([csr_matrix(A), csr_matrix(Lpr @ D2)])
            x, D = lsq(A_aug, b_aug, xi, method)

    # Uncertainty quantification
    if n_stages >= 4:
        Gpo_inv = A.T @ A + Lpr.T @ Lpr
        return x, D, Lpr0, Gpo_inv

    return x, (A_aug, b_aug), D, Lpr0


def twomey(A, b, xi=None, iter=100, f_sigma=True, f_bar=False):
    """
    Performs inversion using the iterative Twomey approach.

    Parameters:
    A : array
        Model matrix or kernel.
    b : array
        Data vector.
    xi : array, optional
        Initial guess for the inversion. Defaults to a vector of ones.
    iter : int, optional
        Number of iterations. Defaults to 100.
    f_sigma : bool, optional
        Flag to check for convergence based on mean square error. Defaults to True.
    f_bar : bool, optional
        Flag to display a progress bar. Defaults to False (not showing bar).

    Returns:
    x : array
        The inverted result.
    """
    
    # Handle default values for parameters
    if xi is None:
        xi = np.ones(A.shape[1])
    
    x = xi

    # Display progress bar if needed
    if f_bar:
        print('Twomey progress:')
    
    # Scaling factors
    s = 1 / np.maximum(np.max(A, axis=1), 1e-10)  # avoid division by zero
    A = (s[:, np.newaxis] * A)  # scale model matrix
    b = np.maximum(b * s, 0)  # remove any negative data and scale

    lam = 1  # factor to adjust step size in Twomey
    
    # Perform Twomey iterations
    f_bar = not f_bar  # invert for input to tqdm
    for kk in tqdm(range(1, iter + 1), disable=f_bar):
        for ii in range(len(b)):
            if b[ii] != 0:
                y = np.dot(A[ii, :], x)
                if y != 0:  # avoid division by zero
                    X = b[ii] / y  # weight factor
                    C = 1 + lam * (X - 1) * A[ii, :]
                    x *= C

        # Check for convergence
        if f_sigma:
            mse = mean_sq_err(A, x, b)
            if mse < 0.01:
                if not f_bar:
                    print('\033[93m' + f'Exited Twomey loop as mean square error reached: iter = {kk}.' + '\033[0m')
                break
    
    return x


def twomark(A, b, n, xi, opt_smooth=2, Sf=1/300):
    """
    Performs inversion using the iterative Twomey-Markowski approach.
    This method adds an intermediate smoothing step to the standard Twomey routine.

    Parameters:
    A : numpy.ndarray
        Model matrix.
    b : numpy.ndarray
        Data vector.
    n : int
        Length of the first dimension of the solution, used in smoothing.
    xi : numpy.ndarray
        Initial guess for the solution.
    opt_smooth : int, optional
        Smoothing option (default is 2).
    Sf : float, optional
        Smoothing factor (default is 1/300).

    Returns:
    x : numpy.ndarray
        The estimated solution vector.
    """
    
    iter_max = 40  # Maximum number of overall Twomey-Markowski iterations
    iter_two = 150  # Maximum number of iterations in Twomey pass

    # Initial Twomey procedure
    x = xi
    x = twomey(A, b, x, iter_two)
    R = roughness(x)  # Roughness vector

    print('Twomey-Markowski progress:')
    for kk in tqdm(range(iter_max)):  # Iterate Twomey and smoothing procedure
        x_temp = np.copy(x)  # Store temporarily for the case that roughness increases
        
        # Perform smoothing
        x, _, err = markowski(A, b, x, n, 1000, opt_smooth)
        if err:
            print(err)
        
        # Perform Twomey
        x = twomey(A, b, x, iter_two)
        
        # Check roughness of solution
        R_new = roughness(x)
        if R_new > R:  # Exit if roughness has stopped decreasing
            print('\033[93m' + f'Exited Twomey-Markowski loop because roughness increased: iter = {kk}.' + '\033[0m')
            x = x_temp  # Restore previous iteration
            break
        
        R = R_new
    
    return x

def roughness(x):
    """
    Computes an estimate of the roughness of the solution.
    This function is used for convergence in the Twomey-Markowski loop 
    and is based on the average, absolute value of the second derivative.

    Parameters:
    x : numpy.ndarray
        The solution vector.

    Returns:
    R : float
        The roughness of the solution.
    """
    
    R = np.abs(x[2:] + x[:-2] - 2 * x[1:-1])
    R = np.sum(R) / (len(x) - 2)
    
    return R

def markowski(A, b, x, n, iter, opt_smooth=3):
    """
    Performs Markowski smoothing, used in the Twomey-Markowski algorithm.

    Parameters:
    A : array
        Model matrix.
    b : array
        Data vector.
    x : array
        Initial guess vector.
    n : int
        The size of the grid or smoothing parameter.
    iter : int
        Number of iterations for the smoothing process.
    opt_smooth : int, optional
        Smoothing option, default is 3 for G3 matrix.

    Returns:
    x : array
        Smoothed output vector.
    G : array
        Smoothing matrix used.
    err : str
        Error message if the algorithm does not converge.
    """
    
    # Set default smoothing option if not provided
    err = None
    
    # Select smoothing matrix based on option
    if opt_smooth == 3:
        G = G3(n)
    else:
        G = G_Markowski(n)

    # Perform smoothing over multiple iterations
    for jj in range(iter):
        x = G @ x  # Apply smoothing
        
        # Exit if mean square error exceeds unity
        if mean_sq_err(A, x, b) > 2:
            return x, G, err
    
    # If not converged after max iterations
    err = f'SMOOTHING algorithm did not necessarily converge after {iter} iteration(s).'
    
    return x, G, err

def G_Markowski(n):
    """
    Generates a smoothing matrix based on the original work of Markowski.

    Parameters:
    n : int
        The size of the smoothing matrix.

    Returns:
    G : array
        Generated smoothing matrix.
    """
    
    G = tikhonov_lpr(2, n, 1)  # Call to Tikhonov method (assumed already implemented)
    
    G = np.abs(G) / 4
    G[0, 1] = 1 / 4
    G[0, 0] = 3 / 4
    G[-1, -2] = 1 / 4
    G[-1, -1] = 3 / 4
    
    return G

def G3(n):
    """
    Generates a G3 smoothing matrix.

    Parameters:
    n : int
        Size of the smoothing matrix.

    Returns:
    G : array
        Generated G3 matrix.
    """
    
    G = np.diag(np.ones(n-2), -2) + 2 * np.diag(np.ones(n-1), -1) + \
        3 * np.diag(np.ones(n)) + 2 * np.diag(np.ones(n-1), 1) + \
        np.diag(np.ones(n-2), 2)
    
    G = G / np.sum(G, axis=1, keepdims=True)
    
    return G