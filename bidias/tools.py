
import numpy as np

def x2y(x, grid_x, fun=None, grid_y=None, axis=1, span_y=None, n_y=200, new_type='', transpose=False):
    """
    Transform a distribution to a different space.
    
    Parameters:
    x : np.ndarray
        Input distribution.
    grid_x : Grid
        Input grid object.
    fun : function, optional
        Transformation function. Default is mass-mobility transformation.
    axis : int, optional
        Dimension to change (default is 1).
    span_y : tuple, optional
        Explicit span of the transformed quantity.
    n_y : int, optional
        Number of elements in the transformed dimension (default is 600).
    new_type : str, optional
        Assigns the type of dimension that is being added (e.g., rho, mp, dm).
    transpose : bool, optional
        Whether to transpose the output distribution before returning (default is False)
    
    Returns:
    y : np.ndarray
        Transformed distribution.
    grid_y : Grid
        New grid for transformed space.
    T : np.ndarray
        Transformation matrix.
    """

    from bidias.Grid import Grid
    
    if fun is None:
        fun, new_type, _, _ = get_transform('mp2rho')

    elif type(fun) == str:
        fun, new_type, _, _ = get_transform(fun)

    dim2 = axis
    dim = 1 - axis  # other dimension (to preserve, switch between 0 and 1)

    if grid_y == None:  # build grid, if not given
        # Estimate span_y if not provided
        if span_y is None:
            f0 = fun(grid_x.elements[:, 0], grid_x.elements[:, 1])
            
            f2 = np.log10(np.max(f0))
            f2 = np.ceil(10 ** (f2 - np.floor(f2)) * 10) / 10 * 10 ** np.floor(f2)
            
            f1 = np.log10(np.min(f0[f0 > 0])) if np.any(f0 > 0) else np.log10(f2 / 1e3)
            f1 = np.floor(10 ** (f1 - np.floor(f1)) * 10) / 10 * 10 ** np.floor(f1)
            
            span_y = (f1, f2)
        
        # Generate new grid for transformed space
        y_n = np.logspace(np.log10(span_y[0]), np.log10(span_y[1]), n_y)  # discretized y space
        
        if dim == 1:
            grid_y = Grid(span=[span_y, grid_x.span[1]], ne=[n_y, len(grid_x.edges[1])])
            grid_y.type = [new_type, grid_x.type[1]]
        else:
            grid_y = Grid(span=[grid_x.span[0], span_y], ne=[len(grid_x.edges[0]), n_y])
            grid_y.type = [grid_x.type[0], new_type]
    
    x_rs = grid_x.reshape(x)
    if dim == 1:
        x_rs = x_rs.T

    # Zero entries that are NaN (i.e., out-of-scope of PartialGrid).
    x_rs[np.isnan(x_rs)] = 0
    
    n_dim = grid_x.ne[dim]
    y = np.zeros((n_dim, grid_y.ne[dim2]))
    
    for ii in range(n_dim):
        T = np.zeros((grid_y.ne[dim2], grid_x.ne[dim2]))
        
        if dim == 1:
            y_old = fun(grid_x.nodes[0], grid_x.edges[1][ii])
        else:
            y_old = fun(grid_x.edges[0][ii], grid_x.nodes[1])
        
        if y_old[1] < y_old[0]:
            y_old = y_old[::-1]
            f_reverse = True
        else:
            f_reverse = False
        
        y_old = np.log10(np.maximum(y_old, 1e-10))
        
        for jj in range(grid_x.ne[dim2]):
            T[:, jj] = np.maximum(
                np.minimum(np.log10(grid_y.nodes[dim2][1:]), y_old[jj + 1]) -
                np.maximum(np.log10(grid_y.nodes[dim2][:-1]), y_old[jj]), 0) / (
                    np.log10(grid_y.nodes[dim2][1:]) - np.log10(grid_y.nodes[dim2][:-1]))
        
        if f_reverse:
            T = np.fliplr(T)
        
        y[ii, :] = T @ x_rs[:, ii]
    
    if dim == 0:
        y = y.T

    y = y.ravel()

    # Doesn't work as function is generic (not required to be power law).
    # if isinstance(grid_x, PartialGrid):
    #     grid_y = PartialGrid(grid_x.span, grid_x.ne, r=grid_x.r, slope=grid_x.slope)
    
    if transpose:
        grid_y, y = grid_y.transpose(y)

    return y, grid_y, T


def get_transform(spec:str):
    
    # Consider preset options, specific by T = str.
    if spec == 'mp2rho':
        c0 = np.array([0, np.log10(6 / np.pi) + 9])
        T = np.array([[1, 0], [-3, 1]])
        fun = lambda b, a: 6 * a / (np.pi * b ** 3) * 1e9
        new_type = 'rho'

    elif spec == 'rho2mp':
        c0 = np.array([0, np.log10(np.pi / 6) - 9])
        T = np.array([[1, 0], [3, 1]])
        fun = lambda b, a: (np.pi/6) * (a*1e-9) * b**3
        new_type = 'mp'

    # elif spec == 'dm2rho':
    #     c0 = np.array([0, np.log10(np.pi / 6) - 9])
    #     T = np.array([[0, 1], [-3, 1]])
    #     fun = lambda b, a: (np.pi/6) * (a*1e-9) * b**3
    #     new_type = 'rho'

    return fun, new_type, T, c0
