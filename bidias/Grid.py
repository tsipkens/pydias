
import numpy as np

from scipy.sparse import coo_matrix, lil_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm

class Grid:
    def __init__(self, span=None, ne=None, edges=None, discrete=['log', 'log'], type=['dm', 'mp']):
        """
        Initialize a 2D grid object.

        Parameters:
        span (list of lists of int): A list of two pairs representing the minimum and maximum values in each dimension.
        ne (list of int): A list of two integers specifying the number of elements in each dimension.
        edges (list of lists): A list of two arrays, each representing the edges of the grid in two dimensions.
        discrete (list of str): A list specifying the discretization method for each dimension ('log' or 'lin').

        Raises:
        ValueError: If neither edges nor span/ne is provided.
        """
        
        # Initiate properties and set some defaults.
        self.discrete = discrete
        self.dim = 2
        self.type = type  # types for the grid (e.g., 'mp', 'dm', 'da', 'mrBC', 'rho')
        self.span = None

        self.ne = None
        self.Ne = None

        self.edges = [None] * self.dim
        self.nodes = [None] * self.dim
        self.nodes_tr = [None] * self.dim  # transformed nodes (used for dr)

        self.elements = None
        self.nelements = None
        self.nelements_tr = None

        self.adj = None  # adjacency matrix

        # Handle input
        if not edges == None:  # Edges provided
            self.edges = edges
            self.ne = [len(edges[0]), len(edges[1])]
            self.span = [[min(edges[0]), max(edges[0])], [min(edges[1]), max(edges[1])]]
        elif not span == None:  # Span provided
            self.span = span
            self.ne = ne
            # Edges assigned during generate_mesh(...) call below. 
        else:
            raise ValueError("Incorrect inputs to form Grid. Either specify EDGES or both SPAN and NE.")
        
        self.Ne = np.prod(self.ne)
        
        self.generate_mesh()
        self.adj = Grid.adjacency(self)

    def generate_mesh(self):
        # Generate edges if not explicitly stated.
        if np.any(self.edges[0] == None):
            for ii in range(self.dim):
                if self.discrete[ii] == 'log':
                    self.edges[ii] = np.logspace(np.log10(self.span[ii][0]), np.log10(self.span[ii][1]), self.ne[ii])
                elif self.discrete[ii] == 'lin':
                    self.edges[ii] = np.linspace(self.span[ii][0], self.span[ii][1], self.ne[ii])

        self.Ne = np.prod(self.ne)

        # Generate nodes
        for ii in range(self.dim):
            if self.discrete[ii] == 'log':
                r_m = np.exp((np.log(self.edges[ii][1:]) + np.log(self.edges[ii][:-1])) / 2)
                self.nodes[ii] = np.concatenate([[np.exp(2 * np.log(self.edges[ii][0]) - np.log(r_m[0]))], r_m, [np.exp(2 * np.log(self.edges[ii][-1]) - np.log(r_m[-1]))]])
                self.nodes_tr[ii] = np.log10(self.nodes[ii])
            elif self.discrete[ii] == 'lin':
                r_m = (self.edges[ii][1:] + self.edges[ii][:-1]) / 2
                self.nodes[ii] = np.concatenate([[2 * self.edges[ii][0] - r_m[0]], r_m, [2 * self.edges[ii][-1] - r_m[-1]]])
                self.nodes_tr[ii] = self.nodes[ii]

        vec1 = np.meshgrid(self.edges[0], self.edges[1])
        self.elements = np.vstack([vec1[0].ravel(), vec1[1].ravel()]).T
        
        vec1 = np.meshgrid(self.nodes[0][:-1], self.nodes[1][:-1])
        vec2 = np.meshgrid(self.nodes[0][1:], self.nodes[1][1:])
        self.nelements = np.vstack([vec1[0].ravel(), vec2[0].ravel(), vec1[1].ravel(), vec2[1].ravel()]).T

        self.nelements_tr = self.nelements.copy()
        for ii in range(self.dim):
            if self.discrete[ii] == 'log':
                self.nelements_tr[:,2*ii:2*ii+2] = np.log10(self.nelements_tr[:,2*ii:2*ii+2])

    def adjacency(self, w=1):
        ind1 = []
        ind2 = []
        vec = []

        for jj in range(np.prod(self.ne)):
            if (jj + 1) % self.ne[0] != 0:  # up pixels
                ind1.append(jj)
                ind2.append(jj + 1)
                vec.append(w)
            if jj % self.ne[0] != 0:  # down pixels
                ind1.append(jj)
                ind2.append(jj - 1)
                vec.append(w)
            if jj >= self.ne[0]:  # left pixels
                ind1.append(jj)
                ind2.append(jj - self.ne[0])
                vec.append(1)
            if jj < (np.prod(self.ne) - self.ne[0]):  # right pixels
                ind1.append(jj)
                ind2.append(jj + self.ne[0])
                vec.append(1)

        adj = coo_matrix((vec, (ind1, ind2)), shape=(np.prod(self.ne), np.prod(self.ne)))
        return adj

    def reshape(self, x):
        return x.reshape([self.ne[1], self.ne[0]])

    def vectorize(self, x):
        return x.flatten()
    
    def logify_r(self, r):

        for ii in range(len(r)):
            if self.discrete[0] == 'log':
                r[ii][0] = np.log10(r[ii][0])
            if self.discrete[1] == 'log':
                r[ii][1] = np.log10(r[ii][1])
        
        return np.asarray(r)

    def dr(self):
        """
        Calculates the differential area of the elements in the grid.

        Returns:
        - dr (np.ndarray): Differential area for the grid.
        - dr1 (np.ndarray): Differential area in the first dimension.
        - dr2 (np.ndarray): Differential area in the second dimension.
        """
        dr_0 = [None] * self.dim  # Initialize list for differential values

        for ii in range(self.dim):
            dr_0[ii] = self.nodes_tr[ii][1:] - self.nodes_tr[ii][:-1]

        # Create grid of differential values using ndgrid equivalent (meshgrid in numpy)
        dr1, dr2 = np.meshgrid(dr_0[0], dr_0[1], indexing='ij')
        
        # Ensure positive differential areas in case of reversed edges
        dr1 = np.abs(dr1)
        dr2 = np.abs(dr2)

        # Flatten the grids and compute element-wise product
        dr = (dr1.ravel() * dr2.ravel())
        
        return dr, dr1, dr2
    
    def l1(self, w=1, bc=1):
        """
        Compute the first-order Tikhonov operator.
        W adds a weight used to reevaluate the adjacency matrix.

        Parameters:
        w: Optional weight used to recalculate the adjacency matrix.
        bc: Optional boundary condition flag. If bc == 0, force zeros at the boundary.

        Returns:
        l1: First-order Tikhonov operator matrix.
        """
        
        adj_local = Grid.adjacency(self, w)  # Reevaluate adjacency with weight
        
        # Compute L1: -diag(sum(tril(adj_local))) + triu(adj_local)
        l1 = -np.diag(np.sum(np.tril(adj_local.todense()), axis=1)) + np.triu(adj_local.todense())
        
        # Add unity on diagonal in final row for stability in square matrix
        l1[-1, -1] = -1
        
        if bc == 0:  # Force zeros at boundary condition
            isedge = np.where(self.elements[:, 1] == self.edges[1][-1])[0]
            isedge = np.concatenate((isedge, np.where(self.elements[:, 0] == self.edges[0][-1])[0]))
            l1[np.ix_(isedge, isedge)] = np.eye(len(isedge))  # Replace entries with identity
        
        return l1
    
    def ray_sum(self, r, slope, f_bar=True):
        """
        Perform a ray sum for a given ray and the current grid.
        Currently assumes a uniform, logarithmic grid and can accommodate partial grids.
        
        Args:
            r (array): A point on each ray.
            slope (float): Slope of the ray.

        Returns:
            C (sparse matrix): Ray-sum matrix.
            rmin (array): Minimum points of the ray intersections.
            rmax (array): Maximum points of the ray intersections.
        """

        f_bar = not f_bar

        slope = np.asarray(slope)
        r = self.logify_r(r)

        #-- Preallocate arrays ---------------------------------------#
        m = np.size(slope)
        C = lil_matrix((m, self.Ne))  # lil_matrix allows efficient row-wise operations

        #-- Compute ray-sum matrix -----------------------------------#
        for ii in tqdm.tqdm(range(m), disable=f_bar):  # loop over multiple rays
            #-- Ray vector -------------#
            dv = np.array([1, slope[ii]])  # Convert slope to step vector along line
            dv = dv / np.linalg.norm(dv)
            dv[dv == 0] = 1e-10  # for stability during division

            #-- Line intersections -----#
            # Parametric representation of the line and finds two intersections for each element
            # Assuming a logarithmic grid
            tmin = (self.nelements_tr[:, [2, 0]] - r) / dv  # min of element
            tmax = (self.nelements_tr[:, [3, 1]] - r) / dv  # max of element

            #-- Corrections ------------#
            # Decide which points correspond to transecting the pixel
            tmin = np.max(tmin, axis=1)
            tmax = np.min(tmax, axis=1)

            #-- Convert back to [x, y] --#
            rmin = r + tmin[:, np.newaxis] * dv  # Intersection with min. of pixel
            rmax = r + tmax[:, np.newaxis] * dv  # Intersection with max. of pixel

            rmin = np.minimum(rmin, self.nelements_tr[:, [3, 1]])
            rmax = np.maximum(rmax, self.nelements_tr[:, [2, 0]])

            chord = np.sqrt(np.sum((rmax - rmin) ** 2, axis=1))  # Chord length
            chord[chord < 1e-15] = 0  # Truncate small values

            #-- Ray-sum matrix ---------#
            jj = np.nonzero(chord)[0]  # Indices of non-zero chords
            a = chord[jj]  # Values of the non-zero chords
            if len(a) > 0:
                C[ii, jj] = lil_matrix((1, len(jj)), dtype=float)
                C[ii, jj] = a  # Store chord lengths in sparse matrix

            #-- Modify rmin and rmax for output -----# 
            rmin = np.fliplr(rmin)  # Flip left-right for output
            rmax = np.fliplr(rmax)

        return C, rmin, rmax
    
    def plot2d(self, x, cmap=sns.color_palette('rocket_r', as_cmap=True), **kwargs):

        xp, yp = np.meshgrid(self.edges[0], self.edges[1])
        
        plt.pcolor(xp, yp, self.reshape(x), cmap=cmap, **kwargs)

        if self.discrete[0] == 'log':
            plt.xscale('log')

        if self.discrete[1] == 'log':
            plt.yscale('log')

        if not self.type == None:
            plt.xlabel(self.type[0])
            plt.ylabel(self.type[1])

        plt.gca().set_box_aspect(1)

    def scatter(self, x, cmap=sns.color_palette('mako_r', as_cmap=True), edgecolors='k', linewidth=0.2, **kwargs):
        
        plt.scatter(self.elements[:,0], self.elements[:,1], 20 + 35 * x / np.max(x), x, \
                    cmap=cmap, edgecolors=edgecolors, linewidth=linewidth, **kwargs)

        if self.discrete[0] == 'log':
            plt.xscale('log')

        if self.discrete[1] == 'log':
            plt.yscale('log')

        if not self.type == None:
            plt.xlabel(self.type[0])
            plt.ylabel(self.type[1])

        plt.colorbar()

    def sweep(self, x, cmap=None, edgecolors=None, **kwargs):
        
        if cmap == None:
            cmap = sns.color_palette('mako_r', as_cmap=True)
        
        if edgecolors == None:
            edgecolors = 'k'

        plt.scatter(self.elements[:,0], self.elements[:,1], 10 + 45 * x / np.max(x), x, \
                    cmap=cmap, edgecolors=edgecolors, **kwargs)

        if self.discrete[0] == 'log':
            plt.xscale('log')

        if self.discrete[1] == 'log':
            plt.yscale('log')

        if not self.type == None:
            plt.xlabel(self.type[0])
            plt.ylabel(self.type[1])

        plt.colorbar()
    
    def transpose(self, x=None):
        grid = Grid(edges=[self.edges[1], self.edges[0]], discrete=np.flip(self.discrete))
        if not self.type == None:
            grid.type = [self.type[1], self.type[0]]

        if np.any(x == None):
            return grid
        else:
            x = (self.reshape(x).T).ravel()
            return grid, x

    
class PartialGrid(Grid):
    def __init__(self, span=None, ne=None, r=[1, np.inf], slope=[1], **kwargs):
        super().__init__(span=span, ne=ne, **kwargs)  # inherit from Grid superclasss
    
        if not type(slope) == list:
            slope = [slope]

        n = np.size(slope)  # number of conditions (1 or 2)
        
        self.r = np.copy(r)
        self.slope = slope
        
        if self.discrete[1] == 'log':
            for ii in range(n):
                r[ii][1] = np.log10(r[ii][1])
                r[ii][0] = np.log10(r[ii][0])
        
        b = [None] * n
        for ii in range(n):
            b[ii] = r[ii][1] - slope[ii] * r[ii][0]
        self.b = b

        f_missing = self.nelements_tr[:, 3] > (self.nelements_tr[:, 1] * slope[0] + b[0])
        if len(slope) > 1:
            f_missing = np.logical_or(f_missing, self.nelements_tr[:, 2] < (self.nelements_tr[:, 0] * slope[1] + b[1]))

        idx = np.arange(self.Ne)
        self.remaining = idx[~f_missing]
        self.missing = idx[f_missing]

        # Update grid properties after truncation
        self.elements = self.elements[self.remaining, :]
        self.nelements = self.nelements[self.remaining, :]
        self.nelements_tr = self.nelements_tr[self.remaining, :]
        self.Ne = self.elements.shape[0]

        self.adj, _ = PartialGrid.adjacency(self)

    
    def adjacency(self, w=1):
        """
        Compute the adjacency matrix for the full grid using a four-point stencil.
        
        Parameters:
        w: Optional weight to apply to vertical pixels.
        
        Returns:
        adj: Adjacency matrix after processing.
        isedge: Boolean array indicating whether an element is adjacent to a new edge.
        """

        # Call inherited adjacency method from the Grid class
        adj = Grid.adjacency(self, w)
        adj = adj.todense()

        # Identify elements next to the new edge
        adju = np.triu(adj, 2)  # Get the upper triangle of the matrix with offset 2
        isedge = np.any(adju[:, self.missing], axis=1)  # Check for adjacency to missing elements
        isedge = np.delete(isedge, self.missing)  # Remove missing elements from the edge flagging
        
        # Remove rows and columns corresponding to missing elements
        adj = adj[self.remaining, :]
        adj = adj[:, self.remaining]
        adj = coo_matrix(adj)

        return adj, isedge
    
    def reshape(self, x):
        x = self.partial2full(x)
        return x.reshape([self.ne[1], self.ne[0]])

    def dr(self):
        """
        Calculates the differential area of the elements in the grid.
        """

        # Call the dr method from the parent Grid class
        dr, dr1, dr2 = super().dr()

        dr = self.full2partial(dr)  # used if lower cut is employed

        # NOTE: Could add areas for partial cells. Currently use whole cells if any part is in the domain. 

        return dr, dr1, dr2
    
    def l1(self, w=1, bc=1):
        """
        Compute the first-order Tikhonov operator.
        
        Parameters:
        w: Optional weight used to re-evaluate the adjacency matrix.
        bc: Boundary condition flag. If 0, forces zeros at the boundary condition.
        
        Returns:
        l1: First-order Tikhonov operator matrix.
        """

        # Re-evaluate adjacency with weight if provided
        adj_local, _ = self.adjacency(w)
        adj_local = adj_local.todense()

        # Calculate l1 matrix
        l1 = -np.diag(np.sum(np.tril(adj_local), axis=1)) + np.triu(adj_local)

        # Add unity on the diagonal in the final row for stability in square matrix
        l1[-1, -1] = -1

        # Force zeros at the boundary condition if bc == 0
        if bc == 0:
            isedge = np.where(self.elements[:, 1] == self.edges[1][-1])[0]
            isedge = np.concatenate((isedge, np.where(self.elements[:, 0] == self.edges[0][-1])[0]))

            # Replace entries with identity matrix for boundary condition
            l1[isedge[:, None], isedge] = np.eye(len(isedge))

        return l1
    
    def partial2full(self, x):
        """
        Convert x defined on a partial grid to the full grid equivalent.
        Fill removed grid points with zeros.
        """
        x_full = np.empty(np.prod(self.ne))
        x_full[:] = np.nan

        x_full[self.remaining] = x
        return x_full

    def full2partial(self, x):
        """
        Convert x defined on a full grid to the partial grid equivalent.
        Removes entries for missing indices.
        """
        return x[self.remaining]

    def plot2d(self, x, **kawrgs):
        """
        Convert x defined on a full grid to the partial grid equivalent.
        Removes entries for missing indices.
        """
        
        x[np.isnan(x)] = 0
        super().plot2d(x, **kawrgs)

        ncut = len(self.slope)

        yl = plt.gca().get_ylim()
        xl = plt.gca().get_xlim()
        for ii in range(ncut):
            if self.discrete[0] == 'log':
                ye = np.log10(xl) * self.slope[ii] + self.b[ii]
            else:
                ye = xl * self.slope[ii] + self.b[ii]

            if self.discrete[1] == 'log':
                ye = 10 ** ye

            plt.plot(xl, ye, color='k', linewidth=0.5, linestyle='--')

        plt.gca().set_ylim(yl)
