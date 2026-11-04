
import numpy as np

import scipy.sparse as sp
from scipy.sparse import coo_matrix, lil_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from cmap import get_colormap

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
        discrete = ['lin' if item.lower() == 'linear' else item for item in discrete]  # convert to shorthand for linear dimensions
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
        self.ielements = None  # elements in terms of index
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
                r_m = np.exp((np.log(self.edges[ii][1:]) + np.log(self.edges[ii][:-1])) / 2)  # mid-point in log-space
                self.nodes[ii] = np.concatenate([[np.exp(2 * np.log(self.edges[ii][0]) - np.log(r_m[0]))], r_m, [np.exp(2 * np.log(self.edges[ii][-1]) - np.log(r_m[-1]))]])
                self.nodes_tr[ii] = np.log10(self.nodes[ii])
            elif self.discrete[ii] == 'lin':
                r_m = (self.edges[ii][1:] + self.edges[ii][:-1]) / 2  # mid-point in linear space
                self.nodes[ii] = np.concatenate([[2 * self.edges[ii][0] - r_m[0]], r_m, [2 * self.edges[ii][-1] - r_m[-1]]])
                self.nodes_tr[ii] = self.nodes[ii]

        vec1 = np.meshgrid(self.edges[0], self.edges[1])
        self.elements = np.vstack([vec1[0].ravel(), vec1[1].ravel()]).T
        
        ivec1 = np.meshgrid(np.arange(self.ne[0]), np.arange(self.ne[1]))
        self.ielements = np.vstack([ivec1[0].ravel(), ivec1[1].ravel()]).T

        vec1 = np.meshgrid(self.nodes[0][:-1], self.nodes[1][:-1])
        vec2 = np.meshgrid(self.nodes[0][1:], self.nodes[1][1:])
        self.nelements = np.vstack([vec1[0].ravel(), vec2[0].ravel(), vec1[1].ravel(), vec2[1].ravel()]).T

        self.nelements_tr = self.nelements.copy()
        for ii in range(self.dim):
            if self.discrete[ii] == 'log':
                self.nelements_tr[:,2*ii:2*ii+2] = np.log10(self.nelements_tr[:,2*ii:2*ii+2])

    def __repr__(self):
        """
        Generates and prints a smooth-cornered ASCII graphic of the grid 
        showing limits (span) and counts (ne).
        
        Parameters:
        scale_factor (int): Multiplier for the internal display width/height 
                            to control the size of the empty box (default is 2).
        """
        
        # --- Determine dimensions and metadata ---
        scale_factor = 4 / np.min(self.ne)

        # Width of the internal data area (scaled)
        inner_width = int(np.ceil(self.ne[0] * scale_factor * 3))  # Factor of 2 for better horizontal stretch
        inner_height = int(np.ceil(self.ne[1] * scale_factor))
        
        # Metadata strings (formatted to fit standard display space)
        x_min_str = f"x={self.span[0][0]:.4g}"
        x_max_str = f"x={self.span[0][1]:.4g}"
        y_min_str = f"y={self.span[1][0]:.4g}"
        y_max_str = f"y={self.span[1][1]:.4g}"
        
        w_count_str = f"nx={self.ne[0]}"
        h_count_str = f"ny={self.ne[1]}"
        
        # Label width on the left (for y_max, Ny, y_min labels)
        label_width = max(len(y_max_str), len(h_count_str), len(y_min_str)) + 1
        
        # --- Build components ---
        
        # Horizontal Border Character
        H_BORDER = '─'
        full_width = label_width + 1 + inner_width  # full width for border line construction
        
        # Helper function to create padding for centered labels
        def create_padding(label, total_width, char=H_BORDER):
            padding_len = total_width - len(label)
            left_pad = padding_len // 2
            right_pad = padding_len - left_pad
            return char * left_pad + label + char * right_pad

        # --- Construct Output Lines ---
        output = ["\033[1mGRID\033[0m:"]

        # Top Line: y_max ╭─────────nx──────────╮
        top_center_label = create_padding(w_count_str, inner_width, char=H_BORDER)
        top_line = f"{y_max_str.ljust(label_width)}╭{top_center_label}╮"
        output.append(top_line)
        
        # Middle Lines: |   (Empty Space)   |
        for i in range(inner_height):
            left_label = ' ' * label_width
            
            # Place ny label near the vertical center
            if i == inner_height // 2:
                left_label = h_count_str.ljust(label_width)
            
            middle_line = f"{left_label}│{' ' * inner_width}│"
            output.append(middle_line)

        # Bottom Line: y_min ╰───────────╯
        bottom_line = f"{y_min_str.ljust(label_width)}╰{H_BORDER * inner_width}╯"
        output.append(bottom_line)

        # X-Limits Line: x_min       nx          x_max
        # Note: x_min and x_max labels need to span the space created by the horizontal border
        x_min_pad_len = label_width + 1 # Space before x_min
        x_max_pad_len = full_width - len(x_max_str) - x_min_pad_len

        x_limit_line = (
            f"{' ' * x_min_pad_len}"
            f"{x_min_str}"
            f"{' ' * (inner_width - len(x_min_str) - len(x_max_str) + 1)}" # Spacing between x_min and x_max
            f"{x_max_str}"
        )
        
        output.append(x_limit_line)

        # Print the final visualization
        return '\n'.join(output)


    @staticmethod
    def adjacency0(ne, anisotropy=1.0):
        """
        Compute the adjacency matrix using a four-point stencil.
        Allows for anisotropy (> 1 weights horizontal more significantly).
        This can be called independently of instance of Grid class.
        """
        Ne = np.prod(ne)
        nx = ne[0]
        
        w_x, w_y = anisotropy, 1.0
        ind1, ind2, vec = [], [], []

        for jj in range(Ne):
            # Horizontal (X)
            if (jj + 1) % nx != 0:
                ind1.extend([jj, jj + 1])
                ind2.extend([jj + 1, jj])
                vec.extend([w_x, w_x])
                
            # Vertical (Y)
            if jj < (Ne - nx):
                ind1.extend([jj, jj + nx])
                ind2.extend([jj + nx, jj])
                vec.extend([w_y, w_y])

        return coo_matrix((vec, (ind1, ind2)))

    def adjacency(self, **kwargs):
        """Bridging function to static method."""
        return Grid.adjacency0(self.ne, **kwargs)

    def isedge(self):
        """
        A function to compute which elements are at the edges of the grid domain.
        """
        pts = self.ielements
        # A set of tuples is the fastest way to check existence in Python
        index_set = set(map(tuple, pts))
        
        # Define the 4 cardinal shifts: Left, Right, Top, Bottom
        # We use broadcasting to calculate all potential neighbor coords at once
        shifts = [
            pts + [-1, 0],  # Left
            pts + [1, 0],   # Right
            pts + [0, -1],  # Top
            pts + [0, 1]    # Bottom
        ]
        
        # Generate the boolean mask for each direction
        # We use a list comprehension over the directions rather than the points
        edge_types = np.array([
            [tuple(neighbor) not in index_set for neighbor in shift]
            for shift in shifts
        ]).T  # Transpose to get (n, 4) shape

        # Get indices where at least one direction is True
        is_edge_indices = np.where(edge_types.any(axis=1))[0]

        return is_edge_indices, edge_types

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
    
    def marginalize(self, x, axis=0):
        """
        Marginalize the size distribution. 
        """
        return np.nansum(self.dr()[2-axis] * self.reshape(x), axis=axis)

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
        dr2, dr1 = np.meshgrid(dr_0[1], dr_0[0], indexing='ij')
        
        # Ensure positive differential areas in case of reversed edges
        dr1 = np.abs(dr1)
        dr2 = np.abs(dr2)

        # Flatten the grids and compute element-wise product
        dr = (dr1.ravel() * dr2.ravel())
        
        return dr, dr1, dr2
    
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
    
    def plot2d(self, x, cmap='rocket_r', bg=True,**kwargs):

        if type(cmap) is str:
            cmap = get_colormap(cmap)

        xp, yp = np.meshgrid(self.edges[0], self.edges[1])
        
        mesh = plt.pcolor(xp, yp, self.reshape(x), cmap=cmap, **kwargs)

        # This creates an empty version of the plot with axes for saving purposes.

        if self.discrete[0] == 'log':
            plt.xscale('log')

        if self.discrete[1] == 'log':
            plt.yscale('log')

        # -- PLOT ONLY AXES WITHOUT FILL --
        if bg==False:
            limx = plt.xlim()  # only get axis limits
            limy = plt.ylim()

            plt.cla()  # clear axes to reset

            plt.plot([], [])  # plot empty axes
            plt.xlim(limx)
            plt.ylim(limy)
            if self.discrete[0] == 'log':
                plt.xscale('log')
            if self.discrete[1] == 'log':
                plt.yscale('log')
        # ---------------------------------

        if not self.type == None:
            plt.xlabel(self.type[0])
            plt.ylabel(self.type[1])

        plt.gca().set_box_aspect(1)

        return mesh
    
    def plot2d_marg(self, x, n=5, **kwargs):
        """
        Bidimensional plot with marginal distributions. 
        """

        ax_main = plt.subplot2grid((n, n), (1, 0), colspan=n, rowspan=n-1)
        mesh = self.plot2d(x, **kwargs)

        ax_top = plt.subplot2grid((n, n), (0, 0), colspan=n, rowspan=1, sharex=ax_main)
        plt.plot(self.edges[0], self.marginalize(x, axis=0))

        ax_top.tick_params(axis='y', left=False, labelleft=False)
        ax_top.tick_params(axis='x', bottom=True, labelbottom=False)
        plt.xscale('log')

        ax_right = plt.subplot2grid((n, n), (1, n-1), colspan=1, rowspan=n-1, sharey=ax_main)
        plt.plot(self.marginalize(x, axis=1), self.edges[1])
        ax_right.tick_params(axis='y', left=False, labelleft=False)
        ax_right.tick_params(axis='x', bottom=True, labelbottom=False)
        plt.yscale('log')

        # --- ALIGN AXES ---
        # Force the top and right plots to match the 'shrunk' dimensions of the 1:1 box
        plt.draw() # Necessary to calculate the box aspect positions

        # Adjust Top Plot width to match Main Plot
        pos_main = ax_main.get_position()
        pos_top = ax_top.get_position()
        ax_top.set_position([pos_main.x0, pos_top.y0, pos_main.width, pos_top.height])

        # Adjust Right Plot height to match Main Plot
        pos_right = ax_right.get_position()
        ax_right.set_position([pos_right.x0, pos_main.y0, pos_right.width, pos_main.height])

        plt.sca(ax_main)

        return ax_main, ax_top, ax_right, mesh

    def scatter(self, x, cmap='mako_r', edgecolors='k', linewidth=0.2, **kwargs):

        if type(cmap) is str:
            cmap = get_colormap(cmap)
        
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
        self.ielements = self.ielements[self.remaining, :]
        self.nelements = self.nelements[self.remaining, :]
        self.nelements_tr = self.nelements_tr[self.remaining, :]
        self.Ne = self.elements.shape[0]

        self.adj = PartialGrid.adjacency(self)


    def adjacency(self, **kwargs):
        # Call inherited adjacency method from the Grid class
        adj = Grid.adjacency(self, **kwargs)
        
        # Remove rows and columns corresponding to missing elements
        adj = adj.tocsr()
        adj = adj[self.remaining, :][:, self.remaining]
        adj = coo_matrix(adj)

        return adj
    
    # def isedge(self):
    #     """
    #     A function to compute which elements are at the edges of the grid domain.
    #     """
    #     # Identify elements next to the new edges.
    #     adju = sp.triu(Grid.adjacency(self, 1), 2).tocsr()  # Get the upper triangle of the matrix with offset 2
    #     missingedge = adju[:, self.missing].sum(axis=1).flatten().astype(bool)  # Check for adjacency to missing elements
    #     missingedge = np.delete(missingedge, self.missing)  # Remove missing elements from the edge flagging
    #     missingedge = np.where(missingedge.A1)[0]

    #     # Finally, return the combination of the 
    #     # original edge elements for full grid with the new edge elements.
    #     return np.concatenate((Grid.isedge(self), missingedge))
    
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
        mesh = super().plot2d(x, **kawrgs)

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

        return mesh
    
    def transpose(self, x=None):
        if np.shape(self.r)[0] == 1:  # if only one line, add second at -np.inf
            r = np.vstack((self.r, [[self.r[0][0],-np.inf]]))
            slope = np.concatenate((self.slope, [1]))
        else:
            r = self.r
            slope = self.slope

        grid = PartialGrid(edges=[self.edges[1], self.edges[0]], discrete=np.flip(self.discrete), 
                           r=np.flip(np.flip(r, axis=1), axis=0), slope=[1/sl for sl in slope])
        if not self.type == None:
            grid.type = [self.type[1], self.type[0]]

        if np.any(x == None):
            return grid
        else:
            x = grid.full2partial((self.reshape(x).T).ravel())
            return grid, x
