
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

import bidias.tools as tools

import warnings


class Phantoms:
    """
    Wraps Phantom to handle cases of multiple modes. 
    """
    def __init__(self, spec=None, mu=None, Sig=None, p=None, massmob=None, w=None):
        # Determine source and argument type
        if mu is not None and Sig is not None:
            self.nmodes = len(mu)
            self.modes = [Phantom(mu=mu[i], Sig=Sig[i]) for i in range(self.nmodes)]
        elif p is not None:
            self.nmodes = len(p)
            self.modes = [Phantom(p=p[i]) for i in range(self.nmodes)]
        elif massmob is not None:
            self.nmodes = len(massmob)
            self.modes = [Phantom(massmob=massmob[i]) for i in range(self.nmodes)]
        else:
            raise ValueError("Phantoms could not be created. Insufficient inputs!")

        # Initialize weights
        self.w = w if w is not None else np.ones(self.nmodes) / self.nmodes
    
    def eval(self, *args, **kwargs):
        return sum(weight * mode.eval(*args, **kwargs) for weight, mode in zip(self.w, self.modes))
    
    def show(self, nx=70, ny=70, nc=40):
        """
        Quick plot of the combined Phantoms modes with weights,
        reusing each Phantom's show logic.
        """
        # Determine combined plotting range
        s1_list = [np.sqrt(mode.Sig[0,0]) for mode in self.modes]
        s2_list = [np.sqrt(mode.Sig[1,1]) for mode in self.modes]
        mu1_list = [mode.mu[0] for mode in self.modes]
        mu2_list = [mode.mu[1] for mode in self.modes]

        x = np.linspace(min(mu1 - 3*s1 for mu1, s1 in zip(mu1_list, s1_list)),
                        max(mu1 + 3*s1 for mu1, s1 in zip(mu1_list, s1_list)), nx)
        y = np.linspace(min(mu2 - 3*s2 for mu2, s2 in zip(mu2_list, s2_list)),
                        max(mu2 + 3*s2 for mu2, s2 in zip(mu2_list, s2_list)), ny)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        # Use each Phantom's PDF instead of duplicating logic
        Z = sum(w * mode.rv.pdf(pos) for w, mode in zip(self.w, self.modes))

        plt.figure()
        plt.contourf(10**X, 10**Y, Z, nc)
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().set_box_aspect(1)


class Phantom:
    def __init__(self, spec=None, mu=None, Sig=None, p=None, massmob=None):
        # Default values
        self.type = spec
        self.mu = []
        self.Sig = []
        self.R = []

        self.p = {}
        self.massmob = {}
        
        self.rv = None

        # -- Determine input type --
        if not mu is None and not Sig is None:  # direct mu + Sig input
            self.mu = np.asarray(mu)
            self.Sig = np.asarray(Sig)

        elif not p is None:  # dictionary of means and GSDs input
            [self.mu, self.Sig] = self.p2mu_sig(**p)

        elif not massmob is None:  # dictionary of mass-mobility parameters
            # Mapping of old keys → new keys
            rename_map = {
                'dg': 'mu1', 'mg': 'mu2',
                'sd': 's1', 'sm': 's2',
                'sm_d': 's2|1', 'sm|d': 's2|1', 
                'Dm': 'pow', 'zet': 'pow', 
            }

            # Create a new dictionary with renamed keys if they exist
            p = {rename_map.get(k, k): v for k, v in massmob.items()}

            # Now build analogous to above. 
            [self.mu, Sig] = self.p2mu_sig(**p)

            # Perform check of covariance.
            if np.abs(self.cov2corr(Sig)[0,1]) > 1:
                warnings.warn('Warning: Phantom correlation exceeded unity. Adjusted to R12 = 1.')
                Sig[0,1] = 1
                Sig[1,0] = 1
            self.Sig = Sig

        else:
            print('Phantom could not be created. Insufficient inputs!')
            return

        self.rv = multivariate_normal(self.mu, self.Sig)

        self.R = self.cov2corr(self.Sig)
        self.p = self.mu_sig2p(self.mu, self.Sig)
        self.massmob = self.p2massmob(self.p)

        self.power_law = lambda x: 10 ** (self.p['pow'] * (np.log10(x) - self.mu[0]) + self.mu[1])

    @staticmethod
    def mu_sig2p(mu, Sig):
        """
        Convert means and covariance (in log10-space) to a p dictionary. 
        """
        R12 = Phantom.cov2corr(Sig)[0, 1]
        return {
            "mu1": 10**mu[0],
            "mu2": 10**mu[1],
            "s1": 10**np.sqrt(Sig[0, 0]),
            "s2": 10**np.sqrt(Sig[1, 1]),
            "s2|1": 10**np.sqrt(Sig[1, 1] * (1 - R12**2)),
            "pow": Sig[0, 1] / Sig[0, 0],
            "R12": R12,
        }

    @staticmethod
    def p2mu_sig(**p):
        """
        Convert a dictionary of properties in p dictionary in log10 mean and covariance. 
        """
        # Apply log10 to the relevant fields. 
        fields = ['mu1', 'mu2', 's1', 's2', 's2|1']
        for field in fields:
            if field in p:
                p[field] = np.log10(p[field])

        # Check if pow is used instead of one of the standard deviations. 
        # Calculate the missing standard deviation.
        if 'pow' in p:
            if 's2|1' in p and not 'R12' in p:
                if 's1' in p:
                    p['R12'] = 1 / np.sqrt(1 + (p['s2|1'] / (p['s1'] * p['pow']))**2)
                elif 's2' in p:
                    p['R12'] = np.sqrt(1 - (p['s2|1'] / (p['s2']))**2)
            
            if p['pow'] == 0:
                p['s2'] = p['s2|1']
            elif not 's1' in p:
                p['s1'] = p['s2'] * p['R12'] / p['pow']
            elif not 's2' in p:
                p['s2'] = p['s1'] / p['R12'] * p['pow']

        return get_cov(**p)

    @staticmethod
    def p2massmob(p):  # extra parameters computed from standard p
        dg, mg = p['mu1'], p['mu2']
        sd, sm = p['s1'], p['s2']
        zet, sm_d = p['pow'], p['s2|1']

        rhog = 6 * mg / (np.pi * dg**3) * 1e9
        rho100 = rhog * (100 / dg)**(zet - 3)
        m100 = 1e-9 * rho100 * np.pi / 6 * 100**3
        k = m100 / (100**zet)

        massmob = {
            "dg": dg, "mg": mg, "sd": sd, "sm": sm,
            "zet": zet, "Dm": zet,
            "rhog": rhog, "rho100": rho100, "m100": m100, "k": k,
            "sm|d": sm_d,
        }

        return massmob

    @staticmethod
    def cov2corr(Sig):
        """
        Converts a covariance matrix to a correlation matrix.

        Parameters:
        ----------
        Sig : numpy.ndarray
            A 2D covariance matrix of shape (2, 2).

        Returns:
        -------
        R : numpy.ndarray
            A 2D correlation matrix of shape (2, 2).
        """
        
        # Calculate the off-diagonal correlation
        R12 = Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1])
        
        # Form the correlation matrix
        R = np.diag([1, 1]) + np.rot90(np.diag([R12, R12]))
        
        return R

    @staticmethod
    def corr2cov(R, s):
        """
        Converts a covariance matrix to a correlation matrix.
        """
        s12 = s[0] * s[1] * R # compute off-diagonal
        return [[s[0] ** 2, s12], [s12, s[1] ** 2]]

    def transform(self, T, c0=np.array([0,0])):
        """
        Transform phantom to a different domain. 

        T: Transformation matrix.
        c: Shift in mean.
        """

        # TO DO: Consider preset options, specific by T = str.
        if type(T) == str:
            _, T, c0 = tools.get_transform(T)

        return Phantom(mu=T @ self.mu + c0, Sig=T @ self.Sig @ T.T)

        
    def show(self, nx=70, ny=70, nc=40):
        """
        Generate a quick plot of the phantom.
        """
        s1, s2 = np.sqrt(self.Sig[0,0]), np.sqrt(self.Sig[1,1])
        x = np.linspace(self.mu[0] - 3*s1, self.mu[0] + 3*s1, nx)
        y = np.linspace(self.mu[1] - 3*s2, self.mu[1] + 3*s2, ny)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        plt.figure()
        plt.contourf(10**X, 10**Y, self.rv.pdf(pos), nc)
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().set_box_aspect(1)

    def overlay(self, *args, **kwargs):
        """A function to bridge the general overlay for the current phantom."""
        overlay(self.mu, self.Sig, *args, **kwargs)

    
    def eval(self, grid=None, elements=None):
        """
        Evaluate the Phantom on a given grid or set of elements. 
        """
        if not grid == None:
            elements = np.log10(grid.elements)

        pos = np.dstack((elements[:,0], elements[:,1]))
        return self.rv.pdf(pos)

    def transpose(self):
        return Phantom('standard', mu=np.flip(self.mu), Sig=np.flip(self.Sig), w=self.w)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):

        # Size of output. 
        h = 10
        w = h * 3  # width

        # Header
        out = "—" * (int(w/2) - 4) + " \033[1mPHANTOM\033[0m " + "—" * (int(w/2) - 4) + "\n"

        # Compact parameter block (two clean lines)
        keys = list(self.p.keys())
        vals = [f"{self.p[k]:.4g}" for k in keys]

        # Split into roughly half
        mid1 = 2
        mid2 = 5

        row1  = "  ".join(f"\033[36m{k}\033[0m={v}" for k, v in zip(keys[:mid1], vals[:mid1]))
        row2 = "  ".join(f"\033[36m{k}\033[0m={v}" for k, v in zip(keys[mid1:mid2], vals[mid1:mid2]))
        row3 = "  ".join(f"\033[36m{k}\033[0m={v}" for k, v in zip(keys[mid2:], vals[mid2:]))

        out += row1 + "\n"
        out += row2 + "\n"
        out += row3 + "\n"

        # Generate ASCII version of the Phantom. 
        out += self.show_ascii(self.R, h, w)

        return out
    
    @staticmethod
    def show_ascii(R, h, w):
        """
        Generate an ASCII representation of the Phantom using the correlation matrix.
        """
        chars = " .:-=+*#%@"

        # Grid
        x_range = np.linspace(-3, 3, w)
        y_range = np.linspace(-3, 3, h)

        # Middle border
        out = "╭" + "—" * w + "╮\n"

        for y in y_range:
            line = ""
            for x in x_range:
                # Bivariate exponent (correlation ellipse).
                term = (x**2 - 2*R[0,1]*x*y + y**2) / (1 - R[0,1]**2)
                density = np.exp(-0.5 * term)

                char_idx = int(density * (len(chars) - 1))
                line += chars[char_idx]

            out += "|" + line[::-1] + "|\n"

        # Bottom border
        out += "╰" + "—" * w + "╯\n"

        return out

def overlay(mu, Sig, sd=2.0, **plot_kwargs):
    """
    Plot an ellipse representing the covariance matrix `Sig`
    centered at `mu`. sd = number of standard deviations.
    """

    ax = plt.gca()

    sd = np.atleast_1d(sd)  # convert to 1D array, handles if multiple std. dev. are given

    # After computing ellipse in log-space
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_coords = np.array([np.cos(theta), np.sin(theta)])
    vals, vecs = np.linalg.eigh(Sig)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    for s in sd:
        A = vecs @ np.diag(np.sqrt(vals) * s)
        ellipse_linear = (A @ ellipse_coords) + mu[:,None]

        # Transform back to linear
        ellipse_linear = 10**ellipse_linear

        ax.plot(ellipse_linear[0,:], ellipse_linear[1,:], **plot_kwargs)
    
    pow = Phantom.mu_sig2p(mu, Sig)['pow']
    fun = lambda x:  10 ** (pow * (np.log10(x) - mu[0]) + mu[1])
    xvec = np.logspace(mu[0] - np.max(sd) * np.sqrt(Sig[0,0]), 
                       mu[0] + np.max(sd) * np.sqrt(Sig[0,0]), 20)
    ax.plot(xvec, fun(xvec), **plot_kwargs)
    

def flines(fun, *args, **kwargs):
    """Plot a line to edge of an axis."""

    limx = plt.xlim()   # get x-limits of current axis
    limy = plt.ylim()  # get x-coordinates of where line intersect y-limits of current axis
    
    limx_y = np.concatenate((10**minimize(lambda x: (limy[0] - fun(10**x)) ** 2, x0=np.log10(limx[0]))['x'], 
                        10**minimize(lambda x: (limy[1] - fun(10**x)) ** 2, x0=np.log10(limx[0]))['x']))
    limx_y = np.sort(limx_y)  # sort (direction dependent)

    xvec = np.logspace(np.log10(np.maximum(limx[0], limx_y[0])), np.log10(np.minimum(limx[1], limx_y[1])), 50)  # resolve the combination

    plt.gca().plot(xvec, fun(xvec), *args, **kwargs)  # finally plot
    plt.ylim(limy)



def get_cov(mu1, mu2, s1, s2, R12, **kwargs):
    """
    Build distribution mean and covariance matrix from constituent parameters. 
    Extra **kwargs allows the function to ignore extra inputs. 
    """
    mu = np.array([mu1, mu2])
    Sig = np.diag([s1**2, s2**2])
    Sig[0,1] = R12 * s1 * s2
    Sig[1,0] = Sig[0,1]
    return mu, Sig

def fit(elements, f):
    """
    Fit a Phantom to a data. 

    Parameters
    ----------
    elements : ndarray (N,2)
        Elements on which the PDF is evaluated, arranged as [x-coordinates, y-coordinates].
    f : ndarray (N,)
        Values representing PDF(x, y) (possibly noisy).

    Returns
    -------
    params : dict
        Dictionary containing mux, muy, sigx, sigy, rho.
    f_fit : ndarray
        Fitted PDF evaluated at (x, y).
    """

    # Flatten data
    x = np.log10(elements[:,0].ravel())
    y = np.log10(elements[:,1].ravel())
    f = np.asarray(f).ravel()

    # Initial estimates from weighted moments
    w = f / (f.sum() + 1e-12)

    mux0 = np.sum(w * x)
    muy0 = np.sum(w * y)
    sigx0 = np.sqrt(np.sum(w * (x - mux0)**2))
    sigy0 = np.sqrt(np.sum(w * (y - muy0)**2))

    # correlation initial guess
    R0 = np.sum(w * (x - mux0) * (y - muy0)) / (sigx0 * sigy0 + 1e-12)
    R0 = np.clip(R0, -0.99, 0.99)

    p0 = [mux0, muy0, sigx0, sigy0, R0, 2]

    # bounds: sigx>0, sigy>0, -1<rho<1
    bounds = ([-np.inf, -np.inf, 1e-8, 1e-8, -0.999, -np.inf],
              [ np.inf,  np.inf, np.inf, np.inf,  0.999, np.inf])
    
    # Function for computing the residuals.
    def residuals(params, x, y):
        return y - 10**params[-1] * multivariate_normal(*get_cov(*params[:-1])).pdf(x)

    # Fit parameters
    res = least_squares(
        residuals,
        p0,
        args=(np.dstack([x, y]), f),
        bounds=(bounds[0], bounds[1])
    )
    
    pha = Phantom('standard', *get_cov(*res.x[:-1]))

    return pha


def fit_gmm(x, edges, n=1):
    """
    Fit to the data by sampling and fitting a Gaussian mixture model.
    """

    # Import GMM package and related on demand. 
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    # ----------------------------------------------------
    # 1. Sample points from the discretized PDF
    # ----------------------------------------------------
    flat_pdf = x / x.sum()  # normalize to ensure it's a proper PDF
    cdf = np.cumsum(flat_pdf)  # get cdf
    
    N = 20000 # number of samples

    # Draw uniform random values in [0,1]
    u = np.random.rand(N)

    # Find grid-cell indices corresponding to sampled CDF positions
    idx = np.searchsorted(cdf, u)

    # Convert 1D indices → 2D grid coordinates
    iy, ix = np.divmod(idx, len(edges[0]))
    samples = np.column_stack((np.log10(edges[0])[ix], np.log10(edges[1])[iy]))

    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)

    # ----------------------------------------------------
    # 2. Fit a Gaussian Mixture Model (GMM)
    # ----------------------------------------------------
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        n_init=10,           # multiple initializations for robustness
        max_iter=500,
        random_state=42
    )
    gmm.fit(samples)

    if n == 1:
        pha = Phantom(mu=gmm.means_[0], Sig=gmm.covariances_[0])
    else:
        pha = Phantoms(mu=gmm.means_, Sig=gmm.covariances_, w=gmm.weights_)

    return pha
