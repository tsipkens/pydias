
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

import bidias.tools as tools

import warnings

class Phantom:
    def __init__(self, spec=None, mu=None, Sig=None, w=None, p=None, massmob=None):
        # Default values
        self.type = spec
        self.mu = []
        self.Sig = []
        self.R = []

        self.w = w

        self.p = {}
        self.massmob = {}

        self.nmodes = 0 # number of modes
        self.rv = None

        if not mu is None and not Sig is None:
            self.mu = np.asarray(mu)
            self.Sig = np.asarray(Sig)

        elif not p is None:
            [self.mu, self.Sig] = self.p2mu_sig(**p)

        elif not massmob is None:
            self.mu = np.asarray([np.log10(massmob['dg']), np.log10(massmob['mg'])])

            # Requires 'sd'!
            Sig = np.array([[0.,0.],[0.,0.]])
            Sig[0,0] = np.log10(massmob['sd']) ** 2

            # Then pick various options for specifying other distribution widths.
            if 'sm' in massmob.keys():
                Sig[1,1] = np.log10(massmob['sm']) ** 2

                if 'R12' in massmob.keys():  # OPTION 1: use (sm, R12)
                    Sig[0,1] = np.sqrt(Sig[0,0] * Sig[1,1]) * massmob['R12']

                elif 'zet' in massmob.keys():  # OPTION 2: use (sm, zet)
                    Sig[0,1] = Sig[0,0] * massmob['zet']

                Sig[1,0] = Sig[0,1]

            else:  # then requires 'zet'
                Sig[0,1] = Sig[0,0] * massmob['zet']

                if 'sm_d' in massmob.keys():  # OPTION 3: use (zet, sm_d)
                    R12 = 1 / np.sqrt(1 + np.log10(massmob['sm_d']) ** 2 / (Sig[0,0] * massmob['zet'] ** 2))

                elif 'R12' in massmob.keys():  # OPTION 4: use (zet, R12)
                    R12 = massmob['R12']
                
                Sig[1,1] = (Sig[0,1] / R12) ** 2 / Sig[0,0]
            
            Sig[1,0] = Sig[0,1]

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

    def mu_sig2p(self, mu, Sig):
        """
        Convert means and covariance (in log10-space) to a p dictionary. 
        """

        R12 = self.cov2corr(Sig)[0, 1]
        p = {
            "mu1": 10**mu[0],
            "mu2": 10**mu[1],
            "s1": 10**np.sqrt(Sig[0, 0]),
            "s2": 10**np.sqrt(Sig[1, 1]),
            "pow": Sig[0, 1] / Sig[0, 0],
            "R12": R12,
            "s2|1": 10**np.sqrt(Sig[1, 1] * (1 - R12**2)),
        }

        return p

    def p2mu_sig(self, **p):
        """
        Convert a dictionary of properties in p dictionary in log10 mean and covariance. 
        """
        # log10 the relevant fields. 
        fields = ['mu1', 'mu2', 's1', 's2']
        for field in fields:
            if field in p:
                p[field] = np.log10(p[field])

        # Check if pow instead of one of the standard deviations. 
        # Calculate the missing standard deviation.
        if 'pow'  in p:
            if not 's1' in p:
                p['s1'] = p['s2'] * p['R12'] / p['pow']
            elif not 's2' in p:
                p['s2'] = p['s1'] / p['R12'] * p['pow']

        return get_cov(**p)

    def p2massmob(self, p):  # extra parameters computed from standard p
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

        # Consider preset options, specific by T = str.
        if type(T) == str:
            _, T, c0 = tools.get_transform(T)

        return Phantom('standard', T @ self.mu + c0, T @ self.Sig @ T.T)

        
    def show(self, nx=70, ny=70, nc=40):
        s1 = np.sqrt(self.Sig[0,0])
        s2 = np.sqrt(self.Sig[1,1])

        x, y = np.meshgrid(self.mu[0] + s1 * np.linspace(-3.5, 3.5, nx), 
                        self.mu[1] + s2 * np.linspace(-3.5, 3.5, ny))
        pos = np.dstack((x, y))
        
        plt.figure()
        plt.contourf(10 ** x, 10 ** y, self.rv.pdf(pos), nc)
        plt.xscale('log')
        plt.yscale('log')

    
    def eval(self, grid=None, v=None):

        if not grid == None:
            v = np.log10(grid.elements)

        pos = np.dstack((v[:,0], v[:,1]))
        return self.rv.pdf(pos)

    def transpose(self):
        return Phantom('standard', mu=np.flip(self.mu), Sig=np.flip(self.Sig), w=self.w)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):

        # Size of output. 
        size = 13
        w = size * 3

        # Header
        out = "———— \033[1mPHANTOM\033[0m " + "—" * (w - 12) + "\n"

        # Compact parameter block (two clean lines)
        keys = list(self.p.keys())
        vals = [f"{self.p[k]:.5g}" for k in keys]

        # Split into roughly half
        mid = len(keys) // 2 + 1

        left  = "  ".join(f"\033[36m{k}\033[0m={v}" for k, v in zip(keys[:mid], vals[:mid]))
        right = "  ".join(f"\033[36m{k}\033[0m={v}" for k, v in zip(keys[mid:], vals[mid:]))

        out += left + "\n"
        out += right + "\n"

        # ASCII representation of a Bivariate Normal
        chars = " .:-=+*#%@"

        # Grid
        x_range = np.linspace(-3, 3, w)
        y_range = np.linspace(-3, 3, size)

        # Middle border
        out += "╭" + "—" * w + "╮\n"

        for y in y_range:
            line = ""
            for x in x_range:
                # Bivariate exponent (correlation ellipse)
                term = (x**2 - 2*self.R[0,1]*x*y + y**2) / (1 - self.R[0,1]**2)
                density = np.exp(-0.5 * term)

                char_idx = int(density * (len(chars) - 1))
                line += chars[char_idx]

            out += "|" + line[::-1] + "|\n"

        # Bottom border
        out += "╰" + "—" * w + "╯\n"

        return out

def overlay(mu, Sig, sd=2.0, **ellipse_kwargs):
    """
    Plot an ellipse representing the covariance matrix `Sig`
    centered at `mu`. sd = number of standard deviations.
    """

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

        plt.gca().plot(ellipse_linear[0,:], ellipse_linear[1,:], **ellipse_kwargs)


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
