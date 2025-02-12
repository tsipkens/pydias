
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

import warnings

class Phantom:
    def __init__(self, spec, mu=None, Sig=None, w=None, prop=None):
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

        if spec == 'standard':
            self.mu = np.asarray(mu)
            self.Sig = np.asarray(Sig)

        elif spec == 'massmob':
            self.mu = [np.log10(prop['dg']), np.log10(prop['mg'])]

            # Requires 'sd'!
            Sig = np.array([[0.,0.],[0.,0.]])
            Sig[0,0] = np.log10(prop['sd']) ** 2

            # Then pick various options for specifying other distribution widths.
            if 'sm' in prop.keys():
                Sig[1,1] = np.log10(prop['sm']) ** 2

                if 'R12' in prop.keys():  # OPTION 1: use (sm, R12)
                    Sig[0,1] = np.sqrt(Sig[0,0] * Sig[1,1]) * prop['R12']

                elif 'zet' in prop.keys():  # OPTION 2: use (sm, zet)
                    Sig[0,1] = Sig[0,0] * prop['zet']

                Sig[1,0] = Sig[0,1]

            else:  # then requires 'zet'
                Sig[0,1] = Sig[0,0] * prop['zet']

                if 'sm_d' in prop.keys():  # OPTION 3: use (zet, sm_d)
                    R12 = 1 / np.sqrt(1 + np.log10(prop['sm_d']) ** 2 / (Sig[0,0] * prop['zet'] ** 2))

                elif 'R12' in prop.keys():  # OPTION 4: use (zet, R12)
                    R12 = prop['R12']
                
                Sig[1,1] = (Sig[0,1] / R12) ** 2 / Sig[0,0]
            
            Sig[1,0] = Sig[0,1]

            if np.abs(self.cov2corr(Sig)[0,1]) > 1:
                warnings.warn('Warning: Phantom correlation exceeded unity. Adjusted to R12 = 1.')
                Sig[0,1] = 1
                Sig[1,0] = 1

            self.Sig = Sig

        elif type(spec) == str:  # then assume preset
            1 == 1

        self.rv = multivariate_normal(self.mu, self.Sig)

        self.R = self.cov2corr(self.Sig)
        self.p = self.mu_sig2p(self.mu, self.Sig)
        self.massmob = self.get_massmob(self.p)


    def mu_sig2p(self, mu, Sig):
        """
        """

        p = {}
        p['m1'] = 10 ** mu[0]
        p['s1'] = 10 ** np.sqrt(Sig[0,0])

        p['m2'] = 10 ** mu[1]
        p['s2'] = 10 ** np.sqrt(Sig[1,1])

        p['pow'] = Sig[0,1] / Sig[0,0]
        
        p['R12'] = self.cov2corr(Sig)[0,1]
        p['s2_1'] = 10 ** np.sqrt(Sig[1,1] * (1 - p['R12'] ** 2))

        return p
    

    def get_massmob(self, p):  # extra parameters computed from standard p
        prop = {}

        prop['dg'] = p['m1']
        prop['sd'] = p['s1']
        prop['mg'] = p['m2']
        prop['sm'] = p['s2']
        prop['zet'] = p['pow']
        prop['Dm'] = p['pow']

        prop['rhog'] = 6 * prop['mg'] / (np.pi * prop['dg'] ** 3) * 1e9
        prop['rho100'] = prop['rhog'] * (100 / prop['dg']) ** (prop['zet'] - 3)
        prop['m100'] = 1e-9 * prop['rho100'] * np.pi / 6 * 100**3
        prop['k'] = prop['m100'] / (100 ** prop['zet'])

        prop['sm_d'] = p['s2_1']

        return prop


    def cov2corr(self, Sig):
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

        
    def show(self):
        s1 = np.sqrt(self.Sig[0,0])
        s2 = np.sqrt(self.Sig[1,1])

        x, y = np.meshgrid(self.mu[0] + s1 * np.linspace(-3.5, 3.5, 70), 
                        self.mu[1] + s2 * np.linspace(-3.5, 3.5, 70))
        pos = np.dstack((x, y))
        
        plt.figure()
        plt.contourf(10 ** x, 10 ** y, self.rv.pdf(pos), 40)
        plt.xscale('log')
        plt.yscale('log')

    
    def eval(self, grid=None, v=None):

        if not grid == None:
            v = np.log10(grid.elements)

        pos = np.dstack((v[:,0], v[:,1]))
        return self.rv.pdf(pos)

    def transpose(self):
        return Phantom('standard', mu=np.flip(self.mu), Sig=np.flip(self.Sig), w=self.w)
