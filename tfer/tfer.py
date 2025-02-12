
import numpy as np

from scipy.special import erf
from scipy.optimize import fsolve, fmin, minimize
from scipy.integrate import odeint
from scipy.sparse import csr_matrix

import os
from tqdm import tqdm

import tabulate # used to show lists of dictionaries (e.g., for setpoint info)

from autils import autils, props

def unpack(sp):
    """
    A function to pack a list of dictionaries describing the setpoint.
    """
    spo = {}
    for key, value in sp[0].items():
        spo[key] = np.zeros((1, len(sp)))
        for ii in range(len(sp)):
            if not np.size(sp[ii][key]) == 0:  # skip empty fields
                spo[key][:,ii] = sp[ii][key]
    
    for key, value in spo.items():
        spo[key] = spo[key].T

    return spo


def pack(sp):
    """
    A function to unpack a dictionary containing lists.
    The output can be visualized using tfer.show().
    """
    spo = [{}] * len(sp)
    for key, value in sp.items():
        for ii in range(len(sp[key])):
            spo[ii][key] = sp[key][ii]
    
    return spo


def show(s):
    if type(s) == dict:
        s = [s]

    header = s[0].keys()
    rows = [si.values() for si in s]
    print(tabulate.tabulate(rows, header))


def shape_inputs(s, s_star=None, z=None):

    if np.ndim(s) == 1:
        s = np.expand_dims(s, 1).T
    
    if np.ndim(s_star) == 1:
        s_star = np.expand_dims(s_star, 1)
    
    if np.ndim(z) == 1:
        z = np.expand_dims(z, 1)

    return s, s_star, z


#== DMA transfer function and helpers =============================#
def dma(d_star, d, z, prop=None, opts=None):
    """
    Calculate the transfer function OMEGA for an integer charge state of Z.

    Parameters:
    ----------
    d_star : array-like
        Setpoint diameters, 1 x n1 (in nm).
    d : array-like
        Particle diameters, n2 x 1 (in nm).
    z : int
        Integer charge state.
    prop_dma : dict, optional
        Properties of the DMA (e.g., radii) as a data structure. This is the 
        preferred method of specifying DMA properties. Defaults to properties 
        specified by a separate `prop_dma()` function.
    opts : dict, optional
        Options structure with the following optional fields:
        - 'diffusion': Boolean, indicates whether to include diffusion.
        - 'solver': String, method used to calculate diffusion.
        - 'param': String, which parameter set to use.

    Returns:
    -------
    Omega : numpy.ndarray
        The calculated transfer function.
    Zp_tilde : numpy.ndarray, optional
        Non-dimensional electrical mobility as a vector.
    prop_dma : dict, optional
        Updated properties structure, containing transfer function-specific 
        information (e.g., DMA resolution).
    sp : dict, optional
        Setpoint information.

    Notes:
    ------
    - `d` should be a n2 x 1 array, and `d_star` should be 1 x n1.
    - If any entries in `prop_dma` are vectors, they should have the same 
      dimensions as `d_star`.

    References:
    ----------
    - Sipkens, Timothy (2018)
    - Some code adapted from Buckley et al. (2017) and Olfert group.
    """
    #-- Parse inputs -------------------------------------------------------#
    if opts is None:
        opts = {}
    opts.setdefault('solver', 'fullydeveloped')
    opts.setdefault('diffusion', 1)
    opts.setdefault('type', 'd_star')

    if prop is None:
        prop = props.dma(opts)
        print('\033[93m' + 'WARNING: Using default DMA dimensions and properties.' + '\033[0m')

    d, d_star, z = shape_inputs(d, d_star, z)

    d = np.array(d) * 1e-9  # convert from nm to meters for calculations

    #-- Compute additional classifier properties --------------------------#
    prop['bet'] = (prop['Qs'] + prop['Qa']) / (prop['Qc'] + prop['Qm'])
    prop['del'] = (prop['Qs'] - prop['Qa']) / (prop['Qs'] + prop['Qa'])

    prop['Rd'] = 1 / prop['bet']  # theoretical resolution

    gam = (prop['R1'] / prop['R2']) ** 2
    kap = prop['L'] / prop['R2']  # Stolzenburg Manuscript, Eq. 9

    #-- Parse setpoint input -----------------------------------------------#
    if opts['type'] == 'd_star':  # default case
        d_star = np.array(d_star) * 1e-9  # convert from nm to meters

        if opts['solver'] == 'buckley':
            _, Zp_star = autils.dm2zp(d_star, 1)
        else:
            _, Zp_star = autils.dm2zp(d_star, 1, prop['T'], prop['p'])

        V = (prop['Qc'] / ((2 * np.pi) * Zp_star * prop['L'])) * np.log(prop['R2'] / prop['R1'])

    elif opts['type'] == 'V':
        V = d_star  # input is actually voltage
        Zp_star = (prop['Qc'] / ((2 * np.pi) * V * prop['L'])) * np.log(prop['R2'] / prop['R1'])
        
        if opts['solver'] == 'buckley':
            d_star = autils.zp2dm(Zp_star, 1)
        else:
            d_star = autils.zp2dm(Zp_star, 1, prop['T'], prop['p'])

    # Generate setpoint structure (for output).
    sp = [{'d_star': d_star[:,ii], 'Zp_star': Zp_star[:,ii], 'V': V[:,ii]} for ii in range(np.size(d_star, 1))]

    #-- Calculate G_DMA ----------------------------------------------------#
    if opts['solver'] == 'buckley':
        I_gamma = (0.25 * (1 - gam**2) * (1 - gam)**2 + (5 / 18) * (1 - gam**3) * np.log(gam)) / \
                   ((1 - gam) * (-0.5 * (1 + gam) * np.log(gam) - (1 - gam))**2)
        prop['G_DMA'] = 4 * (1 + prop['bet'])**2 / (1 - gam) * \
                        (I_gamma + (2 * (1 + prop['bet']) * kap)**(-2))

    elif opts['solver'] in ['olfert', 'fullydeveloped']:
        omega_a = np.zeros(len(prop['bet']))
        omega_s = np.zeros(len(prop['bet']))
        
        for ii in range(len(prop['bet'])):
            fun_a = lambda omega: (
                (((1 - omega)**2 * np.log(gam) / 2 / (1 - gam) + 
                omega * np.log(omega) + (1 - omega)) / 
                ((1 + gam) * np.log(gam) / 2 + (1 - gam)) - 
                prop['bet'][ii] * (1 - prop['del'][ii]) / 2 / (1 + prop['bet'][ii])).item()
            )
            omega_a[ii] = fsolve(fun_a, 0.9)

            fun_s = lambda omega: (
                (1 - ((1 - omega)**2 * np.log(gam) / 2 / (1 - gam) + 
                omega * np.log(omega) + (1 - omega)) / 
                ((1 + gam) * np.log(gam) / 2 + (1 - gam))) - \
                prop['bet'][ii] * (1 + prop['del'][ii]) / 2 / (1 + prop['bet'][ii]).item()
            )
            omega_s[ii] = fsolve(fun_s, 0.3)
        
        A = (-1 / 2 * (1 + gam) * np.log(gam) - (1 - gam))**(-1)
        I_gamma = lambda omega: A**2 / (1 - gam) * (
            -omega**2 * ((1 - gam) * np.log(omega) - (1 - omega) * np.log(gam))**2 / 2 +
            (omega**2 * (1 - gam) / 2 + omega**3 * np.log(gam) / 3) * 
            ((1 - gam) * np.log(omega) - (1 - omega) * np.log(gam)) + 
            (1 - omega**2) * (1 - gam)**2 / 4 +
            5 * (1 - omega**3) * (1 - gam) * np.log(gam) / 18 + 
            (1 - omega**4) * (np.log(gam))**2 / 12
        )
        
        G_o_corr = (4 * (1 + prop['bet'])**2) * \
                    (I_gamma(omega_s) - I_gamma(omega_a)) / (1 - gam) + \
                    (omega_a - omega_s) / kap**2
        prop['G_DMA'] = G_o_corr * np.log(prop['R2'] / prop['R1'])
    
    #-- Loop through charge states and evaluate transfer function ----------#
    Omega = np.zeros((np.size(d_star), np.size(d), len(z)))
    Zp_tilde = np.zeros((np.size(d_star), np.size(d), len(z)))
    
    for ii in range(len(z)):
        Omega[:, :, ii], Zp_tilde[:, :, ii] = tfer_dma0(Zp_star, V, d, z[ii], prop, opts)
    
    return Omega, Zp_tilde, prop, sp

def tfer_dma0(Zp_star, V, d, z, prop, opts):
    """
    Evaluate transfer function for a single charge state.
    """
    kb = 1.38064852e-23  # Boltzmann constant [m^2 kg s^-2 K^-1]
    e = 1.6022E-19  # electron charge [C]

    if opts['solver'] == 'buckley':
        _, Zp = autils.dm2zp(d, z)
    else:
        _, Zp = autils.dm2zp(d, z, prop['T'], prop['p'])
    
    Zp_tilde = Zp / Zp_star

    if opts['diffusion']:
        if opts['solver'] == 'buckley':
            sigma = np.sqrt(prop['G_DMA'] * 2 * np.pi * prop['L'] * prop['D'](Zp) / prop['Qc'])
        else:
            sigma_star = np.sqrt((kb * prop['T']) / (z * e * V) * prop['G_DMA'])
            sigma = np.sqrt(sigma_star**2 * Zp_tilde)

        epsilon = lambda x: x * erf(x) + np.exp(-x**2) / np.sqrt(np.pi)

        Omega = sigma / (np.sqrt(2) * prop['bet'] * (1 - prop['del'])) * (
            epsilon((Zp_tilde - 1 - prop['bet']) / np.sqrt(2) / sigma) +
            epsilon((Zp_tilde - 1 + prop['bet']) / np.sqrt(2) / sigma) -
            epsilon((Zp_tilde - 1 - prop['bet'] * prop['del']) / np.sqrt(2) / sigma) -
            epsilon((Zp_tilde - 1 + prop['bet'] * prop['del']) / np.sqrt(2) / sigma)
        )

    else:
        Omega = 1 / (2 * prop['bet'] * (1 - prop['del'])) * (
            np.abs(Zp_tilde - 1 + prop['bet']) +
            np.abs(Zp_tilde - 1 - prop['bet']) -
            np.abs(Zp_tilde - 1 + prop['bet'] * prop['del']) -
            np.abs(Zp_tilde - 1 - prop['bet'] * prop['del'])
        )

    Omega = np.maximum(Omega, 0)
    Omega[Omega < (1e-7 * np.max(Omega))] = 0

    return Omega, Zp_tilde


#== For charging ==================================================#
def charger(d, z=None, model='hybrid', T=298, opts=None):
    """
    Calculates the fraction of particles with a specific integer charge state.

    Parameters:
    ----------
    d : np.ndarray
        Particle diameter [nm].
    z : np.ndarray, optional
        Integer particle charge state, default is np.arange(0, 7).
    T : float, optional
        Temperature in Kelvin, default is 298 K.
    model : str, optional
        String specifying which model to use, default is 'hybrid'.
    opt : dict, optional
        Additional options for models.

    Returns:
    -------
    fn : np.ndarray
        Fraction of particles with specific integer charge.
    qbar : np.ndarray
        Average charge on particles.
    model : str
        Model used.
    """

    d, _, z = shape_inputs(d, None, z)

    # Parse inputs
    if z is None:
        z = -np.expand_dims(np.arange(0, 7), 1)  # Default charge states 0 to 6
    else:
        z = np.asarray(z)

    # Assign character vector for hybrid case
    if model == 'hybrid':
        model = ''
        for ii in range(np.size(z)):
            if abs(z[ii,0]) < 3:
                model = model + 'g'
            else:
                model = model + 'w'
    model = model.lower()

    if opts is None:
        opts = {}

    # Convert diameter from nm to m
    d = d * 1e-9

    # Constants
    e = 1.602177e-19  # Elementary charge
    epi = 8.85418e-12  # Dielectric constant for air [F/m]
    kB = 1.38065e-23  # Boltzmann constant
    Z_Z = 0.875  # Ion mobility ratio (Wiedensohler, 1988)

    vec_z, vec_d = np.meshgrid(z, d, indexing='ij')  # Create grid for d and z
    fn = np.zeros_like(vec_d)

    # Select model
    indw = np.array([s == 'w' for s in model])
    if np.any(indw):
        # Wiedensohler model
        ind_small_z = np.squeeze(np.abs(z) < 3) & indw

        a = np.array([[-26.3328, -2.3197, -0.0003, -2.3484, -44.4756],
                      [35.9044, 0.6175, -0.1014, 0.6044, 79.3772],
                      [-21.4608, 0.6201, 0.3073, 0.4800, -62.8900],
                      [7.0867, -0.1105, -0.3372, 0.0013, 26.4492],
                      [-1.3088, -0.1260, 0.1023, -0.1553, -5.7480],
                      [0.1051, 0.0297, -0.0105, 0.0320, 0.5049]])

        if np.any(ind_small_z):
            exponent = np.zeros_like(d)
            for ii in range(6):
                exponent = exponent + a[ii, z[ind_small_z] + 2] * (np.log10(d * 1e9) ** ii)
            fn[ind_small_z, :] = 10 ** exponent
            
            fn[(d < 20e-9) & np.abs(z == 2) & indw] = 0  # zero some specific cases

        ind_large_z = np.squeeze(np.abs(z) >= 3) & indw
        if np.any(ind_large_z):
            fn[ind_large_z, :] = (e / np.sqrt(4 * np.pi * epi * kB * T * vec_d[ind_large_z, :])) * np.exp(
                -(vec_z[ind_large_z, :] - (2 * np.pi * epi * kB * T * np.log(Z_Z) * vec_d[ind_large_z, :] / e ** 2)) ** 2
                / (4 * np.pi * epi * kB * T * vec_d[ind_large_z, :] / e ** 2)
            )
            fn[(d < 69.78e-9) & np.expand_dims(ind_large_z, 1)] = 0
            fn[fn <= 6e-5] = 0

    indg = np.array([s == 'g' for s in model])
    if np.any(indg):
        # Gopalakrishnan model
        ind_small_z = np.squeeze((np.abs(z) < 3)) & indg
        if np.any(ind_small_z):
            cond = opts.get('conduction', 1)
            a = get_a(cond)

            exponent = np.zeros_like(d)
            for ii in range(4):
                exponent = exponent + a[ii, z[ind_small_z] + 2] * (np.log(d * 1e9) ** ii)
            fn[ind_small_z, :] = np.exp(exponent)
    
    eps = opts.get('eps', 13.5)  # dielectric constant (default is from T. Johnson)
    nit = opts.get('nit', 1.01e13)  # ion·s/m3
    if model == 'fuchs':
        print('Running Fuchs model:')
        for ii in tqdm(range(np.size(d))):
            pz, qbar = fuchs(d[:,ii][0], max(80, np.round(max(z) * 1.2)), T, 1, nit, eps)
            fn[:, ii] = pz[z[:,0]]
        autils.textdone()

    # Calculate qbar
    qbar = np.sum(fn * z, axis=0) / np.sum(fn, axis=0)

    # Li model can be added similarly

    fn = fn.T

    return fn, qbar, model

def get_a(cond):
    """
    Coefficients from Gopalakrishnan et al.
    
    Parameters:
    ----------
    cond : int
        Conduction mode, 1 for conducting, else non-conducting.

    Returns:
    -------
    a : np.ndarray
        Coefficients matrix.
    """
    if cond == 1:  # Conducting values
        return np.array([[-45.405, -7.8696, -0.3880, -8.0157, -40.714],
                         [20.049, 3.1036, 0.4545, 3.2536, 17.487],
                         [-3.0570, -0.4557, -0.1634, -0.5018, -2.6146],
                         [0.1534, 0.0187, 0.0091, 0.0223, 0.1282]])
    else:  # Non-conducting values
        return np.array([[-63.185, -16.801, -1.212, -16.704, -71.051],
                         [26.833, 7.5947, 1.1068, 7.5438, 31.209],
                         [-3.8723, -1.1975, -0.2934, -1.1938, -4.6696],
                         [0.1835, 0.0590, 0.0169, 0.0589, 0.2301]])


# Constants for Fuchs evaluation
e = 1.6e-19  # electron charge (C)
k = 1.38e-23  # Boltzmann constant (J/K)
KE = 9e9  # Coulomb constant (N*m^2/C^2)
Na = 6.023e23  # Avogadro's number (mol^-1)

def fuchs(d, zmax, T, P, nit, eps):
    b = fuchs_sub(zmax, T, d, P, eps)
    init_dist = ecd(d, T, zmax)  # equilibrium charge distribution
    Z = birth_death(b, nit, init_dist)
    
    dist = Z[-1, 1:]
    partition = np.sum(dist)
    probs = dist / partition

    # Remove negative probabilities due to rounding
    probs[probs < 0] = 0
    
    zmean = np.sum((np.arange(len(probs)) * probs))

    return probs, zmean

def fuchs_sub(nmax, T, dp, P, epsilon):
    a = dp / 2  # convert particle diameter to radius in meters

    # ION PROPERTIES
    mi = 0.109  # ionic molecular weight (kg/mol)
    mg = 0.029  # air molecular weight (kg/mol)
    Z = 0.00014  # electrical mobility of ion (m^2/Vs)
    Zi = Z / P  # electrical mobility at pressure P
    D = k * T * Zi / e  # diffusion coefficient (m^2/s)
    ci = np.sqrt(8 * k * T * Na / np.pi / mi)  # mean speed of ions (m/s)
    li = 1.329 * Zi / e * np.sqrt(k * T * mi * mg / (mi + mg) / Na)  # ionic mean free path

    K = (epsilon - 1) / (epsilon + 1)

    # Limiting sphere radius
    delta = (a**3) / (li**2) * ((1/5) * ((1 + li/a)**5) -
                                (1/3) * (1 + (li**2) / (a**2)) * ((1 + li/a)**3) +
                                (2/15) * ((1 + (li**2) / (a**2))**(5/2)))

    # Numerical integration for PSI
    n = np.arange(nmax)  # charge state array
    r = np.arange(delta, a, -a / 1000)  # spatial array from delta to particle radius

    def int_fun(y, x):  # inherit remaining properties
        if x == 0:
            return np.ones_like(n) * np.finfo(np.float64).max  # handle inf values
        r = a / x
        pot = KE * (e**2) * ((n / r) - K * ((a**3) / (2 * (r**2) * (r**2 - (a**2)))))
        out = np.exp(pot / k / T)
        return out

    PSI = odeint(int_fun, np.zeros_like(n), np.linspace(a/delta/1000, a/delta, 800))[-1]

    pot1 = KE * (e**2) * ((n[np.newaxis].T / delta) - K * ((a**3) / (2 * (delta**2) * (delta**2 - (a**2)))))
    pot2 = KE * (e**2) * ((n[np.newaxis].T / r[np.newaxis]) - K * ((a**3) / (2 * (r**2) * (r**2 - (a**2)))))
    impfac = (r**2) * (1 + (2/(3 * k * T)) * (pot1 - pot2))
    
    bmsqrd = np.min(impfac, axis=1)
    gam = bmsqrd / delta**2
    valid_gam = (gam > 0) & (gam <= 1)

    pot_d = KE * (e**2) * ((n[valid_gam] / delta) - K * (a**3 / (2 * delta**2 * (delta**2 - (a**2)))))
    b = np.zeros_like(gam)
    b[valid_gam] = ((4 * np.pi * a * D) /
                    ((4 * D * a / (gam[valid_gam] * ci * delta**2)) *
                    np.exp(pot_d / k / T) + PSI[valid_gam]))

    return b

def ecd(d, T, nmax):
    partition = np.sqrt(np.pi) / np.sqrt((KE * e**2) / (d * k * T))
    n = np.arange(1, nmax+1)
    f = np.exp(-KE * (e**2) * n**2 / (d * k * T)) / partition
    return f

def birth_death(b, nit_total, dist):
    nit = np.arange(0, nit_total + 1e11, 1e11)
    newm = len(b)

    # This function calculates the fraction of charged particles according 
    # (Boisdron and Brock, 1970, Eqn 18) birth-and-death model. The model 
    # can be used with Combination Coefficients (b) determined by any models 
    # available in the literature (cont., fm, and trans. regime). 
    def charge(y, t):
        dy = np.zeros_like(y)
        dy[0] = -b[0] * y[0]
        dy[1:] = b[:-1] * y[:-1] - b[1:] * y[1:]
        return dy

    Y = odeint(charge, dist, nit)
    Z = np.column_stack((nit, Y))
    
    return Z


def white(deq, nit):
    # Convert diameter from nm to meters
    deq = deq * 1e-9

    T = 300  # Temperature in Kelvin
    mg = 2 * 2.3258671e-26  # Molecular weight in kg

    # Use CGS units as per the original work
    e = 4.8e-10  # Electron charge in CGS units
    kB = 1.3807e-16  # Boltzmann constant in CGS units
    nit1 = nit / 1e6
    mg1 = mg * 1e3
    c1 = np.sqrt(3 * kB * T / mg1)

    # Constants in White's equation
    A = 1 / 2 * (kB * T) / (e**2) * 1e2  # CGS units correction
    B = np.pi / 2 * c1 * e**2 * nit1 / (kB * T) * 1e2

    # White's equation
    qbar1 = deq * A * np.log(1 + deq * B)

    # Optional: Power-law approximation using Taylor series expansion
    qbar_pl = np.empty_like(deq)
    deqs = 1e-7  # 100 nm
    p = B * deqs / ((B * deqs + 1) * np.log(B * deqs + 1))
    k = deqs * A * np.log(1 + deqs * B)
    
    # Power-law approximation
    qbar_pl = k * (deq / (deqs / 1e2)) ** (1 + p)

    return qbar1, qbar_pl, A, B


def diff_product(x, nj, np):
    """
    Computes the product of differences between elements of x.
    """
    fl = np.arange(np + 1) != nj
    return np.prod(x[fl] - x[nj])

def product_term(x, np):
    """
    Computes the product of elements of x.
    """
    return np.prod(x[1:np])

def collkernel2charge(collkernel, ntvec):
    """
    Computes mean charge and fq based on the collision kernel and time vector.
    
    Parameters:
    - collkernel: array of collision kernels
    - ntvec: array of time values
    
    Returns:
    - meancharge0: mean charge for each time value
    - fq: normalized collision kernel
    """
    npmax = len(collkernel) - 1
    collkernelratio = collkernel / collkernel[0]

    # Generate necessary dummy variables
    dummy1 = np.ones((npmax, npmax + 1))
    dummy2 = np.zeros(npmax)
    
    for np in range(1, npmax + 1):
        for nj in range(np + 1):
            dummy1[np - 1, nj] = diff_product(collkernelratio, nj, np)
        dummy2[np - 1] = product_term(collkernelratio, np)
    
    # Initialize output arrays
    meancharge0 = np.zeros(len(ntvec))
    fq = np.zeros((npmax, len(ntvec)))
    
    for nn, nt in enumerate(ntvec):
        meancharge = 0
        fq[0, nn] = np.exp(-collkernel[0] * nt)
        sscheck = fq[0, nn]
        
        for np in range(1, npmax + 1):
            fq[np, nn] = np.sum(np.exp(-collkernel[:np + 1] * nt) / dummy1[np - 1, :np + 1])
            fq[np, nn] *= dummy2[np - 1]
            
            if np.isnan(fq[np, nn]):
                fq[np, nn] = 0
            
            sscheck += fq[np, nn]
            meancharge += np * fq[np, nn]
        
        meancharge0[nn] = meancharge
    
    return meancharge0, fq


def get_power_law(qbar0, d):
    """
    Get power law parameters for high charge counts.

    Parameters:
    - qbar0: Array of average charge values
    - d: Array of particle diameters

    Returns:
    - nu: Exponent of the power law
    - q0: Coefficient of the power law
    - p: Parameters of the fitted power law
    - X: A constant derived from the power law parameters
    """
    # Flag higher charge states. Do not fit when average charge is less than 4.
    fl = qbar0 > 4

    # Sensitivity matrix for least-squares
    A = np.vstack([np.log(d[fl]), np.ones(np.sum(fl))]).T
    b = np.log(qbar0[fl])  # Data to fit to

    # Least-squares fit
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # p = (A' * A) \ (A' * b)

    # Extract power law parameters
    nu = p[0]
    q0 = np.exp(p[1])

    # Calculate X
    X = (1 / q0) ** (1 / nu)

    return nu, q0, p, X


#== PMA transfer functions ========================================#
def pma(sp, m, d, z=None, prop=None, opt=None):
    """
    Bridging function used to evaluate Particle Mass Analyzer (PMA) transfer function.

    Parameters:
    ----------
    sp : array-like
        Setpoint parameters.
    m : array-like
        Particle masses.
    d : array-like
        Particle diameters.
    z : array-like, optional
        Charge states. Defaults to [1, 2, 3, 4].
    prop : dict, optional
        Properties of the PMA. If not provided, defaults are used from `prop_pma()`.
    opt : str, optional
        Option for which solver to use. Defaults to '1C', which uses a Taylor series
        solution about rc without diffusion.

    Returns:
    -------
    Lambda_ii : numpy.ndarray
        PMA transfer function for each setpoint, mass, and charge state.
    prop : dict
        Updated properties of the PMA.
    """

    # Parse inputs
    if opt is None:
        opt = '1C'  # Default to '1C' (Taylor series solution without diffusion)
    
    if prop is None:
        prop = props.pma()  # Call the prop_pma function to get default properties

    if z is None:
        z = np.arange(1, 5)  # Default charge states [1, 2, 3, 4]

    m, _, z = shape_inputs(m, None, z)
    d, _, _ = shape_inputs(d)

    # Dynamically import the appropriate solver function from a module
    fun = eval('tfer_' + opt)

    # For first charge state
    Lambda_i = np.zeros((len(sp), np.size(m), len(z)))

    # Add additional charge states
    for ii in range(len(z)):
        Lambda_i[:, :, ii], _ = fun(unpack(sp), m * 1e-18, d * 1e-9, z[ii], prop)

    return Lambda_i, prop


def get_setpoint(prop, *args):
    """
    Generate setpoint parameter structure from available parameters.
    
    Parameters
    ----------
    prop : dict
        Properties of the particle mass analyzer.
    *args : tuple
        Name-value pairs for setpoint. Two values are required; if one value is specified,
        `Rm` is assumed to be 3. The pairs include:
        - ('m_star', float): Setpoint mass in kg (e.g., 1e-18 for a 1 fg particle).
        - ('Rm', float): Resolution.
        - ('omega1', float): Angular speed of the inner electrode.
        - ('V', float): Setpoint voltage.
    
    Returns
    -------
    tuple
        sp : list of dict
            List of dictionaries containing multiple setpoint parameters (V, alpha, etc.).
        m_star : list of float
            List of setpoint masses, assuming a singly charged particle, in kg.
    """

    if len(args) == 0:  # If empty input
        sp = [{'m_star': [], 'V': [], 'Rm': [], 'omega': [], 'omega1': [], 'omega2': [], 'alpha': [], 'beta': [], 'm_max': []}]
        m_star = []
        return sp, m_star

    if len(args) == 2:  # Default Rm = 3
        args = list(args) + [('Rm', 3)]

    n = max(len(args[1]), len(args[3]))  # Number of setpoints

    if len(args[1]) != n:
        args = list(args)
        args[1] = np.full(n, args[1])
    if len(args[3]) != n:
        args = list(args)
        args[3] = np.full(n, args[3])

    # Default empty structure
    sp = [{}] * n
    m_star = []
    for ii in range(n):  # Loop through setpoints
        sp_i, m_star_i = get_setpoint0(prop, args[0], args[1][ii], args[2], args[3][ii])
        sp[ii] = sp_i
        m_star.append(m_star_i)

    return sp, m_star

def get_setpoint0(prop, *args):
    """
    Get parameters for a single setpoint.

    Parameters
    ----------
    prop : dict
        Properties of the particle mass analyzer.
    *args : tuple
        Name-value pairs for setpoint.
    
    Returns
    -------
    tuple
        sp : dict
            Dictionary containing setpoint parameters (V, alpha, etc.).
        m_star : float
            Setpoint mass in kg.
    """
    # Initialize sp dictionary
    sp = {'m_star': [], 'V': [], 'Rm': [], 'omega': [], 'omega1': [], 
            'omega2': [], 'alpha': [], 'beta': [], 'm_max': []}

    # Parse inputs
    for i in range(0, len(args), 2):
        sp[args[i]] = np.array(args[i + 1])

    e = 1.60218e-19  # Electron charge [C]

    if not sp['m_star']:  # m_star is not specified
        if sp['omega1'] == []:
            sp['omega1'] = sp['omega'] / (((prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)) +
                                         (prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1) / prop['rc'] ** 2))
        sp['alpha'] = sp['omega1'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)
        sp['beta'] = sp['omega1'] * prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)
        sp['m_star'] = sp['V'] / (np.log(1 / prop['r_hat']) / e * (sp['alpha'] * prop['rc'] + sp['beta'] / prop['rc']) ** 2)
        sp['omega'] = sp['alpha'] + sp['beta'] / (prop['rc'] ** 2)
        sp['omega2'] = sp['alpha'] + sp['beta'] / (prop['r2'] ** 2)

    elif sp['omega1']:  # m_star and omega1 are specified
        sp['alpha'] = sp['omega1'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)
        sp['beta'] = sp['omega1'] * prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)
        sp['V'] = sp['m_star'] * np.log(1 / prop['r_hat']) / e * (sp['alpha'] * prop['rc'] + sp['beta'] / prop['rc']) ** 2
        sp['omega2'] = sp['alpha'] + sp['beta'] / (prop['r2'] ** 2)
        sp['omega'] = sp['alpha'] + sp['beta'] / (prop['rc'] ** 2)

    elif sp['omega']:  # m_star and omega are specified
        sp['omega1'] = sp['omega'] / (((prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)) +
                                     (prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1) / prop['rc'] ** 2))
        sp['alpha'] = sp['omega1'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)
        sp['beta'] = sp['omega1'] * prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)
        sp['V'] = sp['m_star'] * np.log(1 / prop['r_hat']) / e * (sp['alpha'] * prop['rc'] + sp['beta'] / prop['rc']) ** 2
        sp['omega2'] = sp['alpha'] + sp['beta'] / (prop['r2'] ** 2)

    elif sp['V']:  # m_star and V are specified
        v_theta_rc = np.sqrt(sp['V'] * e / (sp['m_star'] * np.log(1 / prop['r_hat'])))
        A = (prop['rc'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1) +
             1 / prop['rc'] * (prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)))
        sp['omega1'] = v_theta_rc / A
        sp['alpha'] = sp['omega1'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)
        sp['beta'] = sp['omega1'] * prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)
        sp['omega2'] = sp['alpha'] + sp['beta'] / (prop['r2'] ** 2)
        sp['omega'] = sp['alpha'] + sp['beta'] / (prop['rc'] ** 2)

    elif sp['Rm']:  # m_star and Rm are specified
        n_B = get_nb(sp['m_star'], prop)
        B_star, _, _ = autils.mp2zp(sp['m_star'], 1, prop['T'], prop['p'], prop)
        sp['m_max'] = sp['m_star'] * (1 / sp['Rm'] + 1)
        sp['omega'] = np.sqrt(prop['Q'] / (sp['m_star'] * B_star * 2 * np.pi * prop['rc'] ** 2 * prop['L'] *
                                          ((sp['m_max'] / sp['m_star']) ** (n_B + 1) - (sp['m_max'] / sp['m_star']) ** n_B)))
        sp['omega1'] = sp['omega'] / (((prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)) +
                                     (prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1) / prop['rc'] ** 2))
        sp['alpha'] = sp['omega1'] * (prop['r_hat'] ** 2 - prop['omega_hat']) / (prop['r_hat'] ** 2 - 1)
        sp['beta'] = sp['omega1'] * prop['r1'] ** 2 * (prop['omega_hat'] - 1) / (prop['r_hat'] ** 2 - 1)
        sp['omega2'] = sp['alpha'] + sp['beta'] / (prop['r2'] ** 2)
        sp['V'] = sp['m_star'] * np.log(1 / prop['r_hat']) / e * (sp['alpha'] * prop['rc'] + sp['beta'] / prop['rc']) ** 2

    else:
        raise ValueError('Invalid setpoint parameters specified.')
    
    # if sp['Rm'] == []:  # If resolution is not specified
        # sp['Rm'], sp['m_max'] = get_resolution(sp['m_star'], sp['omega'], prop)

    return sp, sp['m_star']

def get_resolution(m_star, omega, prop):
    """
    Solver to evaluate the resolution from m_star and prop.
    
    Parameters
    ----------
    m_star : float
        Setpoint mass in kg.
    omega : float
        Angular speed at gap center.
    prop : dict
        Properties of the particle mass analyzer.
    
    Returns
    -------
    tuple
        Rm : float
            Resolution.
        m_max : float
            Approximate upper end of non-diffusing transfer function.
    """
    n_B = get_nb(m_star, prop)
    B_star = autils.mp2zp(m_star, 1, prop['T'], prop['p'], prop)
    t0 = prop['Q'] / (m_star * B_star * (2 * np.pi * prop['L']) * omega ** 2 * prop['rc'] ** 2)

    def func(Rm):
        m_rat = 1 / Rm + 1
        return (m_rat ** (n_B + 1) - m_rat ** n_B) - t0

    result = minimize(lambda Rm: func(Rm) ** 2, 5)
    Rm = result.x[0]
    m_max = m_star * (1 / Rm + 1)
    return Rm, m_max

def get_nb(m_star, prop):
    """
    Function to evaluate n_B constant.
    
    Parameters
    ----------
    m_star : float
        Setpoint mass in kg.
    prop : dict
        Properties of the particle mass analyzer.
    
    Returns
    -------
    float
        n_B constant.
    """
    m_high = m_star * 1.001
    m_low = m_star * 0.999
    B_high, _, _ = autils.mp2zp(m_high, 1, prop['T'], prop['p'], prop)
    B_low, _, _ = autils.mp2zp(m_low, 1, prop['T'], prop['p'], prop)
    n_B = np.log10(B_high / B_low) / np.log10(m_high / m_low)
    return n_B


def parse_inputs(sp, m, d, z, prop):
    """
    Parse inputs and calculate the relevant parameters including:
    tau, C0, D, and optionally rs.

    Parameters:
        sp (dict): Contains setpoint parameters (e.g., 'V', 'alpha', 'beta').
        m (float or np.ndarray): Particle mass, can be a vector.
        d (float or np.ndarray): Particle mobility diameter, optional.
        z (int): Integer charge state (default is 1 if not specified).
        prop (dict): Properties of the particle mass analyzer.

    Returns:
        tau (np.ndarray): Product of mechanical mobility and particle mass.
        C0 (np.ndarray): Electrostatic force summary parameter.
        D (np.ndarray): Diffusion coefficient.
        rs (np.ndarray): Equilibrium radius (if requested).
    """

    #-- Parse inputs ---------------------------------------------------------#
    if z is None:
        z = 1  # If integer charge is not specified, use z = 1

    #-- Constants ------------------------------------------------------------#
    e = 1.60218e-19  # Electron charge [C]
    q = z * e  # Particle charge

    #-- Evaluate mechanical mobility -----------------------------------------#
    if d is None:
        # If mobility diameter is not specified, use mass-mobility relation
        print('Invoking mass-mobility relation to determine Zp.')
        B, _, _ = autils.mp2zp(m, z, prop['T'], prop['p'], prop)
    else:
        # If mobility diameter is specified
        B, _ = autils.dm2zp(d, z, prop['T'], prop['p'])

    #-- Evaluate output parameters -------------------------------------------#
    tau = B * m
    D = prop['D'][0](B)  # Diffusion as a function of mechanical mobility

    # Calculate recurring C0 parameter
    C0 = np.array(sp['V']) * q / np.log(1 / prop['r_hat'])

    #-- Optionally calculate equilibrium radius ------------------------------#
    rs = None
    if 'alpha' in sp and 'beta' in sp:
        # Evaluate positive and negative roots for rs
        alpha = np.array(sp['alpha'])
        beta = np.array(sp['beta'])

        C0_over_m = C0 / m
        r_m = (np.sqrt(C0_over_m) - np.sqrt(C0_over_m - 4 * alpha * beta)) / (2 * alpha)
        r_p = (np.sqrt(C0_over_m) + np.sqrt(C0_over_m - 4 * alpha * beta)) / (2 * alpha)

        # Determine which root is closer to the centerline radius
        bo = np.abs(r_m - prop['rc']) > np.abs(r_p - prop['rc'])
        bo[r_m == 0] = True  # Avoid zero values for APM case

        # Assign the appropriate root to rs
        rs = np.where(bo, r_p, r_m)

        # Zero out cases where no equilibrium radius exists (removing complex numbers)
        rs[C0_over_m < 4 * alpha * beta] = 0

    return tau, C0, D, rs


def tfer_1S(sp, m, d, z, prop={}):
    """
    Evaluates the transfer function for a PMA in Case 1S.
    
    Parameters:
    m_star (float): Setpoint particle mass.
    m (float): Particle mass.
    d (float): Particle mobility diameter.
    z (int): Integer charge state.
    prop (dict): Dictionary of device properties (e.g., classifier length).
    **kwargs: Optional name-value pairs for setpoint, with the following keys:
        - 'Rm' (float): Resolution (default is 3).
        - 'omega1' (float): Angular speed of the inner electrode.
        - 'V' (float): Setpoint voltage.
    
    Returns:
    float: Lambda, the transfer function.
    callable: G0, function mapping final to initial radial position.
    """
    tau, _, _, rs = parse_inputs(sp, m, d, z, prop)
    # parse inputs for common parameters

    # -- Estimate device parameter --------------------------------------------#
    lam = 2 * tau * (sp['alpha'] ** 2 - sp['beta'] ** 2 / (rs ** 4)) * prop['L'] / prop['v_bar']
    
    # -- Evaluate G0 and transfer function ------------------------------------#
    G0 = lambda r: rs + (r - rs) * np.exp(-lam)  # define as a lambda function

    ra = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r1'])))
    rb = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r2'])))

    Lambda = (1 / (2 * prop['del'])) * (rb - ra)

    return Lambda, G0


def tfer_1C(sp, m, d, z, prop={}):
    """
    Evaluates the transfer function for a PMA in Case 1C.
    
    Parameters:
    sp (dict): Dictionary defining various setpoint parameters 
               (e.g., m_star, V). Use 'get_setpoint' to generate this structure.
    m (float): Particle mass.
    d (float): Particle mobility diameter.
    z (int): Integer charge state.
    prop (dict): Dictionary of device properties (e.g., classifier length).
    
    Returns:
    float: Lambda, the transfer function.
    callable: G0, function mapping final to initial radial position.
    """
    tau, C0, _, _ = parse_inputs(sp, m, d, z, prop)
    # parse inputs for common parameters

    # -- Taylor series expansion constants ------------------------------------#
    C3 = tau * (sp['alpha'] ** 2 * prop['rc'] + 2 * sp['alpha'] * sp['beta'] / prop['rc'] + \
                sp['beta'] ** 2 / (prop['rc'] ** 3) - C0 / (m * prop['rc']))
    C4 = tau * (sp['alpha'] ** 2 - 2 * sp['alpha'] * sp['beta'] / (prop['rc'] ** 2) - \
                3 * sp['beta'] ** 2 / (prop['rc'] ** 4) + C0 / (m * (prop['rc'] ** 2)))

    # -- Evaluate G0 and transfer function ------------------------------------#
    G0 = lambda r: prop['rc'] + (r - prop['rc'] + C3 / C4) * \
                   np.exp(-C4 * prop['L'] / prop['v_bar']) - C3 / C4

    ra = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r1'])))
    rb = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r2'])))

    Lambda = (1 / (2 * prop['del'])) * (rb - ra)

    return Lambda, G0


def tfer_1C_diff(sp, m, d, z, prop={}):
    """
    Evaluates the transfer function for a PMA in Case 1c (with diffusion).
    
    Parameters:
    sp (dict): Dictionary defining various setpoint parameters 
               (e.g., m_star, V). Use 'get_setpoint' to generate this structure.
    m (float): Particle mass.
    d (float): Particle mobility diameter.
    z (int): Integer charge state.
    prop (dict): Dictionary of device properties (e.g., classifier length).
    
    Returns:
    float: Lambda, the transfer function.
    callable: G0, function mapping final to initial radial position.
    """
    _, _, D, _ = parse_inputs(sp, m, d, z, prop)  # get diffusion coeff.
    sig = np.sqrt(2 * prop['L'] * D / prop['v_bar'])  # diffusive spreading parameter

    # -- Evaluate relevant functions ------------------------------------------#
    _, G0 = tfer_1C(sp, m, d, z, prop)
    # get G0 function for this case

    rho_fun = lambda G, r: (G - r) / (np.sqrt(2) * sig)  # recurring quantity
    kap_fun = lambda G, r: \
        (G - r) * erf(rho_fun(G, r)) + \
        sig * np.sqrt(2 / np.pi) * np.exp(-rho_fun(G, r) ** 2)  # define function for kappa

    # -- Evaluate the transfer function and its terms -------------------------#
    K22 = kap_fun(G0(prop['r2']), prop['r2'])
    K21 = kap_fun(G0(prop['r2']), prop['r1'])
    K12 = kap_fun(G0(prop['r1']), prop['r2'])
    K11 = kap_fun(G0(prop['r1']), prop['r1'])
    Lambda = -1 / (4 * prop['del']) * (K22 - K12 - K21 + K11)

    # f_small = K22 > 1e2 # flag large values out of error fun. eval.
    Lambda[K22 > 1e2] = 0  # remove cases with large values out of error fun. eval.
    Lambda[np.absolute(Lambda) < 1e-10] = 0  # remove cases with roundoff error

    return Lambda, G0


def tfer_W1(sp, m, d, z, prop):
    """
    Evaluates the transfer function for a PMA in Case W1.
    
    Parameters:
    sp (dict): Dictionary defining various setpoint parameters 
               (e.g., m_star, V). Use 'get_setpoint' to generate this structure.
    m (float): Particle mass.
    d (float): Particle mobility diameter.
    z (int): Integer charge state.
    prop (dict): Dictionary of device properties (e.g., classifier length).
    
    Returns:
    float: Lambda, the transfer function.
    callable: G0, function mapping final to initial radial position.
    """
    tau, C0, _, rs = parse_inputs(sp, m, d, z, prop)
    # parse inputs for common parameters

    # -- Estimate device parameter --------------------------------------------#
    lam = 2 * tau * (sp['alpha'] ** 2 - sp['beta'] ** 2 / (rs ** 4)) * prop['L'] / prop['v_bar']

    # -- Evaluate G0 and transfer function ------------------------------------#
    G0 = lambda r: 1 / (sp['omega1'] * np.sqrt(m)) * \
                   np.sqrt((m * sp['omega1'] ** 2 * r ** 2 - C0) * np.exp(-lam) + C0)

    ra = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r1'])))
    rb = np.minimum(prop['r2'], np.maximum(prop['r1'], G0(prop['r2'])))

    Lambda = (1 / (2 * prop['del'])) * (rb - ra)

    return Lambda, G0


def tfer_W1_diff(sp, m, d, z, prop):
    """
    Evaluates the transfer function for a PMA in Case E (with diffusion).
    
    Parameters:
    sp (dict): Setpoint parameters (e.g., 'm_star', 'V').
    m (float): Particle mass.
    d (float): Particle mobility diameter.
    z (int): Integer charge state.
    prop (dict): Device properties (e.g., 'classifier_length').

    Returns:
    Lambda (float): Transfer function value.
    G0 (function): Function mapping final to initial radial position.
    """
    _, _, D, _ = parse_inputs(sp, m, d, z, prop)  # get diffusion coeff.
    sig = np.sqrt(2 * prop['L'] * D / prop['v_bar'])  # diffusive spreading parameter

    # -- Evaluate relevant functions ------------------------------------------#
    _, G0 = tfer_W1(sp, m, d, z, prop)
    # get G0 function for this case

    rho_fun = lambda G, r: (G - r) / (np.sqrt(2) * sig)  # recurring quantity
    kap_fun = lambda G, r: \
        (G - r) * erf(rho_fun(G, r)) + \
        sig * np.sqrt(2 / np.pi) * np.exp(-rho_fun(G, r) ** 2)  # define function for kappa

    # -- Evaluate the transfer function and its terms -------------------------#
    K22 = kap_fun(G0(prop['r2']), prop['r2'])
    K21 = kap_fun(G0(prop['r2']), prop['r1'])
    K12 = kap_fun(G0(prop['r1']), prop['r2'])
    K11 = kap_fun(G0(prop['r1']), prop['r1'])
    Lambda = -1 / (4 * prop['del']) * (K22 - K12 - K21 + K11)

    # f_small = K22 > 1e2 # flag large values out of error fun. eval.
    Lambda[K22 > 1e2] = 0  # remove cases with large values out of error fun. eval.
    Lambda[np.absolute(Lambda) < 1e-10] = 0  # remove cases with roundoff error

    return Lambda, G0


#== Other transfer functions ======================================#
def calc_d_star(tau_star, mu):
    # A function to compute d_star when not directly supplied.
    d_star = np.zeros(len(tau_star))
    for ii in range(len(tau_star)):
        # Minimize to find d_star
        d_star[ii] = fmin(lambda d: np.abs(autils.cc(d) * 1e3 * d ** 2 / (18 * mu) - tau_star[ii]), 100e-9)[0]
    return d_star

def aac(da_star, da, prop, opts=None, *args):
    if opts is None:
        opts = {}

    opts.setdefault('input', 'd_star')  # Default input to 'd_star'
    opts.setdefault('scan', 0)  # Default to steady state, no scanning
    opts.setdefault('model', 'ps')  # Default model to particle-streamline
    opts.setdefault('ideal', False)
    opts.setdefault('diffusion', True)

    da, da_star, _ = shape_inputs(da, da_star)

    # Convert from nm to m for calculations
    da = np.array(da) * 1e-9

    # Compute other classifier properties
    prop['del'] = (prop['Qs'] - prop['Qa']) / (prop['Qs'] + prop['Qa'])
    prop['bet'] = (prop['Qs'] + prop['Qa']) / (prop['Qsh'] + prop['Qexh'])

    if np.all(prop['del'] == prop['del'][0]):
        prop['del'] = prop['del'][0]
    if np.all(prop['bet'] == prop['bet'][0]):
        prop['bet'] = prop['bet'][0]

    # Compute resolution
    prop['Rt'] = 1 / prop['bet']

    # Constants
    mu = 1.82e-5  # Gas viscosity [Pa*s]
    rc = prop['r1'] / prop['r2']  # Ratio of radii
    tf = np.pi * prop['L'] * (prop['r2'] ** 2 - prop['r1'] ** 2) / (prop['Qa'] + prop['Qsh'])
    gam = (prop['Qa'] + prop['Qsh'] - prop['Qs'] * (1 - rc ** 2)) / (prop['Qa'] + prop['Qsh'] - prop['Qsh'] * (1 - rc ** 2))

    # Interpret setpoint
    if opts['input'] == 'd_star':  # d_star given, so compute omega
        da_star = da_star * 1e-9  # Convert input from nm to m
        tau_star = autils.cc(da_star) * 1e3 * da_star ** 2 / (18 * mu)
        omega = np.sqrt((prop['Qsh'] + prop['Qexh']) / (np.pi * (prop['r2'] + prop['r1']) ** 2 * prop['L'] * tau_star))

    elif opts['input'] == 'omega':  # omega given, so compute d_star
        omega = da_star
        tau_star = (prop['Qsh'] + prop['Qexh']) / (np.pi * (prop['r2'] + prop['r1']) ** 2 * prop['L'] * omega ** 2)
        da_star = calc_d_star(tau_star, mu)

    # Relaxation time for da input
    tau = autils.cc(da) * 1e3 * da ** 2 / (18 * mu)

    #-- Constants for scanning -----------------------------------------------%
    #   prop.omega_s is initial angular speed
    #   prop.omega_e is final angular speed
    #   prop.tsc is scan time
    if opts['scan']:
        tau_sc = prop['tsc'] / (2 * np.log(prop['omega_e'] / prop['omega_s']))
        tf = (np.pi * prop['L'] * (prop['r2']**2 - prop['r1']**2)) / (prop['Qa'] + prop['Qsh'])  # Johnson et al., Eq. 13

        c_sc = prop['omega_s']**2 * tau_sc * (1 - np.exp(-tf / tau_sc))

    if opts['model'] in ['lt', 'limited-trajectory']:
        if not opts.get('scan', False):  # steady state
            K = omega**2 * tf
        else:  # scanning
            c_tau_star = 1 / (2 * c_sc) * (np.log(1. / rc) + 1/2 * np.log(gam))
            
            if opts['input'] == 'time':  # OPTION 1: time to d_star
                tm = prop['d_star']  # classifier exit time
                tau_star = c_tau_star * np.exp(-tm / prop['tau_sc'])
                calc_d_star(tau_star, prop['mu'])
            else:  # OPTION 2: d_star to time
                tm = -tau_sc * np.log(tau_star / c_tau_star)

            K = c_sc * np.exp(tm / tau_sc)

        # Transfer function calculation
        f1 = (prop['Qa'] + prop['Qsh'] * rc**2 - 
              np.exp(-2 * tau * K) * (prop['Qa'] + prop['Qsh'] - prop['Qs'] * (1 - rc**2))) / \
             (prop['Qa'] * (1 - rc**2))

        f2 = (prop['Qa'] + prop['Qsh']) / prop['Qa'] * \
             (np.exp(-2 * tau * K) - rc**2) / (1 - rc**2)

        f3 = prop['Qs'] / prop['Qa'] * np.ones_like(f1)

        Lambda = np.maximum(0, np.minimum(np.minimum(f1, f2), np.minimum(f3, 1)))

    # Particle streamline approach
    elif opts['model'] in ['ps', 'particle-streamline']:
        # Transmission constants (Johnson, 2018)
        prop['lambda_e'] = 0.8  # classifier entrance/exit transmission efficiency 
        prop['L_eff'] = 46  # length of a circular tube with the same diffusion deposition as the classifier 
        
        # Diffusion properties
        if opts.get('diffusion', True):
            if 'm0' not in prop:
                print('Warning: Mass-mobility information not given for AAC transfer function. Assuming dm = da.')
                dm = da  # assume dm = da
            else:
                dm = autils.dm2da(da, prop)
            B, _ = autils.dm2zp(dm, None, prop['T'], prop['p'])  # compute mobility
            
            kB = 1.3806488e-23  # Boltzmann's constant
            D = kB * prop['T'] * B  # diffusion coefficient
        else:  # neglect diffusion
            D = 0

        # Transfer width using empirical correlation
        if not opts.get('ideal', False):
            # Incorporate deposition parameter
            del_dep = (prop['L_eff'] * D) / prop['Qa']
            fl = del_dep >= 0.007

            lambda_d = np.zeros_like(del_dep)
            lambda_d[fl] = (0.891 * np.exp(-11.5 * del_dep[fl]) + 
                            0.0975 * np.exp(-70.1 * del_dep[fl]) + 
                            0.0325 * np.exp(-179 * del_dep[fl]))
            lambda_d[~fl] = (1 - 5.50 * (del_dep[~fl]**(2/3)) + 
                             3.77 * del_dep[~fl] + 0.814 * (del_dep[~fl]**(4/3)))

            lambda_tf = prop['lambda_e'] * lambda_d  # total transmission efficiency

            a, b, c = -1.73, -0.0316, 1.999
            mu_tf = a * (da * 1e9)**b + c
        else:
            lambda_tf = 1
            mu_tf = 1

        # User-defined transfer function variables for transmission efficiency
        A0 = (lambda_tf * (mu_tf**2)) / (2 * prop['bet'])
        B0 = prop['bet'] / mu_tf

        tau_norm = tau / tau_star

        Lambda = A0 * (np.abs(tau_norm - (1 + B0)) + 
                       np.abs(tau_norm - (1 - B0)) - 
                       np.abs(tau_norm - (1 + B0 * prop['del'])) - 
                       np.abs(tau_norm - (1 - B0 * prop['del'])))
    
    return Lambda, da_star, prop


def bin(s_star, s, mode='log'):
    """
    Evaluate equivalent kernel/transfer functions for uniformly binned data.
    
    Inputs:
    s_star  : numpy array
        Vector of data bin centers.
    s       : numpy array
        Vector of reconstruction bin centers.
    
    Outputs:
    A : scipy sparse matrix
        Kernel matrix.
    """
    s, s_star, _ = shape_inputs(s_star, s)

    # Number of bins and evaluation points
    N_b = np.size(s_star)
    N_x = np.size(s)

    # Convert to log space if that mode.
    if mode == 'log':
        s_star = np.log(s_star)
        s = np.log(s)

    # Bin widths
    db = s_star[1:,:] - s_star[:-1,:]  # bin width for s_star
    dx = s[:,1:] - s[:,:-1]

    # Nodes for s_star and s
    nodes_b = np.concatenate([s_star[0,:] - db[0,:] / 2,
                              np.squeeze(s_star[1:,:] + s_star[:-1,:]) / 2,
                              s_star[-1,:] + db[-1,:] / 2])[np.newaxis]

    nodes_x = np.concatenate([s[:,0] - dx[:,0] / 2,
                              np.squeeze(s[:,1:] + s[:,:-1]) / 2,
                              s[:,-1] + dx[:,-1] / 2])[np.newaxis].T

    # Initialize sparse kernel matrix
    K = np.zeros((N_x, N_b))

    # Fill the kernel matrix
    for ii in range(np.size(s_star,0)):
        K[:, ii] = np.squeeze(np.maximum(
            np.minimum(nodes_x[1:,:], nodes_b[:,ii + 1]) -
            np.maximum(nodes_x[:-1,:], nodes_b[:,ii]), 0
        ) / (nodes_b[:,ii + 1] - nodes_b[:,ii]))

    # Multiply kernel by element area (~ integration)
    A = K * (nodes_x[1:,:] - nodes_x[:-1,:])

    return A


def elpi(da, prop=None):
    """
    Compute kernel/transfer function for ELPI+ based on Järvinen et al. (2014).

    Parameters:
    ----------
    da : array-like
        Aerodynamic diameters in nanometers.
    prop : dict, optional
        ELPI+ properties. If not provided, defaults are used from `prop_elpi()`.

    Returns:
    -------
    K : numpy.ndarray
        Kernel/transfer function for each aerodynamic diameter.
    d50 : numpy.ndarray
        Cutoff diameters.
    """

    # If prop is not provided, use default properties from prop_elpi function
    if prop is None:
        prop = props.elpi()  # Assuming prop_elpi() is a function that returns default properties
    
    d50 = prop['d50']

    da, _, _ = shape_inputs(da)
    da = da.T

    # Define the function En
    def En(da, d50, s50):
        return 1 / (1 + (d50 / da) ** (2 * s50))

    # Initialize kernel and intermediate variable B
    K = np.zeros((len(d50), np.size(da)))
    B = np.zeros(len(da))

    # Loop through d50 values in reverse to compute the kernel K
    for ii in range(len(d50) - 1, -1, -1):
        t0 = En(da, d50[ii], prop['s50'][ii])[:,0]
        K[ii, :] = t0 - B
        B = 1 - (1 - B) * (1 - t0)

    return K, d50


def tri(s_star0, s, R0, z=None):
    """
    Evaluates a triangular transfer function from a setpoint and resolution.
    
    Parameters:
    s_star0 (array-like): Setpoints.
    s (array-like): Array of sizes.
    R0 (array-like or float): Resolutions or a scalar resolution.
    z (array-like or None): Charge states. If None, defaults to a single charge state.
    
    Returns:
    Lambda (ndarray): Transfer function values.
    """

    if isinstance(s_star0[0], dict):
        s, _, _ = shape_inputs(s)

        # Handle PMA setpoint case
        s_star0 = unpack(s_star0)

        zet = R0
        R0 = np.array(s_star0['Rm'])
        s_star0 = np.array(s_star0['m_star'])
        
        if z is None:
            z = [1]
        else:
            z = np.asarray(z)
        
        Lambda = np.zeros((np.size(s_star0), np.size(s), len(z)))
        for ii, zi in enumerate(z):
            s_star = s_star0 * zi
            R = R0 * zi ** (1 / zet)
            Lambda[:, :, ii] = tfer_tri0(s_star, s, R)
    else:
        s, s_star0, _ = shape_inputs(s, s_star0)

        # Simple evaluation at a single charge state
        if np.isscalar(R0):
            R0 = R0 * np.ones_like(s_star0)
        
        Lambda = tfer_tri0(s_star0, s, R0)
    
    return Lambda

def tfer_tri0(s_star, s, R):
    """
    Simple transfer function evaluation.
    
    Parameters:
    s_star (array-like): Setpoints.
    s (array-like): Array of sizes.
    R (array-like): Resolutions.
    
    Returns:
    Lambda (ndarray): Transfer function values.
    """
    s_star = np.asarray(s_star)
    s = np.asarray(s)
    R = np.asarray(R)
    
    s_max = s_star * (1 / R + 1)
    s_del = s_max - s_star
    s_min = 2 * s_star - s_max
    
    Lambda = np.where(
        s <= s_star,
        np.where(s > s_min, (s - s_min) / s_del, 0),
        np.where(s < s_max, (s_star - s) / s_del + 1, 0)
    )
    
    return Lambda
