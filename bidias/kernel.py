
import numpy as np

from scipy.ndimage import gaussian_filter

from autils import autils
from cmap import textdone
from tfer import tfer

def check_type(type, str):
    """
    Check if type contains str and return position or None.
    """
    
    if any(stri == str for stri in type):
        return type.index(str)
    else:
        return None


def expand_sp(sp0, grid_b, prop_pma, dim=0):
    """
    Expands a setpoint structure for a provided grid.
    """
    # Initialize vector.
    omega = np.zeros_like(grid_b.elements[:,dim])
    v = np.zeros_like(grid_b.elements[:,dim])

    # Loop through setpoints and compare to masses in grid.
    for ii in range(len(sp0)):
        mask = grid_b.elements[:,dim] == sp0[ii]['m_star'] * 1e18
        omega[mask] = sp0[ii]['omega']
        v[mask] = sp0[ii]['V']
    return tfer.get_setpoint(prop_pma, 'V', v, 'omega', omega)[0]  # recompile into new setpoint list


def build(grid_i, spec, z=None, grid_b=None, type=None, detect='number'):
    """
    Interfaces with transfer function and charge fractions to build kernels for bidimensional problems. 
    """
    
    print('\r' + '\033[36m' + '[ BUILDING KERNEL ]' + '\033[0m')

    #== Parse inputs =================================#
    if z is None or len(z) == 0:
        z = np.arange(1, 4)  # default charge states

    if grid_b is not None:
        grid_b = grid_b
    
    if type is None:
        type = grid_i.type
    
    nc = len(spec)  # Number of classifiers
    Lambda = [None] * nc  # Initialize Lambda list
    
    # Get indices of grid dimension types. 
    # By default (if not supplied), assume a mass-mobility grid.
    if type is None:
        type = ['dm', 'mp']

    # Extract indices.
    dm_idx = check_type(type, 'dm')
    mp_idx = check_type(type, 'mp')
    da_idx = check_type(type, 'da')
    rho_idx = check_type(type, 'rho')
    mrbc_idx = check_type(type, 'mrbc')
    frbc_idx = check_type(type, 'frbc')
    
    # ------- MASS CHECK --------
    # Unpack grid elements for transfer function evaluation.
    mp = None
    if rho_idx is not None:
        for ii in np.arange(len(spec)):
            if spec[ii][0] == 'pma':
                prop_p = spec[ii][2]
        
        if mp_idx is None:
            if dm_idx is None:
                da = grid_i.elements[:, da_idx]  # then assume da_idx is available
                dm = autils.da_rhoeff2dm(da * 1e-9, grid_i.elements[:, rho_idx]) * 1e9
            else:
                dm = grid_i.elements[:, dm_idx]  # otherwise, extract dm from grid
            mp = (np.pi/6) * grid_i.elements[:, rho_idx] * dm ** 3 * 1e-9  # now compute mass
            
        if da_idx is None:
            da = autils.dm_rhoeff2da(dm * 1e-9, grid_i.elements[:, rho_idx]) * 1e9

    elif mp_idx is not None:
        mp = grid_i.elements[:, mp_idx]
    
    # -------- MOBILITY CHECK --------
    # Handle cases where mobility diameter isn't given (required for PMA/charging).
    if dm_idx is None:
        # OPTION 1: Use da and mp to compute dm if available
        if da_idx is not None and mp_idx is not None:
            da = grid_i.elements[:, da_idx]
            dm = autils.mp_da2dm(mp * 1e-18, da * 1e-9) * 1e9

        # OPTION 2: Apply mass-mobility relationship
        elif mp_idx is not None:
            for ii in range(len(spec)):
                if spec[ii][0] == 'pma':
                    prop_p = spec[ii][2]

            print('Invoking mass-mobility relationship to determine dm.')
            dm = autils.mp2dm(grid_i.elements[:, mp_idx] * 1e-18, prop_p) * 1e9

    else:  # otherwise use explicit mobility diameter dimension
        dm = grid_i.elements[:, dm_idx]

    # -------- AERODYNAMIC CHECK --------
    if da_idx is not None:
        da = grid_i.elements[:, da_idx]
    elif mp is not None:
        da = autils.dm_mp2da(dm * 1e-9, mp * 1e-18) * 1e9

    # -------- MRBC CHECK --------
    # Handle cases where mrbc is not given. Only relevant to SP2.
    if mrbc_idx is None:
        if frbc_idx is not None:  # then, convert from frbc
            frbc = grid_i.elements[:, frbc_idx]
            mrbc = frbc * grid_i.elements[:, mp_idx]
            
    else:  # otherwise use explicit mrbc dimension
        mrbc = grid_i.elements[:, mrbc_idx]

    # -------- PROCEED TO BUILD KERNEL --------
    for ii in range(nc):  # loop over classifiers
        classifier = spec[ii][0]
        
        if classifier == 'charger':  # add charger and charge fractions
            print('Computing charger contribution ...', end="", flush=True)
            
            d, idx = np.unique(dm, return_inverse=True)

            f_z, _, _ = tfer.charger(d, z, *spec[ii][1:])
            f_z = np.expand_dims(f_z, 0)
            
            # if nc > 1:
            #     size = np.size(Lambda[0], 0)  # extract data length from previous Lambda entry
            # else:
            #     size = 1
            
            Lambda[ii] = np.repeat(f_z, np.size(Lambda[0], 0), axis=0)

            Lambda[ii] = Lambda[ii][:,idx,:]

            textdone()
        
        elif classifier in ['dma', 'smps']:  # differential mobility analyzer
            print('Computing DMA contribution ...', end="", flush=True)

            d_star, idx_star = np.unique(spec[ii][1], return_inverse=True)  # find unique entries to speed computation
            d, idx = np.unique(grid_i.elements[:, dm_idx], return_inverse=True)  # extract corresponding mobility diameters from grid

            Lambda[ii], _, _, _ = tfer.dma(d_star, d, z, spec[ii][2])

            Lambda[ii] = Lambda[ii][idx_star,:,:]
            Lambda[ii] = Lambda[ii][:,idx,:]

            textdone()
        
        elif classifier == 'pma':  # particle mass analyzer (CPMA/APM)
            print('Computing PMA contribution ...', end="", flush=True)

            sp = spec[ii][1]
            if isinstance(sp, list):
                sp = tfer.unpack(sp)

            # Use voltage and angular speed to find unique setpoints.
            # This speeds up computation.
            if isinstance(sp, dict):
                v_star, idx_star = np.unique(np.hstack((sp['V'], sp['omega'])), return_inverse=True, axis=0)  # find unique entries
                sp, _ = tfer.get_setpoint(spec[ii][2], 'V', v_star[:,0], 'omega', v_star[:,1])
            else:
                sp, idx_star = sp.unique()  # find unique entries
                sp = tfer.pack(sp)
            
            v, idx = np.unique(np.vstack((mp, dm)).T, return_inverse=True, axis=0)  # extract corresponding mobility diameters from grid
            # v, idx = np.unique(np.vstack((mp, autils.mp2dm(grid_i.elements[:, mp_idx] * 1e-18, spec[ii][2]) * 1e9)).T, return_inverse=True, axis=0)
            m = v[:,0]
            d = v[:,1]

            Lambda[ii], _ = tfer.pma(sp, m, d, z, spec[ii][2], '1C_diff')

            Lambda[ii] = Lambda[ii][idx_star,:,:]
            Lambda[ii] = Lambda[ii][:,idx,:]

            textdone()
        
        elif classifier in ['sp2', 'bin']:  # binned contributions (also used for SP2)
            print('Computing binned contribution ...', end="", flush=True)

            if classifier == 'sp2':
                s = mrbc
            else:
                s = grid_i.elements[:, spec[ii][2]]
            
            # Separate procedure to find unique values, as order matters.
            def unique2(s):
                idx = np.zeros(len(s), dtype=np.int16)
                idx[0] = 0
                su = np.array(s[0])
                for ss in range(1, len(s)):
                    if s[ss] == s[ss-1]:
                        idx[ss] = idx[ss-1]
                    else:
                        idx[ss] = idx[ss-1] + 1
                        su = np.append(su, s[ss])
                return su, idx

            # Now sort values.
            s_star, idx_star = unique2(spec[ii][1])  # find unique entries to speed computation
            s, idx = unique2(s)  # extract corresponding mobility diameters from grid
            
            # Now compute transfer function.
            Lambda[ii] = tfer.bin(s_star, s)

            Lambda[ii] = Lambda[ii][idx_star,:]
            Lambda[ii] = Lambda[ii][:,idx]

            # Add dimensions in case charging.
            Lambda[ii] = np.expand_dims(Lambda[ii], axis=2)

            textdone()

        elif classifier == 'aac':  # aerodynamic classifier (e.g., AAC)
            print('Computing AAC contribution ....', end="", flush=True)
            if isinstance(spec[ii][1], np.ndarray):
                d_star, idx_star = np.unique(spec[ii][1], return_inverse=True)  # find unique entries to speed computation
            else:
                d_star, idx_star = spec[ii][1].unique()
            
            v, idx = np.unique(np.vstack((da, dm)).T, return_inverse=True, axis=0)  # extract corresponding mobility diameters from grid
            d = v[:,0]
            d2 = v[:,1]  # mobility diameter

            Lambda[ii], _, _= tfer.aac(d_star, d, spec[ii][2], spec[ii][3], dm=d2)

            Lambda[ii] = Lambda[ii][idx_star,:]
            Lambda[ii] = Lambda[ii][:,idx]
            
            # Add dimensions in case charging.
            Lambda[ii] = np.expand_dims(Lambda[ii], axis=2)

            textdone()
    
    # Compile the kernel
    print("Compiling kernel ...", end="", flush=True)
    Ac = Lambda[0]
    for ii in range(1, nc):
        Ac = Ac * Lambda[ii]  # cannot use *= as dimensions may change

    # Zero any NaN values (e.g., can occur when using dm-mp grid with AAC).
    Ac[np.isnan(Ac)] = 0
    
    # If detector measures charge (i.e., electrometer), adjust charge state weighting. 
    if detect == 'charge' and np.shape(Ac)[2] == len(z):
        print(' + charge weighting ...', end="", flush=True)
        A = np.sum(Ac * np.expand_dims(z, [0,1]), axis=2)

    else:  # otherwise sum to get count/number
        A = np.sum(Ac, axis=2)

    # Sum over charge states and multiply by grid area
    A = A * grid_i.dr()[0]
    
    # Convert to sparse matrix
    # A = csr_matrix(A)

    textdone()

    print('\r' + '\033[36m' + '[ COMPLETE! ]' + '\033[0m' + '\n\n')
    
    return A, Ac


def build_charge(grid_i, prop_dma=None, grid_b=None):
    """
    Wrapper to generate code for a kernel to invert the charge distributions 
    instead of the size distribution.
    """

    z = grid_i.edges[check_type(grid_i.type, 'z')]
    d1 = grid_b.elements[:, check_type(grid_b.type, 'dstar1')]
    d2 = grid_b.elements[:, check_type(grid_b.type, 'dstar2')]

    # -- First classifier --
    A1, _ = build(grid_i, [['dma', d1, prop_dma], ['charger']], z)

    # -- Second classifier --
    _, A2c = build(grid_i, [['dma', d2, prop_dma]], z)

    A2d = np.zeros(A2c.shape[:2])

    for ii in range(A2c.shape[1]):
        idx = grid_i.elements[ii, 1]
        A2d[:, ii] = A2c[:, ii, int(idx)-1]

    A = A1 * A2d

    return A


def gen_smps_t(d_star, t_star, d, t, z=None, argin_dma=None, argin_z=None):
    """
    Evaluates the transfer function of a differential mobility analyzer (DMA).

    Parameters:
    ----------
    d_star : np.ndarray
        Particle diameter, measurement set point for DMA [m].
    d : np.ndarray
        Particle diameter, points in the integral, can be a vector [m].
    z : np.ndarray, optional
        Charge states, defaults to range(1, 5).
    argin_dma : dict or list, optional
        DMA properties, can be generated using prop_DMA function.
    argin_z : dict or list, optional
        Charging options for the `charger` function.

    Returns:
    -------
    Omega : np.ndarray
        Transfer function.
    f_z : np.ndarray
        Charging fractions.
    Omega_z : np.ndarray
        Transfer function with individual charge contributions.
    qbar : np.ndarray
        Average charge on particles.
    """

    # Parse inputs
    if z is None:
        z = -np.arange(1, 5)  # Default charge states

    if argin_dma is None:
        argin_dma = []
    elif not isinstance(argin_dma, list):
        argin_dma = [argin_dma]

    if argin_z is None:
        argin_z = []
    elif not isinstance(argin_z, list):
        argin_z = [argin_z]

    # Evaluate particle charging fractions
    f_z, qbar, _ = tfer.charger(d, z, *argin_z)  # Get fraction charged for d

    # Evaluate DMA transfer function
    Omega_z, _, _, _ = tfer.dma(d_star, d, np.abs(z), *argin_dma)  # Call DMA transfer function

    # Incorporate charge fraction
    Omega_z *= np.expand_dims(f_z, axis=0)

    # Sum over multiple charge states
    Omega = np.sum(Omega_z, axis=2)

    Omega = gaussian_filter(Omega, 0.5)  # soften, to avoid kernel noise
    
    # Incorporate time
    tu = np.unique(t)
    B0 = tfer.bin(t_star, tu, mode='linear')
    B0 = gaussian_filter(B0, 1)  # soften, to avoid kernel noise

    B = np.zeros_like(Omega)
    for ii in range(len(tu)):
        nu = sum(t == tu[ii])
        B[:, t == tu[ii]] = np.expand_dims(B0[:, ii], 1) * np.ones((1, nu))

    Omega = Omega * B

    return Omega
