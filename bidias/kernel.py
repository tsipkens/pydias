
import numpy as np

from scipy.ndimage import gaussian_filter

from autils import autils
from tfer import tfer

def check_type(type, str):
    """
    Check if type contains str and return position or None.
    """
    
    if any(stri == str for stri in type):
        return type.index(str)
    else:
        return None


def build(grid_i, spec, z=None, grid_b=None, type=None):
    """
    Interfaces with transfer function and charge fractions to build kernels for bidimensional problems. 
    """

    print('\r' + '\033[36m' + '[== Computing kernel ==]' + '\033[0m' + '\n')

    #== Parse inputs =================================#
    if z is None or len(z) == 0:
        z = np.arange(1, 4)  # default charge states

    if grid_b is not None:
        grid_b = grid_b
    
    if type is None:
        type = grid_i.type
    
    nc = len(spec)  # Number of classifiers
    Lambda = [None] * nc  # Initialize Lambda list
    
    # Get dimension indices from grid_i object
    dm_idx = None
    mp_idx = None
    da_idx = None

    # Get indices of grid dimension types. 
    # By default (if not supplied), assume a mass-mobility grid.
    if type is None:
        dm_idx = 0
        mp_idx = 1
        rho_idx = None
        da_idx = None

    else:
        dm_idx = check_type(type, 'dm')
        mp_idx = check_type(type, 'mp')
        da_idx = check_type(type, 'da')
        rho_idx = check_type(type, 'rho')
    
    # Unpack grid elements for transfer function evaluation.
    if rho_idx is not None:
        for ii in np.arange(len(spec)):
            if spec[ii][0] == 'pma':
                prop_p = spec[ii][2]
        
        if mp_idx is None:
            m = (np.pi/6) * grid_i.elements[:, rho_idx] * \
                grid_i.elements[:, dm_idx] ** 3 * 1e-9

    elif mp_idx is not None:
        m = grid_i.elements[:, mp_idx]
    
    # Handle cases where mobility diameter isn't given (required for PMA/charging)
    if dm_idx is None:
        # OPTION 1: Use da and mp to compute dm if available
        if da_idx is not None and mp_idx is not None:
            da = grid_i.elements[:, da_idx]
            dm = autils.mp_da2dm(m, da)

        # OPTION 2: Apply mass-mobility relationship
        elif mp_idx is not None:
            for ii in range(len(spec)):
                if spec[ii][0] == 'pma':
                    prop_p = spec[ii][2]

            print('Invoking mass-mobility relationship to determine dm.')
            dm = autils.mp2dm(grid_i.elements[:, mp_idx] * 1e-18, prop_p) * 1e9
    else:  # otherwise use explicit mobility diameter dimension
        dm = grid_i.elements[:, dm_idx]
    
    # Loop over classifiers to compute Lambda
    for ii in range(nc):
        classifier = spec[ii][0]
        
        if classifier == 'charger':
            print('Computing charger contribution ...')
            
            d, idx = np.unique(dm, return_inverse=True)

            f_z, _, _ = tfer.charger(d, z)
            f_z = np.expand_dims(f_z, 0)
            
            Lambda[ii] = np.repeat(f_z, np.size(Lambda[0], 0), axis=0)

            Lambda[ii] = Lambda[ii][:,idx,:]

            autils.textdone()
        
        elif classifier in ['dma', 'smps']:
            print('Computing DMA contribution ...')

            d_star, idx_star = np.unique(spec[ii][1], return_inverse=True)  # find unique entries to speed computation
            d, idx = np.unique(grid_i.elements[:, dm_idx], return_inverse=True)  # extract corresponding mobility diameters from grid

            Lambda[ii], _, _, _ = tfer.dma(d_star, d, z, spec[ii][2])

            Lambda[ii] = Lambda[ii][idx_star,:,:]
            Lambda[ii] = Lambda[ii][:,idx,:]

            autils.textdone()
        
        elif classifier == 'pma':
            print('Computing PMA contribution ...')

            sp = tfer.unpack(spec[ii][1])

            v_star, idx_star = np.unique(np.hstack((sp['V'], sp['omega'])), return_inverse=True, axis=0)  # find unique entries to speed computation
            sp, _ = tfer.get_setpoint(spec[ii][2], 'V', v_star[:,0], 'omega', v_star[:,1])
            
            v, idx = np.unique(np.vstack((m, dm)).T, return_inverse=True, axis=0)  # extract corresponding mobility diameters from grid
            m = v[:,0]
            d = v[:,1]

            Lambda[ii], _ = tfer.pma(sp, m, d, z, spec[ii][2], '1C')

            Lambda[ii] = Lambda[ii][idx_star,:,:]
            Lambda[ii] = Lambda[ii][:,idx,:]

            autils.textdone()
        
        elif classifier in ['bin', 'sp2']:
            print('Computing binned contribution ...')

            s_idx = spec[ii][2]

            s_star, idx_star = np.unique(spec[ii][1], return_inverse=True)  # find unique entries to speed computation
            s, idx = np.unique(grid_i.elements[:, s_idx], return_inverse=True)  # extract corresponding mobility diameters from grid

            Lambda[ii] = tfer.bin(s_star, s)

            Lambda[ii] = Lambda[ii][idx_star,:]
            Lambda[ii] = Lambda[ii][:,idx]

            Lambda[ii] = np.expand_dims(Lambda[ii], axis=2)
            Lambda[ii] = np.repeat(Lambda[ii], len(z), axis=2)

            autils.textdone()
        
        # Add other cases for 'pma', 'aac', 'bin', etc., similarly...
    
    # Compile the kernel
    print(" Compiling kernel ...")
    Ac = Lambda[0]
    for ii in range(1, nc):
        Ac = Ac * Lambda[ii]
    
    # Sum over charge states and multiply by grid area
    A = np.sum(Ac, axis=2)
    A = A * grid_i.dr()[0]
    
    # Convert to sparse matrix
    # A = csr_matrix(A)

    autils.textdone()

    print('\r' + '\033[36m' + '[== COMPLETE! ========]' + '\033[0m' + '\n')
    
    return A, Ac


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
