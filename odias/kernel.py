
import numpy as np

from autils import props
from tfer import tfer


def gen_pma(sp, m, d, z=None, prop=None, opt=None, **kwargs):
    """
    Bridging function used to evaluate particle mass analyzer (PMA) transfer function.

    Parameters:
    ----------
    sp : dict
        Structure defining various setpoint parameters (e.g., m_star, V).
        Use 'get_setpoint' method to generate this structure.
    m : float or np.ndarray
        Particle mass.
    d : float or np.ndarray
        Particle mobility diameter.
    z : np.ndarray, optional
        Integer charge state. If not provided, defaults to range(1, 5).
    prop : dict, optional
        Device properties (e.g., classifier length). Defaults to `prop_pma()`.
    opt : str, optional
        Alphanumeric code for transfer function evaluation method.
        If not provided, defaults to '1C'.
    *args : Additional arguments passed to the `charger` function.

    Returns:
    -------
    Lambda : np.ndarray
        Transfer function.
    prop : dict
        CPMA device settings.
    f_z : np.ndarray
        Charge fractions.
    Lambda_z : np.ndarray
        Transfer function with individual charge contributions.
    qbar : np.ndarray
        Average charge on particles (not on transmitted particles).
    """

    # Parse inputs
    if opt is None:
        opt = '1C'  # Default transfer function evaluation method

    if prop is None:
        prop = props.prop_pma()  # Use default PMA properties if not provided

    if z is None:
        z = np.arange(1, 5)  # Default charge states [1, 2, 3, 4]

    # Compute charge state
    f_z, qbar, _ = tfer.charger(d, z, **kwargs)  # Get fraction charged for d

    # PMA transfer function
    Lambda_z, _ = tfer.pma(sp, m, d, z, prop, opt)

    # Incorporate charge fraction
    Lambda_z *= np.expand_dims(f_z, axis=0)

    # Sum over multiple charge states
    Lambda = np.sum(Lambda_z, axis=2)

    # If nargout > 3 equivalent in Python (for additional outputs)
    if Lambda_z.ndim == 3:
        Lambda_z = np.transpose(Lambda_z, (0, 2, 1))

    return Lambda, prop, f_z, Lambda_z, qbar


def gen_smps(d_star, d, z=None, argin_dma=None, argin_z=None):
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
        z = np.arange(1, 5)  # Default charge states

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
    Omega_z, _, _, _ = tfer.dma(d_star, d, z, *argin_dma)  # Call DMA transfer function

    # Incorporate charge fraction
    Omega_z *= np.expand_dims(f_z, axis=0)

    # Sum over multiple charge states
    Omega = np.sum(Omega_z, axis=2)

    # If more outputs are needed
    if Omega_z.ndim == 3:
        Omega_z = np.transpose(Omega_z, (0, 2, 1))

    return Omega, f_z, Omega_z, qbar
