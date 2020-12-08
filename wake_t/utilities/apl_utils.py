""" This module contains tools to work with APLs """

import scipy.constants as ct
from wake_t.data_analysis import analyze_bunch
from wake_t.beamline_elements import PlasmaLens, Beamline, Drift
from copy import copy
from scipy.optimize import minimize_scalar
import time


def calculate_apl_strength(bunch, apl_start_z=0.05, apl_length=0.1,
                          focal_plane=1., apl_radius=None,
                          return_phasespace=True):
    """
    This function finds the exact current required
    to focus a bunch through an APL at focal_plane,
    with the focus_energy being focussed there.

    Parameters
    ----------
    bunch: ParticleBunch object
        The electron bunch that will be focussed

    apl_start_z: float, default is 0
        Start position of the APL along z, in SI

    apl_length: float, default is 0.1
        Length of the APL, in SI

    focal_plane: float, default is 1
        Position of the required focus along z, in m

    apl_radius: float
        The radius of the plasma lens. If given,
        also returns the plasma lens current.

    return_phasespace: bool
        If True, also returns the bunch at the focus
        for the optimised focusing strength

    Returns
    -------
    k_opt: float
        Plasma lens focussing strength required to
        focus an energy at the focal plane

    I_opt: float, optional
        Current required to focus the required energy
        at the provided plane
    """
    start_time = time.time()
    drift1 = Drift(length=apl_start_z, n_out=5)
    drift2 = Drift(length=focal_plane-apl_length-apl_start_z, n_out=5)

    # Define the optimisation function
    def prop_to_foc(k):
        apl = PlasmaLens(length=apl_length, foc_strength=k, n_out=25)
        beamline = Beamline([drift1, apl, drift2])
        _bunch = copy(bunch)
        return beamline.track(_bunch)

    def opt_fun(k):
        if k < 0:  # It tries negative values while bracketing
            return 1e3
        bunch_prop = prop_to_foc(k)
        pms = analyze_bunch(bunch_prop[-1])
        return pms['sigma_x']  # Use sigma_x, as alpha does not vary much

    # Optimise...
    k_opt = minimize_scalar(opt_fun, bounds=(0, 1e5))  #, bracket=(60, 110, 160))#, method='Brent', bounds=(60, 160))

    # If radius given, calculate the current
    if apl_radius is not None:
        target_energy = analyze_bunch(bunch)['avg_ene']
        I = k_opt.x * (2 * ct.pi * apl_radius ** 2 * target_energy * ct.m_e * ct.c) / (ct.mu_0 * ct.e )
        print(f'Required current in APL with radius {apl_radius*1e3:1.1f} mm is {I:1.2f} A')
    else:
        I = None

    # if required, return the bunch, too
    if return_phasespace:
        foc_bunch = prop_to_foc(k_opt.x)[-1]
    else:
        foc_bunch = None

    print(f'Optimisation took {time.time()-start_time:1.2f} seconds.')

    return k_opt.x, I, foc_bunch
