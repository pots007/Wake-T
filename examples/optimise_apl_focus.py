"""
Basic example of optimising the current to
focus a particular energy at the focal plane
"""

from wake_t.beamline_elements import Beamline, Drift, PlasmaLens
from wake_t.utilities.apl_utils import calculate_apl_strength
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t.data_analysis import analyze_bunch
from aptools.data_handling.saving import save_beam
from pprint import pprint


# Set up simplistic beam parameters
target_energy = 127.  # beta*gamma
norm_emitt = 1e-6
spot_size = 1e-6
beta_initial = spot_size ** 2 / (norm_emitt / target_energy)

# And simple APL
r0 = 2e-3  # APL radius, in m
z0 = 0.05  # Start position of APL
L = 0.1    # APL length
zf = 1.    # Desired focal plane

# Generate a bunch
bunch = get_gaussian_bunch_from_twiss(en_x=norm_emitt, en_y=norm_emitt, a_x=0., a_y=0.,
                                      b_x=beta_initial, b_y=beta_initial, ene=target_energy, ene_sp=0.,
                                      s_t=20., xi_c=0., q_tot=10., n_part=1e5)

# Fint the focussing strength required
k_opt, I_opt, bunch_foc = calculate_apl_strength(bunch, apl_start_z=z0, apl_length=L,
                                                 focal_plane=zf, apl_radius=r0)

print(k_opt, I_opt)
print(f'Bunch parameters at focus:')
pprint(analyze_bunch(bunch_foc))

# We can now propagate the bunch to a plane __just__ before focus for ASTRA ICS sim
dz = 100e-6  # Distance before focus
beamline = Beamline((Drift(length=z0, n_out=5),
                     PlasmaLens(length=L, foc_strength=k_opt, n_out=25),
                     Drift(length=zf-L-z0-dz, n_out=5)))
bunch_list = beamline.track(bunch, out_initial=True)

print(f'Bunch parameters {dz*1e6} microns before focus:')
pprint(analyze_bunch(bunch_list[-1]))
# And save it to ASTRA format for ICS stuff
save_beam(code_name='astra', beam_data=bunch_list[-1].get_bunch_matrix(),
          folder_path='.', file_name='bunch_test')
