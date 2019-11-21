import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc
from wavefunction_vmc import wavefunction_singlet
from config_vmc import MC_parameters as config_vmc
import pairings

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def generate_MC_chain(wf):
    for MC_step in range(config_vmc.MC_length):
        wf.perform_MC_step()

    return

config_vmc = config_vmc()
pairings_list = [pairings.on_site_pairings[5]]
for dt in np.logspace(-5, -8, 100):
    wf_1 = wavefunction_singlet(config_vmc, pairings_list, [1.0 - dt])
    wf_2 = wavefunction_singlet(config_vmc, pairings_list, [1.0 + dt])
    print((wf_2.current_det - wf_1.current_det) / dt / (wf_1.current_det + wf_2.current_det) - wf_1.get_O_pairing(0),
          (-wf_2.current_det - wf_1.current_det) / dt / (wf_1.current_det - wf_2.current_det) - wf_1.get_O_pairing(0))  # log derivative calculated explicitly
    # print(wf_1.get_O_pairing(0))  # log devirative calculated from O_k formula

# generate_MC_chain(wf)