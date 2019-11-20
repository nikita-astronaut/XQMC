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
pairings_list = [pairings.on_site_pairings[0]]
wf = wavefunction_singlet(config_vmc, pairings_list)

generate_MC_chain(wf)