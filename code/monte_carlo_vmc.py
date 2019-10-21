import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc
import wavefunction_vmc
from config_vmc import MC_parameters as config_vmc
from copy import deepcopy

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def generate_conf(config_vmc, config):
    total_particles = config.total_dof // 2 + config_vmc.particles_excess
    conf = np.zeros(shape = (config.total_dof))
    spins_up = np.random.choice(np.arange(0, len(conf), 2), config_vmc.total_spin // 2 + total_particles // 2, replace = False)

    spins_down = np.random.choice(np.arange(1, len(conf), 2), -config_vmc.total_spin // 2 + total_particles // 2, replace = False)

    conf[spins_up] = 1
    conf[spins_down] = 1

    return conf

def flip_spin(conf, config):
    while True:
        spin1, spin2 = np.random.choice(config.total_dof, size = 2, replace = False)
        if (spin1 + spin2) % 2 == 0 and conf[spin1] + conf[spin2] == 1:
            break

    conf[spin1] = 1 - conf[spin1]
    conf[spin2] = 1 - conf[spin2]

    return conf

def generate_MC_chain(config_vmc, config, pairing, conf_ini = None):
    if conf_ini == None:
        conf_ini = generate_conf(config_vmc, config)

    current_conf = deepcopy(conf_ini)
    current_A = wavefunction_vmc.get_wavefunction(np.where(current_conf > 0.5)[0], pairing, config)
    MC_chain = np.zeros((0, len(conf_ini)))

    for MC_step in range(config_vmc.MC_length):
        MC_chain = np.concatenate([MC_chain, current_conf[np.newaxis, ...]], axis = 0)
        conf_proposed = flip_spin(deepcopy(current_conf), config)
        A_proposed = wavefunction_vmc.get_wavefunction(np.where(conf_proposed > 0.5)[0], pairing, config)
        ratio = np.min([1, A_proposed / current_A])
        lamb = np.random.uniform(0, 1)
        if lamb < ratio:
            current_conf = conf_proposed
            current_A = A_proposed
        print(A_proposed, current_A)
config = config()
config_vmc = config_vmc()
pairing = config_vmc.pairing(config)

generate_MC_chain(config_vmc, config, pairing)