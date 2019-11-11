import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

# TODO: make this parallel (multiprocessing / GPU-threads)

class singlet_wavefunction(Object):
    def __init__(self, config):
        self.config = config

    def get_O(self, base_state):
        '''
            O_i = \\partial{\\psi_i} / \\psi_i
        '''

    def get_wavefunction(indexes, pairing, config, pfaffian = False):
        A = np.zeros((len(indexes), len(indexes)))
        for n1, idx1 in enumerate(indexes):
            for n2, idx2 in enumerate(indexes):
                A[n1, n2] = pairing.get_pairing_f(idx1, idx2)
        # print(np.sum(np.abs(A + A.T)))
        if not pfaffian:
            return xp.linalg.det(A)  # if we are performing MC-sampling, we only need the WF


# counter = 0
# deltas = []
# config = config()
# pairing = models_vmc.on_site_and_nn_pairing(config)
#for index1 in range(config.n_orbitals * config.n_sublattices * config.Ls ** 2 * 2):
#    for index2 in range(config.n_orbitals * config.n_sublattices * config.Ls ** 2 * 2):
#        delta = pairing.get_pairing_f(index1, index2)
#        if delta != 0.0:
#            counter += 1
#            deltas.append(delta)

#  A = get_wavefunction(np.random.choice(config.total_dof, config.total_dof // 2, replace = False), pairing, config)
#print(A)