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
class wavefunction(Object):
    def __init__(self, config):
        self.config = config
        self.var_params = self._init_var_params()
        self.U_matrix = self._construct_U_matrix()
        self.conf, self.e_positions, self.positions_in_string = self._generate_configuration()

    def get_O(self, base_state):
        '''
            O_i = \\partial{\\psi_i} / \\psi_i
        '''
    def _construct_U_matrix(self):
        K = self.config.model(self.config.L, self.config.mu)
        Delta = self.config.pairing(self.config.L)  # the magnitude of the pairing is not yet included here

        T = np.zeros((2 * K.shape[0], 2 * K.shape[1]))
        T[0:K.shape[0], 0:K.shape[1]] = K
        T[K.shape[0]:2 * K.shape[0], K.shape[1]:2 * K.shape[1]] = -K
        T[0:K.shape[0], K.shape[1]:2 * K.shape[1]] = self.var_params[0] * Delta
        T[K.shape[0]:2 * K.shape[0], 0:K.shape[1]] = self.var_params[0] * Delta.conj().T

        self.E, self.U = np.linalg.eig(T)
        assert(np.allclose(np.diag(energies), U.conj().T.dot(A).dot(U)))  # U^{\dag} T U = E

        lowest_energy_states = np.argpartition(self.E, self.config.N_electrons)  # select lowest-energy orbitals
        U = U[:, lowest_energy_states]  # select only occupied orbitals
        return U 

    def _construct_U_tilde_matrix(self):
        

    def _generate_configuration(self):
        conf = np.zeros(self.config.total_dof)  # 2 * n_sites
        e_positions = np.random.choice(np.arange(self.config.total_dof), size = self.config.N_electrons, replace = False)
        conf[e_positions] = 1
        positions_in_string = np.zeros(self.config.total_dof) - 1
        positions_in_string[e_positions] = np.arange(len(e_positions)) + 1  # the initial state is
                                                                            # d^{\dag}_{e_positions[0]} d^{\dag}_{e_positions[1]} ...|0>

        return conf, e_positions, positions_in_string

