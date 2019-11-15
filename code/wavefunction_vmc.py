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


