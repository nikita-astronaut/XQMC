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
class wavefunction_singlet(Object):
    def __init__(self, config):
        self.config = config
        self.var_params = self._init_var_params()
        self.U_matrix = self._construct_U_matrix()
        self.conf, self.occupied_sites, self.empty_sites, self.positions_in_string = self._generate_configuration()
        self.U_tilde_matrix = self._construct_U_tilde_matrix()
        self.W_GF = self._construct_W_GF()

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
        occupied_sites_string = positions_in_string[positions_in_string > -0.5]
        occupied_sites = np.where(positions_in_string > -0.5)[0]
        # occupied_sites[i] is located in the string at the position occupied_sites_string[i]
        U_tilde = np.zeros((self.config.N_electrons, self.config.N_electrons))

        U_tilde[occupied_sites_string, :] = U[occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        return self.U_matrix.dot(U_tilde_inv)

    def _generate_configuration(self):
        conf = np.zeros(self.config.total_dof)  # 2 * n_sites
        occupied_sites = np.random.choice(np.arange(self.config.total_dof), size = self.config.N_electrons, replace = False)
        empty_sites = np.arange(self.config.total_dof)
        empty_sites[occupied_sites] = -1
        empty_sites = empty_sites[empty_sites > 0.5]

        conf[occupied_sites] = 1
        positions_in_string = np.zeros(self.config.total_dof) - 1
        positions_in_string[occupied_sites] = np.arange(len(occupied_sites))  # the initial state is
                                                                              # d^{\dag}_{e_positions[0]} d^{\dag}_{e_positions[1]} ...|0>

        return conf, occupied_sites, empty_sites, positions_in_string

    def perform_MC_step(self):
        moved_site = np.random.choice(self.e_positions, 1)[0]
        empty_site = np.random.choice(self.empty_sites, 1)[0]

        det_ratio = self.W_GF[empty_site, moved_site]

        if det_ratio ** 2 < np.random.uniform(0, 1):
            return False, 1

        self.positions_in_string[empty_site] = self.positions_in_string[moved_site]
        self.positions_in_string[moved_site] = -1
        self.occupied_sites[self.occupied_sites == moved_site] = empty_site
        self.empty_sites[self.empty_sites == empty_site] == moved_site

        delta = np.zeros(W.shape[1])
        delta[moved_site] = 1
        self.W_GF -= np.einsum('i,k->ik', W_GF[:, moved_site], W[empty_site, :] - delta) / W_GF[empty_site, moved_site]

        return True, det_ratio ** 2