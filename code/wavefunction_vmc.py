import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc
import pairings

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

# TODO: make this parallel (multiprocessing / GPU-threads)
class wavefunction_singlet():
    def __init__(self, config, pairings_list):
        self.config = config
        self.pairings_list = pairings_list
        # self.var_params = self._init_var_params()
        self.var_params = [0.]
        self.U_matrix = self._construct_U_matrix()
        while True:
            self.occupied_sites, self.empty_sites = self._generate_configuration()
            self.U_tilde_matrix = self._construct_U_tilde_matrix()
            if np.linalg.matrix_rank(self.U_tilde_matrix) == self.config.N_electrons:
                break
        self.W_GF = self._construct_W_GF()
        self.current_det = np.linalg.det(self.U_tilde_matrix) ** 2

    def get_O(self, base_state):
        '''
            O_i = \\partial{\\psi_i} / \\psi_i
        '''
    def _construct_U_matrix(self):
        K = self.config.model(self.config.Ls, self.config.mu)
        Delta = pairings.get_total_pairing(self.config, self.pairings_list, self.var_params)  # the magnitude of the pairing is not yet included here

        T = np.zeros((2 * K.shape[0], 2 * K.shape[1])) * 1.0j
        T[0:K.shape[0], 0:K.shape[1]] = K
        T[K.shape[0]:2 * K.shape[0], K.shape[1]:2 * K.shape[1]] = -K
        T[0:K.shape[0], K.shape[1]:2 * K.shape[1]] = Delta
        T[K.shape[0]:2 * K.shape[0], 0:K.shape[1]] = Delta.conj().T

        E, U = np.linalg.eigh(T)
        assert(np.allclose(np.diag(E), U.conj().T.dot(T).dot(U)))  # U^{\dag} T U = E
        lowest_energy_states = np.argpartition(E, self.config.N_electrons)[:self.config.N_electrons]  # select lowest-energy orbitals
        U = U[:, lowest_energy_states]  # select only occupied orbitals
        return U 

    def _construct_U_tilde_matrix(self):
        U_tilde = self.U_matrix[self.occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        return self.U_matrix.dot(U_tilde_inv)

    def _generate_configuration(self):
        occupied_sites = np.random.choice(np.arange(self.config.total_dof), size = self.config.N_electrons, replace = False)
        # electrons are placed in occupied_states as they are in the string d_{R_1} d_{R_2} ... |0>
        empty_sites = np.arange(self.config.total_dof)
        empty_sites[occupied_sites] = -1
        empty_sites = empty_sites[empty_sites > -0.5]

        return occupied_sites, empty_sites

    def perform_MC_step(self):
        moved_site_idx = np.random.choice(np.arange(len(self.occupied_sites)), 1)[0]
        moved_site = self.occupied_sites[moved_site_idx]
        empty_site_idx = np.random.choice(np.arange(len(self.empty_sites)), 1)[0]
        empty_site = self.empty_sites[empty_site_idx]

        det_ratio = self.W_GF[empty_site, moved_site_idx] ** 2

        if det_ratio < np.random.uniform(0, 1):
            return False, 1
        self.current_det *= det_ratio
        print(self.current_det)
        self.occupied_sites[moved_site_idx] = empty_site
        self.empty_sites[empty_site_idx] = moved_site

        ### DEBUG ###
        # U_tilde_new = self._construct_U_tilde_matrix()
        # det_ratio_naive = np.linalg.det(U_tilde_new) ** 2 / np.linalg.det(self.U_tilde_matrix) ** 2
        # print(det_ratio, det_ratio_naive, det_ratio / det_ratio_naive - 1, np.linalg.matrix_rank(self.U_tilde_matrix))
        # self.U_tilde_matrix = U_tilde_new # = self._construct_U_tilde_matrix()

        delta = np.zeros(self.W_GF.shape[1])
        delta[moved_site_idx] = 1
        self.W_GF -= np.einsum('i,k->ik', self.W_GF[:, moved_site_idx], self.W_GF[empty_site, :] - delta) / self.W_GF[empty_site, moved_site_idx]  # TODO this is the most costly operation -- go for GPU
        # W_GF_naive = self.U_matrix.dot(np.linalg.inv(self.U_tilde_matrix))
        # print('W_GF discrepancy =', np.sum(np.abs(self.W_GF - W_GF_naive)))

        return True, det_ratio