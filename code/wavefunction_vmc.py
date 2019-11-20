import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc
import pairings
from copy import deepcopy

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass


class wavefunction_singlet():
    def __init__(self, config, pairings_list):
        self.config = config
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self.config, gap) for gap in pairings_list]
        # self.var_params = self._init_var_params()
        self.var_params = [1.]  # if the parameter is complex, we need to double the gap (repeat it twice in the list, but one of times with the i (real = False))
        self.U_matrix = self._construct_U_matrix()
        while True:
            self.occupied_sites, self.empty_sites = self._generate_configuration()
            self.U_tilde_matrix = self._construct_U_tilde_matrix()
            if np.linalg.matrix_rank(self.U_tilde_matrix) == self.config.N_electrons:
                break
        self.W_GF = self._construct_W_GF()
        self.current_det = np.linalg.det(self.U_tilde_matrix) ** 2

        self.W_k_derivatives = [self._get_W_k_derivative(gap) for gap in self.pairings_list_unwrapped]

        return

    def get_O_pairing(self, pairing_index):
        W_k = self.W_k_derivatives[pairing_index]
        W_GF_complete = np.zeros((self.W_GF.shape[0], self.W_GF.shape[1])) * 1.0j  # TODO: this can be done ONCE for all gaps
        W_GF_complete[:, self.occupied_sites] = self.W_GF

        return -np.trace(W_k.dot(W_GF_complete))  # (6.98) from S.Sorella book

    def _get_W_k_derivative(self, gap):  # obtaining (6.99) from S. Sorella book
        V = np.zeros((2 * self.K.shape[0], 2 * self.K.shape[1])) * 1.0j  # (6.91) in S. Sorella book
        V[0:self.K.shape[0], self.K.shape[1]:2 * self.K.shape[1]] = gap
        V[self.K.shape[0]:2 * self.K.shape[0], 0:self.K.shape[1]] = gap.conj().T

        Vdash = (self.U_full.conj().T).dot(V).dot(self.U_full)  # (6.94) in S. Sorella book

        Vdash_rescaled = np.zeros(shape = Vdash.shape) * 1.0j
        for alpha in range(Vdash.shape[0]):
            for beta in range(Vdash.shape[1]):
                if self.E[alpha] > self.E_fermi and self.E[beta] <= self.E_fermi:
                    Vdash_rescaled[alpha, beta] = Vdash[alpha, beta] / (self.E[alpha] - self.E[beta])
        return self.U_full.dot(Vdash_rescaled).dot(self.U_full.conj().T)  # (6.99) step

    def get_O(self, base_state):
        '''
            O_i = \\partial{\\psi(x)}/ \\partial(w) / \\psi(x)
        '''

    def _construct_U_matrix(self):
        self.K = self.config.model(self.config.Ls, self.config.mu)
        Delta = pairings.get_total_pairing_upwrapped(self.config, self.pairings_list_unwrapped, self.var_params)

        T = np.zeros((2 * self.K.shape[0], 2 * self.K.shape[1])) * 1.0j
        T[0:self.K.shape[0], 0:self.K.shape[1]] = self.K
        T[self.K.shape[0]:2 * self.K.shape[0], self.K.shape[1]:2 * self.K.shape[1]] = -self.K
        T[0:self.K.shape[0], self.K.shape[1]:2 * self.K.shape[1]] = Delta
        T[self.K.shape[0]:2 * self.K.shape[0], 0:self.K.shape[1]] = Delta.conj().T

        E, U = np.linalg.eigh(T)
        assert(np.allclose(np.diag(E), U.conj().T.dot(T).dot(U)))  # U^{\dag} T U = E
        self.U_full = deepcopy(U)
        self.E = E
        lowest_energy_states = np.argpartition(E, self.config.N_electrons)[:self.config.N_electrons]  # select lowest-energy orbitals

        rest_energies = E[np.setdiff1d(np.arange(len(self.E)), lowest_energy_states)]

        U = U[:, lowest_energy_states]  # select only occupied orbitals
        self.E_fermi = np.max(self.E[lowest_energy_states])

        if rest_energies.min() - self.E_fermi < 1e-14:
            print('open shell configuration, we are fucked!')
            print(np.sort(self.E))
            exit(-1)
        else:
            print('Closed shell configuration, gap =', rest_energies.min() - self.E_fermi)
            print(np.sort(self.E))
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
        # print(self.current_det)
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