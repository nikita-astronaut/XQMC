import numpy as np
import time
from config_vmc import MC_parameters as config
import models_vmc
import pairings
from copy import deepcopy

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass


class wavefunction_singlet(object):
    def __init__(self, config, pairings_list, var_params):
        self.config = config
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self.config, gap) for gap in pairings_list]
        # self.var_params = self._init_var_params()
        self.var_params = var_params  # if the parameter is complex, we need to double the gap (repeat it twice in the list, but one of times with the i (real = False))
        self.U_matrix = self._construct_U_matrix()
        while True:
            self.occupied_sites, self.empty_sites, self.place_in_string = self._generate_configuration()
            self.U_tilde_matrix = self._construct_U_tilde_matrix()
            if np.linalg.matrix_rank(self.U_tilde_matrix) == self.config.N_electrons:
                break
        self.W_GF = self._construct_W_GF()
        self.current_det = np.linalg.det(self.U_tilde_matrix)

        self.W_k_derivatives = [self._get_W_k_derivative(gap) for gap in self.pairings_list_unwrapped]
        self.state_dict = {}
        return

    def get_det_ratio(self, i, j):  # i -- moved site (d_i), j -- empty site (d^{\dag}_j)
        return self.W_GF[j, self.place_in_string[i]]

    def get_O_pairing(self, pairing_index):
        W_k = self.W_k_derivatives[pairing_index]
        W_GF_complete = np.zeros((self.W_GF.shape[0], self.W_GF.shape[0])) * 1.0j  # TODO: this can be done ONCE for all gaps
        W_GF_complete[:, self.occupied_sites] = self.W_GF
        return -np.trace(W_k.dot(W_GF_complete))  # (6.98) from S.Sorella book

    def _get_W_k_derivative(self, gap):  # obtaining (6.99) from S. Sorella book
        V = np.zeros((2 * self.K.shape[0], 2 * self.K.shape[1])) * 1.0j  # (6.91) in S. Sorella book
        V[:self.K.shape[0], self.K.shape[1]:] = gap
        V[self.K.shape[0]:, :self.K.shape[1]] = gap.conj().T
        Vdash = (self.U_full.conj().T).dot(V).dot(self.U_full)  # (6.94) in S. Sorella book
        Vdash_rescaled = np.zeros(shape = Vdash.shape) * 1.0j  # (6.94) from S. Sorella book
        for alpha in range(Vdash.shape[0]):
            for beta in range(Vdash.shape[1]):
                if self.E[alpha] > self.E_fermi and self.E[beta] <= self.E_fermi:
                    Vdash_rescaled[alpha, beta] = Vdash[alpha, beta] / (self.E[alpha] - self.E[beta])
        return self.U_full.dot(Vdash_rescaled).dot(self.U_full.conj().T)  # (6.99) step

    def get_O(self):  # derivative over all variational parameters
        '''
            O_i = \\partial{\\psi(x)}/ \\partial(w) / \\psi(x)
        '''

        if tuple(self.state) in self.state_dict:
            print('hit!')
            return self.state_dict[tuple(self.state)]

        ### pairings part ###
        O_pairing = [self.get_O_pairing(pairing_index) for pairing_index in range(len(self.pairings_list_unwrapped))]
        O = O_pairing + []
        self.state_dict[tuple(self.state)] = np.array(O)

        return np.array(O)

    def _construct_U_matrix(self):
        self.K = self.config.model(self.config.Ls, self.config.mu)
        # print(np.sum(self.K))
        Delta = pairings.get_total_pairing_upwrapped(self.config, self.pairings_list_unwrapped, self.var_params)
        T = np.zeros((2 * self.K.shape[0], 2 * self.K.shape[1])) * 1.0j
        T[:self.K.shape[0], :self.K.shape[1]] = self.K
        T[self.K.shape[0]:, self.K.shape[1]:] = -self.K
        T[:self.K.shape[0], self.K.shape[1]:] = Delta
        T[self.K.shape[0]:, :self.K.shape[1]] = Delta.conj().T
        # print(T)
        E, U = np.linalg.eigh(T)
        assert(np.allclose(np.diag(E), U.conj().T.dot(T).dot(U)))  # U^{\dag} T U = E
        self.U_full = deepcopy(U)
        self.E = E
        lowest_energy_states = np.argpartition(E, self.config.N_electrons)[:self.config.N_electrons]  # select lowest-energy orbitals
        print('energy of filled states =', np.sum(self.E[lowest_energy_states]))
        rest_energies = E[np.setdiff1d(np.arange(len(self.E)), lowest_energy_states)]

        U = U[:, lowest_energy_states]  # select only occupied orbitals
        self.E_fermi = np.max(self.E[lowest_energy_states])

        if rest_energies.min() - self.E_fermi < 1e-14:
            print('open shell configuration, consider different pairing or filling!')
            # Es, counts = np.unique(np.around(self.E * 1e+5), return_counts = True)
            # for E, c in zip(Es, counts):
            #     print(E, c)
            # print(np.sort(self.E), np.unique(np.around(self.E * 1e+5), return_counts = True))
            exit(-1)
        else:
            print('Closed shell configuration, gap =', rest_energies.min() - self.E_fermi)
            # Es, counts = np.unique(np.around(self.E * 1e+5), return_counts = True)
            # for E, c in zip(Es, counts):
            #     print(E, c)
            # print(np.sort(self.E), np.unique(np.around(self.E * 1e+5), return_counts = True))
        return U 

    def _construct_U_tilde_matrix(self):
        U_tilde = self.U_matrix[self.occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        return self.U_matrix.dot(U_tilde_inv)

    def _generate_configuration(self):
        occupied_sites = np.random.choice(np.arange(self.config.total_dof), size = self.config.N_electrons, replace = False)  ## FIXME: DEBUG!!!
        # occupied_sites = np.arange(self.config.N_electrons) * 2 + 1
        place_in_string = (np.zeros(self.config.total_dof) - 1).astype(np.int64)
        place_in_string[occupied_sites] = np.arange(len(occupied_sites))
        self.state = np.zeros(self.config.total_dof, dtype=np.int64)
        self.state[occupied_sites] = 1

        # occupied_sites = np.arange(0, self.config.N_electrons)   # REMOVE THIS!
        # electrons are placed in occupied_states as they are in the string d_{R_1} d_{R_2} ... |0>
        empty_sites = np.arange(self.config.total_dof)
        empty_sites[occupied_sites] = -1
        empty_sites = empty_sites[empty_sites > -0.5]

        return occupied_sites, empty_sites, place_in_string


    # Jackstrow factors is the vector v_k, v[0] -- on--site, v[1] -- tree neighbors, v[2] -- more distant sites et cetera (point symmetry group)
    # however, if we introduce the symmetric and invariant v, is that correct with respect to \Delta?
    # anyway: supposing one has \exp (-\sum_{ij} n_i n_j) = \exp(-n^T A n)
    # n'_i = n_i + \delta_{i, es} - \delta_{i, ms}
    # -n'_i A_ij n_j + n_i A_ij n_j = -A_{es, j} n_j - n_i A_{i es} + A_{ms, j} n_j + 
    #                                  n_i A_{i, ms} - A_{es, es} - A_{ms ms} + A_{es ms} + A_{ms es} = 
    #                                 2 (-A_{es,j} + A_{ms,j}) n_j - 2 A_{0,0} + 2 A_{es ms}
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
        self.place_in_string[moved_site] = -1
        self.place_in_string[empty_site] = moved_site_idx

        self.state[moved_site] = 0
        self.state[empty_site] = 1
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
