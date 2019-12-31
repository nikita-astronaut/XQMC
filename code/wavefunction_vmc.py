import numpy as np
from time import time
import pairings
import models
from copy import deepcopy
from numba import jit

class wavefunction_singlet():
    def __init__(self, config, pairings_list, var_mu, var_SDW, var_CDW, 
                 var_params_gap, var_params_Jastrow, \
                 with_previous_state, previous_state):
        self.config = config
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self.config, gap) for gap in pairings_list]
        self.var_params_gap = var_params_gap  # if the parameter is complex, we need to double the gap (repeat it twice in the list, but one of times with the i (real = False))
        self.var_params_Jastrow = var_params_Jastrow
        self.var_mu = var_mu
        self.var_SDW = var_SDW
        self.var_CDW = var_CDW
        self.checkerboard = np.kron(np.ones((config.Ls ** 2 // 2, config.Ls ** 2 // 2)), np.array([[1, -1], [-1, 1]]))

        self.Jastrow_A = config.adjacency_list
        self.Jastrow = np.sum(np.array([A * factor for factor, A in zip(self.var_params_Jastrow, self.Jastrow_A)]), axis = 0)

        self.U_matrix = self._construct_U_matrix()
        self.with_previous_state = with_previous_state

        self.MC_step_index = 0
        self.update = 0.
        self.wf = 0.

        while True:
            if self.with_previous_state:
                self.occupied_sites, self.empty_sites, self.place_in_string = previous_state
                self.state = np.zeros(self.config.total_dof, dtype=np.int64)
                self.state[self.occupied_sites] = 1
                self.occupancy = self.state[:len(self.state) // 2] - self.state[len(self.state) // 2:]
            else:
                self.occupied_sites, self.empty_sites, self.place_in_string = self._generate_configuration()
            self.U_tilde_matrix = self._construct_U_tilde_matrix()
            if np.linalg.matrix_rank(self.U_tilde_matrix) == self.config.total_dof // 2:
                break
            else:
                self.with_previous_state = False  # if previous state failed, reinitialize from scratch
                print('degenerate')

        self.W_GF = self._construct_W_GF()  # green function as defined in (5.80)

        self.a_update_list = []
        self.b_update_list = []  # for delayed GF updates defined in (5.93 -- 5.97)

        self.current_ampl = self.get_cur_det() * self.get_cur_Jastrow_factor()
        self.current_det = self.get_cur_det()
        self.W_mu_derivative = self._get_derivative(self._construct_mu_V())
        self.W_k_derivatives = [self._get_derivative(self._construct_gap_V(gap)) for gap in self.pairings_list_unwrapped]
        self.W_waves_derivatives = [self._get_derivative(self._construct_wave_V((dof // self.config.n_sublattices) % self.config.n_orbitals, 
                                    dof % self.config.n_sublattices, 'SDW')) \
                                    for dof in range(self.config.n_orbitals * self.config.n_sublattices)] + \
                                   [self._get_derivative(self._construct_wave_V((dof // self.config.n_sublattices) % self.config.n_orbitals, 
                                    dof % self.config.n_sublattices, 'CDW')) \
                                    for dof in range(self.config.n_orbitals * self.config.n_sublattices)]
        self._state_dict = {}

        self.random_numbers_acceptance = np.random.random(size = int(1e+6))
        self.random_numbers_move = np.random.randint(0, len(self.occupied_sites), size = int(1e+6))
        self.random_numbers_direction = np.random.randint(0, len(self.adjacency_list[0]), size = int(1e+6))
        return

    def get_Jastrow_ratio(self, alpha, beta, delta_alpha, delta_beta):
        return np.exp(-0.5 * delta_alpha * np.sum((self.Jastrow[alpha, :] + self.Jastrow[:, alpha]) * self.occupancy) - 
                       0.5 * delta_beta * np.sum((self.Jastrow[beta, :] + self.Jastrow[:, beta]) * self.occupancy) - 
                       0.5 * (delta_alpha ** 2 * self.Jastrow[alpha, alpha] + delta_beta ** 2 * self.Jastrow[beta, beta] + 
                             delta_alpha * delta_beta * (self.Jastrow[alpha, beta] + self.Jastrow[beta, alpha])))

    def get_cur_Jastrow_factor(self):
        return np.exp(-0.5 * np.einsum('i,ij,j', self.occupancy, self.Jastrow, self.occupancy))

    def get_cur_det(self):
        return np.linalg.det(self._construct_U_tilde_matrix())

    def get_O_pairing(self, W_k):
        return -np.einsum('ij,ji', W_k, self.W_GF_complete)  # (6.98) from S.Sorella book

    def get_O_Jastrow(self, jastrow_index):
        return -0.5 * np.einsum('i,ij,j', self.occupancy, self.Jastrow_A[jastrow_index], self.occupancy)

    def _construct_gap_V(self, gap):
        V = np.zeros((2 * self.K.shape[0], 2 * self.K.shape[1])) * 1.0j  # (6.91) in S. Sorella book
        V[:self.K.shape[0], self.K.shape[1]:] = gap
        V[self.K.shape[0]:, :self.K.shape[1]] = gap.conj().T
        return V

    def _construct_mu_V(self):
        V = -np.diag([1.0] * self.K.shape[0] + [-1.0] * self.K.shape[0]) + 0.0j
        return V

    def _construct_wave_V(self, orbital, sublattice, wave_type):
        sublattice_matrix = np.zeros((self.config.n_sublattices, self.config.n_sublattices))
        sublattice_matrix[sublattice, sublattice] = 1.

        orbital_matrix = np.zeros((self.config.n_orbitals, self.config.n_orbitals))
        orbital_matrix[orbital, orbital] = 1.            

        dof_matrix = np.kron(np.kron(self.checkerboard, sublattice_matrix), orbital_matrix)

        if wave_type == 'SDW':
            return np.kron(np.eye(2), dof_matrix)
        return np.kron(np.diag([1, -1]), dof_matrix)

    def _get_derivative(self, V):  # obtaining (6.99) from S. Sorella book
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

        self.W_GF_complete = np.zeros((self.W_GF.shape[0], self.W_GF.shape[0])) * 1.0j  # TODO: this can be done ONCE for all gaps
        self.W_GF_complete[:, self.occupied_sites] = self.W_GF

        O_mu = [self.get_O_pairing(self.W_mu_derivative)]
        O_pairing = [self.get_O_pairing(self.W_k_derivatives[pairing_index]) for pairing_index in range(len(self.pairings_list_unwrapped))]
        O_Jastrow = [self.get_O_Jastrow(jastrow_index) for jastrow_index in range(len(self.var_params_Jastrow))]
        O_waves = [self.get_O_pairing(W_wave_derivative) for W_wave_derivative in self.W_waves_derivatives]

        O = O_mu + O_waves + O_pairing + O_Jastrow

        return np.array(O)

    def _construct_U_matrix(self):
        self.K = self.config.model(self.config, self.var_mu)
        self.adjacency_matrix = np.abs(np.asarray(self.config.model(self.config, 0.0))) > 1e-6
        self.big_adjacency_matrix = np.zeros((2 * self.adjacency_matrix.shape[0], 2 * self.adjacency_matrix.shape[1]))
        self.big_adjacency_matrix[:self.adjacency_matrix.shape[0], :self.adjacency_matrix.shape[1]] = self.adjacency_matrix
        self.big_adjacency_matrix[self.adjacency_matrix.shape[0]:, self.adjacency_matrix.shape[1]:] = self.adjacency_matrix

        self.adjacency_list = [np.where(self.big_adjacency_matrix[:, i] > 0)[0] for i in range(self.big_adjacency_matrix.shape[1])]

        Delta = pairings.get_total_pairing_upwrapped(self.config, self.pairings_list_unwrapped, self.var_params_gap)

        ## CONTRUCTION OF H_MF (mean field, denoted as T) ##
        ## standard kinetic term (\mu included) ##
        T = np.kron(np.diag([1, -1]), self.K) + 0.0j

        ## various local pairing terms ##
        T[:self.K.shape[0], self.K.shape[1]:] = Delta
        T[self.K.shape[0]:, :self.K.shape[1]] = Delta.conj().T
        
        ## SDW/CDW is the same for every orbital and sublattice ##

        for dof in range(self.config.n_orbitals * self.config.n_sublattices):
            sublattice = dof % self.config.n_sublattices
            orbital = (dof // self.config.n_sublattices) % self.config.n_orbitals

            T += self._construct_wave_V(orbital, sublattice, 'SDW') * self.var_SDW[dof]
            T += self._construct_wave_V(orbital, sublattice, 'CDW') * self.var_CDW[dof]
            #  delta_cdw_i \sum_xy (-1)^{x + y} [n_up_i(x, y) + n_down_i(x, y)] = \sum_xy (-1)^{x + y} [n_i(x, y) - n_i(x + L, y + L)]
            #  delta_sdw_i \sum_xy (-1)^{x + y} [n_up_i(x, y) - n_down_i(x, y)] = \sum_xy (-1)^{x + y} [n_i(x, y) + n_i(x + L, y + L)]



        E, U = np.linalg.eigh(T)

        assert(np.allclose(np.diag(E), U.conj().T.dot(T).dot(U)))  # U^{\dag} T U = E
        self.U_full = deepcopy(U)
        self.E = E

        lowest_energy_states = np.argsort(E)[:self.config.total_dof // 2]  # select lowest-energy orbitals=
        rest_states = np.setdiff1d(np.arange(len(self.E)), lowest_energy_states)


        U = U[:, lowest_energy_states]  # select only occupied orbitals
        self.n_particles = int(np.rint((np.sum(U[:U.shape[0] // 2] * np.conj(U[:U.shape[0] // 2]))).real))
        self.n_holes = int(np.rint(np.sum(U[U.shape[0] // 2:] * np.conj(U[U.shape[0] // 2:])).real))
        self.E_fermi = np.max(self.E[lowest_energy_states])

        if E[rest_states].min() - self.E_fermi < 1e-14:
            print('open shell configuration, consider different pairing or filling!')
        return U 

    def _construct_U_tilde_matrix(self):
        U_tilde = self.U_matrix[self.occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        return self.U_matrix.dot(U_tilde_inv)

    def _generate_configuration(self):
        doping = (self.config.total_dof // 2 - self.config.N_electrons) // 2  # k
        occupied_sites_particles = np.random.choice(np.arange(self.config.total_dof // 2), 
                                                    size = self.config.total_dof // 4 - doping, replace = False)
        occupied_sites_holes = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof), 
                                                size = self.config.total_dof // 4 + doping, replace = False)
        occupied_sites = np.concatenate([occupied_sites_particles, occupied_sites_holes])

        place_in_string = (np.zeros(self.config.total_dof) - 1).astype(np.int64)
        place_in_string[occupied_sites] = np.arange(len(occupied_sites))
        self.state = np.zeros(self.config.total_dof, dtype=np.int64)
        self.state[occupied_sites] = 1
        self.occupancy = self.state[:len(self.state) // 2] - self.state[len(self.state) // 2:]
        # electrons are placed in occupied_states as they are in the string d_{R_1} d_{R_2} ... |0>
        empty_sites = np.arange(self.config.total_dof)
        empty_sites[occupied_sites] = -1
        empty_sites = set(empty_sites[empty_sites > -0.5])

        return occupied_sites, empty_sites, place_in_string

    def perform_MC_step(self, proposed_move = None, enforce = False):
        self.MC_step_index += 1
        conserving_move = False
        n_attempts = 0

        if proposed_move == None:
            moved_site_idx = self.random_numbers_move[self.MC_step_index]
            moved_site = self.occupied_sites[moved_site_idx]
            # empty_site = random.sample(self.empty_sites,. 1)
            empty_site = self.adjacency_list[moved_site][self.random_numbers_direction[self.MC_step_index]]
        else:  # only in testmode
            moved_site, empty_site = proposed_move
            moved_site_idx = self.place_in_string[moved_site]
            if empty_site not in self.empty_sites or moved_site not in self.occupied_sites:
                return False, 1, moved_site, empty_site

        if empty_site not in self.empty_sites:
            return False, 1, 1, moved_site, empty_site

        t = time()
        det_ratio = self.W_GF[empty_site, moved_site_idx] + \
                    np.sum([a[empty_site, 0] * b[moved_site_idx, 0] for a, b in zip(self.a_update_list, self.b_update_list)])

        Jastrow_ratio = get_Jastrow_ratio(self.Jastrow, self.occupancy, self.state, moved_site, empty_site)

        self.wf += time() - t
        if not enforce and np.abs(det_ratio) ** 2 * (Jastrow_ratio ** 2) < self.random_numbers_acceptance[self.MC_step_index]:
            return False, 1, 1, moved_site, empty_site

        t = time()
        self.current_ampl *= det_ratio * Jastrow_ratio
        self.current_det *= det_ratio
        self.occupied_sites[moved_site_idx] = empty_site

        ## debug
        #det_after = np.linalg.det(self.U_matrix[self.occupied_sites, :])
        # print((self.current_det - det_after) / det_after, 'ratio check')

        self.empty_sites.remove(empty_site)
        self.empty_sites.add(moved_site)

        self.place_in_string[moved_site] = -1
        self.place_in_string[empty_site] = moved_site_idx

        self.state[moved_site] = 0
        self.state[empty_site] = 1
        self.occupancy = self.state[:len(self.state) // 2] - self.state[len(self.state) // 2:]

        a_new = 1. * self.W_GF[:, moved_site_idx]
        if len(self.a_update_list) > 0:
            a_new += np.sum(np.array([a[:, 0] * b[moved_site_idx, 0] \
                            for a, b in zip(self.a_update_list, self.b_update_list)]), axis = 0)  # (5.94)

        delta = np.zeros(self.W_GF.shape[1])
        delta[moved_site_idx] = 1

        b_new = 1. * self.W_GF[empty_site, :]
        W_Kl = self.W_GF[empty_site, moved_site_idx]
        if len(self.a_update_list) > 0:
            b_new += np.sum(np.array([a[empty_site, 0] * b[:, 0] \
                                      for a, b in zip(self.a_update_list, self.b_update_list)]), axis = 0)  # (5.95)
            W_Kl += np.sum(np.array([a[empty_site, 0] * b[moved_site_idx, 0] \
                                     for a, b in zip(self.a_update_list, self.b_update_list)]))  # (5.95)
        b_new = -(b_new - delta) / W_Kl  # (5.95)

        self.a_update_list.append(a_new[..., np.newaxis])
        self.b_update_list.append(b_new[..., np.newaxis])

        if len(self.a_update_list) == self.config.n_delayed_updates:
            self.perform_explicit_GF_update()

        self.update += time() - t 
        return True, det_ratio, Jastrow_ratio, moved_site, empty_site

    def perform_explicit_GF_update(self):
        if len(self.a_update_list) == 0:
            return
        A = np.concatenate(self.a_update_list, axis = 1)
        B = np.concatenate(self.b_update_list, axis = 1)

        self.W_GF += A.dot(B.T)  # (5.97)
        self.a_update_list = []
        self.b_update_list = []

        return

    def get_state(self):
        return self.occupied_sites, self.empty_sites, self.place_in_string

# had to move it outside of the class to speed-up with numba (jitclass is hard!)
@jit(nopython=True)
def get_Jastrow_ratio(Jastrow, occupancy, state, moved_site, empty_site):
    if moved_site == empty_site:
        return 1.0

    delta_alpha = -1 if moved_site < len(state) // 2 else +1
    delta_beta = +1 if empty_site < len(state) // 2 else -1
    alpha, beta = moved_site % (len(state) // 2), empty_site % (len(state) // 2)

    factor = Jastrow[alpha, alpha]
    return np.exp(-np.sum((delta_alpha * Jastrow[alpha, :] + delta_beta * Jastrow[beta, :]) * occupancy) - 
                   0.5 * ((delta_alpha ** 2 + delta_beta ** 2) * factor + 
                          delta_alpha * delta_beta * (Jastrow[alpha, beta] + Jastrow[beta, alpha])))

@jit(nopython=True)
def get_det_ratio(Jastrow, W_GF, place_in_string, state, occupancy, \
                  moved_site, empty_site):  # i -- moved site (d_i), j -- empty site (d^{\dag}_j)
    if moved_site == empty_site:  # if just density correlator <x|d^dag d|Ф> = <Ф|d^dag d |x>^*
        if place_in_string[empty_site] > -1:
            return 1.0 + 0.0j
        return 0.0 + 0.0j

    # if move is impossible, return 0.0
    if place_in_string[moved_site] == -1 or place_in_string[empty_site] > -1:
        return 0.0 + 0.0j

    return W_GF[empty_site, place_in_string[moved_site]]

@jit(nopython=True)
def get_wf_ratio(Jastrow, W_GF, place_in_string, state, occupancy, \
                 moved_site, empty_site):  # i -- moved site (d_i), j -- empty site (d^{\dag}_j)
    Jastrow_ratio = get_Jastrow_ratio(Jastrow, occupancy, state, moved_site, empty_site)
    det_ratio = get_det_ratio(Jastrow, W_GF, place_in_string, state, occupancy, moved_site, empty_site)
    return det_ratio * Jastrow_ratio

@jit(nopython=True)
def density(place_in_string, index):
    return 1.0 if place_in_string[index] > -1 else 0.0

@jit(nopython=True)
def get_wf_ratio_double_exchange(Jastrow, W_GF, place_in_string, state, occupancy, i, j, k, l):
    '''
        this is required for the correlators <\\Delta^{\\dag} \\Delta>
        computes the ratio <x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф> / <x|Ф> = 
        = W(j, I(i)) W(l, I(k)) + (\\delta_jk - W(j, I(k))) W(l, I(i)),
        where I(i) is the position of the occupied site i in the state bitstring
    '''

    L = len(state) // 2
    state_packed = (Jastrow, W_GF, place_in_string, state, occupancy)

    ## have to explicitly work-around degenerate cases ##
    if i == j:
        n_i = density(place_in_string, i)
        return 0.0 + 0.0j if n_i == 0 else get_wf_ratio(*state_packed, k, l)

    if j == k:
        n_j = density(place_in_string, j)
        return 0.0 + 0.0j if n_j == 1 else get_wf_ratio(*state_packed, i, l)

    if l == i and l != k:
        n_l = density(place_in_string, l)
        delta_jk = 1.0 if j == k else 0.0
        return 0.0 + 0.0j if n_l == 0 else delta_jk - get_wf_ratio(*state_packed, k, j)

    if l == k and l != i:
        n_l = density(place_in_string, l)
        return 0.0 + 0.0j if n_l == 0 else get_wf_ratio(*state_packed, i, j)

    if l == k and l == i:
        n_i = density(place_in_string, i)
        return 0.0 + 0.0j if n_i == 1 else get_wf_ratio(*state_packed, i, j)

    ## bus if everything is non-equal... ##
    delta_jk = 1.0 if j == k else 0.0
    ratio_det = get_det_ratio(*state_packed, i, j) * get_det_ratio(*state_packed, k, l) - \
                get_det_ratio(*state_packed, i, l) * get_det_ratio(*state_packed, k, j)
    jastrow = 0.0
    if np.abs(ratio_det) > 1e-10:
        jastrow = get_Jastrow_ratio(Jastrow, occupancy, state, i, j)
        delta_i = 1 if i < L else -1
        delta_j = 1 if j < L else -1
        occupancy[i % L] -= delta_i
        occupancy[j % L] += delta_j

        jastrow *= get_Jastrow_ratio(Jastrow, occupancy, state, k, l)
        occupancy[i % L] += delta_i
        occupancy[j % L] -= delta_j
    return jastrow * ratio_det