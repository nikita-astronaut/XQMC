import numpy as np
from time import time
from opt_parameters import pairings, waves
import models
from copy import deepcopy
from numba import jit
from numba.errors import NumbaPerformanceWarning
import scipy
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class wavefunction_singlet():
    def __init__(self, config, pairings_list, parameters, \
                 with_previous_state, previous_state, orbitals_in_use = None):
        orbitals_in_use = None
        self.config = config
        self.pairings_list_unwrapped = [models.apply_TBC(self.config, deepcopy(gap), inverse = False) \
                                        for gap in self.config.pairings_list_unwrapped]  
        self.reg_gap_term = models.apply_TBC(self.config, deepcopy(self.config.reg_gap_term), inverse = False) * \
                                             self.config.reg_gap_val

        self.var_mu, self.var_f, self.var_waves, self.var_params_gap, self.var_params_Jastrow = config.unpack_parameters(parameters)
        self.var_f = self.var_f if not config.PN_projection else 0.


        ### mean-field Hamiltonian precomputed elements ###
        self.K_up = models.apply_TBC(self.config, deepcopy(self.config.K_0), inverse = False) + \
                        np.eye(self.config.total_dof // 2) * (self.config.mu - self.var_mu)
        self.K_down = models.apply_TBC(self.config, deepcopy(self.config.K_0), inverse = True).T + \
                          np.eye(self.config.total_dof // 2) * (self.config.mu - self.var_mu)


        self.Jastrow_A = np.array([j[0] for j in config.jastrows_list])
        self.Jastrow = np.sum(np.array([A * factor for factor, A in zip(self.var_params_Jastrow, self.Jastrow_A)]), axis = 0)
        assert np.allclose(self.Jastrow, self.Jastrow.T)

        ### diagonalisation of the MF--Hamiltonian ###
        self.U_matrix = self._construct_U_matrix(orbitals_in_use)
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
                print('the rank of this initialization is {:d}'.format(np.linalg.matrix_rank(self.U_tilde_matrix)))
                print('the determinant is', np.linalg.det(self.U_tilde_matrix))
                self.with_previous_state = False  # if previous state failed, reinitialize from scratch
                print('degenerate: will retry the wave function initialisation', flush = True)

        ### delayed-update machinery ###
        self.W_GF = self._construct_W_GF()  # green function as defined in (5.80)

        self.a_update_list = np.zeros((self.W_GF.shape[0], self.config.n_delayed_updates), dtype=np.complex128)
        self.b_update_list = np.zeros((self.W_GF.shape[1], self.config.n_delayed_updates), dtype=np.complex128)
        self.n_stored_updates = 0
        # for delayed GF updates defined in (5.93 -- 5.97)

        self.current_ampl = self.get_cur_det() * self.get_cur_Jastrow_factor()
        self.current_det = self.get_cur_det()

        ### pre-computed W-matrices for fast derivative computation ###
        self.Z = jit_get_Z_factor(self.E, self.occupied_levels)

        self.W_mu_derivative = self._get_derivative(self._construct_mu_V())


        self.W_k_derivatives = np.array([self._get_derivative(self._construct_gap_V(gap)) for gap in self.pairings_list_unwrapped])
        self.W_waves_derivatives = np.array([self._get_derivative(waves.waves_particle_hole(self.config, wave[0])) for wave in self.config.waves_list])
        ### allowed 1-particle moves ###
        self.adjacency_list = self.config.adjacency_transition_matrix 


        ### random numbers for random moves ###
        self._rnd_size = 20000
        self._refresh_rnd()

        self.accepted = 0
        self.rejected_filled = 0
        self.rejected_factor = 0
        return

    def _refresh_rnd(self):
        self.random_numbers_acceptance = np.random.random(size = self._rnd_size)
        self.random_numbers_move = np.random.randint(0, len(self.occupied_sites), size = self._rnd_size)
        self.random_numbers_direction = np.random.randint(0, len(self.adjacency_list[0]), size = self._rnd_size)
        return

    def get_cur_Jastrow_factor(self):
        return np.exp(-0.5 * np.einsum('i,ij,j', self.occupancy, self.Jastrow, self.occupancy) + \
                      self.var_f * np.sum(self.occupancy))

    def total_density(self):
        return np.sum(self.occupancy) + self.config.total_dof // 2

    def get_cur_det(self):
        return np.linalg.det(self._construct_U_tilde_matrix())

    def get_O_pairing(self, W_k):
        return -np.einsum('ij,ji', W_k, self.W_GF_complete)  # (6.98) from S.Sorella book

    def get_O_Jastrow(self, jastrow_index):
        return -0.5 * np.einsum('i,ij,j', self.occupancy, self.Jastrow_A[jastrow_index], self.occupancy)

    def get_O_fugacity(self):
        return np.sum(self.occupancy)

    def _construct_gap_V(self, gap):
        V = np.zeros((self.config.total_dof, self.config.total_dof)) * 1.0j  # (6.91) in S. Sorella book
        V[:self.config.total_dof // 2, self.config.total_dof // 2:] = gap
        V[self.config.total_dof // 2:, :self.config.total_dof // 2] = gap.conj().T
        return V

    def _construct_mu_V(self):
        V = np.ones(self.config.total_dof) + 0.0j
        V[:self.config.total_dof // 2] = -1.0

        return np.diag(V)


    def _get_derivative(self, V):  # obtaining (6.99) from S. Sorella book
        return jit_get_derivative(self.U_full, V, self.Z)

    def get_O(self):  # derivative over all variational parameters
        '''
            O_i = \\partial{\\psi(x)}/ \\partial(w) / \\psi(x)
        '''

        self.W_GF_complete = np.zeros((self.W_GF.shape[0], self.W_GF.shape[0])) * 1.0j
        self.W_GF_complete[:, self.occupied_sites] = self.W_GF

        O_mu = [self.get_O_pairing(self.W_mu_derivative) if self.config.optimize_mu_BCS else 0.0]
        O_fugacity = [self.get_O_fugacity()] if not self.config.PN_projection else []
        O_pairing = jit_get_O_pairing(self.W_k_derivatives, self.W_GF_complete.T) if len(self.W_k_derivatives) > 0 else []
        O_Jastrow = jit_get_O_jastrow(self.Jastrow_A, self.occupancy * 1.0)
        O_waves = jit_get_O_pairing(self.W_waves_derivatives, self.W_GF_complete.T) if len(self.W_waves_derivatives) > 0 else []

        O = O_mu + O_fugacity + O_waves + O_pairing + O_Jastrow

        return np.array(O)

    def _construct_U_matrix(self, orbitals_in_use):
        T = construct_HMF(self.config, self.K_up, self.K_down, \
                          self.pairings_list_unwrapped, self.var_params_gap, self.var_waves, self.reg_gap_term)
        assert np.allclose(T, T.conj().T)
        E, U = np.linalg.eigh(T)

        assert(np.allclose(np.diag(E), U.conj().T.dot(T).dot(U)))  # U^{\dag} T U = E
        self.U_full = deepcopy(U).astype(np.complex128)
        self.E = E

        #if orbitals_in_use is not None:
        #    overlap_matrix = np.abs(np.einsum('ij,ik->jk', U.conj(), orbitals_in_use))
        #    self.lowest_energy_states = np.argmax(overlap_matrix, axis = 0)
        #    print(self.lowest_energy_states)
        #    print(overlap_matrix.max(axis = 0))
        #else:
        
        if self.config.enforce_valley_orbitals:
            plus_valley = np.einsum('ij,ij->j', self.U_full[np.arange(0, self.config.total_dof, 2), ...], self.U_full[np.arange(0, self.config.total_dof, 2), ...].conj())
            plus_valley = plus_valley > 0.99
            assert np.sum(plus_valley) == self.config.total_dof // 2
            assert np.sum(~plus_valley) == self.config.total_dof // 2

            # U_plus_valley = self.U_full[..., plus_valley]; U_minus_valley = self.U_full[..., ~plus_valley];
            # E_plus_valley = self.E[plus_valley]; E_minus_valley = self.E[~plus_valley];

            plus_valley_number = self.config.total_dof // 4 + self.config.valley_imbalance // 2
            minus_valley_number = self.config.total_dof // 4 - self.config.valley_imbalance // 2

            idxs_total = np.argsort(E)
            idxs_plus = idxs_total[plus_valley][:plus_valley_number]
            idxs_minus = idxs_total[~plus_valley][:minus_valley_number]

            U = np.concatenate([self.U_full[..., idxs_plus], self.U_full[..., idxs_minus]], axis = 1)
            self.lowest_energy_states = np.concatenate([idxs_plus, idxs_minus], axis = 0)
        else:
            self.lowest_energy_states = np.argsort(E)[:self.config.total_dof // 2]  # select lowest-energy orbitals
            U = U[:, self.lowest_energy_states]  # select only occupied orbitals

        rest_states = np.setdiff1d(np.arange(len(self.E)), self.lowest_energy_states)
        self.gap = -np.max(E[self.lowest_energy_states]) + np.min(E[rest_states])
        self.E_fermi = np.max(self.E[self.lowest_energy_states])

        self.occupied_levels = np.zeros(len(E), dtype=bool)
        self.occupied_levels[self.lowest_energy_states] = True

        if E[rest_states].min() - self.E_fermi < 1e-14:
            print('open shell configuration, consider different pairing or filling!', flush = True)
            print(self.config.enforce_valley_orbitals, E[rest_states].min(), self.E_fermi)
        return U 

    def _construct_U_tilde_matrix(self):
        U_tilde = self.U_matrix[self.occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        return self.U_matrix.dot(U_tilde_inv)

    def _generate_configuration(self):
        doping = (self.config.total_dof // 2 - self.config.Ne) // 2  # k
        n_particles = self.config.total_dof // 4 - doping
        n_holes = self.config.total_dof // 4 + doping


        if self.config.n_orbitals == 2 and self.config.valley_projection:
            n_particles_plus = n_particles // 2 + self.config.valley_imbalance // 2
            n_particles_minus = n_particles // 2 - self.config.valley_imbalance // 2

            n_holes_plus = n_holes // 2 - self.config.valley_imbalance // 2
            n_holes_minus = n_holes // 2 + self.config.valley_imbalance // 2
            # print('initialisation particles_+ = {:d}, holes_+ = {:d}, particles_- = {:d}, holes_- = {:d}'.format(n_particles_plus, n_holes_plus, n_particles_minus, n_holes_minus))

            particles_plus = np.random.choice(np.arange(0, self.config.total_dof // 2, 2),
                                                        size = n_particles_plus, replace = False)
            particles_minus = np.random.choice(np.arange(0, self.config.total_dof // 2, 2) + 1,
                                                        size = n_particles_minus, replace = False)

            holes_plus = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof, 2),
                                                        size = n_holes_plus, replace = False)
            holes_minus = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof, 2) + 1,
                                                        size = n_holes_minus, replace = False)
            occupied_sites = np.concatenate([particles_plus, particles_minus, holes_plus, holes_minus])        
        else:
            occupied_sites_particles = np.random.choice(np.arange(self.config.total_dof // 2), 
                                                        size = n_particles, replace = False)
            occupied_sites_holes = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof), 
                                                    size = n_holes, replace = False)
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
        rnd_index = self.MC_step_index % self._rnd_size
        #if ind_index == 0:
        #    self._refresh_rnd()        

        if proposed_move == None:
            moved_site_idx = self.random_numbers_move[rnd_index]
            moved_site = self.occupied_sites[moved_site_idx]
            empty_site = self.adjacency_list[moved_site][self.random_numbers_direction[rnd_index]]
        else:  # only in testmode
            moved_site, empty_site = proposed_move
            moved_site_idx = self.place_in_string[moved_site]
            if empty_site not in self.empty_sites or moved_site not in self.occupied_sites:
                return False, 1, moved_site, empty_site

        if empty_site not in self.empty_sites:
            self.rejected_filled += 1
            return False, 1, 1, moved_site, empty_site

        # t = time()
        det_ratio = self.W_GF[empty_site, moved_site_idx] + np.dot(self.a_update_list[empty_site, :self.n_stored_updates],
                                                                   self.b_update_list[moved_site_idx, :self.n_stored_updates])
        Jastrow_ratio = get_Jastrow_ratio(self.Jastrow, self.occupancy, self.state, \
                                          self.var_f, moved_site, empty_site)

        # self.wf += time() - t
        if not enforce and np.abs(det_ratio) ** 2 * (Jastrow_ratio ** 2) < self.random_numbers_acceptance[self.MC_step_index % self._rnd_size]:
            self.rejected_factor += 1
            return False, 1, 1, moved_site, empty_site

        self.accepted += 1
        # t = time()
        self.current_ampl *= det_ratio * Jastrow_ratio
        self.current_det *= det_ratio
        # print(self.current_det, self.current_ampl, self.get_cur_Jastrow_factor())
        self.occupied_sites[moved_site_idx] = empty_site


        self.empty_sites.remove(empty_site)
        self.empty_sites.add(moved_site)

        self.place_in_string[moved_site] = -1
        self.place_in_string[empty_site] = moved_site_idx

        self.state[moved_site] = 0
        self.state[empty_site] = 1
        self.occupancy = self.state[:len(self.state) // 2] - self.state[len(self.state) // 2:]
        # print('valley polarisation', np.sum(self.occupancy[np.arange(0, len(self.occupancy), 2)] - self.occupancy[np.arange(0, len(self.occupancy), 2) + 1]))
        # print('sublattice polarisation', np.sum(self.occupancy[np.arange(0, len(self.occupancy), 4)] + self.occupancy[np.arange(0, len(self.occupancy), 4) + 1] - \
        #                                         self.occupancy[np.arange(0, len(self.occupancy), 4) + 2] + self.occupancy[np.arange(0, len(self.occupancy), 4) + 3] ))
        '''
        det_after = np.linalg.det(self.U_matrix[self.occupied_sites, :])
        jastrow = self.get_cur_Jastrow_factor()
        print((self.current_det - det_after) / det_after, 'ratio check')
        print((self.current_ampl / self.current_det - jastrow) / jastrow, 'jastrow check')
        if np.abs((self.current_ampl / self.current_det - jastrow) / jastrow) > 1e-7:
            print((self.current_ampl / self.current_det - jastrow) / jastrow)
            exit(-1)
        '''
        a_new, b_new = _jit_delayed_update(self.a_update_list, self.b_update_list, self.n_stored_updates, \
                                           self.W_GF, empty_site, moved_site_idx)
        self.a_update_list[..., self.n_stored_updates] = a_new
        self.b_update_list[..., self.n_stored_updates] = b_new
        self.n_stored_updates += 1

        if self.n_stored_updates == self.config.n_delayed_updates:
            self.perform_explicit_GF_update()

        # self.update += time() - t 
        return True, det_ratio, Jastrow_ratio, moved_site, empty_site


    def perform_explicit_GF_update(self):
        if self.n_stored_updates == 0:
            return
        self.W_GF += self.a_update_list[..., :self.n_stored_updates].dot(self.b_update_list[..., :self.n_stored_updates].T)  # (5.97)

        self.a_update_list *= 0.0j
        self.b_update_list *= 0.0j
        self.n_stored_updates = 0
        return

    def get_state(self):
        return self.occupied_sites, self.empty_sites, self.place_in_string

# had to move it outside of the class to speed-up with numba (jitclass is hard!)
@jit(nopython=True)
def get_Jastrow_ratio(Jastrow, occupancy, state, total_fugacity, moved_site, empty_site):
    if moved_site == empty_site:
        return 1.0

    delta_alpha = -1 if moved_site < len(state) // 2 else +1
    delta_beta = +1 if empty_site < len(state) // 2 else -1
    alpha, beta = moved_site % (len(state) // 2), empty_site % (len(state) // 2)

    fugacity_factor = np.exp(total_fugacity * (delta_alpha + delta_beta))

    return np.exp(-np.sum((delta_alpha * Jastrow[alpha, :] + delta_beta * Jastrow[beta, :]) * occupancy) - \
                   0.5 * ((delta_alpha ** 2 * Jastrow[alpha, alpha] + delta_beta ** 2 * Jastrow[beta, beta]) + \
                          delta_alpha * delta_beta * (Jastrow[alpha, beta] + Jastrow[beta, alpha]))) * \
           fugacity_factor

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
                 total_fugacity, moved_site, empty_site):  # i -- moved site (d_i), j -- empty site (d^{\dag}_j)
    Jastrow_ratio = get_Jastrow_ratio(Jastrow, occupancy, state, total_fugacity, moved_site, empty_site)
    det_ratio = get_det_ratio(Jastrow, W_GF, place_in_string, state, occupancy, moved_site, empty_site)
    return det_ratio * Jastrow_ratio

@jit(nopython=True)
def density(place_in_string, index):
    return 1.0 if place_in_string[index] > -1 else 0.0

@jit(nopython=True)
def get_wf_ratio_double_exchange(Jastrow, W_GF, place_in_string, state, occupancy, total_fugacity, i, j, k, l):
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
        return 0.0 + 0.0j if n_i == 0 else get_wf_ratio(*state_packed, total_fugacity, k, l)

    if j == k:
        n_j = density(place_in_string, j)
        return 0.0 + 0.0j if n_j == 1 else get_wf_ratio(*state_packed, total_fugacity, i, l)

    if l == i and l != k:
        n_l = density(place_in_string, l)
        delta_jk = 1.0 if j == k else 0.0
        return 0.0 + 0.0j if n_l == 0 else delta_jk - get_wf_ratio(*state_packed, total_fugacity, k, j)

    if l == k and l != i:
        n_l = density(place_in_string, l)
        return 0.0 + 0.0j if n_l == 0 else get_wf_ratio(*state_packed, total_fugacity, i, j)

    if l == k and l == i:
        n_i = density(place_in_string, i)
        return 0.0 + 0.0j if n_i == 1 else get_wf_ratio(*state_packed, total_fugacity, i, j)

    ## bus if everything is non-equal... ##
    delta_jk = 1.0 if j == k else 0.0
    ratio_det = get_det_ratio(*state_packed, i, j) * get_det_ratio(*state_packed, k, l) - \
                get_det_ratio(*state_packed, i, l) * get_det_ratio(*state_packed, k, j)
    jastrow = 0.0
    if True:#np.abs(ratio_det) > 1e-10:
        jastrow = get_Jastrow_ratio(Jastrow, occupancy, state, total_fugacity, i, j)
        delta_i = 1 if i < L else -1
        delta_j = 1 if j < L else -1
        occupancy[i % L] -= delta_i
        occupancy[j % L] += delta_j

        jastrow *= get_Jastrow_ratio(Jastrow, occupancy, state, total_fugacity, k, l)
        occupancy[i % L] += delta_i
        occupancy[j % L] -= delta_j
    return jastrow * ratio_det


@jit(nopython=True)
def jit_get_derivative(U_full, V, Z):  # obtaining (6.99) from S. Sorella book
    Vdash = (U_full.conj().T).dot(V).dot(U_full)  # (6.94) in S. Sorella book
    # (6.94) from S. Sorella book
    return U_full.dot(Vdash * Z).dot(U_full.conj().T)  # (6.99) step

@jit(nopython=True)
def jit_get_Z_factor(E, occupation):
    Z = np.zeros((len(occupation), len(occupation))) + 0.0j
    for alpha in range(E.shape[0]):
        for beta in range(E.shape[0]):
            if not occupation[alpha] and occupation[beta]:
                Z[alpha, beta] = 1. / (E[alpha] - E[beta])
    return Z

@jit(nopython=True)
def jit_get_O_pairing(W_k_derivatives, W_GF_complete):
    derivatives = []
    for k in range(len(W_k_derivatives)):
        # derivatives.append(-1.0 * np.sum(W_k_derivatives[k] * W_GF_complete.T))
        
        der = 0.0 + 0.0j
        w = W_k_derivatives[k] 
        for i in range(W_GF_complete.shape[1]):
            der -= np.sum(w[i] * W_GF_complete[i])
        derivatives.append(der)
        
    return derivatives

@jit(nopython=True)
def jit_get_O_jastrow(Jastrow_A, occupancy):
    derivatives = []
    for k in range(len(Jastrow_A)):
        derivatives.append(-0.5 * occupancy.dot(Jastrow_A[k].dot(occupancy)))
    return derivatives

def construct_HMF(config, K_up, K_down, pairings_list_unwrapped, var_params_gap,
                  var_waves, reg_gap_term):
    Delta = pairings.get_total_pairing_upwrapped(config, pairings_list_unwrapped, var_params_gap)
    T = scipy.linalg.block_diag(K_up, -K_down) + 0.0j

    ## various local pairing terms ##
    T[:config.total_dof // 2, config.total_dof // 2:] = Delta
    T[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T

    ## regularisation ##
    T[:config.total_dof // 2, config.total_dof // 2:] += reg_gap_term
    T[config.total_dof // 2:, :config.total_dof // 2] += reg_gap_term.conj().T

    for wave, coeff in zip(config.waves_list, var_waves):
        T += waves.waves_particle_hole(config, wave[0]) * coeff
    return T

@jit(nopython = True)
def _jit_delayed_update(a_update_list, b_update_list, n_stored_updates, \
                        W_GF, empty_site, moved_site_idx):
    a_new = 1. * W_GF[:, moved_site_idx]
    if n_stored_updates > 0: # (5.94)
        a_new += a_update_list[..., :n_stored_updates].dot(b_update_list[moved_site_idx, :n_stored_updates])

    delta = np.zeros(W_GF.shape[1])
    delta[moved_site_idx] = 1

    b_new = 1. * W_GF[empty_site, :]
    W_Kl = W_GF[empty_site, moved_site_idx]
    if n_stored_updates > 0:  # (5.95)
        b_new += b_update_list[..., :n_stored_updates].dot(a_update_list[empty_site, :n_stored_updates])
        W_Kl += np.dot(a_update_list[empty_site, :n_stored_updates],
                       b_update_list[moved_site_idx, :n_stored_updates]) # (5.95)
    b_new = -(b_new - delta) / W_Kl  # (5.95)

    return a_new, b_new
