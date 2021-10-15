import numpy as np
from time import time
from opt_parameters import pairings, waves
import models
from copy import deepcopy
from numba import jit
#from numba.core.errors import NumbaPerformanceWarning
import scipy
import warnings
import os
from time import sleep

#warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class wavefunction_singlet():
    def __init__(self, config, pairings_list, parameters, \
                 with_previous_state, previous_state, \
                 orbitals_in_use = None, particle_hole = False, \
                 K_up = None, K_down = None, reg = None, ph_test = False, trs_test = False):
        self.particle_hole = particle_hole
        self.ph_test = ph_test
        self.trs_test = trs_test
        orbitals_in_use = None  # FIXME
        self.with_previous_state = with_previous_state
        # previous_state = None; self.with_previous_state = False;  # FIXME
        self.config = config
        #assert np.allclose(config.pairings_list_unwrapped[0], models.apply_TBC(self.config, self.config.twist, deepcopy(config.pairings_list_unwrapped[0]), inverse = False)) #FIXME

        self.pairings_list_unwrapped = [models.apply_TBC(self.config, self.config.twist, deepcopy(gap), inverse = False) \
                                        for gap in self.config.pairings_list_unwrapped]

        ### check TRS of gaps functions ###
        '''
        TRS = np.concatenate([np.array([2 * i + 1, 2 * i]) for i in range(K_up.shape[0] // 2)], axis=0)
        for gap in self.pairings_list_unwrapped:
            gap_TRS = gap[TRS]
            gap_TRS = gap_TRS[..., TRS].conj()
            assert np.allclose(gap, gap_TRS)
            print('passed')
        exit(-1)
        '''

        self.hoppings_list_TBC_up = [models.apply_TBC(self.config, self.config.twist, deepcopy(h), inverse = False) \
                                     for h in self.config.hoppings]
        self.hoppings_list_TBC_down = [models.apply_TBC(self.config, self.config.twist, deepcopy(h).T, inverse = True) \
                                       for h in self.config.hoppings]



        #print(self.pairings_list_unwrapped[0])
        self.reg_gap_term = reg if reg is not None else models.apply_TBC(self.config, self.config.twist, deepcopy(self.config.reg_gap_term), inverse = False) * \
                                                        self.config.reg_gap_val

        self.var_mu, self.var_f, self.var_hoppings, self.var_params_gap, self.var_params_Jastrow = config.unpack_parameters(parameters)

        self.var_f = 0. #self.var_f if not config.PN_projection else 0.

        # print(K_up, K_down)
        ### mean-field Hamiltonian precomputed elements ###
        plus_valley = np.arange(0, self.config.total_dof // 2, 2)
        minus_valley = plus_valley + 1
        plus_valley_mesh = np.zeros(self.config.total_dof // 2); plus_valley_mesh[plus_valley] = 1
        minus_valley_mesh = np.zeros(self.config.total_dof // 2); minus_valley_mesh[minus_valley] = 1
        if K_up is None:
            self.K_up = models.apply_TBC(self.config, self.config.twist, deepcopy(self.config.K_0), inverse = False)
        else:
            self.K_up = K_up.copy() * 1.0
            #print(self.K_up)
        self.K_up -= np.diag(plus_valley_mesh) * self.var_mu[0]
        self.K_up -= np.diag(minus_valley_mesh) * (self.var_mu[0] + 1e-8)

        if K_down is None:
            self.K_down = models.apply_TBC(self.config, self.config.twist, deepcopy(self.config.K_0).T, inverse = True)
        else:
            self.K_down = K_down.copy() * 1.0
            #print(self.K_down)
        #exit(-1)
        self.K_down -= np.diag(plus_valley_mesh) * self.var_mu[0]
        self.K_down -= np.diag(minus_valley_mesh) * (self.var_mu[0] + 1e-8)

        # assert self.var_mu[0] == 0.0

        print('ABCDEFG', np.linalg.norm(self.K_up- self.K_down.conj()))
        assert np.allclose(self.K_up, self.K_down.conj())

        #print(np.linalg.eigh(self.K_up)[0])

        if particle_hole:
            self.K_up = -np.conj(self.K_up)
            self.K_down = -np.conj(self.K_down)

        self.Jastrow_A = np.array([j[0] for j in config.jastrows_list])
        self.Jastrow = np.sum(np.array([A * factor for factor, A in zip(self.var_params_Jastrow, self.Jastrow_A)]), axis = 0)
        assert np.allclose(self.Jastrow, self.Jastrow.T)

        ### diagonalisation of the MF--Hamiltonian ###
        self.U_matrix = self._construct_U_matrix(orbitals_in_use)
        self.MC_step_index = 0

        self.update = 0.
        self.wf = 0.
        self.t_jastrow = 0
        self.t_det = 0
        self.t_choose_site = 0
        self.t_overhead_after = 0
        self.t_gf_update = 0
        self.t_ab = 0

        ### debug
        self.wf_ampls = []

        while True:
            if self.with_previous_state:
                self.occupied_sites, self.empty_sites, self.place_in_string = previous_state
                self.state = np.zeros(self.config.total_dof, dtype=np.int64)
                self.state[self.occupied_sites] = 1
                self.occupancy = self.state[:len(self.state) // 2] - self.state[len(self.state) // 2:]
            else:
                self.occupied_sites, self.empty_sites, self.place_in_string = self._generate_configuration(particle_hole)
                print(len(self.occupied_sites), 'length of occupied states')
                # print('fresh state')
            self.U_tilde_matrix = self._construct_U_tilde_matrix()
            if np.linalg.matrix_rank(self.U_tilde_matrix) == self.config.total_dof // 2:
                # print('the determinant is', np.linalg.det(self.U_tilde_matrix))
                break
            else:
                print('the rank of this initialization is {:d}'.format(np.linalg.matrix_rank(self.U_tilde_matrix)))
                print('dimensions are', self.U_tilde_matrix.shape)
                print('SHAPEEE', self.U_tilde_matrix.shape, flush=True)
                print('the determinant is', np.linalg.det(self.U_tilde_matrix))
                self.with_previous_state = False  # if previous state failed, reinitialize from scratch
                print('degenerate: will retry the wave function initialisation', flush = True)
                print(self.config.twist)
                #sleep(30)
                #exit(-1)

        ### delayed-update machinery ###
        self.W_GF = self._construct_W_GF()  # green function as defined in (5.80)
        # print(self.W_GF.dtype, 'GF dtype')

        self.a_update_list = np.zeros((self.W_GF.shape[0], self.config.n_delayed_updates), dtype=np.complex128)
        self.b_update_list = np.zeros((self.W_GF.shape[1], self.config.n_delayed_updates), dtype=np.complex128)
        self.n_stored_updates = 0
        # for delayed GF updates defined in (5.93 -- 5.97)

        self.current_ampl = self.get_cur_det() * self.get_cur_Jastrow_factor()
        self.current_det = self.get_cur_det()

        ### pre-computed W-matrices for fast derivative computation ###
        self.Z = jit_get_Z_factor(self.E, self.occupied_levels)

        self.W_mu_derivative = self._get_derivative(self._construct_mu_V(np.arange(0, self.config.total_dof // 2)))

        self.W_k_derivatives = np.array([self._get_derivative(self._construct_gap_V(gap)) for gap in self.pairings_list_unwrapped])
        #self.W_waves_derivatives = np.array([self._get_derivative(waves.waves_particle_hole(self.config, wave)) \
        #                                     for wave in self.config.waves_list_unwrapped])
        self.W_hoppings_derivatives = np.array([self._get_derivative(self._construct_hopping_V(h_up, h_down)) \
                                                for h_up, h_down in zip(self.hoppings_list_TBC_up, self.hoppings_list_TBC_down)])


        ### allowed 1-particle moves ###
        self.adjacency_list = self.config.adjacency_transition_matrix 


        ### random numbers for random moves ###
        self._rnd_size = 1000000
        self._refresh_rnd()

        self.accepted = 0
        self.rejected_filled = 0
        self.rejected_factor = 0
        self.ws = []  # FIXME
        return

    def _refresh_rnd(self):
        self.random_numbers_acceptance = np.random.random(size = self._rnd_size)
        self.random_numbers_move = np.random.randint(0, len(self.occupied_sites), size = self._rnd_size)
        #print(len(self.occupied_sites))
        #exit(-1)
        self.random_numbers_direction = np.random.randint(0, int(1e+7), size = self._rnd_size)
        return

    def get_cur_Jastrow_factor(self):
        return np.exp(-0.5 * np.einsum('i,ij,j', self.occupancy, self.Jastrow, self.occupancy) + \
                      self.var_f * np.sum(self.occupancy))

    def total_density(self):
        return np.sum(self.occupancy) + self.config.total_dof // 2

    def total_plus_density(self):
        return np.sum(self.occupancy[np.arange(0, self.config.total_dof // 2, 2)]) + self.config.total_dof // 2 // 2
    def total_minus_density(self):
        return np.sum(self.occupancy[np.arange(1, self.config.total_dof // 2, 2)]) + self.config.total_dof // 2 // 2

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

    def _construct_hopping_V(self, h_up, h_down):
        V = np.zeros((self.config.total_dof, self.config.total_dof)) * 1.0j  # (6.91) in S. Sorella book
        V[:self.config.total_dof // 2, :self.config.total_dof // 2] = h_up
        V[self.config.total_dof // 2:, self.config.total_dof // 2:] = -h_down

        assert np.allclose(V, V.conj().T)
        return V

    def _construct_mu_V(self, valley):
        V = np.zeros(self.config.total_dof) + 0.0j
        V[valley] = -1.0
        V[valley + self.config.total_dof // 2] = 1.0

        return np.diag(V)


    def _get_derivative(self, V):  # obtaining (6.99) from S. Sorella book
        return jit_get_derivative(self.U_full, V, self.Z)

    def get_O(self):  # derivative over all variational parameters
        '''
            O_i = \\partial{\\psi(x)}/ \\partial(w) / \\psi(x)
        '''

        self.W_GF_complete = np.zeros((self.W_GF.shape[0], self.W_GF.shape[0])) * 1.0j
        self.W_GF_complete[:, self.occupied_sites] = self.W_GF

        O_mu = [self.get_O_pairing(self.W_mu_derivative)]
        O_fugacity = []#[self.get_O_fugacity()] if not self.config.PN_projection else []
        O_pairing = jit_get_O_pairing(self.W_k_derivatives, self.W_GF_complete.T) if len(self.W_k_derivatives) > 0 else []
        O_Jastrow = jit_get_O_jastrow(self.Jastrow_A, self.occupancy * 1.0)
        #O_waves = jit_get_O_pairing(self.W_waves_derivatives, self.W_GF_complete.T) if len(self.W_waves_derivatives) > 0 else []
        O_hoppings = jit_get_O_pairing(self.W_hoppings_derivatives, self.W_GF_complete.T) if len(self.W_hoppings_derivatives) > 0 else []

        #O = O_mu + O_fugacity + O_waves + O_pairing + O_Jastrow
        O = O_mu + O_hoppings + O_pairing + O_Jastrow

        return np.array(O)

    def _construct_U_matrix(self, orbitals_in_use):
        self.T = construct_HMF(self.config, self.K_up, self.K_down, \
                               self.pairings_list_unwrapped, self.var_params_gap, self.hoppings_list_TBC_up, self.hoppings_list_TBC_down, \
                               self.var_hoppings, self.reg_gap_term, \
                               particle_hole = self.particle_hole, ph_test = self.ph_test, trs_test = self.trs_test)


        if self.ph_test:
            self.T = -self.T.conj()
        # assert not self.ph_test  # FIXME
        assert np.allclose(self.T, self.T.conj().T)
        plus_valley = np.arange(0, self.config.total_dof, 2)
        self.T[plus_valley, plus_valley] += 1e-11  # tiny symmetry breaking between valleys -- just so that the orbitals have definite quantum number
        E, U = np.linalg.eigh(self.T)
        print(E, 'energies')
        #print(np.trace(U[:U.shape[0] // 2, :U.shape[0] // 2]), np.trace(U[U.shape[0] // 2:, U.shape[0] // 2:].conj()))
        #assert np.isclose(np.trace(U[:U.shape[0] // 2, :U.shape[0] // 2]), np.trace(U[U.shape[0] // 2:, U.shape[0] // 2:].conj()))
        #assert np.allclose(U[:U.shape[0] // 2, :U.shape[0] // 2], U[U.shape[0] // 2:, U.shape[0] // 2:].conj())
        # exit(-1)
        #print(E.dtype, U.dtype, 'type of energy and U')

        assert(np.allclose(np.diag(E), U.conj().T.dot(self.T).dot(U)))  # U^{\dag} T U = E
        self.U_full = deepcopy(U).astype(np.complex128)
        self.E = E
        # print(self.E); exit(-1)

        if orbitals_in_use is not None:
            print('orbitals in use', flush=True)
            overlap_matrix = np.abs(np.einsum('ij,ik->jk', U.conj(), orbitals_in_use))
            self.lowest_energy_states = np.argmax(overlap_matrix, axis = 0)
            #print(self.lowest_energy_states, len(self.lowest_energy_states))
            #print(np.max(overlap_matrix, axis = 0), 'print maximums of overlaps')
            U = self.U_full[:, self.lowest_energy_states]

        elif self.config.enforce_particle_hole_orbitals:  # enforce all 4 spin species conservation
            # print('Initializing 1st way', flush=True)
            plus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...].conj()).real
            plus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...].conj()).real
            minus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...].conj()).real
            minus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...].conj()).real
            plus_valley_particle = plus_valley_particle > 0.99
            plus_valley_hole = plus_valley_hole > 0.99
            minus_valley_particle = minus_valley_particle > 0.99
            minus_valley_hole = minus_valley_hole > 0.99

            assert np.sum(plus_valley_particle) == self.config.total_dof // 4
            assert np.sum(plus_valley_hole) == self.config.total_dof // 4
            assert np.sum(minus_valley_particle) == self.config.total_dof // 4
            assert np.sum(minus_valley_hole) == self.config.total_dof // 4
            assert np.allclose(np.ones(self.config.total_dof), plus_valley_particle + plus_valley_hole + minus_valley_particle + minus_valley_hole)

            doping = (self.config.total_dof // 2 - self.config.Ne) // 2

            plus_valley_particle_number = self.config.total_dof // 8  - doping // 2
            minus_valley_particle_number = self.config.total_dof // 8 - doping // 2
            plus_valley_hole_number = self.config.total_dof // 8 + doping // 2
            minus_valley_hole_number = self.config.total_dof // 8 + doping // 2
            assert self.config.valley_imbalance == 0
            # print('Construction of Slater determinant: {:d} (+) orbitals and {:d} (-) orbitals'.format(plus_valley_number, minus_valley_number))

            idxs_total = np.argsort(E)
            idxs_plus_particle = idxs_total[plus_valley_particle][:plus_valley_particle_number]
            idxs_plus_hole = idxs_total[plus_valley_hole][:plus_valley_hole_number]
            idxs_minus_particle = idxs_total[minus_valley_particle][:minus_valley_particle_number]
            idxs_minus_hole = idxs_total[minus_valley_hole][:minus_valley_hole_number]

            U = np.concatenate([self.U_full[..., idxs_plus_particle], \
                                self.U_full[..., idxs_minus_particle], \
                                self.U_full[..., idxs_plus_hole], \
                                self.U_full[..., idxs_minus_hole]], axis = 1)
            self.lowest_energy_states = np.concatenate([idxs_plus_particle, idxs_minus_particle, idxs_plus_hole, idxs_minus_hole], axis = 0)
            #check = np.load(self.config.preassigned_orbitals_path)
            #print(self.lowest_energy_states - check)
            #np.save(os.path.join(self.config.workdir, 'saved_orbital_indexes.npy'), self.lowest_energy_states)  # depend only on filling

        elif self.config.enforce_valley_orbitals and not self.config.enforce_particle_hole_orbitals:
            print('Initializing 2nd way', flush=True)
            #print(self.E)
            
            plus_valley = np.einsum('ij,ij->j', self.U_full[np.arange(0, self.config.total_dof, 2), ...], self.U_full[np.arange(0, self.config.total_dof, 2), ...].conj()).real
            print(np.sort(plus_valley))
            plus_valley = plus_valley > 0.99
            print(np.sum(plus_valley), np.sum(~plus_valley))
            
            assert np.sum(plus_valley) == self.config.total_dof // 2
            assert np.sum(~plus_valley) == self.config.total_dof // 2

            plus_valley_number = self.config.total_dof // 4 # + self.config.valley_imbalance // 2
            minus_valley_number = self.config.total_dof // 4 # - self.config.valley_imbalance // 2
            #print('Construction of Slater determinant: {:d} (+) orbitals and {:d} (-) orbitals'.format(plus_valley_number, minus_valley_number))

            idxs_total = np.argsort(E)
            idxs_plus = idxs_total[plus_valley][:plus_valley_number]
            idxs_minus = idxs_total[~plus_valley][:minus_valley_number]

            U = np.concatenate([self.U_full[..., idxs_plus], self.U_full[..., idxs_minus]], axis = 1)
            self.lowest_energy_states = np.concatenate([idxs_plus, idxs_minus], axis = 0)
        
            self.occupied_levels = np.zeros(len(E), dtype=bool)
            self.occupied_levels[self.lowest_energy_states] = True
            #print('Initializing Slater wf: particles {:d}, holes {:d}'.format(np.sum(particles), np.sum(holes)))
            #print('Initializing Slater wf: selected particles {:d}, selected holes {:d}'.format(np.sum(particles * self.occupied_levels), np.sum(holes * self.occupied_levels)))

            plus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...].conj()).real
            plus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...].conj()).real
            minus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...].conj()).real
            minus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...].conj()).real
            #print('Hole-minus-ness = {:.10f}'.format(np.min(np.sort(minus_valley_hole)[-self.config.total_dof // 4:])))
            #print('Hole-plus-ness = {:.10f}'.format(np.min(np.sort(plus_valley_hole)[-self.config.total_dof // 4:])))
            #print('Particle-minus-ness = {:.10f}'.format(np.min(np.sort(minus_valley_particle)[-self.config.total_dof // 4:])))
            #print('Particle-plus-ness = {:.10f}'.format(np.min(np.sort(plus_valley_particle)[-self.config.total_dof // 4:])))

            plus_valley_particle = plus_valley_particle > 0.50
            plus_valley_hole = plus_valley_hole > 0.50
            minus_valley_particle = minus_valley_particle > 0.50
            minus_valley_hole = minus_valley_hole > 0.50

            #print('Initializing Slater wf: particles_+ {:d}, particles_- {:d}, holes _+ {:d}, holes_- {:d}'.format(np.sum(plus_valley_particle), np.sum(minus_valley_particle), np.sum(plus_valley_hole), np.sum(minus_valley_hole)))
            #print('Initializing Slater wf: selected particles_+ {:d}, selected holes_+ {:d}'.format(np.sum(plus_valley_particle * self.occupied_levels), np.sum(plus_valley_hole * self.occupied_levels)))
            #print('Initializing Slater wf: selected particles_- {:d}, selected holes_- {:d}'.format(np.sum(minus_valley_particle * self.occupied_levels), np.sum(minus_valley_hole * self.occupied_levels)))
        elif not self.config.enforce_valley_orbitals and not self.config.enforce_particle_hole_orbitals:
            # print('Initializing free way', flush=True)

            self.lowest_energy_states = np.argsort(E)[:self.config.total_dof // 2]  # select lowest-energy orbitals
            #print(np.argsort(E)[:self.config.total_dof // 2])
            #print(E[self.lowest_energy_states])
            #print(E)
            U = U[:, self.lowest_energy_states]  # select only occupied orbitals

            particles = np.einsum('ij,ij->j', self.U_full[:self.config.total_dof // 2, ...], self.U_full[:self.config.total_dof // 2, ...].conj()).real
            #print('Particleness = {:.3f}'.format(np.min(np.sort(particles)[self.config.total_dof // 2:])))
            #print(np.sort(particles))
            #print(np.sort(E))
            particles = particles > 0.50
            holes = ~particles
            self.occupied_levels = np.zeros(len(E), dtype=bool)
            self.occupied_levels[self.lowest_energy_states] = True
            #print('Initializing Slater wf: particles {:d}, holes {:d}'.format(np.sum(particles), np.sum(holes)))
            #print('Initializing Slater wf: selected particles {:d}, selected holes {:d}'.format(np.sum(particles * self.occupied_levels), np.sum(holes * self.occupied_levels)))
            # exit(-1)
            plus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(0, self.config.total_dof // 2, 2), ...].conj()).real
            plus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2, self.config.total_dof, 2), ...].conj()).real
            minus_valley_particle = np.einsum('ij,ij->j', self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...], \
                                                         self.U_full[np.arange(1, self.config.total_dof // 2, 2), ...].conj()).real
            minus_valley_hole = np.einsum('ij,ij->j', self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...], \
                                                     self.U_full[np.arange(self.config.total_dof // 2 + 1, self.config.total_dof, 2), ...].conj()).real
            #print(plus_valley_particle, plus_valley_hole, minus_valley_particle, minus_valley_hole)
            #print('Hole-minus-ness = {:.10f}'.format(np.min(np.sort(minus_valley_hole)[-self.config.total_dof // 4:])))
            #print('Hole-plus-ness = {:.10f}'.format(np.min(np.sort(plus_valley_hole)[-self.config.total_dof // 4:])))
            #print('Particle-minus-ness = {:.10f}'.format(np.min(np.sort(minus_valley_particle)[-self.config.total_dof // 4:])))
            #print('Particle-plus-ness = {:.10f}'.format(np.min(np.sort(plus_valley_particle)[-self.config.total_dof // 4:])))

            plus_valley_particle = plus_valley_particle > 0.50
            plus_valley_hole = plus_valley_hole > 0.50
            minus_valley_particle = minus_valley_particle > 0.50
            minus_valley_hole = minus_valley_hole > 0.50

            #print('Initializing Slater wf: particles_+ {:d}, particles_- {:d}, holes _+ {:d}, holes_- {:d}'.format(np.sum(plus_valley_particle), np.sum(minus_valley_particle), np.sum(plus_valley_hole), np.sum(minus_valley_hole)))
            #print('Initializing Slater wf: selected particles_+ {:d}, selected holes_+ {:d}'.format(np.sum(plus_valley_particle * self.occupied_levels), np.sum(plus_valley_hole * self.occupied_levels)))
            ##print('Initializing Slater wf: selected particles_- {:d}, selected holes_- {:d}'.format(np.sum(minus_valley_particle * self.occupied_levels), np.sum(minus_valley_hole * self.occupied_levels)))
            
            #exit(-1)

            # exit(-1)

        if self.config.use_preassigned_orbitals:
            print('initialize preassigned', flush=True)
            self.lowest_energy_states = np.load(self.config.preassigned_orbitals_path)
            U = self.U_full[:, self.lowest_energy_states]

        rest_states = np.setdiff1d(np.arange(len(self.E)), self.lowest_energy_states)
        self.gap = -np.max(E[self.lowest_energy_states]) + np.min(E[rest_states])
        self.E_fermi = np.max(self.E[self.lowest_energy_states])

        self.occupied_levels = np.zeros(len(E), dtype=bool)
        self.occupied_levels[self.lowest_energy_states] = True
        # print('smallest gap denominator {:.10f}'.format(np.max(self.E[self.occupied_levels]) - np.min(self.E[~self.occupied_levels])))

        if E[rest_states].min() - self.E_fermi < 1e-14 and not self.config.enforce_valley_orbitals and not self.config.enforce_particle_hole_orbitals and orbitals_in_use is None:
            print('open shell configuration, consider different pairing or filling!', flush = True)
            print(self.config.enforce_valley_orbitals, E[rest_states].min(), self.E_fermi)

        return U 

    def _construct_U_tilde_matrix(self):
        U_tilde = self.U_matrix[self.occupied_sites, :]
        return U_tilde 

    def _construct_W_GF(self):
        U_tilde_inv = np.linalg.inv(self.U_tilde_matrix)
        # print('GF_max = {:.6f}'.format(np.max(np.abs(self.U_matrix.dot(U_tilde_inv)))))
        return self.U_matrix.dot(U_tilde_inv)#.astype(np.complex64)

    def _generate_configuration(self, particle_hole):
        doping = (self.config.total_dof // 2 - self.config.Ne) // 2  # k
        n_particles = self.config.total_dof // 4 - doping
        n_holes = self.config.total_dof // 4 + doping

        if self.config.n_orbitals == 2 and self.config.valley_projection:
            n_particles_plus = n_particles // 2 + self.config.valley_imbalance // 4
            n_particles_minus = n_particles // 2 - self.config.valley_imbalance // 4

            n_holes_plus = n_holes // 2 - self.config.valley_imbalance // 4
            n_holes_minus = n_holes // 2 + self.config.valley_imbalance // 4

            # print('configuration start: ({:d} / {:d})_+, ({:d} / {:d})_-'.format(n_particles_plus, n_holes_plus, n_particles_minus, n_holes_minus))
            # print('initialisation particles_+ = {:d}, holes_+ = {:d}, particles_- = {:d}, holes_- = {:d}'.format(n_particles_plus, n_holes_plus, n_particles_minus, n_holes_minus))

            particles_plus = np.random.choice(np.arange(0, self.config.total_dof // 2, 2),
                                                        size = n_particles_plus, replace = False)
            particles_minus = np.random.choice(np.arange(0, self.config.total_dof // 2, 2) + 1,
                                                        size = n_particles_minus, replace = False)

            holes_plus = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof, 2),
                                                        size = n_holes_plus, replace = False)
            holes_minus = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof, 2) + 1,
                                                        size = n_holes_minus, replace = False)
            if particle_hole:
                particles_plus, holes_plus = holes_plus - self.config.total_dof // 2, particles_plus + self.config.total_dof // 2
                particles_minus, holes_minus = holes_minus - self.config.total_dof // 2, particles_minus + self.config.total_dof // 2
            occupied_sites = np.concatenate([particles_plus, particles_minus, holes_plus, holes_minus])
        else:
            occupied_sites_particles = np.random.choice(np.arange(self.config.total_dof // 2), 
                                                        size = n_particles, replace = False)
            occupied_sites_holes = np.random.choice(np.arange(self.config.total_dof // 2, self.config.total_dof), 
                                                    size = n_holes, replace = False)
            if particle_hole:
                occupied_sites_particles, occupied_sites_holes = occupied_sites_holes - self.config.total_dof // 2, occupied_sites_particles + self.config.total_dof // 2

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

 
    

    def perform_MC_step(self, proposed_move = None, enforce = False, demand_accept = False):
        self.wf_ampls.append(self.current_ampl)
        assert not demand_accept
        if demand_accept:
            t = time()
            MC_step_index_previous = self.MC_step_index
            moved_site, moved_site_idx, empty_site, det_ratio, Jastrow_ratio, self.MC_step_index = \
                _find_acceptance(self.MC_step_index, self._rnd_size, self.adjacency_list, self.state, self.random_numbers_direction,\
                                 self.W_GF, self.Jastrow, self.occupancy, self.var_f, self.random_numbers_acceptance, self.random_numbers_move, self.occupied_sites)
            print('N_accept attempts = {:d}'.format(self.MC_step_index - MC_step_index_previous))
            print('time to find move = {:.10f}'.format(time() - t))
        else:
            self.MC_step_index += 1
            rnd_index = self.MC_step_index % self._rnd_size
            if rnd_index == self._rnd_size - 1:
                self._refresh_rnd()
        
            if proposed_move == None:
                moved_site_idx = self.random_numbers_move[rnd_index]
                moved_site = self.occupied_sites[moved_site_idx]

                #moved_site = self.random_numbers_move[rnd_index]
                #moved_site_idx = self.place_in_string[moved_site]

                t = time()
                empty_site, empty = _choose_site(self.adjacency_list[moved_site], \
                                                 self.state, \
                                                 self.random_numbers_direction[rnd_index])
                #empty_site = self.random_numbers_direction[rnd_index] % len(self.state)
                #empty = self.state[empty_site] == 0
                # print(moved_site, self.adjacency_list[moved_site], self.state, self.random_numbers_direction[rnd_index])
                self.t_choose_site += time() - t
            else:  # only in testmode
                moved_site, empty_site = proposed_move
                moved_site_idx = self.place_in_string[moved_site]
                if empty_site not in self.empty_sites or moved_site not in self.occupied_sites:
                    return False, 1, 1, moved_site, empty_site

            if proposed_move == None and ((not empty) or self.state[moved_site] == 0):
                return False, 1, 1, moved_site, empty_site


            t = time()
            det_ratio = self.W_GF[empty_site, moved_site_idx] + np.dot(self.a_update_list[empty_site, :self.n_stored_updates],
                                                                       self.b_update_list[moved_site_idx, :self.n_stored_updates])
            self.t_det += time() - t
            #print('det = {:.10f}'.format(time() - t))
            t = time()
            Jastrow_ratio = get_Jastrow_ratio(self.Jastrow, self.occupancy, self.state, \
                                              self.var_f, moved_site, empty_site)
            #print('jastrow = {:.10f}'.format(time() - t))
            self.t_jastrow += time() - t
            t = time()

            # self.wf += time() - t
            # if not enforce and np.abs(det_ratio) ** 2 * (Jastrow_ratio ** 2) < self.random_numbers_acceptance[rnd_index]:
            #if np.abs(empty_site - moved_site) >= 144:
            #    print('ph-non conserving move!', np.abs(det_ratio) ** 2, (Jastrow_ratio ** 2))
            #    print(np.sum(np.abs(self.W_GF) > 1e-5))
            #else:
            #    print(np.abs(det_ratio) ** 2, (Jastrow_ratio ** 2))

            #w = (np.abs(det_ratio) ** 2) * ((Jastrow_ratio ** 2))
            #self.ws.append(w)
            # print(w, self.random_numbers_acceptance[rnd_index])
            if (np.abs(det_ratio) ** 2) * ((Jastrow_ratio ** 2)) < self.random_numbers_acceptance[rnd_index] and not enforce:
                self.rejected_factor += 1
                #print('rejected by factor', self.n_stored_updates)
                self.t_overhead_after += time() - t
                #print('overhead_after = {:.10f}'.format(time() - t))
                return False, 1, 1, moved_site, empty_site
        #print('%d --> %d', moved_site, empty_site)
        self.accepted += 1
        #t = time()
        self.current_ampl *= det_ratio * Jastrow_ratio
        self.current_det *= det_ratio
        #         print(self.current_det, self.current_ampl, self.get_cur_Jastrow_factor())
        # print(det_ratio)
        self.occupied_sites[moved_site_idx] = empty_site

        #if np.abs(empty_site - moved_site) > self.config.Ls ** 2 * 4:
        if not self.config.tests:
            assert np.abs(empty_site - moved_site) < self.config.Ls ** 2 * 4
            assert (empty_site - moved_site) % 2 == 0


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
        #print('t before updates = {:.10f}'.format(time() - t))
        t = time()

        a_new, b_new = _jit_delayed_update(self.a_update_list, self.b_update_list, self.n_stored_updates, \
                                           self.W_GF, empty_site, moved_site_idx)
        self.a_update_list[..., self.n_stored_updates] = a_new
        self.b_update_list[..., self.n_stored_updates] = b_new
        self.n_stored_updates += 1
        self.t_ab += time() - t
        #print('t create a, b = {:.10f}'.format(time() - t))
        t = time()

        if self.n_stored_updates == self.config.n_delayed_updates:
            self.perform_explicit_GF_update()

        #print('t update GF = {:.10f}'.format(time() - t))
        #t = time()
        self.t_gf_update += time() - t
        # self.update += time() - t 
        # print(self.get_cur_det(), self.current_det, flush=True)
        # assert np.isclose(self.get_cur_det(), self.current_det) # FIXME
        return True, det_ratio, Jastrow_ratio, moved_site, empty_site


    def perform_explicit_GF_update(self):
        if self.n_stored_updates == 0:
            return
        #self.W_GF = _jit_perform_explicit_GF_update(self.W_GF, self.a_update_list[..., :self.n_stored_updates], self.b_update_list[..., :self.n_stored_updates])# += self.a_update_list[..., :self.n_stored_updates].dot(self.b_update_list[..., :self.n_stored_updates].T)  # (5.97)
        self.W_GF = self.W_GF + self.a_update_list[..., :self.n_stored_updates].dot(self.b_update_list[..., :self.n_stored_updates].T)


        self.a_update_list *= 0.0j
        self.b_update_list *= 0.0j
        self.n_stored_updates = 0
        return

    def get_state(self):
        return self.occupied_sites, self.empty_sites, self.place_in_string

@jit(nopython=True)
def _jit_perform_explicit_GF_update(W, a, b):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            buff = 0.0 + 0.0j
            for k in range(a.shape[1]):
                W[i, j] += a[i, k] * b[j, k]
    return W

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

    return W_GF[empty_site, place_in_string[moved_site]]  # looks correct

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
    '''
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
    '''
    ## but if everything is non-equal... ##
    delta_jk = 1.0 if j == k else 0.0
    ratio_det = get_det_ratio(*state_packed, i, j) * get_det_ratio(*state_packed, k, l) - \
                get_det_ratio(*state_packed, i, l) * get_det_ratio(*state_packed, k, j)
    #print(state)
    #print(get_det_ratio(*state_packed, i, j), get_det_ratio(*state_packed, k, l))
    #print(get_det_ratio(*state_packed, i, l) * get_det_ratio(*state_packed, k, j))
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
    #print(delta_jk, get_det_ratio(*state_packed, i, l))
    return jastrow * (ratio_det + delta_jk * get_det_ratio(*state_packed, i, l))


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

def construct_HMF(config, K_up, K_down, pairings_list_unwrapped, var_params_gap, \
                  hoppings_list_TBC_up, hoppings_list_TBC_down,
                  var_hoppings, reg_gap_term, particle_hole = False, ph_test = False, trs_test = False):
    Delta = pairings.get_total_pairing_upwrapped(config, pairings_list_unwrapped, var_params_gap) * (-1. if ph_test else 1)
    T = scipy.linalg.block_diag(K_up, -K_down) + 0.0j

    #for hop_up, hop_down, coeff in zip(hoppings_list_TBC_up, hoppings_list_TBC_down, var_hoppings):
    #    T[:config.total_dof // 2, :config.total_dof // 2] += hop_up * coeff
    #    T[config.total_dof // 2:, config.total_dof // 2:] += -hop_down * coeff

    ### TEST GAPS REPULSION ###
    '''
    energies, states = np.linalg.eigh(T)
    states = states.T

    Deltamat = T * 0.
    Deltamat[:config.total_dof // 2, config.total_dof // 2:] = Delta
    Deltamat[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T

    repulsion = np.abs(states @ Deltamat @ states.conj().T)
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            if not np.isclose(np.abs(repulsion[i, j]), 0.0):
                print(repulsion[i, j], i, j)
    exit(-1)
    '''
    
    ### END TEST GAPS REPULSION ###


    ## various local pairing terms ##
    T[:config.total_dof // 2, config.total_dof // 2:] = Delta# if not particle_hole else Delta.conj().T
    T[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T# if not particle_hole else Delta

    if trs_test:
        T = T.conj()

    ## regularisation ##
    T[:config.total_dof // 2, config.total_dof // 2:] += reg_gap_term * (-1. if ph_test else 1)
    T[config.total_dof // 2:, :config.total_dof // 2] += reg_gap_term.conj().T * (-1. if ph_test else 1)



    # print(np.linalg.eigh(T)[0], 'energies all of T')    
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

@jit(nopython=True)
def _choose_site(adjacency, state, rnd):
    return adjacency[rnd % len(adjacency)], state[adjacency[rnd % len(adjacency)]] == 0


@jit(nopython=True)
def _find_acceptance(MC_step_index, _rnd_size, adjacency_list, state, random_numbers_direction, \
                     W_GF, Jastrow, occupancy, var_f, random_numbers_acceptance, random_numbers_move, occupied_sites):
    accepted = False
    while not accepted:
        MC_step_index += 1
        rnd_index = MC_step_index % _rnd_size

        moved_site_idx = random_numbers_move[rnd_index]
        moved_site = occupied_sites[moved_site_idx]
        empty_site = _choose_empty_site(adjacency_list[moved_site], \
                                        state, \
                                        random_numbers_direction[rnd_index])

        if empty_site < 0:
            continue


        det_ratio = W_GF[empty_site, moved_site_idx]

        Jastrow_ratio = get_Jastrow_ratio(Jastrow, occupancy, state, var_f, moved_site, empty_site)

        if np.abs(det_ratio) ** 2 * (Jastrow_ratio ** 2) < random_numbers_acceptance[rnd_index]:
            continue
        accepted = True
    return moved_site, moved_site_idx, empty_site, det_ratio, Jastrow_ratio, MC_step_index
