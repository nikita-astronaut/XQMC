import models
import numpy as np
import hamiltonians_vmc
from opt_parameters import pairings, jastrow, waves
import wavefunction_vmc as wfv
from copy import deepcopy

class MC_parameters:
    def __init__(self, irrep_idx):
    	### geometry and general settings ###
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.mu = 0.0
        self.BC_twist = True; self.twist_mesh = 'Baldereschi'  # apply BC-twist
        assert self.BC_twist  # this is always true
        self.twist = np.array([1, 1]); self.n_chains = 8; assert self.twist[0] == 1 and self.twist[1] == 1  # twist MUST be set to [1, 1] here
        
        self.model = models.model_hex_2orb_Koshino
        self.chiral_basis = True
        self.K_0, self.n_orbitals, self.n_sublattices, = self.model(self, self.mu, spin = +1.0)  # K_0 is the tb-matrix, which before twist and particle-hole is the same for spin-up and spin-down

        self.K_0 = models.xy_to_chiral(self.K_0, 'K_matrix', self, self.chiral_basis)  # this option is only valid for Koshino model
        for i in range(self.K_0.shape[0]):
            for j in range(self.K_0.shape[1]):
                if (i + j) % 2 == 1 and np.abs(self.K_0[i, j]) > 1e-9:
                    print(i, j, self.K_0[i, j])
                    assert not self.chiral_basis
                #if (i + j) % 2 == 0 and self.K_0[i, j] != self.K_0[j, i]:
                #    print(i, j, self.K_0[i, j], self.K_0[j, i])
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        Efull, _ = np.linalg.eigh(self.K_0)
        K_0_plus = self.K_0[:, np.arange(0, self.total_dof // 2, 2)]; K_0_plus = K_0_plus[np.arange(0, self.total_dof // 2, 2), :]
        K_0_minus = self.K_0[:, np.arange(1, self.total_dof // 2, 2)]; K_0_minus = K_0_minus[np.arange(1, self.total_dof // 2, 2), :]
        Eplus, _ = np.linalg.eigh(K_0_plus)
        Eminus, _ = np.linalg.eigh(K_0_minus)
        assert np.allclose(Efull, np.sort(np.concatenate([Eplus, Eminus])))

        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)


        ### interaction parameters ###
        self.epsilon = 3.
        self.xi = 0.5
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino
        self.U = 8.

        ### density VQMC parameters ###
        self.Ne = 132
        self.valley_imbalance = 0
        self.enforce_particle_hole_orbitals = False
        self.enforce_valley_orbitals = False  # constructs Slater determinant selecting valley orbitals separately
        self.use_preassigned_orbitals = False; self.preassigned_orbitals_path = '/home/astronaut/Documents/DQMC_TBG/logs/x11/saved_orbital_indexes.npy'
        self.valley_projection = True  # project onto valley imbalance = ...
        # if PN_projection = True, the density is fixed at this number
        self.PN_projection = True  # if PN_projection = False, work in the Grand Canonial approach
        self.optimize_mu_BCS = True

        if self.PN_projection:
            assert 0.0 == self.mu
        self.adjacency_transition_matrix = models.get_transition_matrix_range(self, self.model(self, 0.0, spin = +1.0)[0], \
                                            self.PN_projection, self.n_orbitals, valley_conservation=self.valley_projection) 
        #self.adjacency_transition_matrix = models.get_transition_matrix(self.PN_projection, \
        #                                   self.model(self, 0.0, spin = +1.0)[0], self.n_orbitals, valley_conservation=self.valley_projection)

        ### other parameters ###
        self.visualisation = False; 
        self.workdir = '/galileo/home/userexternal/nastrakh/XQMC/logs/newnewnew/'
        self.tests = True #True
        self.n_cpus = self.n_chains  # the number of processors to use | -1 -- take as many as available
        self.load_parameters = True; self.load_parameters_path = None
        self.offset = 0


        ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.pairings_list = pairings.twoorb_hex_all[irrep_idx]
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.pairings_list_unwrapped = [models.xy_to_chiral(g, 'pairing', \
            self, self.chiral_basis) for g in self.pairings_list_unwrapped]


        ### SDW/CDW parameters setting ###
        waves.obtain_all_waves(self)
        self.waves_list = [] # waves.hex_2orb
        self.waves_list_names = [w[-1] for w in self.waves_list]
        self.waves_list_unwrapped = []

        self.enforce_valley_orbitals = True
        if len(self.pairings_list) == 0:
            self.enforce_valley_orbitals = True
        for name in self.pairings_list_names:
            if '(S_1)' not in name and not '(S_2)' in name:
                self.enforce_valley_orbitals = False
        #for name in self.waves_list_names:
        #    if '(S_1)' not in name and '(S_2)' not in name:
        #        self.enforce_valley_orbitals = False

        self.name_group_dict = pairings.name_group_dict
        print(self.name_group_dict)

        ### jastrow parameters setting ###
        jastrow.obtain_all_jastrows(self)
        #self.jastrows_list = jastrow.jastrow_Koshino_Gutzwiller 
        self.jastrows_list = jastrow.jastrow_Koshino_simple
        self.jastrows_list_names = [j[-1] for j in self.jastrows_list]


        

        ### optimisation parameters ###
        self.MC_chain = 1500000; self.MC_thermalisation = 100000; self.opt_raw = 1500;
        self.optimisation_steps = 1600; self.thermalization = 13000; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = (self.total_dof // 2) * 2
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-3, 4e-2, 1.0005]
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 1
        self.generator_mode = True

        ### regularisation ###
        if not self.enforce_valley_orbitals:
            self.reg_gap_term = models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[1][0]), 'pairing', \
                                                    self, self.chiral_basis)
        else:
            self.reg_gap_term = models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[9][0]), 'pairing', \
                                                    self, self.chiral_basis) + \
                                models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[9][1]), 'pairing', \
                                                    self, self.chiral_basis)
        self.reg_gap_val = 1e-3

        ## initial values definition and layout ###
        self.layout = [2, 1 if not self.PN_projection else 0, len(self.waves_list), len(self.pairings_list), len(self.jastrows_list)]
        ### parameters section ###
        self.initial_parameters = np.concatenate([
            np.array([0.0, 0.0]),  # mu_BCS
            np.array([0.0] if not self.PN_projection else []),  # fugacity
            np.random.uniform(-0.1, 0.1, size = self.layout[2]),  # waves
            np.random.uniform(-0.01, 0.01, size = self.layout[3]),  # gaps
            np.random.uniform(0.01, 0.01, size = self.layout[4]),  # jastrows
        ])
        
        self.initial_parameters[np.sum(self.layout[:-1])] = 1.2
        self.initial_parameters[np.sum(self.layout[:-1]) + 1] = 0.5

        self.all_names = np.concatenate([
            np.array(['mu_BCS_+', 'mu_BCS_-']),  # mu_BCS
            np.array(['fugacity'] if not self.PN_projection else []),  # fugacity
            np.array(self.waves_list_names),  # waves
            np.array(self.pairings_list_names),  # gaps
            np.array(self.jastrows_list_names),
        ])

        self.all_clips = np.concatenate([
            np.ones(self.layout[0]) * 3e-4,  # mu_BCS
            np.array([0.0] if not self.PN_projection else []),  # fugacity
            np.ones(self.layout[2]) * 3e-4,  # waves
            np.ones(self.layout[3]) * 3e-4,  # gaps
            np.ones(self.layout[4]) * 3e-2,  # jastrows
        ])

        self.initial_parameters[:self.layout[0]] = self.select_initial_muBCS_Koshino()

        ### check K-matrix irrep properties ###
        pairings.check_irrep_properties(self, [[self.model(self, self.mu, spin = +1.0)[0], 'K_matrix']], \
            term_type = 'K_matrix', chiral = self.chiral_basis)

    def select_initial_muBCS_Koshino(self, parameters = []):
        if len(parameters) == 0:
            parameters = self.initial_parameters
        _, _, waves, gap, _ = self.unpack_parameters(parameters)
        # T = wfv.construct_HMF(self, self.K_0, self.K_0.T, self.pairings_list_unwrapped, gap, waves, self.reg_gap_term)

        # assert np.allclose(T, T.conj().T)
        if self.twist_mesh == 'Baldereschi':
            twist = [np.exp(2.0j * np.pi * 0.1904), np.exp(2.0j * np.pi * (0.1904 + 0.1))]
        elif self.twist_mesh == 'APBCy':
            twist = [1, -1]
        else:
            twist = [1, 1]
        print(twist)
        K_0_twisted = models.apply_TBC(self, twist, deepcopy(self.K_0), inverse = False)
        K_0_twisted_holes = -models.apply_TBC(self, twist, deepcopy(self.K_0).T, inverse = True)
        assert np.allclose(K_0_twisted, K_0_twisted.conj().T)
        assert np.allclose(K_0_twisted_holes, K_0_twisted_holes.conj().T)
        assert np.allclose(np.linalg.eigh(K_0_twisted_holes)[0], np.sort(-np.linalg.eigh(K_0_twisted)[0]))

        for i in range(K_0_twisted.shape[0]):
            for j in range(K_0_twisted.shape[1]):
                if (i + j) % 2 == 1 and np.abs(K_0_twisted[i, j]) > 1e-12:
                    print(i, j)

        K_0_plus = K_0_twisted[:, np.arange(0, self.total_dof // 2, 2)]; K_0_plus = K_0_plus[np.arange(0, self.total_dof // 2, 2), :]
        K_0_minus = K_0_twisted[:, np.arange(1, self.total_dof // 2, 2)]; K_0_minus = K_0_minus[np.arange(1, self.total_dof // 2, 2), :]

        K_0_plus_holes = K_0_twisted_holes[:, np.arange(0, self.total_dof // 2, 2)]; K_0_plus_holes = K_0_plus_holes[np.arange(0, self.total_dof // 2, 2), :]
        K_0_minus_holes = K_0_twisted_holes[:, np.arange(1, self.total_dof // 2, 2)]; K_0_minus_holes = K_0_minus_holes[np.arange(1, self.total_dof // 2, 2), :]
        print(np.linalg.eigh(K_0_plus)[0] -  np.sort(-np.linalg.eigh(K_0_minus_holes)[0]))
        assert np.allclose(np.linalg.eigh(K_0_plus)[0], np.sort(-np.linalg.eigh(K_0_minus_holes)[0]))
        assert np.allclose(np.linalg.eigh(K_0_minus)[0], np.sort(-np.linalg.eigh(K_0_plus_holes)[0]))


        nu = self.valley_imbalance // 4
        delta = (self.total_dof // 2 - self.Ne) // 4

        Ep, _ = np.linalg.eigh(K_0_plus)  # particle energies
        Ep = np.sort(Ep)
        Em, _ = np.linalg.eigh(K_0_minus)  # particle energies
        Em = np.sort(Em)

        N_particles_plus_below = np.sum(Ep < 0)  # current number of + particle energies below zero
        xi = N_particles_plus_below - (self.total_dof // 8 - delta + nu)  # this many levels must be put up above FS
        print('xi_plus = {:d}'.format(xi))
        dEp = (Ep[N_particles_plus_below - xi - 1] * 0.25 + Ep[N_particles_plus_below - xi] * 0.75) / 1
        #print(Ep, Ep[N_particles_plus_below - xi - 1], Ep[N_particles_plus_below - xi])
        print('initial mu_BCS_1 = {:.10f}'.format(dEp))
        print('N_[articles_plus_below before = {:d}'.format(N_particles_plus_below))
        print(Ep)
        print('N_holes_minus_below before = {:d}'.format(np.sum(np.linalg.eigh(K_0_minus_holes)[0] < 0)))
        Ep, _ = np.linalg.eigh(K_0_plus - np.eye(K_0_plus.shape[0]) * dEp)  # particle energies
        N_particles_plus_below = np.sum(Ep < 0)  # number after proper mu_BCS subtraction
        print('N_particles_plus_below after = {:d}'.format(N_particles_plus_below))
        print('N_holes_minus_below after = {:d}'.format(np.sum(np.linalg.eigh(K_0_minus_holes + np.eye(K_0_plus.shape[0]) * dEp)[0] < 0)))
        # print('holes', np.linalg.eigh(-K_0_plus_holes.T + np.eye(K_0_plus.shape[0]) * dEp)[0])
        # print('particles', Ep)


        N_particles_minus_below = np.sum(Em < 0)  # current number of + particle energies below zero
        xi = N_particles_minus_below - (self.total_dof // 8 - delta - nu)  # this many levels must be put up above FS
        dEm = (Em[N_particles_minus_below - xi - 1] * 0.25 + Em[N_particles_minus_below - xi] * 0.75) / 1
        #print(Em, Em[N_particles_minus_below - xi - 1], Em[N_particles_minus_below - xi])
        print('initial mu_BCS_2 = {:.10f}'.format(dEm))
        print('N_particles_minus_below before = {:d}'.format(N_particles_minus_below))
        print('N_holes_minus_below befire = {:d}'.format(np.sum(np.linalg.eigh(K_0_plus_holes)[0] < 0)))
        Em, _ = np.linalg.eigh(K_0_minus - np.eye(K_0_minus.shape[0]) * dEm)  # particle energies
        N_particles_minus_below = np.sum(Em < 0)  # number after proper mu_BCS subtraction
        print('N_particles_minus_below after = {:d}'.format(N_particles_minus_below))
        print('N_holes_plus_below after = {:d}'.format(np.sum(np.linalg.eigh(K_0_plus_holes + np.eye(K_0_minus.shape[0]) * dEm)[0] < 0)))

        print('!!!!', np.sort(np.concatenate([Em, Ep])))

        print('but I wanted particles_+ {:d}, holes_+ {:d}, particles_-{:d}, holes_- {:d}'.format(self.total_dof // 8 - delta + nu, self.total_dof // 8 + delta - nu, self.total_dof // 8 - delta - nu, self.total_dof // 8 + delta + nu))
        return dEp, dEp + 1e-6

    def unpack_parameters(self, parameters):
        offset = 0
        mu = parameters[offset:offset + self.layout[0]]; offset += self.layout[0]
        fugacity = None if self.PN_projection else parameters[offset]; offset += self.layout[1]

        waves = parameters[offset:offset + self.layout[2]]; offset += self.layout[2]
        gap = parameters[offset:offset + self.layout[3]]; offset += self.layout[3]
        jastrow = parameters[offset:offset + self.layout[4]]; offset += self.layout[4]
        assert offset == len(parameters)

        return mu, fugacity, waves, gap, jastrow
