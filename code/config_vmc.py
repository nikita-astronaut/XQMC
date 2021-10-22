import models
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
#from opt_parameters import pairings, jastrow, waves, hoppings
from opt_parameters import jastrow, pairings, waves
from copy import deepcopy
from scipy import interpolate
import hamiltonians_vmc

class MC_parameters:
    def __init__(self, Ls, Ne, irrep_idx):
    	### geometry and general settings ###
        self.Ls = Ls  # spatial size, the lattice will be of size Ls x Ls
        self.Ne = Ne#Ls ** 2 * 4 - 4 * 7 * 2 
        self.BC_twist = True; self.twist_mesh = 'PBC'  # apply BC-twist
        self.BC = 'PBC'
        self.L_twists_uniform = 6
        self.rank = irrep_idx
        assert self.BC_twist  # this is always true
        self.twist = np.array([1, 1]); self.n_chains = 6; assert self.twist[0] == 1 and self.twist[1] == 1  # twist MUST be set to [1, 1] here
        
        self.model = models.model_hex_2orb_Koshino
        self.chiral_basis = True
        self.K_0, self.n_orbitals, self.n_sublattices, = self.model(self, 0.0, spin = +1.0, BC=self.BC)  # K_0 is the tb-matrix, which before twist and particle-hole is the same for spin-up and spin-down

        self.K_0 = models.xy_to_chiral(self.K_0, 'K_matrix', self, self.chiral_basis)
        print(repr(np.linalg.eigh(self.K_0)[0]))
        exit(-1)

        #print(np.sum(self.K_0) / 0.331)

        check_chirality(self.K_0, self.chiral_basis)
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals

        self.far_indices = models._jit_get_far_indices(self.Ls, self.total_dof, self.n_sublattices, self.n_orbitals)

        Efull, _ = np.linalg.eigh(self.K_0)
        K_0_plus = self.K_0[:, np.arange(0, self.total_dof // 2, 2)]; K_0_plus = K_0_plus[np.arange(0, self.total_dof // 2, 2), :]
        K_0_minus = self.K_0[:, np.arange(1, self.total_dof // 2, 2)]; K_0_minus = K_0_minus[np.arange(1, self.total_dof // 2, 2), :]

        assert np.allclose(K_0_plus, np.conj(K_0_minus))
        Eplus, _ = np.linalg.eigh(K_0_plus)
        Eminus, _ = np.linalg.eigh(K_0_minus)
        assert np.allclose(Efull, np.sort(np.concatenate([Eplus, Eminus])))

        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self, BC=self.BC)

        ### interaction parameters ###
        self.epsilon = 5
        self.xi = 0.10
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino
        self.U = 1.5

        ### density VQMC parameters ###
        self.valley_imbalance = 0
        self.enforce_particle_hole_orbitals = False
        self.use_preassigned_orbitals = False; # self.preassigned_orbitals_path = '/home/astronaut/Documents/DQMC_TBG/logs/x11/saved_orbital_indexes.npy'
        self.valley_projection = True  # project onto valley imbalance = ...

        self.PN_projection = True; #False  # if PN_projection = False, work in the Grand Canonial approach, otherwise Canonical approach

        ### other parameters ###
        self.visualisation = False;
        self.workdir = '/home/astronaut/Documents/DQMC_TBG/logs/wavestest_shortjastrow_neworder/'

        self.tests = False; self.test_gaps = False
        self.n_cpus = self.n_chains  # the number of processors to use | -1 -- take as many as available
        self.load_parameters = True; 
        self.load_parameters_path = None
        self.offset = 0
        self.all_distances = models.get_distances_list(self, BC=self.BC)


        ### hoppings parameters setting ###
        all_Koshino_hoppings_real = []#hoppings.obtain_all_hoppings_Koshino_real(self, pairings)[1:] # exclude the mu_BCS term
        all_Koshino_hoppings_complex = []#hoppings.obtain_all_hoppings_Koshino_complex(self, pairings)
        self.hoppings = [] #[h[-1] + 0.0j for h in all_Koshino_hoppings_real + all_Koshino_hoppings_complex]
        self.hopping_names = [] #[h[0] for h in all_Koshino_hoppings_real + all_Koshino_hoppings_complex]
        for h in self.hoppings:
            projection = np.trace(np.dot(self.K_0.conj().T, h)) / np.trace(np.dot(h.conj().T, h))
            print(projection, name)


         ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.idx_map = []
        self.pairings_list = []# FIXMEpairings.Koshino_united[irrep_idx]#pairings.twoorb_hex_all[1]#irrep_idx] #idx_map[2]] # [irrep[0] for irrep in pairings.twoorb_hex_all[1:]] #[13]
        # self.pairings_list = pairings.twoorb_hex_all[idx_map[irrep_idx]]
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.pairings_list_unwrapped = [models.xy_to_chiral(g, 'pairing', \
            self, self.chiral_basis) for g in self.pairings_list_unwrapped]


        ### SDW/CDW parameters setting ###
        waves.obtain_all_waves(self)
        self.waves_list = waves.hex_2orb
        self.waves_list_unwrapped = [w[0] for w in self.waves_list]
        self.waves_list_names = [w[-1] for w in self.waves_list]
        #self.waves_list_unwrapped = [models.xy_to_chiral(g, 'wave', self, self.chiral_basis) for g in self.waves_list_unwrapped]


        self.enforce_valley_orbitals = False
        self.adjacency_transition_matrix = models.get_transition_matrix(self.PN_projection, self.K_0, \
                                            self.n_orbitals, valley_conservation_K = self.valley_projection, 
                                            valley_conservation_Delta = self.enforce_valley_orbitals)
        print(self.adjacency_transition_matrix)


        ### jastrow parameters setting ###
        jastrow.obtain_all_jastrows(self)
        self.jastrows_list = jastrow.jastrow_Koshino_simple[:10]

        self.jastrows_list_names = [j[-1] for j in self.jastrows_list]

        ### optimisation parameters ###
        self.MC_chain = 400000; self.MC_thermalisation = 100000; self.opt_raw = 1500;
        self.optimisation_steps = 10000; self.thermalization = 100000; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = (self.total_dof // 2) * 5
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-1, 0.001, 1.00]
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 1
        self.generator_mode = True
        self.condensation_energy_check_regime = False#True

        ### regularisation ###
        if not self.enforce_valley_orbitals:
            reg_valley = np.array([[-0.41341, -0.64892], [-0.9587,   0.64063]])
            self.reg_gap_term = np.kron(np.eye(self.total_dof // 2 // 2), reg_valley)
            #models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[1][0]), 'pairing', \
                                #                    self, self.chiral_basis)
                                                    # + models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[9][0]), 'pairing', \
                                                    #self, self.chiral_basis)
        else:
            self.reg_gap_term = models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[9][0]), 'pairing', \
                                                    self, self.chiral_basis) # FIXME
        self.reg_gap_val = 0.000

        ## initial values definition and layout ###
        self.layout = np.array([1, 0, len(self.waves_list_names), len(self.pairings_list_names), len(self.jastrows_list)])
        

        self.initial_parameters = np.concatenate([
            np.array([0.0]),  # mu_BCS
            #np.array([0.0] if not self.PN_projection else []),  # fugacity
            np.array([]),  # no fugacity
            np.random.uniform(2.5e-2, 2.5e-2, size = self.layout[2]),  # waves
            np.random.uniform(1.3e-2, 1.3e-2, size = self.layout[3]),  # gaps
            np.random.uniform(0.0, 0.0, size = self.layout[4]),  # jastrows
        ])

        '''
        self.parameter_fixing = np.concatenate([
            np.array([None]),  # mu_BCS
            np.array(] if not self.PN_projection else []),  # fugacity
            np.array([False] * size = self.layout[2]),  # waves
            np.array([False] * size = self.layout[3]),  # gaps
            np.array([False] * size = self.layout[4]),  # jastrows
        ])
        '''
        
       
        self.all_names = np.concatenate([
            np.array(['mu_BCS']),  # mu_BCS
            np.array(self.waves_list_names),  # hopping matrices
            np.array(self.pairings_list_names),  # gaps
            np.array(self.jastrows_list_names),
        ])

        self.all_clips = np.concatenate([
            np.ones(self.layout[0]) * 3e+4,  # mu_BCS
            #np.array([3e+4] if not self.PN_projection else []),  # fugacity
            np.ones(self.layout[2]) * 3e+4,  # waves
            np.ones(self.layout[3]) * 3e+4,  # gaps
            np.ones(self.layout[4]) * 3e+4,  # jastrows
        ])

        self.initial_parameters[:self.layout[0]], self.mu_BCS_min, self.mu_BCS_max = self.select_initial_muBCS_Koshino(self.Ne)
        print(self.initial_parameters[:self.layout[0]], self.mu_BCS_min, self.mu_BCS_max)

        self.mu = 0.0 #-1.2 #self.initial_parameters[0]

        ### check K-matrix irrep properties ###
        #pairings.check_irrep_properties(self, [[self.model(self, 0.0, spin = +1.0)[0], 'K_matrix']], \
        #    term_type = 'K_matrix', chiral = self.chiral_basis)

    def select_initial_muBCS_Koshino(self, Ne, parameters = []):
        if len(parameters) == 0:
            parameters = self.initial_parameters
        #_, _, waves, gap, _ = self.unpack_parameters(parameters, mode_setting = True)
        # T = wfv.construct_HMF(self, self.K_0, self.K_0.T, self.pairings_list_unwrapped, gap, waves, self.reg_gap_term)

        # assert np.allclose(T, T.conj().T)
        if self.twist_mesh == 'Baldereschi':
            twist = [np.exp(2.0j * np.pi * 0.1904), np.exp(2.0j * np.pi * (0.1904 + 0.1))]
        elif self.twist_mesh == 'APBCy':
            twist = [1, -1]
        else:
            twist = [1, 1]
        #twist = [1.0j, -1.0j]
        print(twist)

        #if twist[0] != 1 or twist[1] != 1:
        K_0_twisted = models.apply_TBC(self, twist, deepcopy(self.K_0), inverse = False)
        energies = np.linalg.eigh(K_0_twisted)[0]
        #print(energies)
        print(twist)
        print('energy_free_theory = ', np.sum(np.sort(energies)[:self.total_dof // 2 // 2] * 2 / (self.total_dof // 2)))
        assert np.allclose(K_0_twisted, K_0_twisted.T.conj())
        print('energy_free_theory = ', np.sum(energies[:self.Ne // 2] * 2) / 16.)


        K_0_twisted_holes = -models.apply_TBC(self, twist, deepcopy(self.K_0).T, inverse = True)
        assert np.allclose(K_0_twisted, K_0_twisted.conj().T)
        assert np.allclose(K_0_twisted_holes, K_0_twisted_holes.conj().T)
        assert np.allclose(np.linalg.eigh(K_0_twisted_holes)[0], np.sort(-np.linalg.eigh(K_0_twisted)[0]))

        for i in range(K_0_twisted.shape[0]):
            for j in range(K_0_twisted.shape[1]):
                if (i + j) % 2 == 1 and np.abs(K_0_twisted[i, j]) > 1e-12:
                    print(i, j)
                    exit(-1)  # chiral basis check failed

        K_0_plus = K_0_twisted[:, np.arange(0, self.total_dof // 2, 2)]; K_0_plus = K_0_plus[np.arange(0, self.total_dof // 2, 2), :]
        K_0_minus = K_0_twisted[:, np.arange(1, self.total_dof // 2, 2)]; K_0_minus = K_0_minus[np.arange(1, self.total_dof // 2, 2), :]

        assert np.allclose(np.conj(K_0_plus), K_0_minus)
        print(np.linalg.eigh(K_0_plus)[0])
        print(np.linalg.eigh(K_0_minus)[0])
        #

        K_0_plus_holes = K_0_twisted_holes[:, np.arange(0, self.total_dof // 2, 2)]; 
        K_0_plus_holes = K_0_plus_holes[np.arange(0, self.total_dof // 2, 2), :]
        
        K_0_minus_holes = K_0_twisted_holes[:, np.arange(1, self.total_dof // 2, 2)]; 
        K_0_minus_holes = K_0_minus_holes[np.arange(1, self.total_dof // 2, 2), :]
        
        #print(np.linalg.eigh(K_0_plus)[0] -  np.sort(-np.linalg.eigh(K_0_minus_holes)[0]))
        assert np.allclose(np.linalg.eigh(K_0_plus)[0], np.sort(-np.linalg.eigh(K_0_minus_holes)[0]))
        assert np.allclose(np.linalg.eigh(K_0_minus)[0], np.sort(-np.linalg.eigh(K_0_plus_holes)[0]))
        #exit(-1)

        nu = self.valley_imbalance // 4
        delta = (self.total_dof // 2 - Ne) // 4

        Ep, _ = np.linalg.eigh(K_0_plus)  # particle energies
        Ep = np.sort(Ep)
        Em, _ = np.linalg.eigh(K_0_minus)  # particle energies
        Em = np.sort(Em)

        N_particles_plus_below = np.sum(Ep < 0)  # current number of + particle energies below zero
        xi = N_particles_plus_below - (self.total_dof // 8 - delta + nu)  # this many levels must be put up above FS
        print('xi_plus = {:d}'.format(xi))
        dEp = (Ep[N_particles_plus_below - xi - 1] * 0.5 + Ep[N_particles_plus_below - xi] * 0.5) / 1
        mu_BCS_min = (Ep[N_particles_plus_below - xi - 1] * 0.999 + Ep[N_particles_plus_below - xi] * 0.001)
        mu_BCS_max = (Ep[N_particles_plus_below - xi - 1] * 0.001 + Ep[N_particles_plus_below - xi] * 0.999)

        print(Ep, Ep[N_particles_plus_below - xi - 1], Ep[N_particles_plus_below - xi])
        print('initial mu_BCS = {:.10f}'.format(dEp))
        #exit(-1)
        print('N_particles_plus_below before = {:d}'.format(N_particles_plus_below))
        #print(Ep)
        #print('N_holes_minus_below before = {:d}'.format(np.sum(np.linalg.eigh(K_0_minus_holes)[0] < 0)))
        Ep, _ = np.linalg.eigh(K_0_plus - np.eye(K_0_plus.shape[0]) * dEp)  # particle energies
        N_particles_plus_below = np.sum(Ep < 0)  # number after proper mu_BCS subtraction
        print('N_particles_plus_below after = {:d}'.format(N_particles_plus_below))
        #print('N_holes_minus_below after = {:d}'.format(np.sum(np.linalg.eigh(K_0_minus_holes + np.eye(K_0_plus.shape[0]) * dEp)[0] < 0)))
        # print('holes', np.linalg.eigh(-K_0_plus_holes.T + np.eye(K_0_plus.shape[0]) * dEp)[0])
        # print('particles', Ep)


        #N_particles_minus_below = np.sum(Em < 0)  # current number of + particle energies below zero
        #xi = N_particles_minus_below - (self.total_dof // 8 - delta - nu)  # this many levels must be put up above FS
        #dEm = (Em[N_particles_minus_below - xi - 1] * 0.49 + Em[N_particles_minus_below - xi] * 0.51) / 1
        #print(Em, Em[N_particles_minus_below - xi - 1], Em[N_particles_minus_below - xi])
        #print('initial mu_BCS_2 = {:.10f}'.format(dEm))
        #print('N_particles_minus_below before = {:d}'.format(N_particles_minus_below))
        #print('N_holes_minus_below before = {:d}'.format(np.sum(np.linalg.eigh(K_0_plus_holes)[0] < 0)))
        #Em, _ = np.linalg.eigh(K_0_minus - np.eye(K_0_minus.shape[0]) * dEm)  # particle energies
        #N_particles_minus_below = np.sum(Em < 0)  # number after proper mu_BCS subtraction
        #print('N_particles_minus_below after = {:d}'.format(N_particles_minus_below))
        #print('N_holes_plus_below after = {:d}'.format(np.sum(np.linalg.eigh(K_0_plus_holes + np.eye(K_0_minus.shape[0]) * dEm)[0] < 0)))

        #print('!!!!', np.sort(np.concatenate([Em, Ep])))

        #print('but I wanted particles_+ {:d}, holes_+ {:d}, particles_-{:d}, holes_- {:d}'.format(self.total_dof // 8 - delta + nu, self.total_dof // 8 + delta - nu, self.total_dof // 8 - delta - nu, self.total_dof // 8 + delta + nu))

        print(dEp)
        return dEp, mu_BCS_min, mu_BCS_max

    def guess_mu_BCS_approximate(self, density):
        densities = np.linspace(0, self.total_dof // 2, (self.total_dof // 2 + 4) // 4) / (self.total_dof // 2)
        print(densities * self.total_dof // 2)
        idx = np.argmin(np.abs(densities - density))
        print(densities[idx])
        return self.select_initial_muBCS_Koshino(int(densities[idx] * (self.total_dof // 2)))


    def unpack_parameters(self, parameters):
        offset = 0
        mu = parameters[offset:offset + self.layout[0]]; offset += self.layout[0]
        fugacity = None if self.PN_projection else parameters[offset]; offset += self.layout[1]

        waves = parameters[offset:offset + self.layout[2]]; offset += self.layout[2]
        gap = parameters[offset:offset + self.layout[3]]; offset += self.layout[3]
        jastrow = parameters[offset:offset + self.layout[4]]; offset += self.layout[4]
        assert offset == len(parameters)

        return mu, fugacity, waves, gap, jastrow


def check_chirality(K_0, chiral_basis):
    for i in range(K_0.shape[0]):
        for j in range(K_0.shape[1]):
            if (i + j) % 2 == 1 and np.abs(K_0[i, j]) > 1e-9:
                print(i, j, K_0[i, j])
                assert not chiral_basis
    return


