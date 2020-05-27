import models
import numpy as np
import hamiltonians_vmc
from opt_parameters import pairings, jastrow, waves
import wavefunction_vmc as wfv

class MC_parameters:
    def __init__(self):
    	### geometry and general settings ###
        self.Ls = 8  # spatial size, the lattice will be of size Ls x Ls
        self.mu = 0.0
        self.BC_twist = True; self.twist_mesh = 'Baldereschi'  # apply BC-twist
        assert self.BC_twist  # this is always true
        self.twist = np.array([1, 1]); self.n_chains = 6; assert self.twist[0] == 1 and self.twist[1] == 1  # twist MUST be set to [1, 1] here
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

        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)


        ### interaction parameters ###
        self.epsilon = 0.3
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino
        self.U = 8.

        ### density VQMC parameters ###
        self.Ne = 124
        self.valley_imbalance = 0
        self.enforce_valley_orbitals = False  # constructs Slater determinant selecting valley orbitals separately
        self.valley_projection = True  # project onto valley imbalance = ...
        # if PN_projection = True, the density is fixed at this number
        self.PN_projection = True  # if PN_projection = False, work in the Grand Canonial approach
        self.optimize_mu_BCS = True

        if self.PN_projection:
            assert 0.0 == self.mu
        self.adjacency_transition_matrix = models.get_transition_matrix(self.PN_projection, \
                                           self.model(self, 0.0, spin = +1.0)[0], self.n_orbitals, valley_conservation=self.valley_projection)

        ### other parameters ###
        self.visualisation = False; 
        self.tests = True
        self.n_cpus = 6  # the number of processors to use | -1 -- take as many as available
        self.workdir = '/s/ls4/users/astrakhantsev/DQMC_TBG/logs/6/'
        self.load_parameters = True; self.load_parameters_path = None
        self.offset = 0


        ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.pairings_list = pairings.twoorb_hex_all[15]
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.pairings_list_unwrapped = [models.xy_to_chiral(g, 'pairing', \
            self, self.chiral_basis) for g in self.pairings_list_unwrapped]
        # for name in self.pairings_list_names:
        #     if '(S_1)' in name or '(S_2)' in name:
        #         self.enforce_valley_orbitals = True

        self.name_group_dict = pairings.name_group_dict
        print(self.name_group_dict)

        ### jastrow parameters setting ###
        jastrow.obtain_all_jastrows(self)
        self.jastrows_list = jastrow.jastrow_Koshino[:2] # remove one jastrow (norm renormalization if PN is conserved)
        self.jastrows_list_names = [j[-1] for j in self.jastrows_list]


        ### SDW/CDW parameters setting ###
        waves.obtain_all_waves(self)
        self.waves_list = [] # waves.hex_2orb
        self.waves_list_names = [w[-1] for w in self.waves_list]


        ### optimisation parameters ###
        self.MC_chain = 1000000; self.MC_thermalisation = 3000; self.opt_raw = 1500;
        self.optimisation_steps = 10000; self.thermalization = 13000; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = 5 * (self.total_dof // 2)
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-4, 6e-2, 1.0005]
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 5
        self.generator_mode = True

        ### regularisation ###
        self.reg_gap_term = models.xy_to_chiral(pairings.combine_product_terms(self, pairings.twoorb_hex_all[0][0]), 'pairing', \
                                                self, self.chiral_basis)
        self.reg_gap_val = 1e-4  # s-wave regularisation magnitude

        ## initial values definition and layout ###
        self.layout = [1, 1 if not self.PN_projection else 0, len(self.waves_list), len(self.pairings_list), len(self.jastrows_list)]
        ### parameters section ###
        self.initial_parameters = np.concatenate([
            np.array([0.0]),  # mu_BCS
            np.array([0.0] if not self.PN_projection else []),  # fugacity
            np.random.uniform(-0.1, 0.1, size = self.layout[2]),  # waves
            np.random.uniform(-0.00001, 0.00001, size = self.layout[3]),  # gaps
            np.random.uniform(0.5, 0.6, size = self.layout[4]),  # jastrows
        ])

        self.all_names = np.concatenate([
            np.array(['mu_BCS']),  # mu_BCS
            np.array(['fugacity'] if not self.PN_projection else []),  # fugacity
            np.array(self.waves_list_names),  # waves
            np.array(self.pairings_list_names),  # gaps
            np.array(self.jastrows_list_names),
        ])

        self.initial_parameters[0] = self.select_initial_muBCS()
        print('mu_BCS was set to {:.5f}'.format(self.initial_parameters[0]))

        ### check K-matrix irrep properties ###
        pairings.check_irrep_properties(self, [[self.model(self, self.mu, spin = +1.0)[0], 'K_matrix']], \
            term_type = 'K_matrix', chiral = self.chiral_basis)

    def select_initial_muBCS(self, parameters = []):
        if len(parameters) == 0:
            parameters = self.initial_parameters
        _, _, waves, gap, _ = self.unpack_parameters(parameters)
        T = wfv.construct_HMF(self, self.K_0, self.K_0.T, self.pairings_list_unwrapped, gap, waves, self.reg_gap_term)

        assert np.allclose(T, T.conj().T)
        E, U = np.linalg.eigh(T)

        assert np.allclose(np.diag(E), U.conj().T.dot(T).dot(U))  # U^{\dag} T U = E

        return (np.sort(E)[self.Ne - 1] + np.sort(E)[self.Ne]) / 2.

    def unpack_parameters(self, parameters):
        offset = 0
        mu = parameters[offset]; offset += self.layout[0]
        fugacity = None if self.PN_projection else parameters[offset]; offset += self.layout[1]

        waves = parameters[offset:offset + self.layout[2]]; offset += self.layout[2]
        gap = parameters[offset:offset + self.layout[3]]; offset += self.layout[3]
        jastrow = parameters[offset:offset + self.layout[4]]; offset += self.layout[4]
        assert offset == len(parameters)

        return mu, fugacity, waves, gap, jastrow
