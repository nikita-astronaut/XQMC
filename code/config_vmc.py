import models
import numpy as np
import hamiltonians_vmc
from opt_parameters import pairings, jastrow, waves


class MC_parameters:
    def __init__(self):
    	### geometry and general settings ###
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.mu = -0.40
        self.BC_twist = True  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.twist = [1, 1]; self.min_num_twists = 4; assert self.twist[0] == 1 and self.twist[1] == 1  # twist MUST be set to [1, 1] here
        self.model = models.model_hex_2orb_Koshino
        self.K_0, self.n_orbitals, self.n_sublattices, = self.model(self, self.mu, spin = +1.0)  # K_0 is the tb-matrix, which before twist and particle-hole is the same for spin-up and spin-down
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)


        ### interaction parameters ###
        self.U = 1.0
        self.V = 1.0
        self.J = (self.U - self.V) / 2  # only used in 2-orbital models, set equal to J'
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino


        ### density VQMC parameters ###
        self.Ne = self.total_dof // 2 - 10 
        # if PN_projection = True, the density is fixed at this number
        self.PN_projection = True  # if PN_projection = False, work in the Grand Canonial approach
        self.adjacency_transition_matrix = models.get_transition_matrix(self.PN_projection, self.model(self, 0.0, spin = +1.0)[0])


        ### other parameters ###
        self.visualisation = False; self.tests = False
        self.n_cpus = 4  # the number of processors to use | -1 -- take as many as available
        self.workdir = '/home/astronaut/Documents/DQMC_TBG/logs/3/'
        self.load_parameters = False; self.load_parameters_path = None#'/home/astronaut/Documents/DQMC_TBG/logs/test/U_1.00_V_1.00_J_0.00_mu_-0.40/last_opt_params.p'



        ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.pairings_list = pairings.twoorb_hex_A1_N + pairings.twoorb_hex_A2_N + pairings.twoorb_hex_E_N + \
                             pairings.twoorb_hex_A1_NN + pairings.twoorb_hex_A2_NN + pairings.twoorb_hex_E_NN  # !!! real ones must (!) come before the imaginary ones
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]


        ### jastrow parameters setting ###
        jastrow.obtain_all_jastrows(self)
        self.jastrows_list = jastrow.jastrow_long_range_2orb_degenerate
        self.jastrows_list_names = [j[-1] for j in self.jastrows_list]


        ### SDW/CDW parameters setting ###
        waves.obtain_all_waves(self)
        self.waves_list = [] #waves.SDW_2orb + waves.CDW_2orb
        self.waves_list_names = [w[-1] for w in self.waves_list]


        ### optimisation parameters ###
        self.MC_chain = 600000; self.MC_thermalisation = 20000; self.opt_raw = 1500;
        self.optimisation_steps = 14000; self.thermalization = 13000; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = self.Ne * 2
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-3, 2e-2, 1.001]
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 5



        self.layout = [1, 1 if not self.PN_projection else 0, len(self.waves_list), len(self.pairings_list), len(self.jastrows_list)]
        ### parameters section ###
        self.initial_parameters = np.concatenate([
            np.array([0.0]),  # mu_BCS
            np.array([0.0] if not self.PN_projection else []),  # fugacity
            np.random.uniform(-0.05, 0.05, size = self.layout[2]),  # waves
            np.random.uniform(-0.001, 0.001, size = self.layout[3]),  # gaps
            np.random.uniform(0.5, 1.0, size = self.layout[4]),  # jastrows
        ])

    def unpack_parameters(self, parameters):
        offset = 0
        mu = parameters[offset]; offset += self.layout[0]

        if self.PN_projection:
            fugacity = None; offset += self.layout[1]
        else:
            fugacity = parameters[offset]; offset += self.layout[1]

        waves = parameters[offset:offset + self.layout[2]]; offset += self.layout[2]
        gap = parameters[offset:offset + self.layout[3]]; offset += self.layout[3]
        jastrow = parameters[offset:offset + self.layout[4]]; offset += self.layout[4]
        assert offset == len(parameters)

        return mu, fugacity, waves, gap, jastrow
