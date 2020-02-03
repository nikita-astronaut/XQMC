import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
    	### geometry and general settings ###
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.mu = np.array([-0.40])
        self.BC_twist = True  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.twist = np.exp(0 * 1.0j * np.random.uniform(0, 1, size = 2) * np.pi * 2); self.min_num_twists = 18;
        print(self.twist)
        self.model = models.model_hex_2orb_Koshino
        _, self.n_orbitals, self.n_sublattices, = self.model(self, self.mu, spin = +1.0)



        ### interaction parameters ###
        self.U = np.array([1.0])
        self.V = np.array([1.0])
        self.J = (self.U - self.V) / 2  # only used in 2-orbital models, set equal to J'
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino



        ### density VQMC parameters ###
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.N_electrons = self.total_dof // 2 - 10 # if PN_projection = True, the density is fixed at this number
        self.PN_projection = False  # if PN_projection = False, work in the Grand Canonial approach


        ### other parameters ###
        self.visualisation = True; self.tests = True
        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.workdir = '/home/astronaut/DQMC_TBG/logs/test/'
        self.load_parameters = True; self.load_parameters_path = '/home/astronaut/Documents/DQMC_TBG/logs/test/U_1.00_V_1.00_J_0.00_mu_-0.40/last_opt_params.p'


        ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.pairings_list = pairings.on_site_2orb_hex_real + pairings.NN_2orb_hex_real # !!! real ones must (!) come before the imaginary ones
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)
        self.initial_mu_parameters = -0.0
        self.initial_fugacity_parameters = 0.03
        self.initial_gap_parameters = np.random.uniform(-0.005, 0.005, size = len(self.pairings_list))
        self.initial_jastrow_parameters = np.random.uniform(0.1, 0.15, size = len(self.adjacency_list))
        self.initial_sdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)
        self.initial_cdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)



        ### optimisation parameters ###
        self.MC_chain = 3000000 // 2; self.MC_thermalisation = 20000; self.opt_raw = 1500;
        self.optimisation_steps = 1400; self.thermalization = 1300; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = self.N_electrons * 3
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-3, 1e-2, 1.003]  
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 5
