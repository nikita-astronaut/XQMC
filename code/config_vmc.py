import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
    	### geometry and general settings ###
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.mu = 0.0
        self.BC_twist = True  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.twist = tuple([1., 1.]); self.num_twists = 576;
        self.model = models.model_hex_2orb_Koshino
        _, self.n_orbitals, self.n_sublattices, = self.model(self, self.mu, spin = +1.0)



        ### interaction parameters ###
        self.U = np.array([2.])
        self.V = np.array([2.])
        self.J = (self.U - self.V) / 2  # only used in 2-orbital models, set equal to J'
        self.hamiltonian = hamiltonians_vmc.hamiltonian_Koshino



        ### density VQMC parameters ###
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.N_electrons = self.total_dof // 2 # only applied if PN_projection = True
        self.PN_projection = False
        self.fugacity = np.array([-0.2])  # if PN_projection = False, work in the Grand Canonial approach



        ### variational parameters settings ###
        pairings.obtain_all_pairings(self)  # the pairings are constructed without twist
        self.pairings_list = pairings.on_site_2orb_hex_real + pairings.NN_2orb_hex_real # !!! real ones must (!) come before the imaginary ones
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)
        self.initial_mu_parameters = -0.0
        self.initial_gap_parameters = np.random.uniform(-0.005, 0.005, size = len(self.pairings_list))
        self.initial_jastrow_parameters = np.random.uniform(0.1, 0.15, size = len(self.adjacency_list))
        self.initial_sdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)
        self.initial_cdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)



        ### optimisation parameters ###
        self.MC_chain = 1000000; self.MC_thermalisation = 40000;
        self.optimisation_steps = 1400; self.thermalization = 1300; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        self.correlation = self.N_electrons * 3
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-3, 1e-2, 1.003]  
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 5



        ### other parameters ###
        self.visualisation = False; self.tests = False
        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.workdir = '/home/astronaut/Documents/DQMC_TBG/logs/new/'
        self.load_parameters = False; self.load_parameters_path = '/home/cluster/niastr/data/DQMC_TBG/code/2.0_new/U_2.00_V_2.00_J_0.00_f_-0.20/last_opt_params.p'
