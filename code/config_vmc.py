import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.U = [4.0, 3.0, 4.0, 5.0, 6.0] # the force of on-site Coulomb repulsion in the units of t1
        self.V = [4.0, 3.0, 4.0, 5.0, 6.0] # the force of on-site Coulomb repulsion in the units of t1
        self.mu = 0.0
        self.BC_twist = False  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.model = models.model_hex_2orb_Kashino
        self.K_matrix, self.n_orbitals, self.n_sublattices, = self.model(self, self.mu)
        self.hamiltonian = hamiltonians_vmc.hamiltonian_4bands

        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.N_electrons = self.total_dof // 2 # only applied if PN_projection = True
        self.PN_projection = True

        self.MC_chain = 180000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)
        self.optimisation_steps = 1400; self.thermalization = 1300; self.obs_calc_frequency = 20
        # thermalisation = steps w.o. observables measurement | obs_calc_frequency -- how often calculate observables (in opt steps)
        
        self.correlation = self.N_electrons * 3
        self.observables_frequency = self.MC_chain // 3  # how often to compute observables
        self.opt_parameters = [1e-3, 1e-2, 1.003]  
        # regularizer for the S_stoch matrix | learning rate | MC_chain increasement rate
        self.n_delayed_updates = 5

        self.visualisation = False; self.tests = False

        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.workdir = '/home/astronaut/Documents/DQMC_TBG/logs/3/'
        self.load_parameters = False  # whether to load previous variational parameters from workdir
        pairings.obtain_all_pairings(self)

        self.pairings_list = pairings.on_site_2orb_hex_real + pairings.on_site_2orb_hex_imag + \
                             pairings.NN_2orb_hex_real + pairings.NN_2orb_hex_imag

        self.pairings_list_names = [p[-1] for p in self.pairings_list]

        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]

        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)
        self.initial_mu_parameters = -0.0
        self.initial_gap_parameters = np.random.uniform(-0.005, 0.005, size = len(self.pairings_list))
        self.initial_jastrow_parameters = np.random.uniform(0.1, 0.15, size = len(self.adjacency_list))
        self.initial_sdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)
        self.initial_cdw_parameters = np.random.uniform(-0.05, 0.05, size = self.n_orbitals * self.n_sublattices)
        

        #if len(self.adjacency_list) < len(self.initial_jastrow_parameters):
        #    self.initial_jastrow_parameters = self.initial_jastrow_parameters[:len(self.adjacency_list)]
        #else:
        #    self.adjacency_list = self.adjacency_list[:len(self.initial_jastrow_parameters)]
