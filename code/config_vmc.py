import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 8  # spatial size, the lattice will be of size Ls x Ls
        self.U = [4., 6.0, 8.0] # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.model_hex_2orb_Kashino
        self.n_orbitals = 2
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_4bands
        self.n_sublattices = 2
        self.MC_chain = 180000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.optimisation_steps = 300
        self.N_electrons = self.Ls ** 2 * self.n_sublattices * self.n_orbitals # only applied if PN_projection = True
        self.correlation = self.N_electrons * 3
        self.opt_parameters = [1e-3, 1e-2]  # regularizer for the S_stoch matrix, learning rate
        self.BC_twist = False  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.PN_projection = True
        self.n_delayed_updates = 5
        self.visualisation = False
        self.tests = True
        self.observables_frequency = 60000  # how often to compute observables
        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.log_name = '/home/astronaut/Documents/DQMC_TBG/logs/log_hex'
        self.observables_log_name = '/home/astronaut/Documents/DQMC_TBG/logs/observables_hex'
        pairings.obtain_all_pairings(self)

        self.pairings_list = [pairings.on_site_2orb_hex_real[0], pairings.on_site_2orb_hex_real[1], pairings.NN_2orb_hex_real[0], \
                              pairings.NN_2orb_hex_real[1], pairings.NN_2orb_hex_imag[1], pairings.NN_2orb_hex_real[2], \
                              pairings.NN_2orb_hex_imag[2], pairings.NN_2orb_hex_real[3], pairings.NN_2orb_hex_imag[3]]

        # self.pairings_list = [pairings.on_site_1orb_square_real[0], pairings.NN_1orb_square_real[0], pairings.NN_1orb_square_real[1], pairings.NN_1orb_square_imag[1], pairings.NN_1orb_square_real[2], pairings.NN_1orb_square_imag[2], pairings.NN_1orb_square_real[3], pairings.NN_1orb_square_imag[3]]
        self.pairings_list_names = ['S', 'Sz', 'S*', 'Dre', 'Dim', 'Pre1', 'Pim1', 'Pre2', 'Pim2']

        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals

        self.initial_mu_parameters = -0.0
        self.initial_gap_parameters = np.random.uniform(-0.1, 0.1, size = len(self.pairings_list))
        self.initial_jastrow_parameters = np.array([1.4, 0.0, 0.3, 0.3, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
        self.initial_sdw_parameters = np.random.uniform(-0.1, 0.1, size = self.n_orbitals * self.n_sublattices)
        self.initial_cdw_parameters = np.random.uniform(-0.1, 0.1, size = self.n_orbitals * self.n_sublattices)
        self.adjacency_list, self.longest_distance = models.get_adjacency_list(self)
        if len(self.adjacency_list) < len(self.initial_jastrow_parameters):
            self.initial_jastrow_parameters = self.initial_jastrow_parameters[:len(self.adjacency_list)]
        else:
            self.adjacency_list = self.adjacency_list[:len(self.initial_jastrow_parameters)]
