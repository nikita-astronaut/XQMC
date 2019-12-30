import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 10  # spatial size, the lattice will be of size Ls x Ls
        self.U = [1, 6, 8, 10] # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.model_square_1orb
        self.n_orbitals = 1
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_2bands
        self.n_sublattices = 1
        self.MC_chain = 180000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.optimisation_steps = 500
        self.N_electrons = self.Ls ** 2 * self.n_sublattices * self.n_orbitals # only applied if PN_projection = True
        self.correlation = self.N_electrons
        self.opt_parameters = [1e-3, 3e-2]  # regularizer for the S_stoch matrix, learning rate
        self.BC_twist = False  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.PN_projection = True
        self.n_delayed_updates = 5
        self.visualisation = False
        self.tests = True
        self.observables_frequency = 60000  # how often to compute observables
        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.log_name = '/home/astronaut/Documents/DQMC_TBG/logs/log'
        self.observables_log_name = '/home/astronaut/Documents/DQMC_TBG/logs/observables'
        pairings.obtain_all_pairings(self)

        self.pairings_list = [pairings.on_site_1orb_square_real[0], pairings.NN_1orb_square_real[0], pairings.NN_1orb_square_real[1], pairings.NN_1orb_square_imag[1]]
        self.pairings_list_names = ['S', 'S*', 'Dre', 'Dim']

        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals

        self.initial_mu_parameters = -0.51
        self.initial_gap_parameters = np.random.uniform(-0.1, 0.1, size = len(self.pairings_list))
        self.initial_jastrow_parameters = np.array([1.4, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
