import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.U = [0, 2, 4, 6, 8, 10] # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.model_hex_2orb_Kashino
        self.n_orbitals = 2
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_2bands
        self.n_sublattices = 2
        self.MC_chain = 18000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.optimisation_steps = 300
        self.N_electrons = 144 # only applied if PN_projection = True
        self.correlation = 100
        self.opt_parameters = [1e-3, 3e-2]  # regularizer for the S_stoch matrix, learning rate
        self.BC_twist = False  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.PN_projection = True
        self.initial_mu_parameters = -0.51
        self.initial_gap_parameters = np.array([0.01, 0.01])
        self.initial_jastrow_parameters = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.n_delayed_updates = 5
        self.visualisation = True
        self.tests = True
        self.observables_frequency = 60000  # how often to compute observables
        self.n_cpus = -1  # the number of processors to use | -1 -- take as many as available
        self.log_name = '/home/astronaut/DQMC_TBG/log_DplusiDwave.dat'
        self.observables_log_name = '/home/astronaut/DQMC_TBG/observables_DplusiDwave.dat'
        pairings.obtain_all_pairings(self)

        self.pairings_list = [pairings.on_site_pairings_2orb_hex_real[0]]
        self.pairings_list_names = ['D-wave_re', 'D-wave_im']

        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals