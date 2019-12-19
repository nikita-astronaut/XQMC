import models
import numpy as np
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.U = 8 # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.model_hex_1orb
        self.n_orbitals = 1
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_2bands
        self.n_sublattices = 2
        self.MC_chain = 180000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.MC_thermalisation = 60000
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.N_electrons = 72 # only applied if PN_projection = True
        self.correlation = 100
        self.optimiser = optimisers.AdamOptimiser
        self.opt_parameters = [0.5, 0.5, 1e-8, 1e-2]
        self.BC_twist = True  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.PN_projection = True
        self.initial_mu_parameters = 0.0
        self.initial_gap_parameters = np.array([0.1])
        self.initial_jastrow_parameters = np.array([1.0])
        self.n_delayed_updates = 5
        self.visualisation = False
        self.tests = True
        self.log_name = '/home/astronaut/Documents/DQMC_TBG/log_Sstarwave.dat'

        pairings.obtain_all_pairings(self)

        self.pairings_list = [pairings.NN_pairings_1orb_hex_real[0]]
        self.pairings_list_names = ['S*-wave_re']
