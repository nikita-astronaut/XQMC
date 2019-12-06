import models
import numpy as np
import models_vmc
import hamiltonians_vmc
import optimisers
import pairings

class MC_parameters:
    def __init__(self):
        self.Ls = 10  # spatial size, the lattice will be of size Ls x Ls
        self.U = 8 # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.model_square_1orb
        self.n_orbitals = 1
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_2bands
        self.n_sublattices = 1
        self.MC_chain = 60000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.MC_thermalisation = 60000
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.N_electrons = 84 # only applied if PN_projection = True
        self.correlation = 100
        self.optimiser = optimisers.AdamOptimiser
        self.opt_parameters = [0.5, 0.5, 1e-8, 1e-2]
        self.BC_twist = True  # whether to apply the BC--twise method (PBC in x direction and APBC in y direction)
        self.PN_projection = True
        self.initial_mu_parameters = -0.51
        self.initial_gap_parameters = np.array([0.08])
        self.initial_jastrow_parameters = np.array([1.16])

config = MC_parameters()
pairings.obtain_all_pairings(config)
config.pairings_list = [pairings.NN_pairings_1orb_square[1]]
