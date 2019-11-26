import models
import numpy as np
import models_vmc
import hamiltonians_vmc
import optimisers

class MC_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.U = 6.  # the force of on-site Coulomb repulsion in the units of t1
        self.V = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.H_TB_Sorella_hexagonal
        self.n_orbitals = 1
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_2bands
        self.n_sublattices = 2
        self.MC_chain = 20000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.MC_thermalisation = 20000
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.particles_excess = 0
        self.total_spin = 0
        self.N_electrons = self.total_dof // 2
        self.correlation = 100
        self.optimiser = optimisers.AdamOptimiser
        self.opt_parameters = [0.5, 0.5, 1e-8, 1e-2]
