import models
import numpy as np
import models_vmc
import hamiltonians_vmc

class MC_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.U = 2.  # the force of on-site Coulomb repulsion in the units of t1
        self.V = 3.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.H_TB_simple
        self.n_orbitals = 2
        self.mu = 0.0
        self.hamiltonian = hamiltonians_vmc.hamiltonian_4bands
        self.n_sublattices = 2
        self.MC_chain = 1000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.MC_thermalisation = 1000
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.particles_excess = 0
        self.total_spin = 0
        self.N_electrons = self.total_dof // 2
        self.learning_rate = 1e-4
