import models
import numpy as np
import models_vmc

class MC_parameters:
    def __init__(self):
        self.Ls = 12  # spatial size, the lattice will be of size Ls x Ls
        self.U = 0.  # the force of on-site Coulomb repulsion in the units of t1
        self.model = models.H_TB_simple
        self.n_orbitals = 2
        self.mu = 0.0
        self.n_sublattices = 2
        self.MC_length = 1000000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)\
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.particles_excess = 0
        self.total_spin = 0
        self.N_electrons = self.total_dof // 2