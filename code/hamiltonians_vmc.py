import numpy as np
import models
from time import time
from wavefunction_vmc import get_wf_ratio
from numba import jit
from time import time

class HubbardHamiltonian(object):
    def __init__(self, config):
        self.config = config
        self.edges_quadratic, self.edges_quadric = self._get_edges()

    def _get_edges(self):
        raise NotImplementedError()

    def __call__(self, wavefunction):
        '''
            performs the summation 
            E_loc(i) = \\sum_{j ~ i} H_{ij} \\psi_j / \\psi_i,
            where j ~ i are the all indixes having non-zero matrix element with i H_{ij}
        '''

        E_loc = 0.0 + 0.0j
        base_state = wavefunction.state
        particles, holes = base_state[:len(base_state) // 2], base_state[len(base_state) // 2:]

        E_loc += get_E_quadratic(base_state, self.edges_quadratic, \
                 (wavefunction.Jastrow, wavefunction.W_GF, wavefunction.place_in_string, wavefunction.state, wavefunction.occupancy))

        E_loc -= self.config.mu * (np.sum(particles) - np.sum(holes) + 1)
        # this sum runs in the real indices space (not 2--extended as above)
        density = particles - holes

        E_loc += np.einsum('i,ij,j', density, self.edges_quadric, density)
        # on-site term U/2 \rho^2 = U n_up n_down + (U/2) (-n_up - n_down + 1)
        # so, at half-filling the standart U n_up n_down and my U/2 \rho^2 must agree

        return E_loc


class hamiltonian_4bands(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)

    def _get_edges(self):
        edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
        K_matrix = self.config.model(self.config, 0.0)
        edges_quadratic = np.zeros((2 * K_matrix.shape[0], 2 * K_matrix.shape[1]))
        edges_quadratic[:K_matrix.shape[0], :K_matrix.shape[1]] = K_matrix
        edges_quadratic[K_matrix.shape[0]:, K_matrix.shape[1]:] = -K_matrix

        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(K_matrix.shape[0]) * self.config.U / 2.0)

        for i in range(K_matrix.shape[0]):
            for j in range(K_matrix.shape[1]):
                orbit1, sublattice1, x1, y1 = models.from_linearized_index(i, self.config.Ls, self.config.n_orbitals)
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(j, self.config.Ls, self.config.n_orbitals)

                if x1 == x2 and y1 == y2 and sublattice1 == sublattice2 and orbit1 != orbit2:
                    edges_quadric[i, j] = self.config.V / 2.0

        return edges_quadratic, edges_quadric

class hamiltonian_2bands(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)

    def _get_edges(self):
        edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
        K_matrix = self.config.model(self.config, 0.0)
        self.offdiagonal_mask = np.ones(K_matrix.shape[0])
        self.offdiagonal_mask -= np.diag(np.diag(self.offdiagonal_mask))

        edges_quadratic = np.kron(np.diag([1, -1]), K_matrix)

        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(K_matrix.shape[0]) * self.config.U / 2.0)

        return edges_quadratic, edges_quadric

@jit(nopython=True)
def get_E_quadratic(base_state, edges_quadratic, wf_state):
    E_loc = 0.0 + 0.0j

    for i in range(len(base_state)):
        for j in range(len(base_state)):
            if not (base_state[i] == 1 and base_state[j] == 0):
                continue
            E_loc += edges_quadratic[i, j] * get_wf_ratio(*wf_state, i, j)
    return E_loc
