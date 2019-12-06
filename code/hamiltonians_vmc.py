import numpy as np
import models
from time import time

class HubbardHamiltonian(object):
    def __init__(self, config):
        self.config = config
        self.edges_quadratic, self.edges_quadric = self._get_edges()
        self._states_dict = {}

    def _get_edges(self):
        raise NotImplementedError()

    def __call__(self, wavefunction):
        '''
            performs the summation 
            E_loc(i) = \\sum_{j ~ i} H_{ij} \\psi_j / \\psi_i,
            where j ~ i are the all indixes having non-zero matrix element with i H_{ij}
        '''
        if tuple(wavefunction.state) in self._states_dict:
            return self._states_dict[tuple(wavefunction.state)]

        E_loc = 0.0 + 0.0j
        base_state = wavefunction.state
        particles, holes = base_state[:len(base_state) // 2], base_state[len(base_state) // 2:]

        edges_contributing = np.where(np.einsum('i,ij,j->ij', base_state == 1, self.edges_quadratic, base_state == 0) != 0.0)
        # print(edges_contributing[0])
        for i, j in zip(edges_contributing[0], edges_contributing[1]):
            E_loc += self.edges_quadratic[i, j] * wavefunction.get_wf_ratio(i, j)

        E_loc -= self.config.mu * (np.sum(particles) - np.sum(holes) + 1)

        # this sum runs in the real indices space (not 2--extended as above)
        E_loc += np.einsum('i,ij,j', particles, self.edges_quadric, 1 - holes)
        self._states_dict[tuple(wavefunction.state)] = E_loc
        return E_loc

    def reset(self):
        self._states_dict = {}

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
        edges_quadric = np.diag([self.config.U] * K_matrix.shape[0])

        for i in range(K_matrix.shape[0]):
            for j in range(K_matrix.shape[1]):
                orbit1, sublattice1, x1, y1 = models.from_linearized_index(i, self.config.Ls, self.config.n_orbitals)
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(j, self.config.Ls, self.config.n_orbitals)

                if x1 == x2 and y1 == y2 and sublattice1 == sublattice2 and orbit1 != orbit2:
                    edges_quadric[i, j] = self.config.V

        return edges_quadratic, edges_quadric

class hamiltonian_2bands(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)

    def _get_edges(self):
        edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
        K_matrix = self.config.model(self.config, 0.0)
        edges_quadratic = np.zeros((2 * K_matrix.shape[0], 2 * K_matrix.shape[1]))
        edges_quadratic[:K_matrix.shape[0], :K_matrix.shape[1]] = K_matrix
        edges_quadratic[K_matrix.shape[0]:, K_matrix.shape[1]:] = -K_matrix

        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag([self.config.U] * K_matrix.shape[0])

        return edges_quadratic, edges_quadric
