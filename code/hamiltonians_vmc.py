import numpy as np
import models

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

        for i, j, Hij in self.edges_quadratic:  # these are non-zero elements of a block-diagonal matrix diag(K, -K)
            if not ((base_state[i] == 1) and base_state[j] == 0):
                continue

            E_loc += Hij * wavefunction.get_det_ratio(i, j)

        E_loc -= self.config.mu * (np.sum(particles) - np.sum(holes))

        # this sum runs in the real indices space (not 2--extended as above)
        for i, j, Vij in self.edges_quadric:  # TODO: this can be parallized
            E_loc += Vij * (particles[i] - holes[i]) * (particles[j] - holes[j])

        return E_loc

class hamiltonian_4bands(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)

    def _get_edges(self):
        edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
        K_matrix = models.H_TB_simple(self.config.Ls, self.config.mu)
        for i in range(K_matrix.shape[0]):
            for j in range(K_matrix.shape[1]):
                if K_matrix[i, j] != 0.0 and i != j:  # only for hoppings, \mu is accounted separately
                    edges_quadratic.append((i, j, K_matrix[i, j]))  # particles pairing
                    edges_quadratic.append((i + K_matrix.shape[0], j + K_matrix.shape[1], -K_matrix[i, j]))  # holes pairing

        edges_quadric = []  # for V_{ij} n_i n_j density--density interactions
        for i in range(K_matrix.shape[0]):  # only on--site for now
            edges_quadric.append((i, i, self.config.U))

        for i in range(K_matrix.shape[0]):
            for j in range(K_matrix.shape[1]):
                orbit1, sublattice1, x1, y1 = models.from_linearized_index(i, self.config.Ls, self.config.n_orbitals)
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(j, self.config.Ls, self.config.n_orbitals)

                if x1 == x2 and y1 == y2 and sublattice1 == sublattice2 and orbit1 != orbit2:
                    edges_quadric.append((i, j, self.config.V))

        return edges_quadratic, edges_quadric
