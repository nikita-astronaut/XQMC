import numpy as np
import models
from time import time
from wavefunction_vmc import get_wf_ratio, density, get_wf_ratio_double_exchange
from numba import jit
from time import time

class HubbardHamiltonian(object):
    def __init__(self, config):
        self.config = config

    def _get_edges(self):
        raise NotImplementedError()
    def __call__(self, wf):
        raise NotImplementedError()


class hamiltonian_Koshino(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)
        self.edges_quadratic, self.edges_quadric, self.orbitsals = self._get_edges()

    def _get_edges(self):
        edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
        K_matrix = self.config.model(self.config, 0.0)[0]

        edges_quadratic = np.zeros((2 * K_matrix.shape[0], 2 * K_matrix.shape[1]))
        edges_quadratic[:K_matrix.shape[0], :K_matrix.shape[1]] = K_matrix
        edges_quadratic[K_matrix.shape[0]:, K_matrix.shape[1]:] = -K_matrix

        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(K_matrix.shape[0]) * self.config.U / 2.0)

        x_orbit, y_orbit = np.arange(0, self.config.total_dof // 2 // 2, 2), \
                           np.arange(1, self.config.total_dof // 2 // 2, 2)  # for Hund exchange terms
                           # x_orbit[i] and y_orbit[i] are the two orbitals residing on the same lattice site

        for i in range(K_matrix.shape[0]):
            for j in range(K_matrix.shape[1]):
                orbit1, sublattice1, x1, y1 = models.from_linearized_index(i, self.config.Ls, self.config.n_orbitals)
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(j, self.config.Ls, self.config.n_orbitals)

                if x1 == x2 and y1 == y2 and sublattice1 == sublattice2 and orbit1 != orbit2:
                    edges_quadric[i, j] = self.config.V / 2.0



        return edges_quadratic, edges_quadric, (x_orbit, y_orbit)

    def __call__(self, wf):
        '''
            performs the summation 
            E_loc(i) = \\sum_{j ~ i} H_{ij} \\psi_j / \\psi_i,
            where j ~ i are the all indixes having non-zero matrix element with i H_{ij}
        '''

        E_loc = 0.0 + 0.0j
        base_state = wf.state
        particles, holes = base_state[:len(base_state) // 2], base_state[len(base_state) // 2:]

        wf_state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

        E_loc += get_E_quadratic(base_state, self.edges_quadratic, wf_state, wf.total_fugacity)

        density = particles - holes
        E_loc -= self.config.mu * np.sum(density + 1)
        # this sum runs in the real indices space (not 2--extended as above)
        E_loc += np.einsum('i,ij,j', density, self.edges_quadric, density)
        # on-site term U/2 \rho^2 = U n_up n_down + (U/2) (-n_up - n_down + 1)
        # so, at half-filling the standart U n_up n_down and my U/2 \rho^2 must agree

        if self.config.J == 0:
            return E_loc

        # Hund terms:
        E_loc += self.config.J * (get_E_J_Hund(self.orbitsals, wf_state, wf.total_fugacity) + \
                                  get_E_Jprime_Hund(self.orbitsals, wf_state, wf.total_fugacity))
        return E_loc


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
def get_E_quadratic(base_state, edges_quadratic, wf_state, total_fugacity):
    E_loc = 0.0 + 0.0j

    for i in range(len(base_state)):
        for j in range(len(base_state)):
            if not (base_state[i] == 1 and base_state[j] == 0):
                continue
            E_loc += edges_quadratic[i, j] * get_wf_ratio(*wf_state, total_fugacity, i, j)
    return E_loc

@jit(nopython=True)
def get_E_J_Hund(orbitals, wf_state, total_fugacity):
    '''
        E_hund = J \\sum_{i, s1, s2} c^{\\dag}_ixs1 c^{\\dag}_iys2 c_ixs2 c_iys1 = 

        1) s1 = s2 = \\up: J \\sum_{i} d^{\\dag}_ix d^{\\dag}_iy d_ix d_iy = -\\sum_i n_{ix} n_{iy}
        2) s1 = \\up, s2 = \\down: \\sum_{i} d^{\\dag}_ix d_{iy + L} d^{\\dag}_{ix + L} d_iy = 
                                =  \\sum_{i} F(ix, iy + L, ix + L, iy)
        3) s1 = \\down, s2 = \\up: J \\sum_{i} d_{ix + L} d^{\\dag}_iy d_ix d^{\\dag}_{iy + L} = 
                                   J \\sum_{i} F(iy, ix + L, iy + L, ix)
        4) s1 = s2 = \\down: J \\sum_{i} d_{ix + L} d_{iy + L} d^{\\dag}_{ix + L} d^{\\dag}_{iy + L} = 
                            = -J \\sum_i d_{ix + L} d^{\\dag}_{ix + L} d_{iy + L} d^{\\dag}_{iy + L} = 
                            = -J \\sum_i (1 - n(ix + L)) (1 - n(iy + L))
    '''
    x_orbital, y_orbital = orbitals
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for x, y in zip(x_orbital, y_orbital):
        E_loc += -density(wf_state[2], x) * density(wf_state[2], y)  # (1)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, x, y + L, x + L, y)  # (2)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, y, x + L, y + L, x)  # (3)
        E_loc += -(1 - density(wf_state[2], x)) * (1 - density(wf_state[2], y))  # (4)
    return E_loc

@jit(nopython=True)
def get_E_Jprime_Hund(orbitals, wf_state, total_fugacity):
    '''
        E_hund_prime = J \\sum_{i, o1 != o2} c^{\\dag}_io1up c^{\\dag}_io1down c_io2down c_io2up = 
                       J \\sum_{i, o1 != o2} d^{\\dag}_io1 d_{io1 + L} d^{\\dag}_{io2 + L} d_{i o2} =
                       J \\sum_{i, o1 != o2} F(io1, io1 + L, io2 + L, io2)
    '''
    x_orbital, y_orbital = orbitals
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for x, y in zip(x_orbital, y_orbital):
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, x, x + L, y + L, y)  # (x-y)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, y, y + L, x + L, x)  # (y-x)
    return E_loc  # in the wide-spread approximation J = J'