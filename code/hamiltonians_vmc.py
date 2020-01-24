import numpy as np
import models
from time import time
from wavefunction_vmc import get_wf_ratio, density, get_wf_ratio_double_exchange
from numba import jit
from time import time
import scipy


class HubbardHamiltonian(object):
    def __init__(self, config):
        self.config = config
        K_matrix_up = self.config.model(self.config, 0.0, spin = +1.0)[0]
        K_matrix_down = self.config.model(self.config, 0.0, spin = -1.0)[0]
        self.edges_quadratic = scipy.linalg.block_diag(K_matrix_up, -K_matrix_down)

    def _get_edges(self):
        raise NotImplementedError()
    def __call__(self, wf):
        raise NotImplementedError()


class hamiltonian_Koshino(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)
        self.edges_quadric, self.x_orbital, self.y_orbital = self._get_interaction()

    def _get_interaction(self):
        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(self.config.total_dof // 2) * self.config.U / 2.0)

        x_orbit, y_orbit = np.arange(0, self.config.total_dof // 2, 2), \
                           np.arange(1, self.config.total_dof // 2, 2)
                           # x_orbit[i] and y_orbit[i] are the two orbitals residing on the same lattice site

        for x, y in zip(x_orbit, y_orbit):
            edges_quadric[x, y] = self.config.V / 2.0
            edges_quadric[y, x] = self.config.V / 2.0

        return edges_quadric, x_orbit, y_orbit

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

        E_loc += get_E_quadratic(base_state, self.edges_quadratic, wf_state, wf.var_f)  # K--term

        density = particles - holes + 1
        E_loc -= self.config.mu * np.sum(density)  # \mu--term
        E_loc += self.config.U * np.sum(particles * (1 - holes))  # U--term
        E_loc += self.config.V * np.sum(density[self.x_orbital] * density[self.y_orbital])
        # the interaction term taken from https://arxiv.org/pdf/1809.06772.pdf, Eq. (2)
        # to ensure the right coefficients for the Kanamori relation

        if self.config.J == 0:
            return E_loc

        # Hund terms (are not affected by the twisted BC):
        E_loc += self.config.J * (get_E_J_Hund(self.x_orbital, self.y_orbital, wf_state, wf.var_f) + \
                                  get_E_Jprime_Hund(self.x_orbital, self.y_orbital, wf_state, wf.var_f))
        return E_loc


class hamiltonian_2bands(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)
        self.edges_quadric = _get_interaction(self)

    def _get_interaction(self):
        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(self.config.total_dof // 2) * self.config.U / 2.0)

        return edges_quadric

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
def get_E_J_Hund(x_orbital, y_orbital, wf_state, total_fugacity):
    '''
        E_hund = J \\sum_{i, s1, s2} c^{\\dag}_ixs1 c_iys1 c^{\\dag}_iys2 c_ixs2  = 

        1) s1 = s2 = \\up: J \\sum_{i} d^{\\dag}_ix d_iy d^{\\dag}_iy d_ix  = J \\sum_i n_{ix} (1 - n_{iy})
        2) s1 = \\up, s2 = \\down: \\sum_{i} d^{\\dag}_ix d_iy d_{iy + L} d^{\\dag}_{ix + L} = 
                                =  -\\sum_{i} F(ix, iy, ix + L, iy + L)
        3) s1 = \\down, s2 = \\up: J \\sum_{i} d_{ix + L} d^{\\dag}_{iy + L} d^{\\dag}_iy d_ix  = 
                                   -J \\sum_{i} F(iy + L, ix + L, iy, ix)
        4) s1 = s2 = \\down: J \\sum_{i} d_{ix + L} d^{\\dag}_{iy + L} d_{iy + L} d^{\\dag}_{ix + L} = 
                            = J \\sum_i (1 - n(ix + L)) n(iy + L)
    '''
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for x, y in zip(x_orbital, y_orbital):
        E_loc += density(wf_state[2], x) * (1 - density(wf_state[2], y))  # (1)
        E_loc += -get_wf_ratio_double_exchange(*wf_state, total_fugacity, x, y, x + L, y + L)  # (2)
        E_loc += -get_wf_ratio_double_exchange(*wf_state, total_fugacity, y + L, x + L, y, x)  # (3)
        E_loc += (1 - density(wf_state[2], x + L)) * density(wf_state[2], y + L)  # (4)
    return E_loc

@jit(nopython=True)
def get_E_Jprime_Hund(x_orbital, y_orbital, wf_state, total_fugacity):
    '''
        E_hund_prime = J \\sum_{i, o1 != o2} c^{\\dag}_io1up c^{\\dag}_io1down c_io2down c_io2up = 
                       J \\sum_{i, o1 != o2} d^{\\dag}_io1 d_{io1 + L} d^{\\dag}_{io2 + L} d_{i o2} =
                       J \\sum_{i, o1 != o2} F(io1, io1 + L, io2 + L, io2)
    '''
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for x, y in zip(x_orbital, y_orbital):
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, x, x + L, y + L, y)  # (x-y)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, y, y + L, x + L, x)  # (y-x)
    return E_loc  # in the wide-spread approximation J = J'