import numpy as np
import models
from time import time
from wavefunction_vmc import get_wf_ratio, density, get_wf_ratio_double_exchange
from numba import jit
from time import time
import scipy
from copy import deepcopy

class HubbardHamiltonian(object):
    def __init__(self, config):
        self.config = config
        K_matrix_up = models.apply_TBC(self.config, self.config.twist, deepcopy(self.config.K_0), inverse = False)
        K_matrix_down = models.apply_TBC(self.config, self.config.twist, deepcopy(self.config.K_0), inverse = True).T

        self.edges_quadratic = scipy.linalg.block_diag(K_matrix_up, -K_matrix_down)
    def _get_edges(self):
        raise NotImplementedError()
    def __call__(self, wf):
        raise NotImplementedError()


class hamiltonian_Koshino(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)
        self.plus_orbital = np.arange(0, self.config.total_dof // 2, 2)  # chiral basis now
        self.minus_orbital = self.plus_orbital + 1
        self.U = 1.
        self.V = 1.
        self.W1 = 0 #2. / 3.
        self.W2 = 0 #1. / 3.
        self.W3 = 0 #1. / 3.
        self.JH = 0 # (self.U - self.V) / 2
        self.J = 0 # 1. / 5.
        self.epsilon = self.config.epsilon
        self.edges_quadric, self.edges_J_same, self.edges_J_updown, self.edges_J_downup = self._get_interaction()

    def _get_interaction(self):
        # for V_{ij} n_i n_j density--density interactions

        #  https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.081102
        #  https://arxiv.org/pdf/2003.09513.pdf
        #  term U / 2 \sum_{nu = +/-} (n_nu)^2
        edges_quadric = np.eye(self.config.total_dof // 2) * self.U / 2.0 / self.epsilon
        #  term V / 2 n_+ n_i + n_- n_+
        edges_quadric += np.kron(np.eye(self.config.total_dof // 2 // 2), np.array([[0, 1], [1, 0]])) * self.V / 2 / self.epsilon

        edges_quadric += np.array([adj[0] for adj in self.config.adjacency_list[3:6]]).sum(axis = 0) * self.W1 / 2 / self.epsilon
        edges_quadric += np.array([adj[0] for adj in self.config.adjacency_list[6:9]]).sum(axis = 0) * self.W2 / 2 / self.epsilon
        edges_quadric += np.array([adj[0] for adj in self.config.adjacency_list[9:12]]).sum(axis = 0) * self.W3 / 2 / self.epsilon

        edges_J_same = np.array([adj[0] for adj in self.config.adjacency_list[3:6]]).sum(axis = 0) * (-self.J / 2) / self.epsilon + 0.0j
        edges_J_updown = models.apply_TBC(self.config, self.config.twist, deepcopy(edges_J_same), inverse = False, factor = 2) / self.epsilon
        edges_J_downup = models.apply_TBC(self.config, self.config.twist, deepcopy(edges_J_same), inverse = True, factor = 2) / self.epsilon

        assert np.allclose(edges_J_updown, edges_J_downup.conj())

        return edges_quadric, edges_J_same, edges_J_updown, edges_J_downup

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

        E_loc += get_E_quadratic(base_state, self.edges_quadratic, wf_state, wf.var_f)  # K--term TODO: wf.state is passed twice
        E_loc += np.dot(particles - holes, self.edges_quadric.dot(particles - holes))

        #E_loc += get_E_C_Koshino(base_state[:len(base_state) // 2], base_state[len(base_state) // 2:], \
        #                         self.config.total_dof // 2, self.config.U, self.config.V)

        # on-site Hund term https://arxiv.org/pdf/2003.09513.pdf (Eq. 2)
        if self.JH != 0.0:
            E_loc += -self.config.JH * (get_E_J_Hund(self.plus_orbital, self.minus_orbital, wf_state, wf.var_f) + \
                                        get_E_J_Hund(self.minus_orbital, self.plus_orbital, wf_state, wf.var_f))

        if self.J != 0.0:
            E_loc += get_E_J_Hund_long(self.edges_J_same, self.edges_J_updown, self.edges_J_updown,\
                                       wf_state, wf.var_f)
        return E_loc


class hamiltonian_1orb_shortrange(HubbardHamiltonian):
    def __init__(self, config):
        super().__init__(config)
        self.edges_quadric = self._get_interaction()

    def _get_interaction(self):
        # for V_{ij} n_i n_j density--density interactions
        edges_quadric = np.diag(np.ones(self.config.total_dof // 2) * self.config.U / 2.0)
        return edges_quadric

    def __call__(self, wf):
        '''
            performs the summation 
            E_loc(i) = \\sum_{j ~ i} H_{ij} \\psi_j / \\psi_i,
            where j ~ i are the all indixes having non-zero matrix element with i H_{ij}
        '''

        E_loc = 0.0 + 0.0j
        base_state = wf.state
        density = base_state[:len(base_state) // 2] - base_state[len(base_state) // 2:]  # TODO: move that to T_C_Koshino

        wf_state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

        E_loc += get_E_quadratic(base_state, self.edges_quadratic, wf_state, wf.var_f)  # K--term TODO: wf.state is passed twice
        return E_loc + 0.5 * self.config.U * np.sum(density ** 2)

@jit(nopython=True)
def get_E_quadratic(base_state, edges_quadratic, wf_state, total_fugacity):
    E_loc = 0.0 + 0.0j

    for i in range(len(base_state)):
        for j in range(len(base_state)):
            if edges_quadratic[i, j] == 0:
                continue

            if i == j:
                E_loc += edges_quadratic[i, i] * density(wf_state[2], i)
                continue
            if not (base_state[i] == 1 and base_state[j] == 0):
                continue
            E_loc += edges_quadratic[i, j] * get_wf_ratio(*wf_state, total_fugacity, i, j)
    return E_loc

@jit(nopython=True)  # http://sces.phys.utk.edu/mostcited/mc1_1114.pdf
def get_E_C_Koshino(electrons, holes, size, U, V):
    density = electrons - holes
    result_U = 0.5 * U * np.sum(density ** 2)
    result_V = V * np.sum(density[np.arange(0, size, 2)] * density[np.arange(0, size, 2) + 1])

    return result_U + result_V


@jit(nopython=True)
def get_E_J_Hund(plus_orbital, minus_orbital, wf_state, total_fugacity):
    '''
        E_hund = J \\sum_{i, s1, s2} c^{\\dag}_ixs1 c^{\\dag}_iys2 c_{ixs2} c_iys1
    '''
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for x, y in zip(plus_orbital, minus_orbital):
        E_loc += -density(wf_state[2], x) * density(wf_state[2], y)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, x, y + L, x + L, y)
        E_loc += get_wf_ratio_double_exchange(*wf_state, total_fugacity, y, x + L, y + L, x)
        E_loc += (1 - density(wf_state[2], x + L)) * (1 - density(wf_state[2], y + L))
    return E_loc

@jit(nopython=True)
def get_E_J_Hund_long(edges_J_same, edges_J_updown, \
                      edges_J_downup, wf_state, total_fugacity):
    L = len(wf_state[3]) // 2
    E_loc = 0.0 + 0.0j

    for i in range(edges_J_same.shape[0] // 2):
        for j in range(edges_J_same.shape[1] // 2):
            if edges_J_same[i * 2, j * 2] == 0:
                continue

            for orb in range(2):
                itotal = i * 2 + orb; jtotal = i * 2 + orb
                E_loc += edges_J_same[itotal, jtotal] * density(wf_state[2], itotal) * density(wf_state[2], jtotal)  # s1 = s2
                E_loc += edges_J_same[itotal, jtotal] * (1 - density(wf_state[2], itotal + L)) * (1 - density(wf_state[2], jtotal + L))  # s1 = s2
                E_loc += -edges_J_updown[itotal, jtotal] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, itotal, jtotal + L, itotal + L, jtotal)
                E_loc += -edges_J_downup[itotal, jtotal] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, jtotal, itotal + L, jtotal + L, itotal)

            for mp in range(2):
                iplus = i * 2 + mp; iminus = i * 2 + 1 - mp; jplus = 2 * j + mp; jminus = j * 2 + 1 - mp;
                E_loc += edges_J_same[iplus, jplus] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, iplus, iminus, jminus, jplus)  # up-up
                E_loc += edges_J_same[iplus, jplus] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, iminus + L, iplus + L, jplus + L, jminus + L)  # down-down
                E_loc += -edges_J_updown[iplus, jplus] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, iplus, jminus + L, iminus + L, jplus)  # up-down
                E_loc += -edges_J_downup[iplus, jplus] * \
                    get_wf_ratio_double_exchange(*wf_state, total_fugacity, jminus, iplus + L, jplus + L, iminus)  # down-up
    return E_loc

@jit(nopython=True)
def get_E_Jprime_Hund(x_orbital, y_orbital, wf_state, total_fugacity):  # valid only for px/py basis
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
