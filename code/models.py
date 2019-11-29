import numpy as np
from copy import deepcopy

abpc = 1.0

def diff_modulo(x, y, L, d):
    if d >= 0:
        return (x - y + L) % L == d  # or (x - y + L) % L == L - d
    else:
        return (x - y + L) % L == L + d

def nearest_neighbor(r1, r2, L, geometry, return_direction):
    if geometry == 'square':
        return nearest_neighbor_square(r1, r2, L, return_direction)
    if geometry == 'hexagonal':
        return nearest_neighbor_hexagonal(r1, r2, L, return_direction)
    print('Geometry', geometry, 'is not supported')
    exit(-1)

def nearest_neighbor_hexagonal(r1, r2, L, return_direction = False):
    if r1[1] == r2[1] and r1[0] == r2[0]:
        if return_direction:
            return True, 1
        return True
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        if return_direction:
            return True, 2
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        if return_direction:
            return True, 3
        return True

    if return_direction:
        return False, 0
    return False

def nearest_neighbor_square(r1, r2, L, return_direction = False):
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        if return_direction:
            return True, 1
        return True
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        if return_direction:
            return True, 2
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -1):
        if return_direction:
            return True, 3
        return True
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, -1):
        if return_direction:
            return True, 4
        return True

    if return_direction:
        return False, 0
    return False

def nearest_neighbor_hexagonal_dir(r1, r2, L):
    if r1[1] == r2[1] and r1[0] == r2[0]:
        return 1
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        return 2
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return 3
    return -1
'''
def nearest_neighbor_square(r1, r2, L):
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True
    # if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, -1):
    #    return True
    #if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -1):
    #    return True
    return False
'''
def fifth_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -2):
        return True
    if diff_modulo(r1[0], r2[0], L, -2) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    return False

def from_linearized_index(index, L, n_orbitals, n_sublattices = 2):
    orbit = index % n_orbitals
    coord = index // n_orbitals
    sublattice = coord % n_sublattices
    coord = coord // n_sublattices

    x = coord // L
    y = coord % L
    return orbit, sublattice, x, y

def to_linearized_index(x, y, sublattice, orbit, L, n_orbitals, n_sublattices = 2):
    return orbit + n_orbitals * (sublattice + n_sublattices * (y + x * L))

def H_TB_simple(L, mu, only_NN = False):
    n_orbitals = 2
    t1, t2 = 0.331, (-0.010 + 1.0j * 0.097)
    if only_NN:
        t2 = 0.0 + 0.0j

    K = np.zeros((2 * n_orbitals * L * L, 2 * n_orbitals * L * L))
    for first in range(2 * n_orbitals * L * L):
        for second in range(2 * n_orbitals * L * L):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), L, n_orbitals)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), L, n_orbitals)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, L) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

            if orbit2 == orbit1 and fifth_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
                K[first, second] = np.real(t2)
            if orbit2 != orbit1 and fifth_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
                if orbit1 == 0 and orbit2 == 1:
                    K[first, second] = np.imag(t2)
                else:
                    K[first, second] = -np.imag(t2)

    K = K + K.conj().T
    K = K - np.diag(mu * np.ones(2 * n_orbitals * L * L))
    return K

def H_TB_Sorella_hexagonal(L, mu):
    t1 = 1.
    n_orbitals = 1
    K = np.zeros((2 * n_orbitals * L * L, 2 * n_orbitals * L * L))
    for first in range(2 * n_orbitals * L * L):
        for second in range(2 * n_orbitals * L * L):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), L, n_orbitals)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), L, n_orbitals)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, L) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

    K = K + K.conj().T
    K = K - np.diag(mu * np.ones(2 * n_orbitals * L * L))
    return K

def interorbital_mod(A, n_orbitals):
    res = []
    if n_orbitals == 1:
        return [A]
    return [np.kron(A, np.array([[1, 0], [0, 1]])), np.kron(A, np.array([[0, 1], [1, 0]]))]  # now only symmetric

def get_adjacency_list(config, max_len):
    if config.n_sublattices == 2:
        K_matrix = H_TB_Sorella_hexagonal(config.Ls, 0.0)  # only nearest-neighbors
    else:
        K_matrix = H_TB_Sorella_square(config.Ls, 0.0)  # only nearest-neighbors

    A = np.abs(np.asarray(K_matrix)) > 1e-6

    adjacency_list = []
    adj = np.diag(np.ones(len(np.diag(A))))
    seen_elements = adj * 0
    while len(adjacency_list) < max_len:
        adjacency_list = adjacency_list + interorbital_mod(adj, config.n_orbitals)
        seen_elements += adj
        adj = adj.dot(A)
        adj = np.logical_and(seen_elements == 0, adj > 0) * 1.
    return adjacency_list


def H_TB_Sorella_square(L, mu, BC_twist = False):
    t1 = 1.
    n_orbitals = 1
    n_sublattices = 1
    K = np.zeros((n_sublattices * n_orbitals * L * L, n_sublattices * n_orbitals * L * L))
    for first in range(n_sublattices * n_orbitals * L * L):
        for second in range(n_sublattices * n_orbitals * L * L):
            _, _, x1, y1 = from_linearized_index(deepcopy(first), L, n_orbitals, n_sublattices = 1)
            _, _, x2, y2 = from_linearized_index(deepcopy(second), L, n_orbitals, n_sublattices = 1)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            bc_factor = 1.0
            if BC_twist and r2[1] < r2[1]:
                bc_factor = -1

            if nearest_neighbor_square(r1, r2, L):
                K[first, second] = t1# * bc_factor

    # K = K + K.conj().T # already counted
    print(np.sum(K))
    K = K - np.diag(mu * np.ones(n_sublattices * n_orbitals * L * L))
    return K
