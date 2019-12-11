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

def model_hex_2orb_Kashino(config, mu, only_NN = False):
    t1, t2 = 0.331, 0 * (-0.010 + 1.0j * 0.097)
    if only_NN:
        t2 = 0.0 + 0.0j

    K = np.zeros((config.total_dof // 2, config.total_dof // 2))
    for first in range(config.total_dof // 2):
        for second in range(config.total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals, config.n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, config.Ls) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

            if orbit2 == orbit1 and fifth_nearest_neighbor(r1, r2, config.Ls) and sublattice2 == sublattice1:
                K[first, second] = np.real(t2)
            if orbit2 != orbit1 and fifth_nearest_neighbor(r1, r2, config.Ls) and sublattice2 == sublattice1:
                if orbit1 == 0 and orbit2 == 1:
                    K[first, second] = np.imag(t2)
                else:
                    K[first, second] = -np.imag(t2)

    K = K + K.conj().T
    K = K - mu * np.eye(K.shape[0])
    return apply_twisted_periodic_conditions(config, K)

def model_hex_1orb(config, mu):
    t1 = 1.
    K = np.zeros((config.total_dof // 2, config.total_dof // 2))
    for first in range(config.total_dof // 2):
        for second in range(config.total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals, config.n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, config.Ls) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

    K = K + K.conj().T
    K = K - mu * np.eye(K.shape[0])
    return apply_twisted_periodic_conditions(config, K)

def interorbital_mod(A, n_orbitals):
    res = []
    if n_orbitals == 1:
        return [A]
    return [np.kron(A, np.array([[1, 0], [0, 1]])), np.kron(A, np.array([[0, 1], [1, 0]]))]  # now only symmetric

def get_adjacency_list(config, max_len):
    if config.n_sublattices == 2:
        K_matrix = model_hex_1orb(config, 0.0)  # only nearest-neighbors
    else:
        K_matrix = model_square_1orb(config, 0.0)  # only nearest-neighbors

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


def model_square_1orb(config, mu):
    t1 = 1.
    K = np.zeros((config.total_dof // 2, config.total_dof // 2))
    for first in range(config.total_dof // 2):
        for second in range(config.total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals, config.n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if nearest_neighbor_square(r1, r2, config.Ls):
                K[first, second] = t1 # * bc_factor

    # K = K + K.conj().T # already counted
    K = K - mu * np.eye(K.shape[0])
    return apply_twisted_periodic_conditions(config, K)

def apply_twisted_periodic_conditions(config, K):
    '''
        if config.BC_twist == True, we demand that if j >= L, c_{j} = -c_{j % L} (only in y--direction)
    '''
    if not config.BC_twist:
        return K

    for first in range(K.shape[0]):
        for second in range(K.shape[1]):
            if K[first, second] == 0.0:
                continue
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals, config.n_sublattices)

            if np.abs(y1 - y2) > config.Ls // 2:  # for sufficiently large lattices, this is the critetion of going beyond the boundary
                K[first, second] *= -1
    return K