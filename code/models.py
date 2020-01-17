import numpy as np
from copy import deepcopy

abpc = 1.0

R_hexagonal = np.array([[np.sqrt(3) / 2, 1 / 2.], [np.sqrt(3) / 2., -1 / 2.]])
G_hexagonal = 2 * np.pi * np.array([[1 / np.sqrt(3), 1.], [1. / np.sqrt(3), -1.]])

R_square = np.array([[1, 0], [0, 1]])
G_square = 2 * np.pi * np.array([[1, 0], [0, 1]])

def diff_modulo(x, y, L, d):
    if d >= 0:
        return (x - y + L) % L == d  # or (x - y + L) % L == L - d
    else:
        return (x - y + L) % L == L + d

def lattice_to_physical(lattice, geometry):
    if geometry == 'square':
        return lattice

    x = lattice[0] * np.sqrt(3) + lattice[1] * np.sqrt(3) / 2.
    y = lattice[1] * 3 / 2.

    if lattice[2] == 1:
        y += 1.
    return x, y

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

def model_hex_2orb_Koshino(config, mu, only_NN = False):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = config.Ls ** 2 * n_orbitals * n_sublattices * 2
    t1, t2 = 0.331, (-0.010 + 1.0j * 0.097)
    if only_NN:
        t2 = 0.0 + 0.0j

    K = np.zeros((total_dof // 2, total_dof // 2))
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, n_orbitals, n_sublattices)

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
    return apply_twisted_periodic_conditions(config, n_orbitals, n_sublattices, K), n_orbitals, n_sublattices

def model_hex_1orb(config, mu):
    t1 = 1.
    n_orbitals = 1
    n_sublattices = 2
    total_dof = config.Ls ** 2 * n_orbitals * n_sublattices * 2
    K = np.zeros((total_dof // 2, total_dof // 2))
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, config.Ls) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

    K = K + K.conj().T
    K = K - mu * np.eye(K.shape[0])
    return apply_twisted_periodic_conditions(config, n_orbitals, n_sublattices, K), n_orbitals, n_sublattices

def interorbital_mod(A, n_orbitals, dist):
    result = []
    for i_orbital in range(n_orbitals):
        for j_orbital in range(i_orbital, n_orbitals):
            coupling = np.zeros((n_orbitals, n_orbitals))
            coupling[i_orbital, j_orbital] = 1
            coupling[j_orbital, i_orbital] = 1

            result.append([np.kron(A, coupling), i_orbital, j_orbital, dist])
    return result

def get_bc_copies(r, R, Ls, sublattice):
    copies = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            copies.append(np.array([r[0] + x * Ls, r[1] + y * Ls]).dot(R) + sublattice * np.array([1, 0]) / np.sqrt(3))
    return copies

def get_adjacency_list(config):
    if config.n_sublattices == 2:
        R = R_hexagonal
    else:
        R = R_square

    A = np.zeros((config.total_dof // 2 // config.n_orbitals, config.total_dof // 2 // config.n_orbitals))

    for first in range(config.total_dof // 2 // config.n_orbitals):  # omit orbitals for the time being
        for second in range(config.total_dof // 2 // config.n_orbitals):
            _, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, 1, config.n_sublattices)
            _, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, 1, config.n_sublattices)
            
            r1 = np.array([x1, y1]).dot(R) + sublattice1 * np.array([1, 0]) / np.sqrt(3)  # always 0 in the square case
            r2s = get_bc_copies([x2, y2], R, config.Ls, sublattice2)
            A[first, second] = np.min([np.sum((r1 - r2) ** 2) for r2 in r2s])
    distances = np.sort(np.unique(A.round(decimals=10)))
    adjacency_list = []
    longest_distance = None
    for dist in distances:
        adj = (A.round(decimals=10) == dist).astype(np.float32)
        adjacency_list = adjacency_list + interorbital_mod(adj, config.n_orbitals, dist)

        if dist == distances.max():
            longest_distance = np.kron(adj, np.ones((config.n_orbitals, config.n_orbitals)))
    return adjacency_list, longest_distance


def model_square_1orb(config, mu):
    t1 = 1.
    n_orbitals = 1
    n_sublattices = 1
    total_dof = config.Ls ** 2 * n_orbitals * n_sublattices * 2
    K = np.zeros((total_dof // 2, total_dof // 2))
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if nearest_neighbor_square(r1, r2, config.Ls):
                K[first, second] = t1 # * bc_factor

    # K = K + K.conj().T # already counted
    K = K - mu * np.eye(K.shape[0])
    return apply_twisted_periodic_conditions(config, n_orbitals, n_sublattices, K), n_orbitals, n_sublattices

def apply_twisted_periodic_conditions(config, n_orbitals, n_sublattices, K):
    '''
        if config.BC_twist == True, we demand that if j >= L, c_{j} = -c_{j % L} (only in y--direction)
    '''
    if not config.BC_twist:
        return K

    for first in range(K.shape[0]):
        for second in range(K.shape[1]):
            if K[first, second] == 0.0:
                continue
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), config.Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), config.Ls, n_orbitals, n_sublattices)

            if np.abs(y1 - y2) > config.Ls // 2:  # for sufficiently large lattices, this is the critetion of going beyond the boundary
                K[first, second] *= -1
    return K

def spatial_checkerboard(config):
    checkerboard = np.zeros(config.Ls ** 2)
    for x in range(config.Ls):
        for y in range(config.Ls):
            lin_index = x * config.Ls + y
            checkerboard[lin_index] = (-1) ** (x + y)
    return np.diag(checkerboard)
