import numpy as np
from numba import jit

R_hexagonal = np.array([[np.sqrt(3) / 2, 1 / 2.], [np.sqrt(3) / 2., -1 / 2.]])
G_hexagonal = 2 * np.pi * np.array([[1 / np.sqrt(3), 1.], [1. / np.sqrt(3), -1.]])

R_square = np.array([[1., 0.], [0., 1.]])
G_square = 2 * np.pi * np.array([[1, 0], [0, 1]])

U_xy_to_chiral = None


@jit(nopython=True)
def diff_modulo(x, y, L, d, BC='PBC'):
    if BC == 'OBC':
        return x - y == d

    if d >= 0:
        return (x - y + L) % L == d  # or (x - y + L) % L == L - d
    return (x - y + L) % L == L + d

def lattice_to_physical(lattice, geometry):
    if geometry == 'square':
        return [lattice[0], lattice[1]]

    x = (lattice[0] + lattice[1]) * np.sqrt(3) / 2
    y = (lattice[0] - lattice[1]) / 2

    if lattice[2] == 1:
        x += 1. / np.sqrt(3)
    return [x, y]

def physical_to_lattice(physical, geometry):
    if geometry == 'square':
        return [physical[0], physical[1], 0]

    x, y = physical
    lattice = [0, 0, 0]
    if np.abs(int(np.rint(2 * x / np.sqrt(3))) - 2 * x / np.sqrt(3)) >= 1e-5:
        lattice[2] = 1
        x = x - 1. / np.sqrt(3)


    lattice[1] = (x - np.sqrt(3) * y) / np.sqrt(3)
    lattice[0] = (x + np.sqrt(3) * y) / np.sqrt(3)

    return lattice


@jit(nopython=True)
def nearest_neighbor(r1, r2, L, geometry, BC='PBC'):
    if geometry == 'square':
        return nearest_neighbor_square(r1, r2, L, BC)
    if geometry == 'hexagonal':
        return nearest_neighbor_hexagonal(r1, r2, L, BC)
    print('Geometry', geometry, 'is not supported!!! Terminate.')
    return False, 0


@jit(nopython=True)
def nearest_neighbor_hexagonal(r1, r2, L, BC):
    if r1[1] == r2[1] and r1[0] == r2[0]:
        return True, 1
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1, BC):
        return True, 2
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1, BC):
        return True, 3
    return False, 0


@jit(nopython=True)
def nearest_neighbor_square(r1, r2, L, BC):
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True, 1
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        return True, 2
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -1):
        return True, 3
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, -1):
        return True, 4

    return False, 0


@jit(nopython=True)
def next_nearest_neighbor(r1, r2, L, geometry):
    if geometry == 'hexagonal':
        return next_nearest_neighbor_hexagonal(r1, r2, L)
    print('Geometry', geometry, 'is not supported!!! Terminate.')
    raise NotImplementedError()


@jit(nopython=True)
def next_nearest_neighbor_hexagonal(r1, r2, L):
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True, 1

    if diff_modulo(r1[0], r2[0], L, 1) and r1[1] == r2[1]:
        return True, 2

    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -1):
        return True, 4

    if diff_modulo(r1[0], r2[0], L, -1) and r1[1] == r2[1]:
        return True, 5

    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -1):
        return True, 3

    if diff_modulo(r1[0], r2[0], L, -1) and diff_modulo(r1[1], r2[1], L, 1):
        return True, 6

    return False, -1


@jit(nopython=True)
def fifth_nearest_neighbor(r1, r2, L, BC):
    if diff_modulo(r1[0], r2[0], L, 1, BC) and diff_modulo(r1[1], r2[1], L, -2, BC):
        return True
    if diff_modulo(r1[0], r2[0], L, -2, BC) and diff_modulo(r1[1], r2[1], L, 1, BC):
        return True
    if diff_modulo(r1[0], r2[0], L, 1, BC) and diff_modulo(r1[1], r2[1], L, 1, BC):
        return True
    return False

@jit(nopython=True)
def fourth_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 1) and r1[1] == r2[1]:
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -2):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -2):
        return True
    if diff_modulo(r1[0], r2[0], L, -2) and r1[1] == r2[1]:
        return True
    if diff_modulo(r1[0], r2[0], L, -2) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    return False

@jit(nopython=True)
def sixth_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 2) and r1[1] == r2[1]:
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 2):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, 1):
        return True


    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -3):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -3):
        return True
    if diff_modulo(r1[0], r2[0], L, 2) and diff_modulo(r1[1], r2[1], L, -3):
        return True


    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, -3):
        return True
    if diff_modulo(r1[1], r2[1], L, 1) and diff_modulo(r1[0], r2[0], L, -3):
        return True
    if diff_modulo(r1[1], r2[1], L, 2) and diff_modulo(r1[0], r2[0], L, -3):
        return True
    return False

@jit(nopython=True)
def from_linearized_index(index, L, n_orbitals, n_sublattices = 2):
    orbit = index % n_orbitals
    coord = index // n_orbitals
    sublattice = coord % n_sublattices
    coord = coord // n_sublattices

    x = coord // L
    y = coord % L
    return orbit, sublattice, x, y


@jit(nopython=True)
def to_linearized_index(x, y, sublattice, orbit, L, n_orbitals, n_sublattices = 2):
    return orbit + n_orbitals * (sublattice + n_sublattices * (y + x * L))

@jit(nopython=True)
def _jit_get_far_indices(Ls, total_dof, n_sublattices, n_orbitals):
    far_indices = [(0, 0)]
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            _, _, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            _, _, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)
            if np.abs(x1 - x2) >= Ls // 2 or np.abs(y1 - y2) >= Ls // 2:
                far_indices.append((first, second))
    return far_indices[1:]


def xy_to_chiral(M, term_type, config, chiral = False):
    if not chiral:
        return M
    global U_xy_to_chiral
    if U_xy_to_chiral is None:
        U_xy_to_chiral = np.kron(np.eye(M.shape[0] // 2), np.array([[1, 1], [-1.0j, +1.0j]]) / np.sqrt(2))
    if term_type == 'K_matrix' or term_type == 'bilinear':
        M_chiral = U_xy_to_chiral.conj().T.dot(M).dot(U_xy_to_chiral)
    elif term_type == 'wave':
        M_chiral = np.kron(np.eye(2), U_xy_to_chiral).conj().T.dot(M).dot(np.kron(np.eye(2), U_xy_to_chiral))
        print(M[:2, :2], M_chiral[:2, :2], np.kron(np.eye(2), U_xy_to_chiral)[:2, :2])
    else:
        print(U_xy_to_chiral.shape, M.shape)
        M_chiral = U_xy_to_chiral.T.dot(M).dot(U_xy_to_chiral)  # pairing

    #if term_type == 'pairing':
    #    for i in range(M.shape[0]):
    #        for j in range(M.shape[1]):
    #            if M_chiral[i, j] != 0.0 and i % 2 != j % 2:
    #                print('ACHTUNG')
    #print('passed')
    if term_type == 'K_matrix':
        assert np.isclose(np.sum(np.abs(M_chiral[np.arange(0, \
            M_chiral.shape[0], 2), ...][..., np.arange(0, \
            M_chiral.shape[0], 2) + 1])), 0)
    return M_chiral


@jit(nopython=True)
def _model_hex_2orb_Koshino(Ls, twist, tlong, mu, spin, BC):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    t1, t2, t5, t4 = 0.331, (-0.010 * 0. + 1.0j * tlong), 0 * 0.119, 0 * 0.036


    K = np.zeros((total_dof // 2, total_dof // 2)) * 1.0j
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, Ls, BC)[0] and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1


            if orbit2 == orbit1 and fifth_nearest_neighbor(r1, r2, Ls, BC) and sublattice2 == sublattice1:
                K[first, second] = np.real(t2)
            if orbit2 != orbit1 and fifth_nearest_neighbor(r1, r2, Ls, BC) and sublattice2 == sublattice1:
                if orbit1 == 0 and orbit2 == 1:
                    K[first, second] = np.imag(t2)
                else:
                    K[first, second] = -np.imag(t2)

    K = K + K.conj().T

    inverse = False if spin > 0 else True
    return K, n_orbitals, n_sublattices 


def model_hex_2orb_Koshino(config, mu, spin = +1.0, BC='PBC'):
    return _model_hex_2orb_Koshino(config.Ls, config.twist, config.tlong, mu, spin, BC)


@jit(nopython=True)
def _model_hex_1orb(Ls, twist, mu, spin):
    t1 = 1.
    n_orbitals = 1
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    K = np.zeros((total_dof // 2, total_dof // 2)) * 1.0j
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, Ls)[0] and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

    K = K + K.conj().T
    K = K - mu * np.eye(K.shape[0])

    inverse = False if spin > 0 else True
    return _apply_TBC(Ls, n_orbitals, n_sublattices, K, twist, inverse = inverse), n_orbitals, n_sublattices

def model_hex_1orb(config, mu, spin = +1.0):
    return _model_hex_1orb(config.Ls, config.twist, mu, spin)


def interorbital_mod(A, n_orbitals, dist):
    result = []
    for i_orbital in range(n_orbitals):
        for j_orbital in range(i_orbital, n_orbitals):
            coupling = np.zeros((n_orbitals, n_orbitals))
            coupling[i_orbital, j_orbital] = 1
            coupling[j_orbital, i_orbital] = 1

            result.append([np.kron(A, coupling), i_orbital, j_orbital, dist])
    return result

@jit(nopython=True)
def get_bc_copies(r, R, Ls, sublattice):
    copies = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            copies.append(np.array([r[0] + x * Ls, r[1] + y * Ls]).dot(R) + sublattice * np.array([1, 0]) / np.sqrt(3))
    return copies


def get_adjacency_list(config, orbital_mod = True, BC='PBC'):
    if config.n_sublattices == 2:
        R = R_hexagonal
    else:
        R = R_square

    A = np.zeros((config.total_dof // 2 // config.n_orbitals, config.total_dof // 2 // config.n_orbitals))

    for first in range(config.total_dof // 2 // config.n_orbitals):  # omit orbitals for the time being
        for second in range(config.total_dof // 2 // config.n_orbitals):
            _, sublattice1, x1, y1 = from_linearized_index(first, config.Ls, 1, config.n_sublattices)
            _, sublattice2, x2, y2 = from_linearized_index(second, config.Ls, 1, config.n_sublattices)
            
            r1 = np.array([x1, y1]).dot(R) + sublattice1 * np.array([1, 0]) / np.sqrt(3)  # always 0 in the square case
            if BC == 'PBC':
                r2s = get_bc_copies(np.array([1.0 * x2, 1.0 * y2]), R, config.Ls, sublattice2)
                A[first, second] = np.min(np.array([np.sum((r1 - r2) ** 2) for r2 in r2s]))
            else:
                r2 = np.array([x2, y2]).dot(R) + sublattice2 * np.array([1, 0]) / np.sqrt(3)
                A[first, second] = np.sum((r1 - r2) ** 2)




    A_rounded = A.round(decimals = 10)
    distances = np.sort(np.unique(A_rounded))
    adjacency_list = []
    longest_distance = None
    for dist in distances:
        adj = (A_rounded == dist).astype(np.float32)
        if orbital_mod:
            adjacency_list = adjacency_list + interorbital_mod(adj, config.n_orbitals, dist)
        else:
            adjacency_list.append(adj)

        if dist == distances.max():
            longest_distance = np.kron(adj, np.ones((config.n_orbitals, config.n_orbitals)))
    return adjacency_list, longest_distance


def get_distances_list(config, BC = 'PBC'):
    # returns |r_i - r_j|^2 for all sites i, j
    if config.n_sublattices == 2:
        R = R_hexagonal
    else:
        R = R_square

    A = np.zeros((config.total_dof // 2, config.total_dof // 2))

    for first in range(config.total_dof // 2):
        for second in range(config.total_dof // 2):
            _, sublattice1, x1, y1 = from_linearized_index(first, config.Ls, config.n_orbitals, config.n_sublattices)
            _, sublattice2, x2, y2 = from_linearized_index(second, config.Ls, config.n_orbitals, config.n_sublattices)
            
            r1 = np.array([x1, y1]).dot(R) + sublattice1 * np.array([1, 0]) / np.sqrt(3)  # always 0 in the square case
            if BC == 'PBC':
                r2s = get_bc_copies(np.array([1.0 * x2, 1.0 * y2]), R, config.Ls, sublattice2)
                A[first, second] = np.min(np.array([np.sum((r1 - r2) ** 2) for r2 in r2s]))  # account for PBC
            else:
                r2 = np.array([x2, y2]).dot(R) + sublattice2 * np.array([1, 0]) / np.sqrt(3)
                A[first, second] = np.sum((r1 - r2) ** 2)
    return A


def get_reduced_adjacency_matrix(config, max_distance, with_t5=False):
    A = get_adjacency_list(config)[0]
    reduced_A = np.zeros((config.total_dof // 2, config.total_dof // 2))

    for adj in A:
        if adj[-1] > max_distance + 1e-5:
            continue
        reduced_A += adj[0]

    #if with_t5:
    #    for adj in A:
    #        if np.isclose(adj[-1], 3.0):
    #            reduced_A += adj[0]
    return np.array([np.where(reduced_A[i, :] > 0.5)[0] for i in range(reduced_A.shape[0])])


@jit(nopython=True)
def _model_square_1orb(Ls, twist, mu, spin):
    t1 = 1.
    n_orbitals = 1
    n_sublattices = 1
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    K = np.zeros((total_dof // 2, total_dof // 2)) * 1.0j
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if nearest_neighbor_square(r1, r2, Ls)[0]:
                K[first, second] = t1

    # K = K + K.conj().T # already counted
    K = K - mu * np.eye(K.shape[0])
    inverse = False if spin > 0 else True
    return _apply_TBC(Ls, n_orbitals, n_sublattices, K, twist, inverse = inverse), n_orbitals, n_sublattices


def model_square_1orb(config, mu, spin = +1.0):
    return _model_square_1orb(config.Ls, config.twist, mu, spin)

#@jit(nopython = True)
def get_transition_matrix(PN_projection, K, n_orbitals = 1, \
                          valley_conservation_K=True, valley_conservation_Delta=True):
    assert valley_conservation_K and PN_projection


    adjacency_matrix = np.zeros(K.shape)
    unit_matrix = np.eye(n_orbitals) if valley_conservation_K else np.ones((n_orbitals, n_orbitals))
    for i in range(K.shape[0] // n_orbitals):
        for j in range(K.shape[0] // n_orbitals):
            if np.sum(np.abs(K[i * n_orbitals:i * n_orbitals + n_orbitals, j * n_orbitals : j * n_orbitals + n_orbitals])) > 1e-7:
                adjacency_matrix[i * n_orbitals:i * n_orbitals + n_orbitals, \
                                 j * n_orbitals:j * n_orbitals + n_orbitals] = unit_matrix  # valley-charge conservation
    big_adjacency_matrix = np.kron(np.eye(2), adjacency_matrix)

    

    '''
    if not PN_projection:
        adjacency_matrix_Delta = np.kron(np.eye(K.shape[0] // n_orbitals), \
            np.eye(n_orbitals) if valley_conservation_Delta else np.array([[0, 1], [1, 0]]))
        big_adjacency_matrix += np.kron(np.array([[0, 1], [1, 0]]), adjacency_matrix_Delta)
    '''



    adjacency_list = [np.where(big_adjacency_matrix[:, i] > 0)[0] \
                      for i in range(big_adjacency_matrix.shape[1])]

    return adjacency_list

def get_transition_matrix_range(config, K, PN_projection, n_orbitals = 1, valley_conservation=True):
    adjacency_matrix = np.zeros(K.shape)
    unit_matrix = np.eye(n_orbitals) if valley_conservation else np.ones((n_orbitals, n_orbitals))
    distances = get_adjacency_list(config, orbital_mod=False)[0]
    for dist in distances[:2]:
        adjacency_matrix += np.kron(dist, unit_matrix)
    adjacency_matrix[np.arange(K.shape[0]), np.arange(K.shape[0])] = 0.0
    
    
    big_adjacency_matrix = np.kron(np.eye(2), adjacency_matrix)
    if not PN_projection:  # not only particle-conserving moves
        big_adjacency_matrix += np.kron(np.array([[0, 1], [1, 0]]), adjacency_matrix) # np.eye(adjacency_matrix.shape[0]))
        # on-site pariticle<->hole transitions

    adjacency_list = [np.where(big_adjacency_matrix[:, i] > 0)[0] \
                      for i in range(big_adjacency_matrix.shape[1])]

    return adjacency_list

@jit(nopython=True)  # TODO: check this is valid for gaps
def _apply_TBC(Ls, n_orbitals, n_sublattices, K, twist, far_indices, \
               inverse = False, factor = 1, chiral_basis=True):  # inverse = True in the case of spin--down
    x_factor = twist[0] if not inverse else 1. / twist[0]
    y_factor = twist[1] if not inverse else 1. / twist[1]

    if factor != 1:
        x_factor = x_factor ** factor
        y_factor = y_factor ** factor
    #for first, second in far_indices:
    for first in range(K.shape[0]):
        for second in range(K.shape[1]):
            orbit1, sublattice1, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)

            if np.abs(K[first, second]) < 1e-7:
                continue

            if np.abs(y1 - y2) > Ls // 2:  # for sufficiently large lattices, this is the critetion of going beyond the boundary
            #if (first // 2, second // 2) == (2, 5) or (first // 2, second // 2) == (5, 2) or \
            #   (first // 2, second // 2) == (1, 6) or (first // 2, second // 2) == (6, 1):
                if y2 > y1:
                    if chiral_basis and orbit1 == 0:
                        K[first, second] *= y_factor
                    elif chiral_basis and orbit1 == 1:
                        K[first, second] *= np.conj(y_factor)
                    else:
                        K[first, second] *= y_factor
                else:
                    if chiral_basis and orbit2 == 0:
                        K[first, second] *= np.conj(y_factor)
                    elif chiral_basis and orbit2 == 1:
                        K[first, second] *= y_factor
                    else:
                        K[first, second] *= np.conj(y_factor)

            if np.abs(x1 - x2) > Ls // 2:  # for sufficiently large lattices, this is the critetion of going beyond the boundary
            #if (first // 2, second // 2) == (0, 7) or (first // 2, second // 2) == (7, 0) or \
            #   (first // 2, second // 2) == (0, 5) or (first // 2, second // 2) == (5, 0) or \
            #   (first // 2, second // 2) == (2, 7) or (first // 2, second // 2) == (7, 2) or \
            #   (first // 2, second // 2) == (2, 5) or (first // 2, second // 2) == (5, 2):
                #if np.abs(K[first, second]) < 1e-1:
                #    print(first, second, 'connection outer', y1, y2)
                if x2 > x1:
                    if chiral_basis and orbit1 == 0:
                        K[first, second] *= x_factor
                    elif chiral_basis and orbit1 == 1:
                        K[first, second] *= np.conj(x_factor)
                    else:
                        K[first, second] *= x_factor
                else:
                    if chiral_basis and orbit2 == 0:
                        K[first, second] *= np.conj(x_factor)
                    elif chiral_basis and orbit2 == 1:
                        K[first, second] *= x_factor
                    else:
                        K[first, second] *= np.conj(x_factor)
    return K


def apply_TBC(config, twist, K, inverse = False, factor = 1, chiral_basis=True):
    return _apply_TBC(config.Ls, config.n_orbitals, config.n_sublattices, K, \
                      twist, config.far_indices, inverse, factor, chiral_basis)


@jit(nopython=True)
def spatial_checkerboard(Ls):
    checkerboard = np.zeros(Ls ** 2)
    for x in range(Ls):
        for y in range(Ls):
            lin_index = x * Ls + y
            checkerboard[lin_index] = (-1) ** (x + y)
    return np.diag(checkerboard)


def spatial_uniform(Ls):
    return np.diag(np.ones(Ls ** 2))
