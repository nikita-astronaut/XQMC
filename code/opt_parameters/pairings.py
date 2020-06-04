import numpy as np
import models
from copy import deepcopy

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

sigma_1 = (Zpauli - 1.0j * Xpauli)
sigma_2 = (Zpauli + 1.0j * Xpauli)

delta_hex_AB, delta_hex_BA = [], []
delta_hex_NNN_AA, delta_hex_NNN_BB = [], []
delta_square = []

identity = np.array([[1]])

name_group_dict = {}

def construct_onsite_delta(config):
    return np.eye(config.Ls ** 2 * config.n_sublattices)

'''
    1) in the case of square lattice constructs just the adjacency matrix with links looking onle in one direction (e.g. only up)
    2) for hexagonal lattice the geometry is harder. We can have links of two kinds (A->B and B->A), and every link has three
       possible directions. This function only gives AB--links, BA links are obtained via the transposition of the result
'''
def construct_NN_delta(config, direction, geometry):
    if geometry == 'square':
        interlattice = 0
        n_sublattices = 1
    else:
        interlattice = 1  # interlattice -- sublattice2 - sublattice1
        n_sublattices = 2

    delta = np.zeros((config.Ls ** 2 * n_sublattices, config.Ls ** 2 * n_sublattices))

    for first in range(delta.shape[0]):
        for second in range(delta.shape[1]):
            sublattice1, sublattice2 = first % n_sublattices, second % n_sublattices
            space1, space2 = first // n_sublattices, second // n_sublattices
            x1, y1 = space1 // config.Ls, space1 % config.Ls
            x2, y2 = space2 // config.Ls, space2 % config.Ls
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if interlattice != sublattice2 - sublattice1:
                continue

            if direction == models.nearest_neighbor(r1, r2, config.Ls, geometry)[1]:
                delta[first, second] = 1

    return delta


def construct_NNN_delta(config, direction, geometry, sublattice = 0):
    if geometry == 'square':
        n_sublattices = 1
    else:
        n_sublattices = 2


    delta = np.zeros((config.Ls ** 2 * n_sublattices, config.Ls ** 2 * n_sublattices))

    for first in range(delta.shape[0]):
        for second in range(delta.shape[1]):
            sublattice1, sublattice2 = first % n_sublattices, second % n_sublattices
            space1, space2 = first // n_sublattices, second // n_sublattices
            x1, y1 = space1 // config.Ls, space1 % config.Ls
            x2, y2 = space2 // config.Ls, space2 % config.Ls
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if sublattice1 == sublattice and sublattice2 == sublattice:
                if direction == models.next_nearest_neighbor(r1, r2, config.Ls, geometry)[1]:
                    delta[first, second] = 1

    return delta


def construct_vi_hex(vi, delta_hex):
    '''
        v[1] = delta1 + delta2 + delta3
        v[2] = delta1 + omega delta2 + omega^* delta3
        v[3] = delta1 + omega^* delta2 + omega delta3

        v[1] furnishes the A1--representation in the NN-space,
        v[2]uv[3] furnish the E--representation
    '''

    phase_factor = 1.0 + 0.0j
    if vi == 2:
        phase_factor = np.exp(2.0 * np.pi / 3. * 1.0j)
    if vi == 3:
        phase_factor = np.exp(-2.0 * np.pi / 3. * 1.0j)

    return delta_hex[0] * (1.0 + 0.0j) + delta_hex[1] * phase_factor + delta_hex[2] * phase_factor ** 2


def construct_ui_hex(ui, delta_hex):
    omega = np.exp(-2.0 * np.pi / 3. * 1.0j)

    if ui == 1:
        return (delta_hex[2] + delta_hex[5]) + (delta_hex[1] + delta_hex[4]) + (delta_hex[0] + delta_hex[3])
    if ui == 2:
        return (delta_hex[2] + delta_hex[5]) + omega * (delta_hex[1] + delta_hex[4]) + np.conj(omega) * (delta_hex[0] + delta_hex[3])
    if ui == 3:
        return (delta_hex[2] + delta_hex[5]) + np.conj(omega) * (delta_hex[1] + delta_hex[4]) + omega * (delta_hex[0] + delta_hex[3])
    raise ValueError("NNN has only 3 possible links numerated 1, 2, 3")


def construct_vi_square(vi):
    '''
        v[1] = delta1 + delta2 + delta3 + delta4
        v[2] = delta1 + omega delta2 + omega^2 delta3 + omega^3 delta4
        v[3] = delta1 + omega^* delta2 + omega^*2 delta3 + omega^*3 delta4
        v[4] = delta1 - delta2 + delta3 - delta4
        v[1] furnishes the A1--representation in the NN-space,
        v[4] furnishes the B1--representation in the NN-space,
        v[2]uv[3] furnish the E--representation
    '''
    global delta_square
    phase_factor = 1.0 + 0.0j
    if vi == 2:
        phase_factor = np.exp(2.0 * np.pi / 4. * 1.0j)
    if vi == 3:
        phase_factor = np.exp(-2.0 * np.pi / 4. * 1.0j)
    if vi == 4:
        phase_factor = -1.0 + 0.0j

    return delta_square[0] * (1.0 + 0.0j) + delta_square[1] * phase_factor + \
           delta_square[2] * phase_factor ** 2 + delta_square[3] * phase_factor ** 3

def get_total_pairing_upwrapped(config, pairings_list_unwrapped, var_params):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for gap, vp in zip(pairings_list_unwrapped, var_params):
        Delta += gap * vp

    return Delta

def expand_tensor_product(config, sigma_l1l2, sigma_o1o2, delta_ij, spin_ij = np.eye(1)):
    '''
        returns the matrix of the dimention L x L (L -- total number of sites including orbit),
        sigma_ll \\otimes sigma_oo \\otimes delta_ii
    '''

    spin_factor = spin_ij.shape[0]
    if spin_factor > 1:
        Delta = np.zeros((config.total_dof, config.total_dof)) * 1.0j
    else:
        Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j

    for first in range(Delta.shape[0]):
        for second in range(Delta.shape[1]):
            spin1 = first // (config.total_dof // 2)
            spin2 = second // (config.total_dof // 2)

            orbit1, sublattice1, x1, y1 = models.from_linearized_index(deepcopy(first % (config.total_dof // 2)), \
                                                 config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second % (config.total_dof // 2)), \
                                                 config.Ls, config.n_orbitals, config.n_sublattices)

            space1 = (x1 * config.Ls + y1) * config.n_sublattices + sublattice1
            space2 = (x2 * config.Ls + y2) * config.n_sublattices + sublattice2

            if config.n_sublattices == 2:
                if sublattice2 - sublattice1 == 1:  # AB pairings (only in the hexagonal case)
                    delta_s1s2 = delta_ij[0]
                elif sublattice2 - sublattice1 == -1:  # BA pairings (only in the hexagonal case)
                    delta_s1s2 = delta_ij[1]
                elif sublattice1 == 0 and len(delta_ij) == 2:
                    delta_s1s2 = delta_ij[0]  # AA pairing (only hex case, not on-site)
                elif sublattice1 == 1 and len(delta_ij) == 2:
                    delta_s1s2 = delta_ij[1]  # BB pairing (only hex case, not on-site)
                else:
                    delta_s1s2 = delta_ij
            else:
                delta_s1s2 = delta_ij  # only square case

            # otherwise (subl2 = subl1 means this is a square lattice or hex on-site, just use the delta_ij matrix)
            if sigma_l1l2[sublattice1, sublattice2] != 0.0:
                Delta[first, second] = sigma_l1l2[sublattice1, sublattice2] * \
                                       sigma_o1o2[orbit1, orbit2] * \
                                       delta_s1s2[space1, space2] * \
                                       spin_ij[spin1, spin2]
    return Delta + 0.0j

def combine_product_terms(config, pairing):
    if len(pairing) == 1:
        pairing = pairing[0]
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    if len(pairing) > 2:  # pairings
        for sigma_ll, sigma_oo, delta_ii, C in pairing[:-2]:
            Delta += C * expand_tensor_product(config, sigma_ll, sigma_oo, delta_ii)  # C -- the coefficient corresponding for the right D_3 transformation properties (irrep)
        return Delta * pairing[-2]  # the overal coefficient of this irrep (can be 1 or i)

    # waves
    sigma_ll, sigma_oo, delta_ii, _ = pairing[0]
    Delta += expand_tensor_product(config, sigma_ll, sigma_oo, delta_ii)  # C -- the coefficient corresponding for the right D_3 transformation properties (irrep)
    return Delta
 

def get_total_pairing(config, pairings, var_params):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for gap, vp in zip(pairings, var_params):
        Delta += vp * combine_product_terms(config, gap)

    return Delta

def get_total_pairing_upwrapped(config, pairings_unwrapped, var_params):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for gap, vp in zip(pairings_unwrapped, var_params):
        Delta += vp * gap

    return Delta

def construct_on_site_2orb_hex(config, real = True):
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    return on_site

def construct_2orb_hex(config, NNN=True, real = True):
    '''
    each element of the result is the tensor--decomposed gap in the sublattice x orbitals x neighbors space
    the 2D--irreps go in pairs written in the single line -- one MUST include them both into the total gap

    the "real" parameter only multiplies the gap by i or not. This is because the magnitude can be complex,
    but the SR method only works with real parameters -- so in order to emulate complex magnitude, one should
    add the gap twice -- with real = True and real = False
    '''
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    global delta_hex_AB, delta_hex_BA
    global delta_hex_NNN_AA, delta_hex_NNN_BB
    delta_hex_AB = [construct_NN_delta(config, direction, geometry='hexagonal') for direction in range(1, 4)]
    delta_hex_BA = [delta.T for delta in delta_hex_AB]
    onsite = construct_onsite_delta(config)

    delta_hex_NNN_AA = [construct_NNN_delta(config, direction, geometry='hexagonal', sublattice = 0) for direction in range(1, 7)]
    delta_hex_NNN_BB = [construct_NNN_delta(config, direction, geometry='hexagonal', sublattice = 1) for direction in range(1, 7)]


    v1_AB = construct_vi_hex(1, delta_hex_AB)
    v2_AB = construct_vi_hex(2, delta_hex_AB)
    v3_AB = construct_vi_hex(3, delta_hex_AB)

    v1_BA = construct_vi_hex(1, delta_hex_BA)
    v2_BA = construct_vi_hex(2, delta_hex_BA)
    v3_BA = construct_vi_hex(3, delta_hex_BA)

    v1 = (v1_AB, v1_BA)
    v2 = (v2_AB, v2_BA)
    v3 = (v3_AB, v3_BA)

    u1 = (construct_ui_hex(1, delta_hex_NNN_AA), construct_ui_hex(1, delta_hex_NNN_BB))
    u2 = (construct_ui_hex(2, delta_hex_NNN_AA), construct_ui_hex(2, delta_hex_NNN_BB))
    u3 = (construct_ui_hex(3, delta_hex_NNN_AA), construct_ui_hex(3, delta_hex_NNN_BB))

    A1_N_singlet = [
        [(Ipauli, Ipauli, onsite, 1), factor, addstring + '(S_0)x(S_0&S_x)x(δ)'],
    ]
    print('Testing the A1_N_singlet properties')
    [check_irrep_properties(config, A1_N_singlet[i:i + 1]) for i in range(len(A1_N_singlet))]
    [check_irrep_properties(config, A1_N_singlet[i:i + 1], chiral = True) for i in range(len(A1_N_singlet))]

    A1_N_triplet = [
        [(Zpauli, iYpauli, onsite, 1), factor, addstring + '(S_z)x(iS_y&S_y)x(δ)'],
    ]
    print('Testing the A1_N_triplet properties')
    [check_irrep_properties(config, A1_N_triplet[i:i + 1]) for i in range(len(A1_N_triplet))]
    [check_irrep_properties(config, A1_N_triplet[i:i + 1], chiral = True) for i in range(len(A1_N_triplet))]

    A1_NN_singlet = [
        [(Xpauli, Ipauli, v1, 1.0), factor, addstring + '(S_x)x(S_0&S_x)x(v_1)'],
        [(iYpauli, iYpauli, v1, 1.0), factor, addstring + '(iS_y)x(iS_y&S_y)x(v_1)'],
        [(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, 1.0), factor, addstring + '[(S_x)x(S_1)x(v_2)+(S_x)x(S_2)x(v_3)]'],
    ]
    print('Testing the A1_NN_singlet properties')
    [check_irrep_properties(config, A1_NN_singlet[i:i + 1]) for i in range(len(A1_NN_singlet))]
    [check_irrep_properties(config, A1_NN_singlet[i:i + 1], chiral = True) for i in range(len(A1_NN_singlet))]
    
    A1_NN_triplet = [
        [(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, -1.0), factor, addstring + '[(iS_y)x(S_1)x(v_2)-(iS_y)x(S_2)x(v_3)]'],
    ]
    print('Testing the A1_NN_triplet properties')
    [check_irrep_properties(config, A1_NN_triplet[i:i + 1]) for i in range(len(A1_NN_triplet))]
    [check_irrep_properties(config, A1_NN_triplet[i:i + 1], chiral = True) for i in range(len(A1_NN_triplet))]

    A2_N_singlet = [
        [(Zpauli, Ipauli, onsite, 1), factor, addstring + '(S_z)x(S_0&S_x)x(δ)'],
    ]
    print('Testing the A2_N_singlet properties')
    [check_irrep_properties(config, A2_N_singlet[i:i + 1]) for i in range(len(A2_N_singlet))]
    [check_irrep_properties(config, A2_N_singlet[i:i + 1], chiral = True) for i in range(len(A2_N_singlet))]

    A2_N_triplet = [
        [(Ipauli, iYpauli, onsite, 1), factor, addstring + '(S_0)x(iS_y&S_y)x(δ)'],
    ]
    print('Testing the A2_N_triplet properties')
    [check_irrep_properties(config, A2_N_triplet[i:i + 1]) for i in range(len(A2_N_triplet))]
    [check_irrep_properties(config, A2_N_triplet[i:i + 1], chiral = True) for i in range(len(A2_N_triplet))]


    A2_NN_singlet = [
        [(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, -1.0), factor, addstring + '[(S_x)x(S_1)x(v_2)-(S_x)x(S_2)x(v_3)]'],
    ]
    print('Testing the A2_NN_singlet properties')
    [check_irrep_properties(config, A2_NN_singlet[i:i + 1]) for i in range(len(A2_NN_singlet))]
    [check_irrep_properties(config, A2_NN_singlet[i:i + 1], chiral = True) for i in range(len(A2_NN_singlet))]

    A2_NN_triplet = [
        [(Xpauli, iYpauli, v1, 1.0), factor, addstring + '(S_x)x(iS_y&S_y)x(v_1)'],
        [(iYpauli, Ipauli, v1, 1.0), factor, addstring + '(iS_y)x(S_0&S_x)x(v_1)'],
        [(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, 1.0), factor, addstring + '[(iS_y)x(S_1)x(v_2)+(iS_y)x(S_2)x(v_3)]'],
    ]
    print('Testing the A2_NN_triplet properties')
    [check_irrep_properties(config, A2_NN_triplet[i:i + 1]) for i in range(len(A2_NN_triplet))]
    [check_irrep_properties(config, A2_NN_triplet[i:i + 1], chiral = True) for i in range(len(A2_NN_triplet))]

    E_N_singlet = [
        [(Ipauli, sigma_1, onsite, 1.0), factor, addstring + '(S_0)x(S_1)x(δ)'], [(Ipauli, sigma_2, onsite, 1.0), factor, addstring + '(S_0)x(S_2)x(δ)'],
        [(Zpauli, sigma_1, onsite, 1.0), factor, addstring + '(S_z)x(S_1)x(δ)'], [(Zpauli, sigma_2, onsite, 1.0), factor, addstring + '(S_z)x(S_2)x(δ)'],
    ]
    print('Testing the E_N_singlet properties')
    [check_irrep_properties(config, E_N_singlet[2 * i:2 * i + 2]) for i in range(len(E_N_singlet) // 2)]
    [check_irrep_properties(config, E_N_singlet[2 * i:2 * i + 2], chiral = True) for i in range(len(E_N_singlet) // 2)]

    E_NN_singlet = [
        [(Xpauli, sigma_1, v1, 1.0), factor, addstring + '(S_x)x(S_1)x(v_1)'], [(Xpauli, sigma_2, v1, 1.0), factor, addstring + '(S_x)x(S_2)x(v_1)'],
        [(Xpauli, Ipauli, v2, 1.0), factor, addstring + '(S_x)x(S_0&S_x)x(v_2)'], [(Xpauli, Ipauli, v3, 1.0), factor, addstring + '(S_x)x(S_0&S_x)x(v_3)'],
        [(iYpauli, iYpauli, v2, 1.0), factor, addstring + '(iS_y)x(iS_y&S_y)x(v_2)'], [(iYpauli, iYpauli, v3, 1.0), factor, addstring + '(iS_y)x(iS_y&S_y)x(v_3)'],
        [(Xpauli, sigma_1, v3, 1.0), factor, addstring + '(S_x)x(S_1)x(v_3)'], [(Xpauli, sigma_2, v2, 1.0), factor, addstring + '(S_x)x(S_2)x(v_2)'],
    ]
    print('Testing the E_NN_singlet properties')
    [check_irrep_properties(config, E_NN_singlet[2 * i:2 * i + 2]) for i in range(len(E_NN_singlet) // 2)]
    [check_irrep_properties(config, E_NN_singlet[2 * i:2 * i + 2], chiral = True) for i in range(len(E_NN_singlet) // 2)]

    E_NN_triplet = [
        [(Xpauli, iYpauli, v2, 1.0), factor, addstring + '(S_x)x(iS_y)x(v_2)'], [(Xpauli, iYpauli, v3, 1.0), factor, addstring + '(S_x)x(iS_y)x(v_3)'],
        [(iYpauli, sigma_1, v3, 1.0), factor, addstring + '(iS_y)x(S_1)x(v_3)'], [(iYpauli, sigma_2, v2, 1.0), factor, addstring + '(iS_y)x(S_2)x(v_2)'],
        [(iYpauli, Ipauli, v2, 1.0), factor, addstring + '(iS_y)x(S_0&S_x)x(v_2)'], [(iYpauli, Ipauli, v3, 1.0), factor, addstring + '(iS_y)x(S_0&S_x)x(v_3)'],
        [(iYpauli, sigma_1, v1, 1.0), factor, addstring + '(iS_y)x(S_1)x(v_1)'], [(iYpauli, sigma_2, v1, 1.0), factor, addstring + '(iS_y)x(S_2)x(v_1)'],
    ]
    print('Testing the E_NN_triplet properties')
    [check_irrep_properties(config, E_NN_triplet[2 * i:2 * i + 2]) for i in range(len(E_NN_triplet) // 2)]
    [check_irrep_properties(config, E_NN_triplet[2 * i:2 * i + 2], chiral = True) for i in range(len(E_NN_triplet) // 2)]

    if not NNN:
        return A1_N_singlet, A1_N_triplet, A2_N_singlet, A2_N_triplet, E_N_singlet, \
               A1_NN_singlet, A1_NN_triplet, A2_NN_singlet, A2_NN_triplet, E_NN_singlet, E_NN_triplet


    A1_NNN_singlet = [
        [(Ipauli, Ipauli, u1, 1.0), factor, addstring + '(S_0)x(S_0&S_x)x(u_1)'],
        [(Ipauli, sigma_1, u2, 1.0), (Ipauli, sigma_2, u3, 1.0), factor, addstring + '[(S_0)x(S_1)x(u_2)+(S_0)x(S_2)x(u_3)]'],
        [(Zpauli, sigma_1, u2, 1.0), (Zpauli, sigma_2, u3, -1.0), factor, addstring + '[(S_z)x(S_1)x(u_2)-(S_z)x(S_2)x(u_3)]'],
    ]
    print('Testing the A1_NNN_singlet properties')
    [check_irrep_properties(config, A1_NNN_singlet[i:i + 1]) for i in range(len(A1_NNN_singlet))]
    [check_irrep_properties(config, A1_NNN_singlet[i:i + 1], chiral = True) for i in range(len(A1_NNN_singlet))]

    A1_NNN_triplet = [
        [(Zpauli, iYpauli, u1, 1.0), factor, addstring + '(S_z)x(iS_y&S_y)x(u_1)'],
    ]
    print('Testing the A1_NNN_triplet properties')
    [check_irrep_properties(config, A1_NNN_triplet[i:i + 1]) for i in range(len(A1_NNN_triplet))]
    [check_irrep_properties(config, A1_NNN_triplet[i:i + 1], chiral = True) for i in range(len(A1_NNN_triplet))]

    A2_NNN_singlet = [
        [(Ipauli, sigma_1, u2, 1.0), (Ipauli, sigma_2, u3, -1.0), factor, addstring + '[(S_0)x(S_1)x(u_2)-(S_0)x(S_2)x(u_3)]'],
        [(Zpauli, Ipauli, u1, 1.0), factor, addstring + '(S_z)x(S_0&S_x)x(u_1)'],
        [(Zpauli, sigma_1, u2, 1.0), (Zpauli, sigma_2, u3, 1.0), factor, addstring + '[(S_z)x(S_1)x(u_2)+(S_z)x(S_2)x(u_3)]'],
    ]
    print('Testing the A2_NNN_singlet properties')
    [check_irrep_properties(config, A2_NNN_singlet[i:i + 1]) for i in range(len(A2_NNN_singlet))]
    [check_irrep_properties(config, A2_NNN_singlet[i:i + 1], chiral = True) for i in range(len(A2_NNN_singlet))]

    A2_NNN_triplet = [
        [(Ipauli, iYpauli, u1, 1.0), factor, addstring + '(S_0)x(iS_y&S_y)x(u_1)'],
    ]
    print('Testing the A2_NNN_triplet properties')
    [check_irrep_properties(config, A2_NNN_triplet[i:i + 1]) for i in range(len(A2_NNN_triplet))]
    [check_irrep_properties(config, A2_NNN_triplet[i:i + 1], chiral = True) for i in range(len(A2_NNN_triplet))]

    E_NNN_singlet = [
        [(Ipauli, sigma_1, u1, 1.0), factor, addstring + '(S_0)x(S_1)x(u_1)'], [(Ipauli, sigma_2, u1, 1.0), factor, addstring + '(S_0)x(S_2)x(u_1)'],
        [(Ipauli, Ipauli, u2, 1.0), factor, addstring + '(S_0)x(S_0&S_x)x(u_2)'], [(Ipauli, Ipauli, u3, 1.0), factor, addstring + '(S_0)x(S_0&S_x)x(u_3)'],
        [(Ipauli, sigma_1, u3, 1.0), factor, addstring + '(S_0)x(S_1)x(u_3)'], [(Ipauli, sigma_2, u2, 1.0), factor, addstring + '(S_0)x(S_2)x(u_2)'],
        [(Zpauli, iYpauli, u2, 1.0), factor, addstring + '(S_z)x(iS_y&S_y)x(u_2)'], [(Zpauli, iYpauli, u3, 1.0), factor, addstring + '(S_z)x(iS_y&S_y)x(u_3)'],
    ]
    print('Testing the E_NNN_singlet properties')
    [check_irrep_properties(config, E_NNN_singlet[2 * i:2 * i + 2]) for i in range(len(E_NNN_singlet) // 2)]
    [check_irrep_properties(config, E_NNN_singlet[2 * i:2 * i + 2], chiral = True) for i in range(len(E_NNN_singlet) // 2)]

    E_NNN_triplet = [
        [(Zpauli, sigma_1, u1, 1.0), factor, addstring + '(S_z)x(S_1)x(u_1)'], [(Zpauli, sigma_2, u1, 1.0), factor, addstring + '(S_z)x(S_2)x(u_1)'],
        [(Zpauli, Ipauli, u2, 1.0), factor, addstring + '(S_z)x(S_0&S_x)x(u_2)'], [(Zpauli, Ipauli, u3, 1.0), factor, addstring + '(S_z)x(S_0&S_x)x(u_3)'],
        [(Ipauli, iYpauli, u2, 1.0), factor, addstring + '(S_0)x(iS_y)x(u_2)'], [(Ipauli, iYpauli, u3, 1.0), factor, addstring + '(S_0)x(iS_y)x(u_3)'],
        [(Zpauli, sigma_1, u3, 1.0), factor, addstring + '(S_z)x(S_1)x(u_3)'], [(Zpauli, sigma_2, u2, 1.0), factor, addstring + '(S_z)x(S_2)x(u_2)'],
    ]
    print('Testing the E_NNN_triplet properties')
    [check_irrep_properties(config, E_NNN_triplet[2 * i:2 * i + 2]) for i in range(len(E_NNN_triplet) // 2)]
    [check_irrep_properties(config, E_NNN_triplet[2 * i:2 * i + 2], chiral = True) for i in range(len(E_NNN_triplet) // 2)]


    check_P_symmetry(config, E_NN_singlet[2], E_NN_triplet[0])
    check_P_symmetry(config, E_NN_singlet[3], E_NN_triplet[1])
    #check_P_symmetry(config, A1_NN_singlet[0], A2_NN_triplet[1])

    return A1_N_singlet, A1_N_triplet, A2_N_singlet, A2_N_triplet, E_N_singlet, \
           A1_NN_singlet, A1_NN_triplet, A2_NN_singlet, A2_NN_triplet, E_NN_singlet, E_NN_triplet, \
           A1_NNN_singlet, A1_NNN_triplet, A2_NNN_singlet, A2_NNN_triplet, E_NNN_singlet, E_NNN_triplet

def construct_1orb_hex(config, real = True):
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    global delta_hex_AB, delta_hex_BA
    delta_hex_AB = [construct_NN_delta(config, direction, geometry='hexagonal') for direction in range(1, 4)]
    delta_hex_BA = [delta.conj().T for delta in delta_hex_AB]


    v1_AB = construct_vi_hex(1, delta_hex_AB)
    v2_AB = construct_vi_hex(2, delta_hex_AB)
    v3_AB = construct_vi_hex(3, delta_hex_AB)

    v1_BA = construct_vi_hex(1, delta_hex_BA)
    v2_BA = construct_vi_hex(2, delta_hex_BA)
    v3_BA = construct_vi_hex(3, delta_hex_BA)

    v1 = (v1_AB, v1_BA)
    v2 = (v2_AB, v2_BA)
    v3 = (v3_AB, v3_BA)
    onsite = construct_onsite_delta(config)

    A1_N_singlet = [
        [(Ipauli, identity, onsite, 1), factor, addstring + 'S_0xIxδ'],
    ]
    print('Testing the A1_N_singlet properties')
    [check_irrep_properties(config, A1_N_singlet[i:i + 1]) for i in range(len(A1_N_singlet))]

    A2_N_singlet = [
        [(Zpauli, identity, onsite, 1), factor, addstring + 'S_zxIxδ'],
    ]
    print('Testing the A2_N_singlet properties')
    [check_irrep_properties(config, A2_N_singlet[i:i + 1]) for i in range(len(A2_N_singlet))]

    A1_NN_singlet = [
        [(Xpauli, identity, v1, 1.0), factor, addstring + '(S_x)xIxv_1'],
    ]
    print('Testing the A1_NN_singlet properties')
    [check_irrep_properties(config, A1_NN_singlet[i:i + 1]) for i in range(len(A1_NN_singlet))]

    A2_NN_triplet = [
        [(iYpauli, identity, v1, 1.0), factor, addstring + '(iS_y)xIxv_1'],
    ]
    print('Testing the A2_NN_triplet properties')
    [check_irrep_properties(config, A2_NN_triplet[i:i + 1]) for i in range(len(A2_NN_triplet))]

    E_NN_singlet = [
        [(Xpauli, identity, v2, 1.0), factor, addstring + '(S_x)xIxv_2'],
        [(Xpauli, identity, v3, 1.0), factor, addstring + '(S_x)xIxv_3']
    ]
    print('Testing the E_NN_singlet properties')
    [check_irrep_properties(config, E_NN_singlet[i:i + 2]) for i in range(len(E_NN_singlet) // 2)]

    E_NN_triplet = [
        [(iYpauli, identity, v2, 1.0), factor, addstring + '(iS_y)xIxv_2'],
        [(iYpauli, identity, v3, 1.0), factor, addstring + '(iS_y)xIxv_3']
    ]
    print('Testing the E_NN_triplet properties')
    [check_irrep_properties(config, E_NN_triplet[i:i + 2]) for i in range(len(E_NN_triplet) // 2)]
    return A1_N_singlet, A2_N_singlet, A1_NN_singlet, A2_NN_triplet, E_NN_singlet, E_NN_triplet


def construct_1orb_square(config, real = True):
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    global delta_square
    delta_square = [construct_NN_delta(config, direction, geometry='square') for direction in range(1, 5)]

    v1 = construct_vi_square(1)
    v2 = construct_vi_square(2)
    v3 = construct_vi_square(3)
    v4 = construct_vi_square(4)

    onsite = construct_onsite_delta(config)
    
    A1_N_singlet = [
        [(identity, identity, onsite, 1), factor, addstring + 'IxIxδ'],
    ]
    print('Testing the A1_N_singlet properties')
    [check_irrep_properties(config, A1_N_singlet[i:i + 1]) for i in range(len(A1_N_singlet))]

    A1_NN_singlet = [
        [(identity, identity, v1, 1.0), factor, addstring + 'IxIxv_1'],
    ]
    print('Testing the A1_NN_singlet properties')
    [check_irrep_properties(config, A1_NN_singlet[i:i + 1]) for i in range(len(A1_NN_singlet))]

    A2_NN_singlet = [
        [(identity, identity, v4, 1.0), factor, addstring + 'IxIxv_4'],
    ]
    print('Testing the A1_NN_singlet properties')
    [check_irrep_properties(config, A2_NN_singlet[i:i + 1]) for i in range(len(A2_NN_singlet))]

    E_NN_triplet = [
        [(identity, identity, v2, 1.0), factor, addstring + 'IxIxv_2'],
        [(identity, identity, v3, 1.0), factor, addstring + 'IxIxv_3'],
    ]
    print('Testing the E_NN_triplet properties')
    [check_irrep_properties(config, E_NN_triplet[i:i + 2]) for i in range(len(E_NN_triplet) // 2)]
    return A1_N_singlet, A1_NN_singlet, A2_NN_singlet, E_NN_triplet


def get_C2y_symmetry_map(config, chiral = False):
    if config.n_sublattices == 2:
        geometry = 'hexagonal'
    else:
        geometry = 'square'

    mapping = np.zeros((config.total_dof // 2, config.total_dof // 2)) + 0.0j  # trivial mapping

    for preindex in range(config.total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)     

        if config.n_orbitals == 2:
            if not chiral:
                orbit_image = orbit_preimage
                coefficient = -1.0 if orbit_image == 0 else 1.0
            else:
                orbit_image = 1 - orbit_preimage
                coefficient = -1.0
        else:
            orbit_image = orbit_preimage
            coefficient = 1.0

        r_preimage = np.array(models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], geometry))
        if geometry == 'hexagonal':
            r_preimage -= np.array([1. / np.sqrt(3) / 2, 0.0])
            r_image = np.array([-r_preimage[0], r_preimage[1]]) + np.array([1. / np.sqrt(3) / 2, 0.0])
        else:
            r_image = np.array([-r_preimage[0], r_preimage[1]])

        x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, geometry)
        x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
        x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)
        
        index = models.to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                           config.Ls, config.n_orbitals, config.n_sublattices)

        mapping[preindex, index] += coefficient

    assert np.sum(np.abs(mapping.dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_2y^2 = I
    return mapping + 0.0j

def get_C3z_symmetry_map(config, chiral = False):
    assert config.n_sublattices == 2
    geometry = 'hexagonal'

    mapping = np.zeros((config.total_dof // 2, config.total_dof // 2)) + 0.0j  # trivial mapping
    rotation_matrix = np.array([[np.cos(2 * np.pi / 3.), np.sin(2 * np.pi / 3.)], \
                                [-np.sin(2 * np.pi / 3.), np.cos(2 * np.pi / 3.)]])
    if config.n_orbitals == 2:
        if not chiral:
            rotation_matrix_orbital = rotation_matrix
        else:
            rotation_matrix_orbital = np.diag([np.exp(2.0j * np.pi / 3), np.exp(-2.0j * np.pi / 3)])
    else:
        rotation_matrix_orbital = np.eye(1)

    for preindex in range(config.total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)

        orbit_preimage_vector = np.zeros(config.n_orbitals); orbit_preimage_vector[orbit_preimage] = 1.
        r_preimage = models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], geometry)

        orbit_image_vector = np.einsum('ij,j->i', rotation_matrix_orbital, orbit_preimage_vector)

        r_image = np.einsum('ij,j->i', rotation_matrix, r_preimage)
        
        x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, geometry)

        x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
        x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)

        for orbit_image in range(config.n_orbitals):
            coefficient = orbit_image_vector[orbit_image]
            index = models.to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               config.Ls, config.n_orbitals, config.n_sublattices)
            mapping[preindex, index] += coefficient
    assert np.sum(np.abs(mapping.dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_3z^3 = I
    return mapping + 0.0j


def get_C4z_symmetry_map(config):
    assert config.n_sublattices == 1
    geometry = 'square'

    mapping = np.zeros((config.total_dof // 2, config.total_dof // 2)) + 0.0j  # trivial mapping
    rotation_matrix = np.array([[np.cos(2 * np.pi / 4.), np.sin(2 * np.pi / 4.)], \
                                [-np.sin(2 * np.pi / 4.), np.cos(2 * np.pi / 4.)]])
    if config.n_orbitals == 2:
        rotation_matrix_orbital = rotation_matrix
    else:
        rotation_matrix_orbital = np.eye(1)

    for preindex in range(config.total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)

        orbit_preimage_vector = np.zeros(config.n_orbitals); orbit_preimage_vector[orbit_preimage] = 1.
        r_preimage = models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], geometry)

        orbit_image_vector = np.einsum('ij,j->i', rotation_matrix_orbital, orbit_preimage_vector)

        r_image = np.einsum('ij,j->i', rotation_matrix, r_preimage)
        
        x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, geometry)

        x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
        x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)

        for orbit_image in range(config.n_orbitals):
            coefficient = orbit_image_vector[orbit_image]
            index = models.to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               config.Ls, config.n_orbitals, config.n_sublattices)
            mapping[preindex, index] += coefficient
    assert np.sum(np.abs(mapping.dot(mapping).dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_4z^4 = I
    return mapping + 0.0j

def check_P_symmetry(config, gap_singlet, gap_triplet):
    def norm_sc(b, a):
        return np.sum(a * b.conj()) / np.sum(np.abs(b ** 2))
    df = config.total_dof
    P = np.ones(df)
    P[np.arange(df // 2, df, 2) + 1] = -1
    P = np.diag(P)
    assert np.allclose(np.eye(df), P.dot(P))

    gs = combine_product_terms(config, gap_singlet)
    gt = combine_product_terms(config, gap_triplet)
    gs = models.xy_to_chiral(gs, 'pairing', config, chiral = True)
    gt = models.xy_to_chiral(gt, 'pairing', config, chiral = True)
    print(np.unique(gs), np.unique(gt))

    delta_s_extended = np.zeros((df, df), dtype=np.complex128)
    delta_s_extended[df // 2:, :df // 2] = gs
    delta_s_extended[:df // 2, df // 2:] = -gs.T

    delta_t_extended = np.zeros((df, df), dtype=np.complex128)
    delta_t_extended[df // 2:, :df // 2] = gt
    delta_t_extended[:df // 2, df // 2:] = -gt.T

    delta_s_extended = P.T.dot(delta_s_extended).dot(P)
    coeff = norm_sc(delta_s_extended.flatten(), delta_t_extended.flatten())
    print(coeff)
    assert np.isclose(np.abs(coeff), 1.0)
    print('P^T {:s} P = {:s}'.format(gap_singlet[-1], gap_triplet[-1]))


def check_irrep_properties(config, irrep, term_type = 'pairing', chiral = False):
    return
    global C2y_symmetry_map, C3z_symmetry_map, C4z_symmetry_map
    global C2y_symmetry_map_chiral, C3z_symmetry_map_chiral
    global name_group_dict
    if not config.tests:
        return

    if chiral:
        reflection = C2y_symmetry_map_chiral
    else:
        reflection = C2y_symmetry_map

    if config.n_sublattices == 2:
        if chiral:
            rotation = C3z_symmetry_map_chiral
        else:
            rotation = C3z_symmetry_map
    else:
        rotation = C4z_symmetry_map

    def norm_sc(b, a):
        return np.sum(a * b.conj()) / np.sum(np.abs(b ** 2))
    def in_close(a, array):
        return np.any(np.abs(array - a) < 1e-5)

    for irr in irrep:
        print(irr[-1])
        pairing_group = 0

        if type(irr[0]) == tuple:
            print('λ_spin = {:d}'.format(check_parity(config, irr)))
            pairing_group += (check_parity(config, irr) + 1) // 2

        gap = combine_product_terms(config, irr) if type(irr[0]) == tuple else irr[0]
        i = 1
        for j in range(gap.shape[1] // 2):
            if np.sum(np.abs(gap[2 * i:2 * i + 2, 2 * j:2 * j + 2])) > 0:
                print(gap[2 * i:2 * i + 2, 2 * j:2 * j + 2], 'before', i, j, chiral)
            break
        gap = models.xy_to_chiral(gap, term_type, config, chiral = chiral)  # can do nothing or make chiral transform
        i = 1
        for j in range(gap.shape[1] // 2):
            if np.sum(np.abs(gap[2 * i:2 * i + 2, 2 * j:2 * j + 2])) > 0:
                print(gap[2 * i:2 * i + 2, 2 * j:2 * j + 2], 'after', i, j, chiral)
            break

        if term_type != 'pairing':
            gap_image = (reflection).dot(gap).dot(reflection.conj().T)
        else:
            gap_image = (reflection).dot(gap).dot(reflection.T)

        norm = np.sum(np.abs(gap_image ** 2))
        gap_image = gap_image.flatten()
        for irr_decompose in irrep:
            gap_decompose = combine_product_terms(config, irr_decompose) if type(irr_decompose[0]) == tuple else irr_decompose[0]
            gap_decompose = models.xy_to_chiral(gap_decompose, term_type, config, chiral = chiral)  # can do nothing or make chiral transform
            coeff = norm_sc(gap_decompose.flatten(), gap_image)
            # print('<{:s}|M|{:s}> = '.format(irr[-1], irr_decompose[-1]) + str(coeff))
            if np.abs(coeff) > 1e-5:
                if not (np.isclose(coeff, 1) or np.isclose(coeff, -1)):
                    print('Strange eigenvalue M')
                    exit(-1)
                print('λ_M = {:d}'.format(int(np.sign(coeff))))

                pairing_group += (int(np.sign(coeff)) + 1) // 2 * 2
            gap_image = gap_image - gap_decompose.flatten() * coeff
            norm = np.sum(np.abs(gap_image ** 2))
        assert norm < 1e-5

        if term_type != 'pairing':
            gap_image = (rotation).dot(gap).dot(rotation.conj().T)
        else:
            gap_image = (rotation).dot(gap).dot(rotation.T)
        norm = np.sum(np.abs(gap_image ** 2))

        gap_image = gap_image.flatten()
        for irr_decompose in irrep:
            gap_decompose = combine_product_terms(config, irr_decompose) if type(irr_decompose[0]) == tuple else irr_decompose[0]
            gap_decompose = models.xy_to_chiral(gap_decompose, term_type, config, chiral = chiral)  # can do nothing or make chiral transform
            coeff = norm_sc(gap_decompose.flatten(), gap_image.flatten())
            # print('<{:s}|R|{:s}> = '.format(irr[-1], irr_decompose[-1]) + str(coeff))
            if np.abs(coeff) > 1e-5:
                if np.isclose(coeff, np.exp(2.0j * np.pi / 3)):
                    print('λ_R = ω')
                    pairing_group += 4
                elif np.isclose(coeff, np.exp(-2.0j * np.pi / 3)):
                    print('λ_R = ω*')
                    pairing_group += 8
                elif np.isclose(coeff, 1):
                    print('λ_R = 1')
                    pairing_group += 0
                else:
                    print(coeff)
                    print('Strange eigenvalue R')
                    exit(-1)
            gap_image = gap_image - gap_decompose.flatten() * coeff
            norm = np.sum(np.abs(gap_image ** 2))
        assert norm < 1e-5
        print('group = {:d}'.format(pairing_group))

        if len(irr) > 2:
            name_group_dict[irr[-1]] = pairing_group  # only if not waves
    print('test passed'); 

def check_parity(config, pairing):
    gap = combine_product_terms(config, pairing)
    if np.allclose(gap + gap.T, 0):
        return -1
    elif np.allclose(gap - gap.T, 0):
        return +1
    print('Non parity-symmetric pairing' + gap[-1] + 'appeared the set')
    exit(-1)

twoorb_hex_A1_N_singlet = None; twoorb_hex_A1_N_triplet = None;
twoorb_hex_A2_N_singlet = None; twoorb_hex_A2_N_triplet = None;
twoorb_hex_E_N_singlet = None;

twoorb_hex_A1_NN_singlet = None; twoorb_hex_A1_NN_triplet = None; 
twoorb_hex_A2_NN_singlet = None; twoorb_hex_A2_NN_triplet = None;
twoorb_hex_E_NN_singlet = None; twoorb_hex_E_NN_triplet = None; 

twoorb_hex_A1_NNN_singlet = None; twoorb_hex_A1_NNN_triplet = None; 
twoorb_hex_A2_NNN_singlet = None; twoorb_hex_A2_NNN_triplet = None;
twoorb_hex_E_NNN_singlet = None; twoorb_hex_E_NNN_triplet = None; 

twoorb_hex_all = None;
twoorb_hex_all_dqmc = None;

oneorb_hex_A1_N_singlet = None; oneorb_hex_A2_N_singlet = None; 
oneorb_hex_A1_NN_singlet = None; oneorb_hex_A2_NN_triplet = None; 
oneorb_hex_E_NN_singlet = None; oneorb_hex_E_NN_triplet = None;

oneorb_hex_all = None;

oneorb_square_A1_N_singlet = None; oneorb_square_A2_N_singlet = None; 
oneorb_square_A1_NN_singlet = None; oneorb_square_A2_NN_triplet = None; 
oneorb_square_E_NN_singlet = None; oneorb_square_E_NN_triplet = None;
oneorb_square_A1_N_singlet = None; oneorb_square_A1_NN_singlet = None;
oneorb_square_A2_NN_singlet = None; oneorb_square_E_NN_triplet = None;

oneorb_square_all = None

C2y_symmetry_map = None; C2y_symmetry_map_chiral = None;
C3z_symmetry_map = None; C3z_symmetry_map_chiral = None;
C4z_symmetry_map = None;

def obtain_all_pairings(config):
    global C2y_symmetry_map, C3z_symmetry_map, C4z_symmetry_map, C3z_symmetry_map_chiral, C2y_symmetry_map_chiral
    global twoorb_hex_A1_N_singlet, twoorb_hex_A1_N_triplet, twoorb_hex_A2_N_singlet, twoorb_hex_A2_N_triplet, twoorb_hex_E_N_singlet, \
           twoorb_hex_A1_NN_singlet, twoorb_hex_A1_NN_triplet, twoorb_hex_A2_NN_singlet, twoorb_hex_A2_NN_triplet, \
           twoorb_hex_E_NN_singlet, twoorb_hex_E_NN_triplet,\
           twoorb_hex_A1_NNN_singlet, twoorb_hex_A1_NNN_triplet, twoorb_hex_A2_NNN_singlet, twoorb_hex_A2_NNN_triplet, \
           twoorb_hex_E_NNN_singlet, twoorb_hex_E_NNN_triplet
    global oneorb_hex_A1_N_singlet, oneorb_hex_A2_N_singlet, oneorb_hex_A1_NN_singlet, oneorb_hex_A2_NN_triplet, \
           oneorb_hex_E_NN_singlet, oneorb_hex_E_NN_triplet

    global oneorb_square_A1_N_singlet, oneorb_square_A1_NN_singlet, oneorb_square_A2_NN_singlet, oneorb_square_E_NN_triplet

    global twoorb_hex_all, oneorb_hex_all, oneorb_square_all, twoorb_hex_all_dqmc

    C2y_symmetry_map = get_C2y_symmetry_map(config)
    if config.n_orbitals == 2 and config.n_sublattices == 2:
        C3z_symmetry_map = get_C3z_symmetry_map(config)
        C3z_symmetry_map_chiral = get_C3z_symmetry_map(config, chiral=True)
        C2y_symmetry_map_chiral = get_C2y_symmetry_map(config, chiral=True)

        twoorb_hex_A1_N_singlet, twoorb_hex_A1_N_triplet, twoorb_hex_A2_N_singlet, twoorb_hex_A2_N_triplet, twoorb_hex_E_N_singlet, \
            twoorb_hex_A1_NN_singlet, twoorb_hex_A1_NN_triplet, twoorb_hex_A2_NN_singlet, twoorb_hex_A2_NN_triplet, \
            twoorb_hex_E_NN_singlet, twoorb_hex_E_NN_triplet, \
            twoorb_hex_A1_NNN_singlet, twoorb_hex_A1_NNN_triplet, twoorb_hex_A2_NNN_singlet, twoorb_hex_A2_NNN_triplet, \
           twoorb_hex_E_NNN_singlet, twoorb_hex_E_NNN_triplet = construct_2orb_hex(config, NNN = True, real = True)

        twoorb_hex_all_dqmc = twoorb_hex_A1_N_singlet + twoorb_hex_A1_N_triplet + twoorb_hex_A2_N_singlet + twoorb_hex_A2_N_triplet + twoorb_hex_E_N_singlet + \
            twoorb_hex_A1_NN_singlet + twoorb_hex_A1_NN_triplet + twoorb_hex_A2_NN_singlet + twoorb_hex_A2_NN_triplet + \
            twoorb_hex_E_NN_singlet + twoorb_hex_E_NN_triplet# + \
           # twoorb_hex_A1_NNN_singlet + twoorb_hex_A1_NNN_triplet + twoorb_hex_A2_NNN_singlet + twoorb_hex_A2_NNN_triplet + \
           #twoorb_hex_E_NNN_singlet + twoorb_hex_E_NNN_triplet

        _, _, _, _, twoorb_hex_E_N_singlet_im, _, _, _, _, twoorb_hex_E_NN_singlet_im, twoorb_hex_E_NN_triplet_im = \
                                                                 construct_2orb_hex(config, NNN = False, real = False)

        twoorb_hex_all = [[]] + \
                         [[gap] for gap in twoorb_hex_A1_N_singlet] + \
                         [[gap] for gap in twoorb_hex_A1_N_triplet] + \
                         [[gap] for gap in twoorb_hex_A2_N_singlet] + \
                         [[gap] for gap in twoorb_hex_A2_N_triplet] + \
                         [[gap] for gap in twoorb_hex_A1_NN_singlet] + \
                         [[gap] for gap in twoorb_hex_A1_NN_triplet] + \
                         [[gap] for gap in twoorb_hex_A2_NN_singlet] + \
                         [[gap] for gap in twoorb_hex_A2_NN_triplet] + \
                         [[irrep_re_1, irrep_re_2, irrep_im_1] for irrep_re_1, irrep_re_2, irrep_im_1 in \
                             zip(twoorb_hex_E_N_singlet[0::2], twoorb_hex_E_N_singlet[1::2], twoorb_hex_E_N_singlet_im[0::2])] + \
                         [[irrep_re_1, irrep_re_2, irrep_im_1] for irrep_re_1, irrep_re_2, irrep_im_1 in \
                             zip(twoorb_hex_E_NN_singlet[0::2], twoorb_hex_E_NN_singlet[1::2], twoorb_hex_E_NN_singlet_im[0::2])] + \
                         [[irrep_re_1, irrep_re_2, irrep_im_1] for irrep_re_1, irrep_re_2, irrep_im_1 in \
                             zip(twoorb_hex_E_NN_triplet[0::2], twoorb_hex_E_NN_triplet[1::2], twoorb_hex_E_NN_triplet_im[0::2])]

        for idx, element in enumerate(twoorb_hex_all):
            need_cut = False
            for gap in element:
                if 'S_1' in gap[-1] or 'S_2' in gap[-1]:
                    if len(gap[-1]) < 30:
                        need_cut = True
            if need_cut:
                del twoorb_hex_all[idx][-1]
        for idx, element in enumerate(twoorb_hex_all):
            print('Irrep optimisation No. {:d}'.format(idx))
            for gap in element:
                print(gap[-1])
            print(' ')

        print('there are in total {:d} different irreps in the Koshino model'.format(len(twoorb_hex_all)))
        names = []
        pairings = []
        for irrep in twoorb_hex_all:
            for pairing in irrep:
                pairings.append(combine_product_terms(config, pairing))
                names.append(pairing[-1])
        np.save('all_hex_pairings.npy', np.array(pairings))
        np.save('all_hex_names.npy', np.array(names))

        return


    if config.n_orbitals == 1 and config.n_sublattices == 2:
        C3z_symmetry_map = get_C3z_symmetry_map(config)
        oneorb_hex_A1_N_singlet, oneorb_hex_A2_N_singlet, oneorb_hex_A1_NN_singlet, oneorb_hex_A2_NN_triplet, \
            oneorb_hex_E_NN_singlet, oneorb_hex_E_NN_triplet = construct_1orb_hex(config, real = True)
        
        oneorb_hex_all = oneorb_hex_A1_N_singlet + oneorb_hex_A2_N_singlet + oneorb_hex_A1_NN_singlet + oneorb_hex_A2_NN_triplet + \
            oneorb_hex_E_NN_singlet + oneorb_hex_E_NN_triplet

        names = []
        pairings = []
        for pairing in oneorb_hex_all:
            pairings.append(combine_product_terms(config, pairing))
            names.append(pairing[-1])
        np.save('all_hex_1orb_pairings.npy', np.array(pairings))
        np.save('all_hex_1orb_names.npy', np.array(names))
        return


    if config.n_orbitals == 1 and config.n_sublattices == 1:
        C4z_symmetry_map = get_C4z_symmetry_map(config)
        oneorb_square_A1_N_singlet, oneorb_square_A1_NN_singlet, oneorb_square_A2_NN_singlet, \
            oneorb_square_E_NN_triplet = construct_1orb_square(config, real = True)
        oneorb_square_all = oneorb_square_A1_N_singlet + oneorb_square_A1_NN_singlet + oneorb_square_A2_NN_singlet + \
            oneorb_square_E_NN_triplet
        return
    raise NotImplementedError()
