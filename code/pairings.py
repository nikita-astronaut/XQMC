import numpy as np
import models
from copy import deepcopy

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

sigma_1 = (Zpauli + 1.0j * Xpauli) / np.sqrt(2)
sigma_2 = (Zpauli - 1.0j * Xpauli) / np.sqrt(2)

delta_hex_AB, delta_hex_BA = [], []
delta_square = []

identity = np.array([[1]])

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

            if direction == models.nearest_neighbor(r1, r2, config.Ls, geometry, return_direction = True)[1]:
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

def expand_tensor_product(config, sigma_l1l2, sigma_o1o2, delta_ij):
    '''
        returns the matrix of the dimention L x L (L -- total number of sites including orbit),
        sigma_ll \\otimes sigma_oo \\otimes delta_ii
    '''
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for first in range(Delta.shape[0]):
        for second in range(Delta.shape[1]):
            orbit1, sublattice1, x1, y1 = models.from_linearized_index(deepcopy(first), \
                                                 config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), \
                                                 config.Ls, config.n_orbitals, config.n_sublattices)
            space1 = (x1 * config.Ls + y1) * config.n_sublattices + sublattice1
            space2 = (x2 * config.Ls + y2) * config.n_sublattices + sublattice2

            if sublattice2 - sublattice1 == 1:  # AB pairings (only in the hexagonal case)
                delta_s1s2 = delta_ij[0]
            elif sublattice2 - sublattice1 == -1:  # BA pairings (only in the hexagonal case)
                delta_s1s2 = delta_ij[1]
            else:
                delta_s1s2 = delta_ij

            # otherwise (subl2 = subl1 means this is a square lattice or hex on-site, just use the delta_ij matrix)
            if sigma_l1l2[sublattice1, sublattice2] != 0.0:
                # print(np.sum(delta_s1s2), sublattice1, sublattice2)
                Delta[first, second] = sigma_l1l2[sublattice1, sublattice2] * \
                                       sigma_o1o2[orbit1, orbit2] * \
                                       delta_s1s2[space1, space2]
    return Delta

def combine_product_terms(config, pairing):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for sigma_ll, sigma_oo, delta_ii, C in pairing[:-2]:
        Delta += C * expand_tensor_product(config, sigma_ll, sigma_oo, delta_ii)  # C -- the coefficient corresponding for the right D_3 transformation properties (irrep)
    return models.apply_twisted_periodic_conditions(config, config.n_orbitals, config.n_sublattices, Delta * pairing[-2])  # the overal coefficient of this irrep (can be 1 or i)

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

    onsite = construct_onsite_delta(config)
    on_site = []

    on_site.append([(Ipauli, Ipauli, onsite, 1), factor, addstring + 'σ_0⊗σ_0⊗δ'])  # A1 0
    on_site.append([(Zpauli, iYpauli, onsite, 1), factor, addstring + 'σ_z⊗jσ_y⊗δ'])  # A1 1
    on_site.append([(Ipauli, iYpauli, onsite, 1), factor, addstring + 'σ_0⊗jσ_y⊗δ'])  # A2 2
    on_site.append([(Zpauli, Ipauli, onsite, 1), factor, addstring + 'σ_z⊗I⊗δ'])  # A2 3

    on_site.append([(Ipauli, Xpauli, onsite, 1.0), factor, addstring + 'σ_0⊗σ_x⊗δ']); on_site.append([(Ipauli, Zpauli, onsite, 1.0), factor, addstring + 'σ_0⊗σ_z⊗δ'])  # E 4-5
    on_site.append([(Zpauli, Xpauli, onsite, 1.0), factor, addstring + 'σ_z⊗σ_x⊗δ']); on_site.append([(Zpauli, Zpauli, onsite, 1.0), factor, addstring + 'σ_z⊗σ_z⊗δ'])  # E 6-7

    return on_site

def construct_NN_2orb_hex(config, real = True):
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
    delta_hex_AB = [construct_NN_delta(config, direction, geometry='hexagonal') for direction in range(1, 4)]
    delta_hex_BA = [delta.T for delta in delta_hex_AB]


    v1_AB = construct_vi_hex(1, delta_hex_AB)
    v2_AB = construct_vi_hex(2, delta_hex_AB)
    v3_AB = construct_vi_hex(3, delta_hex_AB)

    v1_BA = construct_vi_hex(1, delta_hex_BA)
    v2_BA = construct_vi_hex(2, delta_hex_BA)
    v3_BA = construct_vi_hex(3, delta_hex_BA)

    v1 = (v1_AB, v1_BA)
    v2 = (v2_AB, v2_BA)
    v3 = (v3_AB, v3_BA)

    # print(np.unique(v1), np.unique(v2), np.unique(v3))
    NN = []

    NN.append([(Xpauli, Ipauli, v1, 1.0), factor, addstring + 'σ_x⊗σ_0⊗v_1'])  # A1 (A1 x A1 x A1) 0
    NN.append([(iYpauli, iYpauli, v1, 1.0), factor, addstring + '(iσ_y)⊗(iσ_y)⊗v_1'])  # A1 (A2 x A2 x A1) 1 

    NN.append([(Xpauli, iYpauli, v1, 1.0), factor, addstring + 'σ_x⊗(iσ_y)⊗v_1'])  # A2 (A2 x A1 x A1) 2
    NN.append([(iYpauli, Ipauli, v1, 1.0), factor, addstring + '(iσ_y)⊗σ_0⊗v_1'])  # A2 (A1 x A2 x A1) 3

    NN.append([(Xpauli, Xpauli, v1, 1.0), factor, addstring + 'σ_x⊗σ_x⊗v_1']); NN.append([(Xpauli, Zpauli, v1, 1.0), factor, addstring + 'σ_x⊗σ_z⊗v_1'])  # E (A1 x E x A1) 4-5
    NN.append([(iYpauli, Xpauli, v1, 1.0), factor, addstring + '(iσ_y)⊗σ_x⊗v_1']); NN.append([(iYpauli, Zpauli, v1, 1.0), factor, addstring + '(iσ_y)⊗σ_z⊗v_1'])  # E (A2 x E x A1) 6-7

    NN.append([(Xpauli, Ipauli, v2, 1.0), factor, addstring + 'σ_x⊗σ_0⊗v_2']); NN.append([(Xpauli, Ipauli, v3, 1.0), factor, addstring + 'σ_x⊗σ_0⊗v_3'])  # E (A1 x A1 x E) 8-9
    NN.append([(iYpauli, Ipauli, v2, 1.0), factor, addstring + '(iσ_y)⊗σ_0⊗v_2']); NN.append([(iYpauli, Ipauli, v3, 1.0), factor, addstring + '(iσ_y)⊗σ_0⊗v_3'])  # E (A2 x A1 x E) 10-11
    NN.append([(Xpauli, iYpauli, v2, 1.0), factor, addstring + 'σ_x⊗(iσ_y)⊗v_2']); NN.append([(Xpauli, iYpauli, v3, 1.0), factor, addstring + 'σ_x⊗(iσ_y)⊗v_3'])  # E (A1 x A2 x E) 12-13
    NN.append([(iYpauli, iYpauli, v2, 1.0), factor, addstring + '(iσ_y)⊗(iσ_y)⊗v_2']); NN.append([(iYpauli, iYpauli, v3, 1.0), factor, addstring + '(iσ_y)⊗(iσ_y)⊗v_3'])  # E (A2 x A2 x E) 14-15

    NN.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, 1.0), factor, addstring + '[σ_x⊗σ_1⊗v_2 + σ_x⊗σ_2⊗v_3]'])  # A1 (A1 x E x E) 16
    NN.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, -1.0), factor, addstring + '[(iσ_y)⊗σ_1⊗v_2 - (iσ_y)⊗σ_2⊗v_3]'])  # A1 (A2 x E x E) 17

    NN.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, -1.0), factor, addstring + '[σ_x⊗σ_1⊗v_2 - σ_x⊗σ_2⊗v_3]'])  # A2 (A1 x E x E) 18
    NN.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, 1.0), factor, addstring + '[(iσ_y)⊗σ_1⊗v_2 + (iσ_y)⊗σ_2⊗v_3]'])  # A2 (A2 x E x E) 19

    NN.append([(Xpauli, sigma_1, v3, 1.0), factor, addstring + 'σ_x⊗σ_1⊗v_3']); NN.append([(Xpauli, sigma_2, v2, 1.0), factor, addstring + 'σ_x⊗σ_2⊗v_2'])  # E (A1 x E x E) 20-21
    NN.append([(iYpauli, sigma_1, v3, 1.0), factor, addstring + '(iσ_y)⊗σ_1⊗v_3']); NN.append([(iYpauli, sigma_2, v2, 1.0), factor, addstring + '(iσ_y)⊗σ_2⊗v_2'])  # E (A2 x E x E) 22-23

    return NN


def construct_on_site_1orb_hex(config, real = True):
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    onsite = construct_onsite_delta(config)
    on_site = []
    on_site.append([(Ipauli, identity, onsite, 1), factor, addstring + 'σ_0⊗I⊗δ'])  # A1 0
    on_site.append([(Zpauli, identity, onsite, 1), factor, addstring + 'σ_z⊗I⊗δ'])  # A2 1

    return on_site


def construct_NN_1orb_hex(config, real = True):
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

    NN = []

    NN.append([(Xpauli, identity, v1, 1.0), factor, addstring + 'σ_x⊗I⊗v_1'])  # A1 (A1 x A1 x A1) 0
    NN.append([(iYpauli, identity, v1, 1.0), factor, addstring + '(iσ_y)⊗I⊗v_1'])  # B2 (A1 x A2 x A1) 1

    NN.append([(Xpauli, identity, v2, 1.0), factor, addstring + 'σ_x⊗I⊗v_2']); NN.append([(Xpauli, identity, v3, 1.0), factor, addstring + 'σ_x⊗I⊗v_3'])  # E1 (A1 x A1 x E) 2-3
    NN.append([(iYpauli, identity, v2, 1.0), factor, addstring + '(iσ_y)⊗I⊗v_2']); NN.append([(iYpauli, identity, v3, 1.0), factor, addstring + '(iσ_y)⊗I⊗v_3'])  # E2 (A2 x A1 x E) 4-5
    return NN


def construct_on_site_1orb_square(config, real = True):
    factor = 1.0
    addstring = ''
    if not real:
        factor = 1.0j
        addstring = 'j'

    onsite = construct_onsite_delta(config)
    on_site = []
    on_site.append([(identity, identity, onsite, 1), factor, addstring + 'I⊗I⊗δ'])  # A1 0

    return on_site


def construct_NN_1orb_square(config, real = True):
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

    NN = []

    NN.append([(identity, identity, v1, 1.0), factor, addstring + 'I⊗I⊗v_1'])  # A1 (A1 x A1 x A1) 0
    NN.append([(identity, identity, v4, 1.0), factor, addstring + 'I⊗I⊗v_4'])  # B2 (A1 x A1 x A2) 1

    NN.append([(identity, identity, v2, 1.0), factor, addstring + 'I⊗I⊗v_2']); NN.append([(identity, identity, v3, 1.0), factor, addstring + 'I⊗I⊗v_3'])  # E (A1 x A1 x E) 2-3
    return NN



def check_parity(config, pairing):
    gap = combine_product_terms(config, pairing)
    print(np.sum(np.abs(gap)) / config.Ls ** 2 / config.n_sublattices)
    # print(np.unique(gap))
    if np.allclose(gap + gap.T, 0):
        return 'triplet'
    elif np.allclose(gap - gap.T, 0):
        return 'singlet'
    return 'WTF is this shit?!'


on_site_2orb_hex_real = None
NN_2orb_hex_real = None

on_site_1orb_hex_real = None
NN_1orb_hex_real = None

on_site_1orb_square_real = None
NN_1orb_square_real = None

on_site_2orb_hex_imag = None
NN_2orb_hex_imag = None

on_site_1orb_hex_imag = None
NN_1orb_hex_imag = None

on_site_1orb_square_imag = None
NN_1orb_square_imag = None

def obtain_all_pairings(config):
    global on_site_2orb_hex_imag, NN_2orb_hex_imag, \
           on_site_1orb_hex_imag, NN_1orb_hex_imag, \
           on_site_1orb_square_imag, NN_1orb_square_imag
    global on_site_2orb_hex_real, NN_2orb_hex_real, \
           on_site_1orb_hex_real, NN_1orb_hex_real, \
           on_site_1orb_square_real, NN_1orb_square_real

    on_site_2orb_hex_real = construct_on_site_2orb_hex(config)
    NN_2orb_hex_real = construct_NN_2orb_hex(config)

    on_site_1orb_hex_real = construct_on_site_1orb_hex(config)
    NN_1orb_hex_real = construct_NN_1orb_hex(config)

    on_site_1orb_square_real = construct_on_site_1orb_square(config)
    NN_1orb_square_real = construct_NN_1orb_square(config)

    on_site_2orb_hex_imag = construct_on_site_2orb_hex(config, real = False)
    NN_2orb_hex_imag = construct_NN_2orb_hex(config, real = False)

    on_site_1orb_hex_imag = construct_on_site_1orb_hex(config, real = False)
    NN_1orb_hex_imag = construct_NN_1orb_hex(config, real = False)

    on_site_1orb_square_imag = construct_on_site_1orb_square(config, real = False)
    NN_1orb_square_imag = construct_NN_1orb_square(config, real = False)


    # for n, pairing in enumerate(NN_1orb_square_real):
    #     print(check_parity(config, pairing), n)
    return
