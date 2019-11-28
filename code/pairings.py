import numpy as np
import models
from config_vmc import MC_parameters as config
from copy import deepcopy
config = config()

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

sigma_1 = Zpauli + 1.0j * Xpauli
sigma_2 = Zpauli - 1.0j * Xpauli

delta_hex = []
delta_square = []

identity = np.array([1])

def construct_onsite_delta(config):
    return np.eye(config.Ls ** 2)

def construct_NN_delta(config, direction, geometry):
    delta = np.zeros((config.Ls ** 2, config.Ls ** 2))

    for first in range(delta.shape[0]):
        for second in range(delta.shape[1]):
            x1, y1 = first // config.Ls, first % config.Ls
            x2, y2 = second // config.Ls, second % config.Ls
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if direction == models.nearest_neighbor(r1, r2, config.Ls, geometry, return_direction = True)[1]:
                delta[first, second] = 1
    ret = delta + delta.T
    ret[np.where(ret != 0)] = 1
    return ret

def construct_vi_hex(vi):
    global delta_hex
    phase_factor = 1.0 + 0.0j
    if vi == 2:
        phase_factor = np.exp(2.0 * np.pi / 3. * 1.0j)
    if vi == 3:
        phase_factor = np.exp(-2.0 * np.pi / 3. * 1.0j)

    return delta_hex[0] * (1.0 + 0.0j) + delta_hex[1] * phase_factor + delta_hex[2] * phase_factor ** 2

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
            orbit1, sublattice1, x1, y1 = models.from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals, config.n_sublattices)
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals, config.n_sublattices)
            space1 = x1 * config.Ls + y1
            space2 = x2 * config.Ls + y2
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            Delta[first, second] = sigma_l1l2[sublattice1, sublattice2] * \
                                   sigma_o1o2[orbit1, orbit2] * \
                                   delta_ij[space1, space2]
    return Delta

def combine_product_terms(config, pairing):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2)) * 1.0j
    for sigma_ll, sigma_oo, delta_ii, C in pairing[:-1]:
        Delta += C * expand_tensor_product(config, sigma_ll, sigma_oo, delta_ii)  # C -- the coefficient corresponding for the right D_3 transformation properties (irrep)
    return Delta * pairing[-1]  # the overal coefficient of this irrep (can be 1 or i)

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



def construct_on_site_pairings_2orb_hex(config, real = True):
    factor = 1.0
    if not real:
        factor = 1.0j

    onsite = construct_onsite_delta(config)
    on_site_pairings = []
    on_site_pairings.append([(Ipauli, Ipauli, onsite, 1), factor])  # A1 0
    on_site_pairings.append([(Zpauli, iYpauli, onsite, 1), factor])  # A1 1
    on_site_pairings.append([(Ipauli, iYpauli, onsite, 1), factor])  # A2 2
    on_site_pairings.append([(Zpauli, Ipauli, onsite, 1), factor])  # A2 3

    on_site_pairings.append([(Ipauli, Xpauli, onsite, 1.0), factor]); on_site_pairings.append([(Ipauli, Zpauli, onsite, 1.0), factor])  # E 4-5
    on_site_pairings.append([(Zpauli, Xpauli, onsite, 1.0), factor]); on_site_pairings.append([(Zpauli, Zpauli, onsite, 1.0), factor])  # E 6-7

    return on_site_pairings


def construct_NN_pairings_2orb_hex(config, real = True):
    factor = 1.0
    if not real:
        factor = 1.0j
    global delta_hex
    delta_tex = [construct_NN_delta(config, direction) for direction in range(1, 4)]
    delta1 = construct_NN_delta(config, 1)
    delta2 = construct_NN_delta(config, 2)  
    delta3 = construct_NN_delta(config, 3)

    v1 = construct_vi(1)
    v2 = construct_vi(2)
    v3 = construct_vi(3)
    # print(np.unique(v1), np.unique(v2), np.unique(v3))
    NN_pairings = []

    NN_pairings.append([(Xpauli, Ipauli, v1, 1), factor])  # A1 (A1 x A1 x A1) 0
    NN_pairings.append([(iYpauli, iYpauli, v1, 1), factor])  # A1 (A2 x A2 x A1) 1 

    NN_pairings.append([(Xpauli, iYpauli, v1, 1), factor])  # A2 (A2 x A1 x A1) 2
    NN_pairings.append([(iYpauli, Ipauli, v1, 1), factor])  # A2 (A1 x A2 x A1) 3

    NN_pairings.append([(Xpauli, Xpauli, v1, 1.0), factor]); NN_pairings.append([(Xpauli, Zpauli, v1, 1.0), factor])  # E (A1 x E x A1) 4-5
    NN_pairings.append([(iYpauli, Xpauli, v1, 1.0), factor]); NN_pairings.append([(iYpauli, Zpauli, v1, 1.0), factor])  # E (A2 x E x A1) 6-7

    NN_pairings.append([(Xpauli, Ipauli, v2, 1.0), factor]); NN_pairings.append([(Xpauli, Ipauli, v3, 1.0), factor])  # E (A1 x A1 x E) 8-9
    NN_pairings.append([(iYpauli, Ipauli, v2, 1.0), factor]); NN_pairings.append([(iYpauli, Ipauli, v3, 1.0), factor])  # E (A2 x A1 x E) 10-11
    NN_pairings.append([(Xpauli, iYpauli, v2, 1.0), factor]); NN_pairings.append([(Xpauli, iYpauli, v3, 1.0), factor])  # E (A1 x A2 x E) 12-13
    NN_pairings.append([(iYpauli, iYpauli, v2, 1.0), factor]); NN_pairings.append([(iYpauli, iYpauli, v3, 1.0), factor])  # E (A2 x A2 x E) 14-15

    NN_pairings.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, 1.0), factor])  # A1 (A1 x E x E) 16
    NN_pairings.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, -1.0), factor])  # A1 (A2 x E x E) 17

    NN_pairings.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, -1.0), factor])  # A2 (A1 x E x E) 18
    NN_pairings.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, 1.0), factor])  # A2 (A2 x E x E) 19

    NN_pairings.append([(Xpauli, sigma_1, v3, 1.0), factor]); NN_pairings.append([(Xpauli, sigma_2, v2, 1.0), factor])  # E (A1 x E x E) 20-21
    NN_pairings.append([(iYpauli, sigma_1, v3, 1.0), factor]); NN_pairings.append([(iYpauli, sigma_2, v2, 1.0), factor])  # E (A2 x E x E) 22-23

    return NN_pairings


def construct_on_site_pairings_1orb_hex(config, real = True):
    factor = 1.0
    if not real:
        factor = 1.0j

    onsite = construct_onsite_delta(config)
    on_site_pairings = []
    on_site_pairings.append([(Ipauli, identity, onsite, 1), factor])  # A1 0
    on_site_pairings.append([(Zpauli, identity, onsite, 1), factor])  # A2 1

    return on_site_pairings


def construct_NN_pairings_1orb_hex(config, real = True):
    factor = 1.0
    if not real:
        factor = 1.0j
    global delta_hex
    delta_tex = [construct_NN_delta(config, direction) for direction in range(1, 4)]

    v1 = construct_vi(1)
    v2 = construct_vi(2)
    v3 = construct_vi(3)
    # print(np.unique(v1), np.unique(v2), np.unique(v3))
    NN_pairings = []

    NN_pairings.append([(Xpauli, identity, v1, 1), factor])  # A1 (A1 x A1 x A1) 0
    NN_pairings.append([(iYpauli, identity, v1, 1), factor])  # A2 (A1 x A2 x A1) 1
    # TODODOOOOOOOOOOOOOo
    NN_pairings.append([(Xpauli, Ipauli, v2, 1.0), factor]); NN_pairings.append([(Xpauli, Ipauli, v3, 1.0), factor])  # E (A1 x A1 x E) 8-9
    NN_pairings.append([(iYpauli, Ipauli, v2, 1.0), factor]); NN_pairings.append([(iYpauli, Ipauli, v3, 1.0), factor])  # E (A2 x A1 x E) 10-11
    NN_pairings.append([(Xpauli, iYpauli, v2, 1.0), factor]); NN_pairings.append([(Xpauli, iYpauli, v3, 1.0), factor])  # E (A1 x A2 x E) 12-13
    NN_pairings.append([(iYpauli, iYpauli, v2, 1.0), factor]); NN_pairings.append([(iYpauli, iYpauli, v3, 1.0), factor])  # E (A2 x A2 x E) 14-15

    NN_pairings.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, 1.0), factor])  # A1 (A1 x E x E) 16
    NN_pairings.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, -1.0), factor])  # A1 (A2 x E x E) 17

    NN_pairings.append([(Xpauli, sigma_1, v2, 1.0), (Xpauli, sigma_2, v3, -1.0), factor])  # A2 (A1 x E x E) 18
    NN_pairings.append([(iYpauli, sigma_1, v2, 1.0), (iYpauli, sigma_2, v3, 1.0), factor])  # A2 (A2 x E x E) 19

    NN_pairings.append([(Xpauli, sigma_1, v3, 1.0), factor]); NN_pairings.append([(Xpauli, sigma_2, v2, 1.0), factor])  # E (A1 x E x E) 20-21
    NN_pairings.append([(iYpauli, sigma_1, v3, 1.0), factor]); NN_pairings.append([(iYpauli, sigma_2, v2, 1.0), factor])  # E (A2 x E x E) 22-23

    return NN_pairings

def check_parity(config, pairing):
    gap = combine_product_terms(config, pairing)
    # print(np.unique(gap))
    if np.allclose(gap + gap.T, 0):
        return 'triplet'
    elif np.allclose(gap - gap.T, 0):
        return 'singlet'
    return 'WTF is this shit?!'

on_site_pairings_2orb_hex = construct_on_site_pairings_2orb_hex(config)
NN_pairings_2orb_hex = construct_NN_pairings_2orb_hex(config)

on_site_pairings_1orb_hex = construct_on_site_pairings_1orb_hex(config)
NN_pairings_1orb_hex = construct_NN_pairings_1orb_hex(config)

# for pairing in NN_pairings:
#     print(check_parity(config, pairing))
