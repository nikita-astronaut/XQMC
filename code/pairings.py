import numpy as np
import models

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

sigma_1 = Zpauli + 1.0j * Xpauli
sigma_2 = Zpauli - 1.0j * Xpauli

delta1, delta2, delta3 = None, None, None



def construct_onsite_delta(config):
    return np.eye((config.Ls ** 2, config.Ls ** 2))

def construct_NN_delta(config, direction):
    delta = np.zeros((config.Ls ** 2, config.Ls ** 2))

    for first in range(delta.shape[0]):
        for second in range(delta.shape[1]):
            x1, y1 = first // config.Ls, first % config.Ls
            x2, y2 = second // config.Ls, second % config.Ls
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if direction == models.nearest_neighbor_hexagonal(r1, r2, config.Ls, return_direction = True)[1]:
                delta[first, second] = 1

def construct_vi(vi):
    global delta1, delta2, delta3
    phase_factor = 1.0 + 0.0j
    if vi == 2:
        phase_factor = np.exp(2.0 * np.pi / 3.)
    if vi == 3:
        phase_factor = np.exp(-2.0 * np.pi / 3.)

    return delta1 * (1.0 + 0.0j) + delta2 * phase_factor + delta3 * phase_factor ** 2

def expand_tensor_product(config, sigma_l1j2, sigma_o1o2, delta_ij):
    '''
        returns the matrix of the dimention L x L (L -- total number of sites including orbit),
        sigma_ll \\otimes sigma_oo \\otimes delta_ii
    '''

    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2))
    for first in range(Delta.shape[0]):
        for second in range(Delta.shape[1]):
            orbit1, sublattice1, x1, y1 = models.from_linearized_index(deepcopy(first), config.Ls, config.n_orbitals)
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), config.Ls, config.n_orbitals)
            space1 = x1 * self.config.Ls + y1
            space2 = x2 * self.config.Ls + y2
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            Delta[first, second] = sigma_l1j2[sublattice1, sublattice2] * \
                                   sigma_o1o2[orbit1, orbit2] * 
                                   delta_ij[space1, space2]
    return Delta

def combine_product_terms(config, terms):
    Delta = np.zeros((config.total_dof // 2, config.total_dof // 2))
    for sigma_ll, sigma_oo, delta_ii, C in terms:
        Delta += C * unwrap_sigma_combination(config, sigma_ll, sigma_oo, delta_ii)
    return Delta


def construct_on_site_pairings(config):
    onsite = construct_onsite_delta(config)
    on_site_pairings = []
    on_site_pairings.append([[Ipauli, Ipauli, onsite, 1, False]])  # A1
    on_site_pairings.append([[Zpauli, iYpauli, onsite, 1, False]])  # A1
    on_site_pairings.append([[Ipauli, iYpauli, onsite, 1, False]])  # A2
    on_site_pairings.append([[Zpauli, Ipauli, onsite, 1, False]])  # A2

    on_site_pairings.append([[Ipauli, Xpauli, onsite, 1.0, True], [Ipauli, Zpauli, onsite, 1.0, True]])  # E
    on_site_pairings.append([[Zpauli, Xpauli, onsite, 1.0, True], [Zpauli, Zpauli, onsite, 1.0, True]])  # E

    return on_site_pairings



def construct_NN_pairings(config):
    global delta1, delta2, delta3
    delta1 = construct_NN_delta(config, 1)
    delta2 = construct_NN_delta(config, 2)
    delta3 = construct_NN_delta(config, 3)

    v1 = construct_vi(config, 1)
    v2 = construct_vi(config, 2)
    v3 = construct_vi(config, 3)
    NN_pairings = []

    NN_pairings.append([[Xpauli, Ipauli, v1, 1, False]])  # A1 (A1 x A1 x A1)
    NN_pairings.append([[iYpauli, iYpauli, v1, 1, False]])  # A1 (A2 x A2 x A1) 

    NN_pairings.append([[Xpauli, iYpauli, v1, 1, False]])  # A2 (A2 x A1 x A1)
    NN_pairings.append([[iYpauli, Ipauli, v1, 1, False]])  # A2 (A1 x A2 x A1)

    NN_pairings.append([[Xpauli, Xpauli, v1, 1.0, True], [Xpauli, Zpauli, v1, 1.0, True]])  # E (A1 x E x A1)
    NN_pairings.append([[iYpauli, Xpauli, v1, 1.0, True], [iYpauli, Zpauli, v1, 1.0, True]])  # E (A2 x E x A1)

    NN_pairings.append([[Xpauli, Ipauli, v2, 1.0, True], [Xpauli, Ipauli, v3, 1.0, True]])  # E (A1 x A1 x E)
    NN_pairings.append([[iYpauli, Ipauli, v2, 1.0, True], [iYpauli, Ipauli, v3, 1.0, True]])  # E (A2 x A1 x E)
    NN_pairings.append([[Xpauli, iYpauli, v2, 1.0, True], [Xpauli, iYpauli, v3, 1.0, True]])  # E (A1 x A2 x E)
    NN_pairings.append([[iYpauli, iYpauli, v2, 1.0, True], [iYpauli, iYpauli, v3, 1.0, True]])  # E (A2 x A2 x E)

    NN_pairings.append([[Xpauli, sigma_1, v2, 1.0, False], [Xpauli, sigma_2, v3, 1.0, False]])  # A1 (A1 x E x E)
    NN_pairings.append([[iYpauli, sigma_1, v2, 1.0, False], [iYpauli, sigma_2, v3, -1.0, False]])  # A1 (A2 x E x E)

    NN_pairings.append([[Xpauli, sigma_1, v2, 1.0, False], [Xpauli, sigma_2, v3, -1.0, False]])  # A2 (A1 x E x E)
    NN_pairings.append([[iYpauli, sigma_1, v2, 1.0, False], [iYpauli, sigma_2, v3, 1.0, False]])  # A2 (A2 x E x E)

    NN_pairings.append([[Xpauli, sigma_1, v3, 1.0, True], [Xpauli, sigma_2, v2, 1.0, True]])  # E (A1 x E x E)
    NN_pairings.append([[iYpauli, sigma_1, v3, 1.0, True], [iYpauli, sigma_2, v2, 1.0, True]])  # E (A2 x E x E)

    return NN_pairings