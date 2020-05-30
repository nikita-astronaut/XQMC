import numpy as np
import models
from opt_parameters import pairings
from copy import deepcopy

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

sigma_1 = (Zpauli - 1.0j * Xpauli)
sigma_2 = (Zpauli + 1.0j * Xpauli)
delta_hex_AB = None 
delta_hex_BA = None
identity = np.array([[1]])


def construct_2orb_hex(config):
    # under particle-hole
    onsite = pairings.construct_onsite_delta(config)

    orders_on_site = [
        [(Zpauli, Ipauli, onsite, Zpauli), '(S_z)x(S_0)x(S_z)'],
        [(Zpauli, Ipauli, onsite, Ipauli), '(S_z)x(S_0)x(S_0)'],
        [(Ipauli, sigma_1, onsite, Ipauli), '(S_0)x(S_1)x(S_0)'], [(Ipauli, sigma_2, onsite, Ipauli), '(S_0)x(S_2)x(S_0)'],
        [(Zpauli, sigma_1, onsite, Ipauli), '(S_z)x(S_1)x(S_0)'], [(Zpauli, sigma_2, onsite, Ipauli), '(S_z)x(S_2)x(S_0)'],
        [(Ipauli, sigma_1, onsite, Zpauli), '(S_0)x(S_1)x(S_z)'], [(Ipauli, sigma_2, onsite, Zpauli), '(S_0)x(S_2)x(S_z)'],
        [(Zpauli, sigma_1, onsite, Zpauli), '(S_z)x(S_1)x(S_z)'], [(Zpauli, sigma_2, onsite, Zpauli), '(S_z)x(S_2)x(S_z)'],
    ]
    print('Checking S_zxS_0xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[0:1], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[0:1], term_type = 'bilinear', chiral = True)

    print('Checking S_zxS_0xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[1:2], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[1:2], term_type = 'bilinear', chiral = True)

    print('Checking S_0x(S_x)xS_0/S_0x(S_z)xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[2:4], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[2:4], term_type = 'bilinear', chiral = True)

    print('Checking S_zx(S_x)xS_0/S_zx(S_z)xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[4:6], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[4:6], term_type = 'bilinear', chiral = True)

    print('Checking S_0x(S_x)xS_z/S_0x(S_z)xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[6:8], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[6:8], term_type = 'bilinear', chiral = True)

    print('Checking S_zx(S_x)xS_z/S_zx(S_z)xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[8:10], term_type = 'bilinear')
    pairings.check_irrep_properties(config, orders_on_site[8:10], term_type = 'bilinear', chiral = True)

    global delta_hex_AB, delta_hex_BA
    delta_hex_AB = [pairings.construct_NN_delta(config, direction, geometry='hexagonal') for direction in range(1, 4)]
    delta_hex_BA = [delta.T for delta in delta_hex_AB]

    v1_AB = pairings.construct_vi_hex(1, delta_hex_AB)
    v1_BA = pairings.construct_vi_hex(1, delta_hex_BA)
    v1 = (v1_AB, v1_BA)

    orders_NN = [
        [(Xpauli, Ipauli, v1, Ipauli), '(S_x)xS_0xv_1xS_0'],  # renormalization of t_1
    ]
    print('Checking (S_x)xS_0xv_1xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_NN[0:1])

    delta_hex_AA = [pairings.construct_NNN_delta(config, direction, 'hexagonal', sublattice = 0) for direction in range(1, 7)]
    delta_hex_BB = [pairings.construct_NNN_delta(config, direction, 'hexagonal', sublattice = 1) for direction in range(1, 7)]

    u1_AA = pairings.construct_ui_hex(1, delta_hex_AA)
    u1_BB = pairings.construct_ui_hex(1, delta_hex_BB)
    u1 = (u1_AA, u1_BB)

    orders_NNN = [
        [(Ipauli, Ipauli, u1, Ipauli), 'S_0xS_0xu_1xS_0']
    ]

    print('Checking S_0xS_0xu_1xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_NNN[0:1])

    orders_unwrapped = []

    for order in orders_on_site:# + orders_NN + orders_NNN:
        order_unwrapped = pairings.expand_tensor_product(config, *(order[0]))
        orders_unwrapped.append([(order_unwrapped + order_unwrapped.conj().T) / 2., order[1]])  # TODO: remove Hermitisation?
    

    return orders_unwrapped


def construct_1orb_hex(config):
    # under particle-hole
    onsite = pairings.construct_onsite_delta(config)

    orders_on_site = [
        [(Zpauli, identity, onsite, Zpauli), 'S_zxIxS_z'],  # sublattice polarisation (SDW)
        [(Zpauli, identity, onsite, Ipauli), 'S_zxIxS_0'],  # sublattice CDW
    ]
    print('Checking S_zxIxS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[0:1])

    print('Checking S_zxIxS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[1:2])

    orders_unwrapped = []

    for order in orders_on_site:
        order_unwrapped = pairings.expand_tensor_product(config, *(order[0]))
        orders_unwrapped.append([(order_unwrapped + order_unwrapped.conj().T) / 2., order[1]])  # TODO: remove Hermitisation?
    

    return orders_unwrapped

def construct_1orb_square(config):
    onsite = pairings.construct_onsite_delta(config)

    orders_on_site = [
        [(identity, identity, onsite, Zpauli), 'IxIxS_z'],  # local AFM response
    ]
    print('Checking IxIxS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[0:1])

    orders_unwrapped = []

    for order in orders_on_site:
        order_unwrapped = pairings.expand_tensor_product(config, *(order[0]))
        orders_unwrapped.append([(order_unwrapped + order_unwrapped.conj().T) / 2., order[1]])  # TODO: remove Hermitisation?
    

    return orders_unwrapped


def waves_particle_hole(config, m):
    m_ph = m.copy()

    m_ph[:m.shape[0] // 2, :m.shape[1] // 2] = models.apply_TBC(config, config.twist, deepcopy(m_ph[:m.shape[0] // 2, :m.shape[1] // 2]), inverse = False)
    m_ph[m.shape[0] // 2:, m.shape[1] // 2:] = -models.apply_TBC(config, config.twist, deepcopy(m_ph[m.shape[0] // 2:, m.shape[1] // 2:]), inverse = True).T
    return m_ph


hex_2orb = None
hex_1orb = None
square_1orb = None

def obtain_all_waves(config):
    global hex_2orb, hex_1orb, square_1orb
    if config.n_orbitals == 2 and config.n_sublattices == 2:
        hex_2orb = construct_2orb_hex(config)
        return
    if config.n_orbitals == 1 and config.n_sublattices == 2:
        hex_1orb = construct_1orb_hex(config)
        return
    if config.n_orbitals == 1 and config.n_sublattices == 1:
        square_1orb = construct_1orb_square(config)
        return
    raise NotImplementedError()
