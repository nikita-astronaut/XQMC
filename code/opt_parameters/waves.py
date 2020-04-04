import numpy as np
import models
from opt_parameters import pairings

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])
delta_hex_AB = None 
delta_hex_BA = None

def construct_2orb_hex(config):
    # under particle-hole
    onsite = pairings.construct_onsite_delta(config)

    orders_on_site = [
        [(Zpauli, Ipauli, onsite, Zpauli), 'S_zxS_0xS_z'],
        [(Zpauli, Ipauli, onsite, Ipauli), 'S_zxS_0xS_0'],
        [(Ipauli, Xpauli, onsite, Ipauli), 'S_0x(S_x)xS_0'], [(Ipauli, Zpauli, onsite, Ipauli), 'S_0x(S_z)xS_0'],
        [(Zpauli, Xpauli, onsite, Ipauli), 'S_zx(S_x)xS_0'], [(Zpauli, Zpauli, onsite, Ipauli), 'S_zx(S_z)xS_0'],
        [(Ipauli, Xpauli, onsite, Zpauli), 'S_0x(S_x)xS_z'], [(Ipauli, Zpauli, onsite, Zpauli), 'S_0x(S_z)xS_z'],
        [(Zpauli, Xpauli, onsite, Zpauli), 'S_zx(S_x)xS_z'], [(Zpauli, Zpauli, onsite, Zpauli), 'S_zx(S_z)xS_z'],
    ]
    print('Checking S_zxS_0xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[0:1])

    print('Checking S_zxS_0xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[1:2])

    print('Checking S_0x(S_x)xS_0/S_0x(S_z)xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[2:4])

    print('Checking S_zx(S_x)xS_0/S_zx(S_z)xS_0 wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[4:6])

    print('Checking S_0x(S_x)xS_z/S_0x(S_z)xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[6:8])

    print('Checking S_zx(S_x)xS_z/S_zx(S_z)xS_z wave symmetries')
    pairings.check_irrep_properties(config, orders_on_site[8:10])

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

    for order in orders_on_site + orders_NN + orders_NNN:
        order_unwrapped = pairings.expand_tensor_product(config, *(order[0]))
        orders_unwrapped.append([(order_unwrapped + order_unwrapped.conj().T) / 2., order[1]])  # TODO: remove Hermitisation?
    

    return orders_unwrapped

def waves_particle_hole(m):
    m_ph = m.copy()
    m_ph[m.shape[0] // 2:, m.shape[1] // 2:] = -1. * m[m.shape[0] // 2:, m.shape[1] // 2:].T
    return m_ph

def construct_wave_V(config, orbital, sublattice, wave_type):
    if config.n_sublattices == 2:
        pattern = models.spatial_uniform(config.Ls)
    else:
        pattern = models.spatial_checkerboard(config.Ls)

    sublattice_matrix = np.zeros((config.n_sublattices, config.n_sublattices))
    sublattice_matrix[sublattice, sublattice] = 1.

    orbital_matrix = np.zeros((config.n_orbitals, config.n_orbitals))
    orbital_matrix[orbital, orbital] = 1.            

    dof_matrix = np.kron(np.kron(pattern, sublattice_matrix), orbital_matrix)

    if wave_type == 'SDW':
        return [np.kron(np.eye(2), dof_matrix) + 0.0j, wave_type + '-' + str(orbital) + '-' + str(sublattice)]
    if wave_type == 'CDW':
        return [np.kron(np.diag([1, -1]), dof_matrix) + 0.0j, wave_type + '-' + str(orbital) + '-' + str(sublattice)]
    return [np.diag(dof_matrix), str(orbital) + '-' + str(sublattice)]

hex_2orb = None
SDW_1orb_hex = None
CDW_1orb_hex = None
SDW_1orb_square = None
CDW_1orb_square = None

def obtain_all_waves(config):
    global hex_2orb, SDW_1orb_hex, CDW_1orb_hex, SDW_1orb_square, CDW_1orb_square
    if config.n_orbitals == 2 and config.n_sublattices == 2:
        hex_2orb = construct_2orb_hex(config)
        return
    if config.n_orbitals == 1 and config.n_sublattices == 2:
        SDW_1orb_hex = [construct_wave_V(config, 0, dof, 'SDW') for dof in range(config.n_sublattices)]
        CDW_1orb_hex = [construct_wave_V(config, 0, dof, 'CDW') for dof in range(config.n_sublattices)]
        return
    if config.n_orbitals == 1 and config.n_sublattices == 1:
        SDW_1orb_square = [construct_wave_V(config, 0, 0, 'SDW')]
        CDW_1orb_square = [construct_wave_V(config, 0, 0, 'CDW')]
        return
    raise NotImplementedError()
