import numpy as np
import models
from opt_parameters import pairings

Ipauli = np.array([[1, 0], [0, 1]])
Xpauli = np.array([[0, 1], [1, 0]])
iYpauli = np.array([[0, 1], [-1, 0]])
Zpauli = np.array([[1, 0], [0, -1]])

def construct_2orb_hex(config):
    onsite = pairings.construct_onsite_delta(config)

    orders_on_site = [
        [(Zpauli, Ipauli, onsite, Zpauli), 'S_zxS_0xS_z'],
        #[(Ipauli, iYpauli, onsite, Zpauli), 'S_0x(iS_y)xS_z'],
        #[(Zpauli, iYpauli, onsite, Ipauli), 'S_zx(iS_y)xS_0'],
        #[(Ipauli, iYpauli, onsite, Ipauli), 'S_0x(iS_y)xS_0'],
        [(Zpauli, Ipauli, onsite, Ipauli), 'S_zxS_0xS_0'],
        #[(Zpauli, iYpauli, onsite, Zpauli), 'S_zx(iS_y)xS_z'],
        [(Ipauli, Xpauli, onsite, Ipauli), 'S_0x(S_x)xS_0'], [(Ipauli, Zpauli, onsite, Ipauli), 'S_0x(S_z)xS_0'],
        [(Zpauli, Xpauli, onsite, Ipauli), 'S_zx(S_x)xS_0'], [(Zpauli, Zpauli, onsite, Ipauli), 'S_zx(S_z)xS_0'],
        [(Ipauli, Xpauli, onsite, Zpauli), 'S_0x(S_x)xS_z'], [(Ipauli, Zpauli, onsite, Zpauli), 'S_0x(S_z)xS_z'],
        [(Zpauli, Xpauli, onsite, Zpauli), 'S_zx(S_x)xS_z'], [(Zpauli, Zpauli, onsite, Zpauli), 'S_zx(S_z)xS_z'],
    ]

    orders_unwrapped = []

    for order in orders_on_site:
        order_unwrapped = pairings.expand_tensor_product(config, *(order[0]))

        orders_unwrapped.append([(order_unwrapped + order_unwrapped.conj().T) / 2., order[1]])
    return orders_unwrapped


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
