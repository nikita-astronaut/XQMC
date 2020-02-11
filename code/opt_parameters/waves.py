import numpy as np
import models

def construct_wave_V(config, orbital, sublattice, wave_type):
    checkerboard = models.spatial_checkerboard(config.Ls)

    sublattice_matrix = np.zeros((config.n_sublattices, config.n_sublattices))
    sublattice_matrix[sublattice, sublattice] = 1.

    orbital_matrix = np.zeros((config.n_orbitals, config.n_orbitals))
    orbital_matrix[orbital, orbital] = 1.            

    dof_matrix = np.kron(np.kron(checkerboard, sublattice_matrix), orbital_matrix)

    if wave_type == 'SDW':
        return [np.kron(np.eye(2), dof_matrix) + 0.0j, wave_type + '-' + str(orbital) + '-' + str(sublattice)]
    return [np.kron(np.diag([1, -1]), dof_matrix) + 0.0j, wave_type + '-' + str(orbital) + '-' + str(sublattice)]

SDW_2orb = None
CDW_2orb = None
SDW_1orb_hex = None
CDW_1orb_hex = None
SDW_1orb_square = None
CDW_1orb_square = None

def obtain_all_waves(config):
    global SDW_2orb, CDW_2orb, SDW_1orb_hex, CDW_1orb_hex, SDW_1orb_square, CDW_1orb_square
    if config.n_orbitals == 2 and config.n_sublattices == 2:
        SDW_2orb = [construct_wave_V(config, (dof // config.n_sublattices) % config.n_orbitals, \
                                      dof % config.n_sublattices, 'SDW') for dof in range(config.n_sublattices * config.n_orbitals)]
        CDW_2orb = [construct_wave_V(config, (dof // config.n_sublattices) % config.n_orbitals, \
                                      dof % config.n_sublattices, 'CDW') for dof in range(config.n_sublattices * config.n_orbitals)]
        return
    if config.n_orbitals == 1 and config.n_sublattices == 2:
        SDW_1orb_hex = [construct_wave_V(config, 0, dof, 'SDW') for dof in config.n_sublattices]
        CDW_1orb_hex = [construct_wave_V(config, 0, dof, 'CDW') for dof in config.n_sublattices]
        return
    if config.n_orbitals == 1 and config.n_sublattices == 1:
        SDW_1orb_square = [construct_wave_V(config, 0, 0, 'SDW')]
        CDW_1orb_square = [construct_wave_V(config, 0, 0, 'CDW')]
        return
    raise NotImplementedError()