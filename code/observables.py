import numpy as np
import models
xp = np  # by default the code is executed on the CPU
gpu = False
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
    gpu = True
except ImportError:
    pass

import auxiliary_field

def get_B_sublattice_mask(config):
    return xp.asarray(1.0 * np.array([models.from_linearized_index(index, 2 * config.n_orbitals * config.Ls ** 2, config.n_orbitals)[1] for index in range(2 * config.n_orbitals * config.Ls ** 2)]))

def density_spin(h_configuration, K, spin, config):
    G_function = auxiliary_field.get_green_function(h_configuration, K, spin, config)
    return xp.trace(G_function) / (2 * config.n_orbitals * config.Ls ** 2)

def total_density(h_configuration, K, config):
    return (density_spin(h_configuration, K, 1.0, config) + density_spin(h_configuration, K, -1.0, config)) / 2.

# this is currently only valid for the Sorella simplest model
def kinetic_energy(h_configuration, K, K_matrix, config):
    A = np.abs(K_matrix) > 1e-6
    G_function_up = auxiliary_field.get_green_function(h_configuration, K, +1.0, config)
    G_function_down = auxiliary_field.get_green_function(h_configuration, K, -1.0, config)

    K_mean = config.main_hopping * xp.einsum('ij,ji', G_function_up + G_function_down, A) / (2 * config.n_orbitals * config.Ls ** 2)
    return K_mean

def double_occupancy(h_configuration, K, config):
    G_function_up = auxiliary_field.get_green_function(h_configuration, K, +1.0, config)
    G_function_down = auxiliary_field.get_green_function(h_configuration, K, -1.0, config)

    return xp.trace(G_function_up * G_function_down) / (2 * config.n_orbitals * config.Ls ** 2)

def staggered_magnetisation(h_configuration, K, config):
    def staggered_magnetisation_ij(G_function_up, G_function_down, i_sublattice, j_sublattice, config):
        i_sublattice_mask = get_B_sublattice_mask(config)
        if i_sublattice == 0:
            i_sublattice_mask = 1. - i_sublattice_mask
        j_sublattice_mask = 1.0 * i_sublattice_mask
        if j_sublattice != i_sublattice:
            j_sublattice_mask = 1. - j_sublattice_mask

        i_sublattice_disconnected = xp.sum((xp.diag(G_function_up) - xp.diag(G_function_down)) * i_sublattice_mask)
        j_sublattice_disconnected = xp.sum((xp.diag(G_function_up) - xp.diag(G_function_down)) * j_sublattice_mask)

        connected_up = xp.einsum('ij,ji,j,i', G_function_up, G_function_up, i_sublattice_mask, j_sublattice_mask)
        connected_down = xp.einsum('ij,ji,j,i', G_function_down, G_function_down, i_sublattice_mask, j_sublattice_mask)

        contact_term = 0.0
        if i_sublattice == j_sublattice:
            contact_term = 3 * xp.sum((xp.diag(G_function_up) + xp.diag(G_function_down)) * i_sublattice_mask)

        connected_up_down = 2. * xp.einsum('ij,ji,j,i', G_function_up, G_function_down, i_sublattice_mask, j_sublattice_mask)
        connected_down_up = 2. * xp.einsum('ij,ji,j,i', G_function_down, G_function_up, i_sublattice_mask, j_sublattice_mask)
        
        return i_sublattice_disconnected * j_sublattice_disconnected + contact_term - connected_up - connected_down - connected_up_down - connected_down_up

    G_function_up = auxiliary_field.get_green_function(h_configuration, K, +1.0, config)
    G_function_down = auxiliary_field.get_green_function(h_configuration, K, -1.0, config)

    AA = staggered_magnetisation_ij(G_function_up, G_function_down, 0, 0, config)
    BB = staggered_magnetisation_ij(G_function_up, G_function_down, 1, 1, config)
    AB = staggered_magnetisation_ij(G_function_up, G_function_down, 0, 1, config)
    BA = staggered_magnetisation_ij(G_function_up, G_function_down, 1, 0, config)
    
    return (AA + BB - AB - BA) / 4. / (2 * config.n_orbitals * config.Ls ** 2) ** 1

def SzSz_onsite(h_configuration, K, config):
    G_function_up = auxiliary_field.get_green_function(h_configuration, K, +1.0, config)
    G_function_down = auxiliary_field.get_green_function(h_configuration, K, -1.0, config)

    return (-2.0 * xp.sum((xp.diag(G_function_up) * xp.diag(G_function_down))) + xp.sum(xp.diag(G_function_down)) + xp.sum(xp.diag(G_function_up))) / (2 * config.n_orbitals * config.Ls ** 2)
