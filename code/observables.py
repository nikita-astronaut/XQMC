import numpy as np
import models
from time import time
import auxiliary_field
from numba import jit

xp = np  # by default the code is executed on the CPU
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass



def get_B_sublattice_mask(config):
    return xp.asarray(1.0 * np.array([models.from_linearized_index(index, config.Ls, config.n_orbitals)[1] for index in range(config.n_sublattices * config.n_orbitals * config.Ls ** 2)]))

def density_spin(phi_field, spin):
    if spin == +1:
        G_function = phi_field.current_G_function_up
    else:
        G_function = phi_field.current_G_function_down
    return xp.trace(G_function) / (phi_field.config.total_dof // 2)

def total_density(phi_field):
    return density_spin(phi_field, +1) + density_spin(phi_field, -1)

# this is currently only valid for the Sorella simplest model
def kinetic_energy(phi_field, K_matrix):
    A = np.abs(K_matrix) > 1e-6
    G_function_up = phi_field.get_current_G_function(+1.0)
    G_function_down = phi_field.get_current_G_function(-1.0)

    K_mean = phi_field.config.main_hopping * xp.einsum('ij,ji', G_function_up + G_function_down, A) / (phi_field.config.total_dof // 2)
    return K_mean

def double_occupancy(h_configuration, K, config):
    G_function_up = auxiliary_field.get_green_function(h_configuration, K, +1.0, config)
    G_function_down = auxiliary_field.get_green_function(h_configuration, K, -1.0, config)

    return xp.trace(G_function_up * G_function_down) / (config.n_sublattices * config.n_orbitals * config.Ls ** 2)

def staggered_magnetisation(phi):
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

    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    AA = staggered_magnetisation_ij(G_function_up, G_function_down, 0, 0, phi.config)
    BB = staggered_magnetisation_ij(G_function_up, G_function_down, 1, 1, phi.config)
    AB = staggered_magnetisation_ij(G_function_up, G_function_down, 0, 1, phi.config)
    BA = staggered_magnetisation_ij(G_function_up, G_function_down, 1, 0, phi.config)
    
    return (AA + BB - AB - BA) / (phi.config.total_dof // 2) ** 2 / 4.

def SzSz_onsite(phi_field):
    G_function_up = phi_field.current_G_function_up
    G_function_down = phi_field.current_G_function_down

    return (-2.0 * xp.sum((xp.diag(G_function_up) * xp.diag(G_function_down))) + xp.sum(xp.diag(G_function_down)) + xp.sum(xp.diag(G_function_up))) / (phi_field.config.total_dof // 2)

def get_n_adj(K_matrix, distance):
    A = xp.abs(xp.asarray(K_matrix)) > 1e-6
    adj = xp.diag(xp.ones(len(xp.diag(A))))

    seen_elements = adj
    for i in range(distance - 1):
        adj = adj.dot(A)
        seen_elements += adj
    return xp.logical_and(seen_elements == 0, adj.dot(A) > 0) * 1.

def SzSz_n_neighbor(phi, adj):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    return (xp.einsum('i,j,ij', xp.diag(G_function_up) - xp.diag(G_function_down), xp.diag(G_function_up) - xp.diag(G_function_down), adj) - \
            xp.einsum('ij,ji,ij', G_function_up, G_function_up, adj) - xp.einsum('ij,ji,ij', G_function_down, G_function_down, adj)) / xp.sum(adj)

def n_up_n_down_correlator(phi, adj):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    return xp.einsum('i,j,ij', xp.diag(G_function_up), xp.diag(G_function_down), adj) / xp.sum(adj)

def kinetic_energy(phi):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    return xp.einsum('ij,ij', phi.K_matrix, G_function_up + G_function_down) / G_function_up.shape[0]

@jit(nopython=True)
def gap_gap_correlator(current_G_function_up, current_G_function_down, gap, adj):
    '''
        ⟨\\Delta^{\\dag} \\Delta⟩ = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
                                  = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} [\\delta_{jl} - G^{down}(l, j)] [\\delta_{ik} - G^{up}_{k, i}]
                                  (i ~ j | k ~ l)_{delta}, (i ~ k)_{adj}
    '''

    G_function_up = np.eye(current_G_function_up.shape[0]) - current_G_function_up
    G_function_down = np.eye(current_G_function_down.shape[0]) - current_G_function_down

    result = 0.0 + 0.0j
    for i in range(gap.shape[0]):
        for k in np.where(adj[i, :] > 0)[0]:
            for j in np.where(np.abs(gap[i, :]) != 0.)[0]:
                for l in np.where(np.abs(gap[k, :]) != 0.)[0]:
                    result += np.conj(gap[i, j]) * gap[k, l] * G_function_down[l, j] * G_function_up[k, i] * adj[i, k]

    return np.real(result)
    # return xp.einsum('ij,kl,lj,ki,ik', np.conj(gap), gap, xp.eye(G_function_down.shape[0]) - G_function_down,
    #                                    xp.eye(G_function_up.shape[0]) - G_function_up, adj)

def Coloumb_energy(phi):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    energy_coloumb = (phi.config.U / 2.) * xp.sum((xp.diag(G_function_up) + xp.diag(G_function_down) - 1.) ** 2).item() \
                     / G_function_up.shape[0]
    if phi.config.n_orbitals == 1:
        return energy_coloumb

    orbital_1 = xp.arange(0, G_function_up.shape[0], 2)
    orbital_2 = xp.arange(1, G_function_up.shape[0], 2)
    energy_coloumb += phi.config.V * xp.einsum('i,i', (xp.diag(G_function_up)[orbital_1] + xp.diag(G_function_down)[orbital_1] - 1),
                                                      (xp.diag(G_function_up)[orbital_2] + xp.diag(G_function_down)[orbital_2] - 1)).item() \
                                                      / G_function_up.shape[0]

    return energy_coloumb

def compute_light_observables(phi):
    observables = [total_density(phi).item(), kinetic_energy(phi).item(), Coloumb_energy(phi), \
                   kinetic_energy(phi).item() + Coloumb_energy(phi)]
    names = ['⟨density⟩', '⟨E_K⟩', '⟨E_C⟩', '⟨E_T⟩']
    return observables, names

def compute_heavy_observables(phi):
    adj_list_density = phi.adj_list[:phi.config.n_adj_density]  # on-site and nn
    adj_list_pairings = phi.adj_list[-phi.config.n_adj_pairings:]  # only largest distance
    observables = []
    names = ['⟨nupndown⟩_density'] + ['⟨' + p + '⟩_pairing' for p in phi.config.pairings_list_names]

    for adj in adj_list_density:
        observables.append(n_up_n_down_correlator(phi, adj[0]).item())

    for pairing_unwrapped in phi.config.pairings_list_unwrapped:
        for n, adj in enumerate(adj_list_pairings):
            observables.append(gap_gap_correlator(phi.current_G_function_up, phi.current_G_function_down, \
                                                  pairing_unwrapped, adj[0]))
    return observables, names
