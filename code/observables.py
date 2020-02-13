import numpy as np
import models
from time import time
import auxiliary_field
from numba import jit
import os

xp = np  # by default the code is executed on the CPU
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

class Observables:
    def __init__(self, phi, local_workdir):
        self.config = phi.config 
        self.local_workdir = local_workdir
        
        self.log_file = open(os.path.join(self.local_workdir, 'general_log.dat'), 'w')
        self.gap_file = open(os.path.join(self.local_workdir, 'gap_log.dat'), 'w')
        self.density_file = open(os.path.join(self.local_workdir, 'density_log.dat'), 'w')

        self.refresh_light_logs()
        self.refresh_heavy_logs()

        self.init_light_log_file()
        self.init_heavy_logs_files()
        return
        
    def init_light_log_file(self):
        self.log_file.write('n_sweep ' + '⟨ratio⟩ ' + '⟨acc⟩ ' + '⟨sign_gen⟩ ' + '⟨sign_obs_l⟩ ' + \
                                         '⟨density⟩ ' + '⟨E_K⟩ ' + '⟨E_C⟩ ' + '⟨E_T⟩\n')
        return

    def init_heavy_logs_files(self):
        self.gap_file.write('step sign_obs ')
        self.density_file.write('step sign_obs ')

        for key, _ in self.density_corr_list.items():
            self.density_file.write(key + ' ')
        for key, _ in self.gap_observables_list.items():
            if 'corr' in key or 'chi' in key:
                self.gap_file.write(key + ' ')

        self.gap_file.write('\n')
        self.density_file.write('\n')
        return

    def update_history(self, ratio, accepted, sign):
        self.ratio_history.append(ratio)
        self.acceptance_history.append(accepted)
        self.sign_history.append(sign)

    def refresh_heavy_logs(self):
        self.density_file.flush()
        self.gap_file.flush()

        self.density_corr_list = {}
        self.gap_observables_list = {}

        adj_list = self.config.adj_list[:self.config.n_adj_density]  # only largest distance

        chi_shape = (self.config.total_dof // 2, self.config.total_dof // 2, self.config.Nt)
        for gap_name in self.config.pairings_list_names:
            self.gap_observables_list[gap_name + '_D1'] = np.zeros(chi_shape) + 0.0j  # susceptibility part D1
            self.gap_observables_list[gap_name + '_D2'] = np.zeros(chi_shape) + 0.0j  # susceptibility part D2
            self.gap_observables_list[gap_name + '_C'] = np.zeros(chi_shape) + 0.0j  # susceptibility part C

            self.gap_observables_list[gap_name + '_chi'] = []
            self.gap_observables_list[gap_name + '_corr'] = []  # large-distance correlation function averaged over orbitals

        density_adj_list = self.config.adj_list[:self.config.n_adj_density]  # only smallest distance
        for adj in adj_list:
            self.density_corr_list["n_up_n_down_({:d}/{:d}/{:.2f})".format(*adj[1:])] = []

        self.heavy_signs_history = []

        return

    def refresh_light_logs(self):
        self.log_file.flush()
        self.light_observables_list = {
            '⟨density⟩' : [], 
            '⟨E_K⟩' : [], 
            '⟨E_C⟩' : [],
            '⟨E_T⟩' : []
        }

        self.light_signs_history = []

        self.ratio_history = []
        self.acceptance_history = []
        self.sign_history = []

        return

    def print_greerings(self):
        print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
              "lattice = {:d}^2 x {:d}".format(self.config.start_type, 1.0 / self.config.dt / self.config.Nt, \
                                               self.config.mu, self.config.Ls, self.config.Nt))
        print('# sweep ⟨r⟩ ⟨acc⟩ ⟨sign⟩ ⟨n⟩ ⟨E_K⟩ ⟨E_C⟩ ⟨E_T⟩')
        return

    def print_std_logs(self, n_sweep):
        print("{:d} {:.5f} {:.2f} {:.3f} {:.5f} {:.5f} {:.5f}".format(
            n_sweep, 
            np.mean(self.ratio_history),
            np.mean(self.acceptance_history),
            np.mean(self.sign_history),
            np.mean(self.light_observables_list['⟨density⟩']),
            np.mean(self.light_observables_list['⟨E_K⟩']),
            np.mean(self.light_observables_list['⟨E_C⟩']),
            np.mean(self.light_observables_list['⟨E_T⟩']),
        ))
        return

    def measure_light_observables(self, phi, current_det_sign):
        self.light_signs_history.append(current_det_sign)

        k = kinetic_energy(phi).item()
        C = Coloumb_energy(phi)
        density = total_density(phi).item()

        self.light_observables_list['⟨density⟩'].append(density)
        self.light_observables_list['⟨E_K⟩'].append(k)
        self.light_observables_list['⟨E_C⟩'].append(C)
        self.light_observables_list['⟨E_T⟩'].append(k + C)
        return

    def signs_avg(self, array, signs):
        return np.mean(np.array(array) * signs)

    def write_light_observables(self, config, n_sweep):
        signs = np.array(self.light_signs_history)

        data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), np.mean(self.sign_history),
                np.mean(self.light_signs_history)] + [self.signs_avg(val, signs) for _, val in self.light_observables_list.items()]

        self.log_file.write(("{:d} " + "{:.6f} " * (len(data) - 1) + '\n').format(n_sweep, *data[1:]))
        self.refresh_light_logs()
        return


    def measure_heavy_observables(self, phi, current_det_sign):
        self.heavy_signs_history.append(current_det_sign)

        adj_list_density = self.config.adj_list[:self.config.n_adj_density]  # on-site and nn
        adj_list_pairings = self.config.adj_list[-self.config.n_adj_pairings:]  # only largest distance

        for pairing_unwrapped, gap_name in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            D1, D2, C = susceptibility_local(phi, pairing_unwrapped)
            self.gap_observables_list[gap_name + '_D1'] += D1 * current_det_sign
            self.gap_observables_list[gap_name + '_D2'] += D2 * current_det_sign
            self.gap_observables_list[gap_name + '_C'] += C * current_det_sign

            averaged_correlator = 0.0 + 0.0j
            for adj in adj_list_pairings:
                averaged_correlator += gap_gap_correlator(phi.current_G_function_up, phi.current_G_function_down, \
                                                          pairing_unwrapped, adj[0])
            self.gap_observables_list[gap_name + '_corr'].append(averaged_correlator.real / len(adj_list_pairings))


        for adj in adj_list_density:
            self.density_corr_list["n_up_n_down_({:d}/{:d}/{:.2f})".format(*adj[1:])].append(n_up_n_down_correlator(phi, adj[0]).item())
    
        return


    def write_heavy_observables(self, config, n_sweep):
        signs = np.array(self.heavy_signs_history)
        density_data = [n_sweep, np.mean(signs)] + [self.signs_avg(val, signs) for _, val in self.density_corr_list.items()]

        gap_data = [n_sweep, np.mean(signs)]

        for pairing_unwrapped, gap_name in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            chi = np.sum(self.gap_observables_list[gap_name + '_C'] - self.gap_observables_list[gap_name + '_D1'] * self.gap_observables_list[gap_name + '_D2']).real
            gap_data.append(chi / (config.total_dof // 2))
            gap_data.append(self.signs_avg(self.gap_observables_list[gap_name + '_corr'], signs))


        self.density_file.write(("{:d} " + "{:.6f} " * (len(density_data) - 1) + '\n').format(n_sweep, *density_data[1:]))
        self.gap_file.write(("{:d} " + "{:.6f} " * (len(gap_data) - 1) + '\n').format(n_sweep, *gap_data[1:]))
        self.refresh_heavy_logs()

        return

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

    counter = 0

    n_bonds = np.sum(np.abs(gap) > 0) / gap.shape[0]

    result = 0.0 + 0.0j
    for i in range(gap.shape[0]):
        for k in np.where(adj[i, :] > 0)[0]:
            counter += 1
            for j in np.where(np.abs(gap[i, :]) != 0.)[0]:
                for l in np.where(np.abs(gap[k, :]) != 0.)[0]:
                    result += np.conj(gap[i, j]) * gap[k, l] * G_function_down[l, j] * G_function_up[k, i] * adj[i, k]
    return np.real(result) / counter / n_bonds

@jit(nopython=True)
def corr_fix_tau(G_up, G_down, gap):
    D_1 = np.zeros(G_up.shape, dtype = np.complex128)
    D_2 = np.zeros(G_up.shape, dtype = np.complex128)
    C = np.zeros(G_up.shape, dtype = np.complex128)

    for i in range(gap.shape[0]):
        for k in range(gap.shape[0]):
            D_2[i, k] = G_up[i, k]
            c_accumulant = 0.0 + 0.0j
            d1_accumulant = 0.0 + 0.0j
            for j in np.where(gap[i, :] != 0.0)[0]:
                for l in np.where(gap[k, :] != 0.0)[0]:
                    d1_accumulant += np.conj(gap[i, j]) * gap[k, l] * G_down[j, l]
                    c_accumulant += np.conj(gap[i, j]) * gap[k, l] * G_down[j, l] * G_up[i, k]
            C[i, k] = c_accumulant
            D_1[i, k] = d1_accumulant
    return D_1, D_2, C


def susceptibility_local(phi, gap):    
    GFs_up = phi.get_nonequal_time_GFs(+1.0)
    GFs_down = phi.get_nonequal_time_GFs(-1.0)

    D_1_total = np.zeros((GFs_up[0].shape[0], GFs_up[0].shape[1], len(GFs_up)), dtype = np.complex128)
    D_2_total = np.zeros((GFs_up[0].shape[0], GFs_up[0].shape[1], len(GFs_up)), dtype = np.complex128)
    C_total = np.zeros((GFs_up[0].shape[0], GFs_up[0].shape[1], len(GFs_up)), dtype = np.complex128)

    for i in range(len(GFs_up)):
        D1, D2, C = corr_fix_tau(GFs_up[i], GFs_down[i], gap)
        D_1_total[..., i] = D1; D_2_total[..., i] = D2; C_total[..., i] = C
    
    norm = np.sum(np.abs(gap) > 0)
    return np.sum(D_1_total) / norm, np.sum(D_2_total) / norm, np.sum(C_total) / norm


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