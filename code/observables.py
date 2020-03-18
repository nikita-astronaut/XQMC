import numpy as np
import models
from time import time
import auxiliary_field
from numba import jit
import os
from collections import OrderedDict
from opt_parameters import waves

try:
    import cupy as cp
except ImportError:
    pass

class Observables:
    def __init__(self, phi, local_workdir):
        self.config = phi.config 
        self.local_workdir = local_workdir

        open_mode = 'a' if phi.config.start_type == 'presaved' else 'w'
        
        self.log_file = open(os.path.join(self.local_workdir, 'general_log.dat'), open_mode)
        self.gap_file = open(os.path.join(self.local_workdir, 'gap_log.dat'), open_mode)
        # self.density_file = open(os.path.join(self.local_workdir, 'density_log.dat'), open_mode)
        self.corr_file = open(os.path.join(self.local_workdir, 'corr_log.dat'), open_mode)

        self.refresh_light_logs()
        self.refresh_heavy_logs()

        self.init_light_log_file()
        self.init_heavy_logs_files()

        self.gfs_data = []
        self.gfs_equal_data = np.zeros((phi.config.total_dof // 2, phi.config.total_dof // 2))
        self.num_equal = 0
        self.global_average_sign = []


        # for Gap-Gap susceptibility
        self.reduced_A = models.get_reduced_adjacency_matrix(self.config)
        self.ijkl = np.array(get_idxs_list(self.reduced_A))
        self.C_ijkl = np.zeros(len(self.ijkl))

        self.cur_buffer_size = 0; self.max_buffer_size = 100
        self.GF_up_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.GF_down_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

        self.GF_up_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.GF_down_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

        # for gap-gap correlator measurement
        self.PHI_ijkl = np.zeros(len(self.ijkl))
        self.adj_list_marking = np.zeros((phi.config.total_dof // 2, phi.config.total_dof // 2)).astype(np.int64)
        for idx, adj in enumerate(self.config.adj_list):
            self.adj_list_marking[adj[0] > 0.5] = idx

        return

    def init_light_log_file(self):
        self.log_file.write('n_sweep ' + '⟨ratio⟩ ' + '⟨acc⟩ ' + '⟨sign_gen⟩ ' + '⟨sign_obs_l⟩ ')
        for key, _ in self.light_observables_list.items():
            self.log_file.write(key + ' ')
        self.log_file.write('\n')
        return

    def init_heavy_logs_files(self):
        self.gap_file.write('step sign_obs ')
        # self.density_file.write('step sign_obs ')
        self.corr_file.write('name step sign_obs ')

        # for key, _ in self.density_corr_list.items():
        #     self.density_file.write(key + ' ')
        for key, _ in self.gap_observables_list.items():
            if 'chi' in key:
                self.gap_file.write(key + ' ')
        for r_index in np.arange(0, len(self.config.adj_list), self.config.n_adj_pairings):
            r = self.config.adj_list[r_index][-1]
            self.corr_file.write('{:2f} '.format(r))

        self.gap_file.write('\n')
        # self.density_file.write('\n')
        self.corr_file.write('\n')
        return

    def update_history(self, ratio, accepted, sign):
        self.ratio_history.append(ratio)
        self.acceptance_history.append(accepted)
        self.sign_history.append(sign)

    def refresh_heavy_logs(self):
        # self.density_file.flush()
        self.gap_file.flush()

        # self.density_corr_list = OrderedDict()
        self.gap_observables_list = OrderedDict()

        adj_list = self.config.adj_list[:self.config.n_adj_density]  # only largest distance

        chi_shape = (self.config.total_dof // 2, self.config.total_dof // 2, self.config.Nt)

        self.num_chi_samples = 0
        for gap_name in self.config.pairings_list_names:
            self.gap_observables_list[gap_name + '_D1'] = 0.0  # susceptibility part D1
            self.gap_observables_list[gap_name + '_D2'] = 0.0  # susceptibility part D2
            self.gap_observables_list[gap_name + '_C'] = 0.0  # susceptibility part C

            self.gap_observables_list[gap_name + '_chi'] = 0.0
            self.gap_observables_list[gap_name + '_chi_total'] = 0.0

            for r_index in np.arange(0, len(self.config.adj_list), self.config.n_adj_pairings):
                r = self.config.adj_list[r_index][-1]
                self.gap_observables_list[gap_name + '{:2f}_corr'.format(r)] = 0.0  # large-distance correlation function averaged over orbitals

        density_adj_list = self.config.adj_list[:self.config.n_adj_density]  # only smallest distance
        # for adj in adj_list:
        #     self.density_corr_list["n_up_n_down_({:d}/{:d}/{:.2f})".format(*adj[1:])] = []

        self.heavy_signs_history = []

        return

    def refresh_light_logs(self):
        self.names_waves = []

        if self.config.n_orbitals == 2 and self.config.n_sublattices == 2:
            self.names_waves = ['⟨SDW_l⟩', '⟨SDW_o⟩', '⟨SDW_lo⟩', '⟨CDW_l⟩', '⟨CDW_o⟩', '⟨CDW_lo⟩']

        self.log_file.flush()
        self.light_observables_list = OrderedDict({
            '⟨density⟩' : [], 
            '⟨E_K⟩' : [], 
            '⟨E_C⟩' : [],
            '⟨E_T⟩' : [],
        })
        for name in self.names_waves:
            self.light_observables_list[name] = []

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
        print("{:d} {:.5f} {:.2f} {:.3f} {:.5f} {:.5f} {:.5f} {:.5f}".format(
            n_sweep, 
            np.mean(self.ratio_history),
            np.mean(self.acceptance_history),
            np.mean(self.global_average_sign),
            np.mean(self.light_observables_list['⟨density⟩']),
            np.mean(self.light_observables_list['⟨E_K⟩']),
            np.mean(self.light_observables_list['⟨E_C⟩']),
            np.mean(self.light_observables_list['⟨E_T⟩']),
        ), flush = True)
        return

    def measure_light_observables(self, phi, current_det_sign):
        self.light_signs_history.append(current_det_sign)
        self.gfs_equal_data += (phi.current_G_function_up + phi.current_G_function_down) / 2.
        
        # print(np.mean(np.trace((phi.current_G_function_up + phi.current_G_function_down) / 2.)), '!!!')
        # assert np.abs(np.mean(np.trace((phi.current_G_function_up + phi.current_G_function_down) / 2.)) - 18.) < 1e-4
        self.num_equal += 1

        k = kinetic_energy(phi).item()
        C = Coloumb_energy(phi)
        density = total_density(phi).item()

        if self.config.n_orbitals == 2 and self.config.n_sublattices == 2:
            waves = waves_twoorb_hex(phi)
        else:
            waves = []

        self.light_observables_list['⟨density⟩'].append(density)
        self.light_observables_list['⟨E_K⟩'].append(k)
        self.light_observables_list['⟨E_C⟩'].append(C)
        self.light_observables_list['⟨E_T⟩'].append(k + C)
        for name, wave in zip(self.names_waves, waves):
            self.light_observables_list[name].append(wave)

        return

    def signs_avg(self, array, signs):
        return np.mean(np.array(array) * signs)


    def signs_std(self, array, signs):
        return (np.std(np.array(array) * signs) / np.mean(signs) - \
               np.std(signs) * np.mean(np.array(array) * signs) / np.mean(signs) ** 2) / np.sqrt(len(signs))

    def write_light_observables(self, config, n_sweep):
        signs = np.array(self.light_signs_history)

        data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), np.mean(self.sign_history),
                np.mean(self.light_signs_history)] + [self.signs_avg(val, signs) for _, val in self.light_observables_list.items()]

        # print(len(self.light_observables_list['⟨density⟩']), len(signs), np.mean(self.light_observables_list['⟨density⟩']), np.mean(signs), np.mean(np.array(self.light_observables_list['⟨density⟩']) * signs))

        self.log_file.write(("{:d} " + "{:.6f} " * (len(data) - 1) + '\n').format(n_sweep, *data[1:]))
        self.log_file.flush()
        self.global_average_sign.append(np.mean(signs))
        self.refresh_light_logs()
        return

    def measure_green_functions(self, phi, current_det_sign):
        self.num_chi_samples += 1
        self.heavy_signs_history.append(current_det_sign)
        t = time()
        phi.copy_to_GPU()
        phi.current_G_function_up = phi.get_G_no_optimisation(+1, -1)[0]
        phi.current_G_function_down = phi.get_G_no_optimisation(-1, -1)[0]

        GFs_up = np.array(phi.get_nonequal_time_GFs(+1.0, phi.current_G_function_up))
        GFs_down = np.array(phi.get_nonequal_time_GFs(-1.0, phi.current_G_function_down))
        phi.copy_to_CPU()
        
        self.GF_up_stored[self.cur_buffer_size, ...] = GFs_up * current_det_sign
        self.GF_down_stored[self.cur_buffer_size, ...] = GFs_down * current_det_sign
        print('obtaining of non-equal Gfs takes', time() - t)

        self.cur_buffer_size += 1
        if self.cur_buffer_size == self.max_buffer_size:
            self.refresh_gfs_buffer()
        return


    def refresh_gfs_buffer(self):
        t = time()
        self.C_ijkl += measure_gfs_correlator(self.GF_up_stored[:self.cur_buffer_size, ...], \
            self.GF_down_stored[:self.cur_buffer_size, ...], self.ijkl)
        self.PHI_ijkl += measure_gfs_correlator(self.GF_up_stored[:self.cur_buffer_size, 0:1, ...], \
            self.GF_down_stored[:self.cur_buffer_size, 0:1, ...], self.ijkl)
        print('measurement of C_ijkl, PHI_ijkl correlators takes', time() - t)
        self.GF_up_sum += np.sum(self.GF_up_stored[:self.cur_buffer_size, ...], axis = 0)
        self.GF_down_sum += np.sum(self.GF_down_stored[:self.cur_buffer_size, ...], axis = 0)

        self.cur_buffer_size = 0
        return


    def measure_heavy_observables(self, phi):
        self.refresh_gfs_buffer()
        t = time()
        mean_signs = np.mean(self.heavy_signs_history)
        # adj_list_density = self.config.adj_list[:self.config.n_adj_density]  # on-site and nn

        for gap, gap_name in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            norm = np.sum(gap != 0.0)  # N_alpha x N_s
            N_alpha = np.sum(gap[0, :] != 0.0)

            total_chi = get_gap_susceptibility(gap, self.ijkl, self.C_ijkl) / (self.num_chi_samples * mean_signs)
            free_chi = np.sum([np.trace(self.GF_up_sum[tau, ...].dot(gap).dot(self.GF_down_sum[tau, ...].T).dot(gap.T.conj())) \
                               for tau in range(self.config.Nt)]) / ((self.num_chi_samples * mean_signs) ** 2)
            self.gap_observables_list[gap_name + '_chi'] = (total_chi - free_chi) / norm
            self.gap_observables_list[gap_name + '_chi_total'] = total_chi / norm

            corr_list = gap_gap_correlator(gap, self.ijkl, self.PHI_ijkl, self.adj_list_marking)

            for r_index in np.arange(0, len(self.config.adj_list), self.config.n_adj_pairings):
                c = np.sum([adj[0] for adj in self.config.adj_list[r_index:r_index + self.config.n_adj_pairings]])
                averaged_correlator = np.sum(np.abs(corr_list[r_index:r_index + self.config.n_adj_pairings])) / c
                r = self.config.adj_list[r_index][-1]
                
                self.gap_observables_list[gap_name + '{:2f}_corr'.format(r)] = \
                    averaged_correlator / self.num_chi_samples / mean_signs / N_alpha
            
        print('obtaining of gap corrs takes', time() - t)
        # for adj in adj_list_density:
        #     self.density_corr_list["n_up_n_down_({:d}/{:d}/{:.2f})".format(*adj[1:])].append(n_up_n_down_correlator(phi, adj[0]).item())
    
        return


    def write_heavy_observables(self, phi, n_sweep):
        config = phi.config
        self.measure_heavy_observables(phi)
        signs = np.array(self.heavy_signs_history)
        # density_data = [n_sweep, np.mean(signs)] + [self.signs_avg(val, signs) / np.mean(signs) for _, val in self.density_corr_list.items()]

        gap_data = [n_sweep, np.mean(signs)]

        for pairing_unwrapped, gap_name in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            gap_data.append(self.gap_observables_list[gap_name + '_chi']) # norm already accounted
            gap_data.append(self.gap_observables_list[gap_name + '_chi_total'])

            corr_data = [n_sweep, np.mean(signs)]
            for r_index in np.arange(0, len(self.config.adj_list), self.config.n_adj_pairings):
                r = self.config.adj_list[r_index][-1]
                corr_data.append(self.gap_observables_list[gap_name + '{:2f}_corr'.format(r)])
            self.corr_file.write(gap_name + (" {:d} " + "{:.6f} " * (len(corr_data) - 1) + '\n').format(n_sweep, *corr_data[1:]))

        # self.density_file.write(("{:d} " + "{:.6f} " * (len(density_data) - 1) + '\n').format(n_sweep, *density_data[1:]))
        self.gap_file.write(("{:d} " + "{:.6f} " * (len(gap_data) - 1) + '\n').format(n_sweep, *gap_data[1:]))

        # self.density_file.flush()
        self.gap_file.flush()
        self.corr_file.flush()

        return

def get_B_sublattice_mask(config):
    return xp.asarray(1.0 * np.array([models.from_linearized_index(index, config.Ls, config.n_orbitals)[1] for index in range(config.n_sublattices * config.n_orbitals * config.Ls ** 2)]))

def density_spin(phi, spin):
    if spin == +1:
        G_function = phi.current_G_function_up
    else:
        G_function = phi.current_G_function_down
    return phi.la.trace(G_function) / (phi.config.total_dof // 2)

def total_density(phi_field):
    return density_spin(phi_field, +1) + density_spin(phi_field, -1)

# this is currently only valid for the Sorella simplest model
def kinetic_energy(phi_field, K_matrix):
    A = np.abs(K_matrix) > 1e-6
    G_function_up = phi_field.current_G_function_up
    G_function_down = phi_field.current_G_function_down

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

    return phi.la.einsum('i,j,ij', phi.la.diag(G_function_up), phi.la.diag(G_function_down), adj) / phi.la.sum(adj)

def kinetic_energy(phi):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    return phi.la.einsum('ij,ij', phi.K_matrix, G_function_up + G_function_down) / G_function_up.shape[0]

'''
@jit(nopython=True)
def gap_gap_correlator(current_G_function_up, current_G_function_down, gap, adj):

        ⟨\\Delta^{\\dag} \\Delta⟩ = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
                                  = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} G^{down}(l, j) G^{up}_{k, i}
                                  (i ~ j | k ~ l)_{delta}, (i ~ k)_{adj}

    G_function_up = current_G_function_up + 0.0j
    G_function_down = current_G_function_down + 0.0j
    adj_complex = adj + 0.0j

    counter = np.sum(adj > 0)
    n_bonds = np.sum(np.abs(gap) > 0) / gap.shape[0]

    return np.sum((G_function_up * adj_complex.T).dot(np.conj(gap)).dot(G_function_down.T) * gap) / n_bonds, counter


@jit(nopython=True)
def corr_fix_tau(G_up, G_down, gap):
    D_1 = np.conj(gap).dot(G_down).dot(gap.T)
    C = G_up * D_1
    return D_1, C


@jit(nopython=True)
def susceptibility_local(gap, GFs_up, GFs_down): 
    D_1_total = np.zeros((GFs_up.shape[1], GFs_up.shape[2], GFs_up.shape[0]), dtype = np.complex128)
    D_2_total = np.zeros((GFs_up.shape[1], GFs_up.shape[2], GFs_up.shape[0]), dtype = np.complex128)
    C_total = np.zeros((GFs_up.shape[1], GFs_up.shape[2], GFs_up.shape[0]), dtype = np.complex128)

    for i in range(GFs_up.shape[0]):
        D1, C = corr_fix_tau(GFs_up[i, ...], GFs_down[i, ...], gap)  # 0.0j for jit complex
        D_1_total[..., i] = D1; D_2_total[..., i] = GFs_up[i, ...]; C_total[..., i] = C
    
    norm = np.sum(np.abs(gap) > 0)
    return D_1_total / np.sqrt(norm), D_2_total / np.sqrt(norm), C_total / norm
'''

def Coloumb_energy(phi):
    G_function_up = phi.current_G_function_up
    G_function_down = phi.current_G_function_down

    energy_coloumb = (phi.config.U / 2.) * phi.la.sum((phi.la.diag(G_function_up) + phi.la.diag(G_function_down) - 1.) ** 2).item() \
                     / G_function_up.shape[0]
    if phi.config.n_orbitals == 1:
        return energy_coloumb

    orbital_1 = phi.la.arange(0, G_function_up.shape[0], 2)
    orbital_2 = phi.la.arange(1, G_function_up.shape[0], 2)
    energy_coloumb += phi.config.V * phi.la.einsum('i,i', (phi.la.diag(G_function_up)[orbital_1] + phi.la.diag(G_function_down)[orbital_1] - 1),
                                                      (phi.la.diag(G_function_up)[orbital_2] + phi.la.diag(G_function_down)[orbital_2] - 1)).item() \
                                                      / G_function_up.shape[0]

    return energy_coloumb

def waves_twoorb_hex(phi):
    dl_up = phi.la.diag(phi.current_G_function_up)  # local density
    dl_down = phi.la.diag(phi.current_G_function_down)

    Ax = waves.construct_wave_V(phi.config, 0, 0, wave_type = 'none')[0] > 0.5
    Bx = waves.construct_wave_V(phi.config, 0, 1, wave_type = 'none')[0] > 0.5
    Ay = waves.construct_wave_V(phi.config, 1, 0, wave_type = 'none')[0] > 0.5
    By = waves.construct_wave_V(phi.config, 1, 1, wave_type = 'none')[0] > 0.5
    sdw_l = phi.la.mean((dl_up[Ax] + dl_up[Ay] - dl_up[Bx] - dl_up[By]) - (dl_down[Ax] + dl_down[Ay] - dl_down[Bx] - dl_down[By])) ** 2
    sdw_o = phi.la.mean((dl_up[Ax] + dl_up[Bx] - dl_up[Ay] - dl_up[By]) - (dl_down[Ax] + dl_down[Bx] - dl_down[Ay] - dl_down[By])) ** 2
    sdw_lo = phi.la.mean((dl_up[Ax] + dl_up[By] - dl_up[Ay] - dl_up[Bx]) - (dl_down[Ax] + dl_down[By] - dl_down[Ay] - dl_down[Bx])) ** 2

    cdw_l = phi.la.mean((dl_up[Ax] + dl_up[Ay] - dl_up[Bx] - dl_up[By]) + (dl_down[Ax] + dl_down[Ay] - dl_down[Bx] - dl_down[By])) ** 2
    cdw_o = phi.la.mean((dl_up[Ax] + dl_up[Bx] - dl_up[Ay] - dl_up[By]) + (dl_down[Ax] + dl_down[Bx] - dl_down[Ay] - dl_down[By])) ** 2
    cdw_lo = phi.la.mean((dl_up[Ax] + dl_up[By] - dl_up[Ay] - dl_up[Bx]) + (dl_down[Ax] + dl_down[By] - dl_down[Ay] - dl_down[Bx])) ** 2
    return sdw_l, sdw_o, sdw_lo, cdw_l, cdw_o, cdw_lo

@jit(nopython=True, parallel=True)
def measure_gfs_correlator(GF_up, GF_down, ijkl):
    C_ijkl = np.zeros(len(ijkl), dtype=np.float64)
    idx = 0

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        C_ijkl[xi] = np.sum(GF_up[:, :, i, k] * GF_down[:, :, j, l])

    return C_ijkl

@jit(nopython=True, parallel=True)
def get_idxs_list(reduced_A):
    ijkl = []

    for i in range(reduced_A.shape[0]):
        for k in range(reduced_A.shape[0]):
            for j in reduced_A[i, ...]:
                for l in reduced_A[k, ...]:
                    ijkl.append(np.array([i, j, k, l]))
    return ijkl

@jit(nopython=True, parallel=True)
def get_gap_susceptibility(gap, ijkl, C_ijkl):
    corr = 0.0 + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        corr += np.conj(gap[i, j]) * gap[k, l] * C_ijkl[xi]
    return corr.real

@jit(nopython=True, parallel=True)
def gap_gap_correlator(gap, ijkl, PHI_ijkl, adj_marking):
    corr_list = np.zeros(len(adj_marking)) + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        adj_index = adj_marking[i, k]
        corr_list[adj_index] += np.conj(gap[i, j]) * gap[k, l] * PHI_ijkl[xi]
    return corr_list.real
