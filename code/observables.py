import numpy as np
import models
from time import time
import auxiliary_field
from numba import jit
import os
from collections import OrderedDict
from opt_parameters import waves
import pickle

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
        # self.corr_file = open(os.path.join(self.local_workdir, 'corr_log.dat'), open_mode)

        self.refresh_light_logs()

        self.init_light_log_file()
        
        self.gfs_data = []
        self.gfs_equal_data = np.zeros((phi.config.total_dof // 2, phi.config.total_dof // 2))
        self.num_equal = 0
        self.global_average_sign = []


        # for Gap-Gap susceptibility
        self.reduced_A_gap = models.get_reduced_adjacency_matrix(self.config, \
            self.config.max_square_pairing_distance)
        self.ijkl = np.array(get_idxs_list(self.reduced_A_gap))

        # for order-order susceptibility
        self.reduced_A_order = models.get_reduced_adjacency_matrix(self.config, \
            self.config.max_square_order_distance)
        self.ijkl_order = np.array(get_idxs_list(self.reduced_A_order))
        self.ik_marking = get_ik_marking(self.config)
        self.distances_list = models.get_distances_list(self.config)  # pairwise distances |r_i - r_j|^2

        # for fast correlator computation
        self.adj_list_marking = np.zeros((phi.config.total_dof // 2, phi.config.total_dof // 2)).astype(np.int64)
        for idx, adj in enumerate(self.config.adj_list):
            self.adj_list_marking[adj[0] > 0.5] = idx


        self.refresh_heavy_logs()
        self.init_heavy_logs_files()
        return

    def init_light_log_file(self):
        self.log_file.write('n_sweep ' + '⟨ratio⟩ ' + '⟨acc⟩ ' + '⟨sign_gen⟩ ' + '⟨sign_obs_l⟩ ')
        for key, _ in self.light_observables_list.items():
            self.log_file.write(key + ' ')
        self.log_file.write('\n')
        return

    def init_heavy_logs_files(self):
        self.gap_file.write('step sign_obs ')
        #self.corr_file.write('name step sign_obs ')


        for key, _ in self.gap_observables_list.items():
            if 'chi' in key:
                self.gap_file.write(key + ' ')
        '''
        for r_index in np.arange(0, len(self.config.adj_list), self.config.n_adj_pairings):
            r = self.config.adj_list[r_index][-1]
            self.corr_file.write('{:2f} '.format(r))
        '''

        self.gap_file.write('\n')
        # self.corr_file.write('\n')
        return

    def update_history(self, ratio, accepted, sign):
        self.ratio_history.append(ratio)
        self.acceptance_history.append(accepted)
        self.sign_history.append(sign)

    def refresh_heavy_logs(self):
        self.gap_file.flush()

        self.C_ijkl = np.zeros(len(self.ijkl))

        self.cur_buffer_size = 0; self.max_buffer_size = 100
        self.GF_up_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.GF_down_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

        self.GF_up_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.GF_down_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

        # for gap-gap correlation length measurement
        self.PHI_ijkl = np.zeros(len(self.ijkl))

        # for order-order correlator measurement
        self.Z_uu_ijkl = np.zeros(len(self.ijkl_order))
        self.Z_dd_ijkl = np.zeros(len(self.ijkl_order))

        self.X_uu_ijkl = np.zeros(len(self.ijkl_order))
        self.X_ud_ijkl = np.zeros(len(self.ijkl_order))
        self.X_du_ijkl = np.zeros(len(self.ijkl_order))
        self.X_dd_ijkl = np.zeros(len(self.ijkl_order))


        self.gap_observables_list = OrderedDict()
        self.order_observables_list = OrderedDict()

        adj_list = self.config.adj_list[:self.config.n_adj_density]  # only largest distance

        self.num_chi_samples = 0
        for gap_name_alpha in self.config.pairings_list_names:
            self.gap_observables_list[gap_name_alpha + '_corr_length'] = 0.0
            self.gap_observables_list[gap_name_alpha + '_Sq0'] = 0.0
            self.gap_observables_list[gap_name_alpha + '_Pq0'] = 0.0

            for gap_name_beta in self.config.pairings_list_names:
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_real'] = 0.0
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_real'] = 0.0
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_imag'] = 0.0
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_imag'] = 0.0


        for order_name in self.config.waves_list_names:
            self.order_observables_list[order_name + '_order'] = 0.0
        density_adj_list = self.config.adj_list[:self.config.n_adj_density]  # only smallest distance


        self.heavy_signs_history = []

        return

    def refresh_light_logs(self):
        self.log_file.flush()
        self.light_observables_list = OrderedDict({
            '⟨density⟩' : [], 
            '⟨E_K⟩' : [], 
            '⟨E_C⟩' : [],
            '⟨E_T⟩' : [],
        })

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
        

        self.num_equal += 1

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


    def signs_std(self, array, signs):
        return (np.std(np.array(array) * signs) / np.mean(signs) - \
               np.std(signs) * np.mean(np.array(array) * signs) / np.mean(signs) ** 2) / np.sqrt(len(signs))

    def write_light_observables(self, config, n_sweep):
        signs = np.array(self.light_signs_history)

        data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), np.mean(self.sign_history),
                np.mean(self.light_signs_history)] + [self.signs_avg(val, signs) for _, val in self.light_observables_list.items()]

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
        self.GF_up_stored[self.cur_buffer_size, ...] = GFs_up
        self.GF_down_stored[self.cur_buffer_size, ...] = GFs_down
        print('obtaining of non-equal Gfs takes', time() - t)

        self.cur_buffer_size += 1
        if self.cur_buffer_size == self.max_buffer_size:
            self.refresh_gfs_buffer()
        return


    def refresh_gfs_buffer(self):
        if self.cur_buffer_size == 0:
            return

        print('current buffer size = {:d}'.format(self.cur_buffer_size))
        t = time()
        signs = np.array(self.heavy_signs_history[-self.cur_buffer_size:])[..., np.newaxis]
        signs = np.repeat(signs, self.config.Nt, axis = 1)
        
        shape = self.GF_up_stored[:self.cur_buffer_size, ...].shape
        print(len(self.ijkl), (shape[0] * shape[1], shape[2], shape[3]))
        G_up_prepared = np.asfortranarray(np.einsum('ijkl,ij->ijkl', \
                       self.GF_up_stored[:self.cur_buffer_size, ...], signs).reshape((shape[0] * shape[1], shape[2], shape[3])))
        G_down_prepared = np.asfortranarray(self.GF_down_stored[:self.cur_buffer_size, ...].reshape((shape[0] * shape[1], shape[2], shape[3])))

        t = time()
        self.C_ijkl += measure_gfs_correlator(G_up_prepared, G_down_prepared, self.ijkl)
        print('C_ijkl take', time() - t)
        self.PHI_ijkl += measure_gfs_correlator(np.asfortranarray(np.einsum('ijkl,ij->ijkl', \
                       self.GF_up_stored[:self.cur_buffer_size, 0:1, ...], signs[..., 0:1]).reshape((shape[0] * 1, shape[2], shape[3]))), \
            np.asfortranarray(self.GF_down_stored[:self.cur_buffer_size, 0:1, ...].reshape((shape[0] * 1, shape[2], shape[3]))), self.ijkl)

        t = time()
        self.Z_uu_ijkl = measure_Z_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)
        self.Z_dd_ijkl = measure_Z_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)

        print('Z_ss_ijkl take', time() - t)
        t = time()

        self.X_uu_ijkl = measure_X_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], \
            self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)
        self.X_ud_ijkl = measure_X_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], \
            self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)
        self.X_du_ijkl = measure_X_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], \
            self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)
        self.X_dd_ijkl = measure_X_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], \
            self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs[:, 0], self.ijkl_order)
        print('X_s1s2_ijkl take', time() - t)

        print('measurement of C_ijkl, PHI_ijkl correlators takes', time() - t)
        self.GF_up_sum += np.einsum('ijkl,i->jkl', self.GF_up_stored[:self.cur_buffer_size, ...], signs[..., 0])
        self.GF_down_sum += np.einsum('ijkl,i->jkl', self.GF_down_stored[:self.cur_buffer_size, ...], signs[..., 0])
        self.cur_buffer_size = 0
        return


    def measure_heavy_observables(self, phi):
        print('refreshing gfs buffers...')
        self.refresh_gfs_buffer()
        t = time()
        mean_signs = np.mean(self.heavy_signs_history)

        idx_alpha = 0
        for gap_alpha, gap_name_alpha in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            idx_beta = 0
            for gap_beta, gap_name_beta in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names): 
                if self.config.name_group_dict[gap_name_alpha] != self.config.name_group_dict[gap_name_beta]:
                    continue
                if idx_beta > idx_alpha:
                    continue

                norm = gap_alpha.shape[0]  # N_s
                N_alpha = 1.0 * np.sum(np.abs(gap_alpha) ** 2) / norm  # N_alpha
                N_beta = 1.0 * np.sum(np.abs(gap_beta) ** 2) / norm  # N_beta
                
                total_chi = self.config.dt * get_gap_susceptibility(gap_alpha, gap_beta, \
                    self.ijkl, self.C_ijkl, np.ones(shape = gap_alpha.shape)) / (self.num_chi_samples * mean_signs)
                free_chi = self.config.dt * np.sum([np.trace(self.GF_up_sum[tau, ...].dot(gap_beta).dot(self.GF_down_sum[tau, ...].T).dot(gap_alpha.T.conj())) \
                                   for tau in range(self.config.Nt)]) / ((self.num_chi_samples * mean_signs) ** 2)


                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_real'] = \
                    np.real((total_chi - free_chi) / norm / np.sqrt(N_alpha * N_beta))
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_real'] = \
                    np.real(total_chi / norm / np.sqrt(N_alpha * N_beta))

                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_imag'] = \
                    np.imag((total_chi - free_chi) / norm / np.sqrt(N_alpha * N_beta))
                self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_imag'] = \
                    np.imag(total_chi / norm / np.sqrt(N_alpha * N_beta))
                idx_beta += 1
            idx_alpha += 1

        for gap, gap_name in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            norm = gap.shape[0]
            N = 1.0 * np.sum(np.abs(gap) ** 2) / norm
            total_chi_n = get_gap_susceptibility(gap, gap, \
                self.ijkl, self.PHI_ijkl, self.distances_list) / (self.num_chi_samples * mean_signs)
            free_chi_n = np.trace((self.GF_up_sum[0, ...] * self.distances_list).dot(gap).dot(self.GF_down_sum[0, ...].T).dot(gap.T.conj())) \
                 / ((self.num_chi_samples * mean_signs) ** 2)

            corr_length_gap = total_chi_n - free_chi_n
            total_chi_d = get_gap_susceptibility(gap, gap, \
                self.ijkl, self.PHI_ijkl, np.ones(shape=gap.shape)) / (self.num_chi_samples * mean_signs)
            free_chi_d = np.trace((self.GF_up_sum[0, ...]).dot(gap).dot(self.GF_down_sum[0, ...].T).dot(gap.T.conj())) \
                 / ((self.num_chi_samples * mean_signs) ** 2)
            corr_length_gap = corr_length_gap / (total_chi_d - free_chi_d) / 4

            self.gap_observables_list[gap_name + '_corr_length'] = corr_length_gap.real
            self.gap_observables_list[gap_name + '_Sq0'] = (total_chi_d - free_chi_d).real / norm / N
            self.gap_observables_list[gap_name + '_Pq0'] = (total_chi_n - free_chi_n).real / norm / N

        for order_list, order_name in zip(self.config.waves_list, self.config.waves_list_names):
            order = order_list[0]
            norm = order.shape[0]  # N_s
            N = 1.0 * np.sum(np.abs(order) ** 2) / norm  # N_alpha

            order_up = order[:order.shape[0] // 2, :order.shape[1] // 2]
            order_down = order[order.shape[0] // 2:, order.shape[1] // 2:]
            order_correlator = get_order_average_disconnected(order_up, order_up, self.ijkl_order, self.X_uu_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average_disconnected(order_up, order_down, self.ijkl_order, self.X_ud_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average_disconnected(order_down, order_up, self.ijkl_order, self.X_du_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average_disconnected(order_down, order_down, self.ijkl_order, self.X_dd_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average_connected(order_up, self.ijkl_order, self.Z_uu_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average_connected(order_down, self.ijkl_order, self.Z_dd_ijkl, self.ik_marking, self.config.Ls)
            order_correlator = order_correlator.reshape((self.config.Ls, self.config.Ls))

            order_correlator /= (self.num_chi_samples * norm * N * mean_signs)
            self.order_observables_list[order_name + '_order'] = order_correlator

        print('obtaining of observables', time() - t)
        return


    def write_heavy_observables(self, phi, n_sweep):
        config = phi.config
        self.measure_heavy_observables(phi)
        signs = np.array(self.heavy_signs_history)

        gap_data = [n_sweep, np.mean(signs)]
        name = os.path.join(self.local_workdir, 'chi_vertex_{:d}.npy'.format(n_sweep))
        idx_alpha = 0; idx_beta = 0
        chi_vertex = np.zeros((len(self.config.pairings_list_names), len(self.config.pairings_list_names)), dtype=np.complex64)
        chi_total = np.zeros((len(self.config.pairings_list_names), len(self.config.pairings_list_names)), dtype=np.complex64)

        corr_lengths = []
        Sq0 = []
        Pq0 = []
        for _, gap_name_alpha in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
            corr_lengths.append(self.gap_observables_list[gap_name_alpha + '_corr_length'])
            Sq0.append(self.gap_observables_list[gap_name_alpha + '_Sq0'])
            Pq0.append(self.gap_observables_list[gap_name_alpha + '_Pq0'])
            idx_beta = 0
            for _, gap_name_beta in zip(self.config.pairings_list_unwrapped, self.config.pairings_list_names):
                chi_vertex[idx_alpha, idx_beta] = self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_real'] + \
                                           1.0j * self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_imag']
                chi_total[idx_alpha, idx_beta] = self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_real'] + \
                                           1.0j * self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_imag']
                gap_data.append(self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_real']) # norm already accounted
                gap_data.append(self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_real'])
                gap_data.append(self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_vertex_imag']) # norm already accounted
                gap_data.append(self.gap_observables_list[gap_name_alpha + '*' + gap_name_beta + '_chi_total_imag'])
                idx_beta += 1
            idx_alpha += 1
        name = os.path.join(self.local_workdir, 'chi_vertex_{:d}.npy'.format(n_sweep))
        np.save(name, (chi_vertex + chi_vertex.conj().T) / 2.)
        name = os.path.join(self.local_workdir, 'chi_total_{:d}.npy'.format(n_sweep))
        np.save(name, (chi_total + chi_total.conj().T) / 2.)
        name = os.path.join(self.local_workdir, 'Sq0_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(Sq0))
        name = os.path.join(self.local_workdir, 'Pq0_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(Pq0))
        name = os.path.join(self.local_workdir, 'corr_lengths_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(corr_lengths))

        np.save(os.path.join(self.local_workdir, 'gap_names.npy'), np.array(self.config.pairings_list_names))
        f = open(os.path.join(self.local_workdir, 'name_group_dict.pkl'), "wb")
        pickle.dump(self.config.name_group_dict, f)
        f.close()


        orders = np.zeros((len(self.config.waves_list_names), self.config.Ls, self.config.Ls), dtype=np.complex64)

        idx_order = 0
        for order_name in self.config.waves_list_names:
            orders[idx_order, ...] = self.order_observables_list[order_name + '_order']
            idx_order += 1

        np.save(os.path.join(self.local_workdir, 'orders_names.npy'), np.array(self.config.waves_list_names))
        name = os.path.join(self.local_workdir, 'order_{:d}.npy'.format(n_sweep))
        np.save(name, orders)


        self.gap_file.write(("{:d} " + "{:.6f} " * (len(gap_data) - 1) + '\n').format(n_sweep, *gap_data[1:]))
        self.gap_file.flush()
        self.refresh_heavy_logs()
        return

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



@jit(nopython=True)
def measure_gfs_correlator(GF_up, GF_down, ijkl):
    C_ijkl = np.zeros(len(ijkl), dtype=np.float64)
    idx = 0

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        C_ijkl[xi] = np.dot(GF_up[:, i, k], GF_down[:, j, l])
    return C_ijkl



# <(delta_ij - G^up(l, j)) G^up(i, k)>
@jit(nopython=True)
def measure_Z_correlator(GF_sigma, signs, ijkl):
    Z_ijkl = np.zeros(len(ijkl), dtype=np.float64)
    idx = 0

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        delta_lj = 0 if l != j else 1
        Z_ijkl[xi] = np.sum((-GF_sigma[:, l, j] + delta_lj) * GF_sigma[:, i, k] * signs)

    return Z_ijkl


@jit(nopython=True)
def measure_X_correlator(GF_sigma1, GF_sigma2, signs, ijkl):
    X_ijkl = np.zeros(len(ijkl), dtype=np.float64)
    idx = 0

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        delta_ij = 0 if i != j else 1
        delta_kl = 0 if k != l else 1

        X_ijkl[xi] = np.sum((GF_sigma1[:, i, j] - delta_ij) * (GF_sigma2[:, l, k] - delta_kl) * signs)

    return X_ijkl


@jit(nopython=True)
def get_idxs_list(reduced_A):
    ijkl = []

    for i in range(reduced_A.shape[0]):
        for k in range(reduced_A.shape[0]):
            for j in reduced_A[i, ...]:
                for l in reduced_A[k, ...]:
                    ijkl.append(np.array([i, j, k, l]))
    return ijkl

@jit(nopython=True)
def get_gap_susceptibility(gap_alpha, gap_beta, ijkl, C_ijkl, weight):
    corr = 0.0 + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        corr += np.conj(gap_alpha[i, j]) * gap_beta[k, l] * C_ijkl[xi] * weight[i, k]
    return corr


@jit(nopython=True)
def get_order_average_disconnected(order_s1, order_s2, ijkl, X_s1s2_ijkl, ik_marking, Ls):
    corr = np.zeros(Ls * Ls) + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        corr[ik_marking[i, k]] += np.conj(order_s1[i, j]) * order_s2[k, l] * X_s1s2_ijkl[xi]
    return corr


@jit(nopython=True)
def get_order_average_connected(order_s, ijkl, Z_ss_ijkl, ik_marking, Ls):
    corr = np.zeros(Ls * Ls) + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        corr[ik_marking[i, k]] += np.conj(order_s[i, j]) * order_s[k, l] * Z_ss_ijkl[xi]
    return corr


@jit(nopython=True)
def gap_gap_correlator(gap, ijkl, PHI_ijkl, adj_marking):
    corr_list = np.zeros(len(adj_marking)) + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        adj_index = adj_marking[i, k]
        corr_list[adj_index] += np.conj(gap[i, j]) * gap[k, l] * PHI_ijkl[xi]
    return corr_list.real


def get_ik_marking(config):
    return _get_ik_marking(config.Ls, config.n_orbitals, config.n_sublattices, config.total_dof)

@jit(nopython=True)
def _get_ik_marking(Ls, n_orbitals, n_sublattices, total_dof):
    A = np.zeros((total_dof // 2, total_dof // 2), dtype = np.int64)
    for i in range(total_dof // 2):
        oi, si, xi, yi = models.from_linearized_index(i, Ls, n_orbitals, n_sublattices)
        for j in range(total_dof // 2):
            oj, sj, xj, yj = models.from_linearized_index(j, Ls, n_orbitals, n_sublattices)

            dx = (xi - xj) % Ls
            dy = (yi - yj) % Ls
            index = dy * Ls + dx
            A[i, j] = index
    return A
