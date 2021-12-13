import numpy as np
import models
from time import time
import auxiliary_field
from numba import jit
import os
from collections import OrderedDict
from opt_parameters import waves
import pickle


@jit(nopython=True)
def get_fft(N):
    W = np.zeros((N ** 2, N ** 2), dtype=np.complex128)
    for kx in range(N):
        for ky in range(N):
            for x in range(N):
                for y in range(N):
                    W[x * N + y, kx * N + ky] = np.exp(2.0j * np.pi / N * kx * x + 2.0j * np.pi / N * ky * y)
    return np.kron(W, np.eye(4)).conj()

try:
    import cupy as cp
except ImportError:
    pass

class Observables:
    def __init__(self, phi, local_workdir, local_workdir_heavy):
        self.config = phi.config 
        self.local_workdir = local_workdir
        self.local_workdir_heavy = local_workdir_heavy

        open_mode = 'a' if phi.config.obs_start_type == 'presaved' else 'w'
        
        self.log_file = open(os.path.join(self.local_workdir, 'general_log.dat'), open_mode)
        self.gap_file = open(os.path.join(self.local_workdir, 'gap_log.dat'), open_mode)
        self.gf_file = open(os.path.join(self.local_workdir, 'gf_log.dat'), open_mode)

        self.X_ijkl_collinear_filename = os.path.join(self.local_workdir_heavy, 'C_ijkl_collinear')
        self.X_ijkl_anticollinear_filename = os.path.join(self.local_workdir_heavy, 'C_ijkl_anticollinear')
        self.Z_ijkl_collinear_filename = os.path.join(self.local_workdir_heavy, 'Phi_ijkl_collinear')
        self.Z_ijkl_anticollinear_filename = os.path.join(self.local_workdir_heavy, 'Phi_ijkl_anticollinear')
        self.SC_ijkl_filename = os.path.join(self.local_workdir_heavy, 'SC_ijkl')


        self.G_sum_filename = os.path.join(self.local_workdir_heavy, 'G_sum')
        self.G_sum_equaltime_filename = os.path.join(self.local_workdir_heavy, 'G_sum_equaltime')
        self.G_sum_backwards_filename = os.path.join(self.local_workdir_heavy, 'G_sum_backwards')

        self.num_samples_filename = os.path.join(self.local_workdir, 'n_samples')

        self.refresh_light_logs()
        self.init_light_log_file()
        self.global_average_sign = 0.
        self.global_average_sign_measurements = 0

        # for Gap-Gap susceptibility
        self.reduced_A_gap = models.get_reduced_adjacency_matrix(self.config, self.config.max_square_order_distance)
        self.ijkl = np.array(get_idxs_list(self.reduced_A_gap))
        # print(self.ijkl)
        # print(len(self.ijkl))
        # exit(-1)
        np.save('ijkl_{:d}.npy'.format(self.config.Ls), self.ijkl)
        #exit(-1)

        # for order-order susceptibility
        self.reduced_A_order = models.get_reduced_adjacency_matrix(self.config, \
            self.config.max_square_order_distance)
        self.ik_marking = get_ik_marking(self.config)
        self.distances_list = models.get_distances_list(self.config)  # pairwise distances |r_i - r_j|^2

        # for fast correlator computation
        self.adj_list_marking = np.zeros((phi.config.total_dof // 2, phi.config.total_dof // 2)).astype(np.int64)
        for idx, adj in enumerate(self.config.adj_list):
            self.adj_list_marking[adj[0] > 0.5] = idx

        NN = models.get_adjacency_list(self.config, orbital_mod = False)[0][1]

        self.NN = phi.connectivity
        self.chiral_to_xy = np.kron(np.eye(self.config.total_dof // 2 // 2), np.array([[1., 1.0j], [1.0, -1.0j]]) / np.sqrt(2))
        self.violation_vals = []
        self.violation_signs = []

        self.load_presaved_GF_data()
        self.refresh_heavy_logs()
        self.init_heavy_logs_files()


        self.init_light_cumulants()
        self.n_saved_times = 0

        self.fft = get_fft(self.config.Ls)

        self.phi = phi
        return

    def init_light_log_file(self):
        self.log_file.write('n_sweep ' + '⟨ratio⟩ ' + '⟨acc⟩ ' + 'Re_⟨sign_gen⟩ Im_⟨sign_gen⟩ Re_⟨sign_obs_l⟩ Im_⟨sign_obs_l⟩ ⟨Re n⟩ ⟨Im n⟩ ⟨Re E_K⟩ ⟨Im E_K⟩ ⟨Re E_CU⟩ ⟨Im E_CU⟩ ⟨Re E_T⟩ ⟨Im E_T⟩ ⟨Re m_z^2⟩ ⟨Im m_z^2⟩\n')
        return

    def init_heavy_logs_files(self):
        self.gap_file.write('step sign_obs ')

        self.gap_file.write('\n')
        return

    def update_history(self, ratio, accepted, sign):
        self.ratio_history.append(ratio)
        self.acceptance_history.append(accepted)
        self.sign_history.append(sign)

    def load_presaved_GF_data(self):
        ### load pveroously stored GF data if exists ###
        loaded = False

        # the data is stored in two copies to avoid the bug with file corruption
        # upon restart we are happy to load any copy
        if self.config.obs_start_type == 'presaved' and (os.path.isfile(self.G_sum_filename + '.npy') \
                                                  or os.path.isfile(self.G_sum_filename + '_dump.npy')):
            try:
                self.GF_sum = np.load(self.G_sum_filename + '.npy')
                self.GF_sum_backwards = np.load(self.G_sum_backwards_filename + '.npy')
                self.GF_sum_equaltime = np.load(self.G_sum_equaltime_filename + '.npy')
                self.X_ijkl_collinear = np.load(self.X_ijkl_collinear_filename + '.npy')
                self.X_ijkl_anticollinear = np.load(self.X_ijkl_anticollinear_filename + '.npy')
                self.Z_ijkl_collinear = np.load(self.Z_ijkl_collinear_filename + '.npy')
                self.Z_ijkl_anticollinear = np.load(self.Z_ijkl_anticollinear_filename + '.npy')
                self.SC_ijkl = np.load(self.SC_ijkl_filename + '.npy')

                self.num_chi_samples = np.load(self.num_samples_filename + '.npy')[0]

                print('Successfully loaded saved GFs files from default location')
                loaded = True
            except Exception:
                print('Failed to load from default location: will try to load from dump')

                try:
                    self.GF_sum = np.load(self.G_sum_filename + '_dump.npy')
                    self.GF_sum_backwards = np.load(self.G_sum_backwards_filename + '_dump.npy')
                    self.GF_sum_equaltime = np.load(self.G_sum_equaltime_filename + '_dump.npy')

                    self.X_ijkl_collinear = np.load(self.X_ijkl_collinear_filename + '_dump.npy')
                    self.X_ijkl_anticollinear = np.load(self.X_ijkl_anticollinear_filename + '_dump.npy')

                    self.Z_ijkl_collinear = np.load(self.Z_ijkl_collinear_filename + '_dump.npy')
                    self.Z_ijkl_anticollinear = np.load(self.Z_ijkl_anticollinear_filename + '_dump.npy')

                    self.SC_ijkl = np.load(self.SC_ijkl_filename + '_dump.npy')

                    self.num_chi_samples = np.load(self.num_samples_filename + '_dump.npy')[0]

                    print('Successfully loaded saved GFs files from dump location')
                    loaded = True
                except Exception:
                    print('Failed to load from dump location: will start from scratch')

        if not loaded:
            print('Initialized GFs buffer from scratch')
            self.GF_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)
            self.GF_sum_backwards = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)
            self.GF_sum_equaltime = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)
            self.num_chi_samples = 0
            self.X_ijkl_collinear = np.zeros((self.config.Nt, len(self.ijkl)), dtype=np.complex64)
            self.X_ijkl_anticollinear = np.zeros((self.config.Nt, len(self.ijkl)), dtype=np.complex64)
            self.Z_ijkl_collinear = np.zeros((self.config.Nt, len(self.ijkl)), dtype=np.complex64)
            self.Z_ijkl_anticollinear = np.zeros((self.config.Nt, len(self.ijkl)), dtype=np.complex64)

            self.SC_ijkl = np.zeros((self.config.Nt, len(self.ijkl)), dtype=np.complex64)
        return

    def save_GF_data(self):
        ### save GF data ###
        addstring = '_dump.npy' if self.n_saved_times % 2 == 0 else '.npy'

        np.save(self.G_sum_filename + addstring, self.GF_sum)
        np.save(self.G_sum_backwards_filename + addstring, self.GF_sum_backwards)
        np.save(self.G_sum_equaltime_filename + addstring, self.GF_sum_equaltime)
        np.save(self.X_ijkl_collinear_filename + addstring, self.X_ijkl_collinear)
        np.save(self.X_ijkl_anticollinear_filename + addstring, self.X_ijkl_anticollinear) 
        np.save(self.Z_ijkl_collinear_filename + addstring, self.Z_ijkl_collinear)
        np.save(self.Z_ijkl_anticollinear_filename + addstring, self.Z_ijkl_anticollinear)
        np.save(self.SC_ijkl_filename + addstring, self.SC_ijkl)

        np.save(self.num_samples_filename + addstring, np.array([self.num_chi_samples]))
        self.n_saved_times += 1
        return

    def refresh_heavy_logs(self):
        self.gap_file.flush()

        ### buffer for efficient GF-measurements ###
        self.cur_buffer_size = 0; self.max_buffer_size = 5
        self.GF_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)
        self.GF_stored_backwards = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)
        self.GF_stored_equaltime = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2), dtype=np.complex64)

        # self.gap_observables_list = OrderedDict()
        # self.order_observables_list = OrderedDict()

        adj_list = self.config.adj_list[:self.config.n_adj_density]  # only largest distance

        density_adj_list = self.config.adj_list[:self.config.n_adj_density]  # only smallest distance
        self.heavy_signs_history = []

        return

    def init_light_cumulants(self):
        self.light_cumulants = OrderedDict({
            '⟨density⟩' : 0.0 + 0.0j, 
            '⟨E_K⟩' : 0.0 + 0.0j, 
            '⟨E_CU⟩' : 0.0 + 0.0j,
            '⟨E_T⟩' : 0.0 + 0.0j,
            '⟨m_z^2⟩' : 0.0 + 0.0j,
            })
        self.n_cumulants = 0
        return


    def refresh_light_logs(self):
        # self.log_file.flush()
        self.light_observables_list = OrderedDict({
            '⟨density⟩' : [], 
            '⟨E_K⟩' : [], 
            '⟨E_CU⟩' : [],
            '⟨E_T⟩' : [],
            '⟨m_z^2⟩' : [],
            })

        self.light_signs_history = []

        self.ratio_history = []
        self.acceptance_history = []
        self.sign_history = []

        return

    def print_greerings(self):
        print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
                "lattice = {:d}^2 x {:d}".format(self.config.obs_start_type, 1.0 / self.config.dt / self.config.Nt, \
                        self.config.mu, self.config.Ls, self.config.Nt))
        print('# sweep ⟨r⟩ ⟨acc⟩ ⟨Re_sign⟩ ⟨Im_sign⟩ ⟨Re n⟩ ⟨Im n⟩ ⟨Re E_K⟩ ⟨Im E_K⟩ ⟨Re E_CU⟩ ⟨Im E_CU⟩ ⟨Re E_T⟩ ⟨Im E_T⟩ ⟨Re m_z^2⟩ ⟨Im m_z^2⟩')
        return

    def print_std_logs(self, n_sweep):
        if self.n_cumulants == 0 or self.global_average_sign == 0:
            return
        print("{:d} {:.5f} {:.3f} {:.6f} {:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}".format(
            n_sweep, 
            np.mean(self.ratio_history),
            np.mean(self.acceptance_history),
            np.real(self.global_average_sign),
            np.imag(self.global_average_sign),
            np.real(self.light_cumulants['⟨density⟩'] / self.n_cumulants / self.global_average_sign),
            np.imag(self.light_cumulants['⟨density⟩'] / self.n_cumulants / self.global_average_sign),
            np.real(self.light_cumulants['⟨E_K⟩'] / self.n_cumulants / self.global_average_sign),
            np.imag(self.light_cumulants['⟨E_K⟩'] / self.n_cumulants / self.global_average_sign),
            np.real(self.light_cumulants['⟨E_CU⟩'] / self.n_cumulants / self.global_average_sign),
            np.imag(self.light_cumulants['⟨E_CU⟩'] / self.n_cumulants / self.global_average_sign),
            np.real(self.light_cumulants['⟨E_T⟩'] / self.n_cumulants / self.global_average_sign),
            np.imag(self.light_cumulants['⟨E_T⟩'] / self.n_cumulants / self.global_average_sign),
            np.real(self.light_cumulants['⟨m_z^2⟩'] / self.n_cumulants / self.global_average_sign),
            np.imag(self.light_cumulants['⟨m_z^2⟩'] / self.n_cumulants / self.global_average_sign),
            ))
        return



    def measure_light_observables(self, phi, current_det_sign, n_sweep, print_gf = False):
        self.light_signs_history.append(current_det_sign)  

        k = kinetic_energy(phi).item()
        CU, CV = Coloumb_energy(phi)
        density = total_density(phi).item()

        G_up, G_down = phi.get_equal_time_GF()

        self.light_observables_list['⟨density⟩'].append(density * current_det_sign)
        self.light_observables_list['⟨E_K⟩'].append(k * current_det_sign)
        self.light_observables_list['⟨E_CU⟩'].append(CU * current_det_sign)
        self.light_observables_list['⟨E_T⟩'].append((k + CU + CV) * current_det_sign)
        self.light_observables_list['⟨m_z^2⟩'].append(total_mz_squared(G_down, G_up) * current_det_sign)

        #self.gf_nonhermicity = np.sum(np.abs(phi.G_up_sum.conj().T - phi.G_up_sum)) / phi.n_gf_measures
        #self.gf_imaginary = np.max(np.abs(phi.G_up_sum.imag)) / phi.n_gf_measures

        self.n_cumulants += 1
        self.light_cumulants['⟨density⟩'] += (density * current_det_sign)
        self.light_cumulants['⟨E_K⟩'] += (k * current_det_sign)
        self.light_cumulants['⟨E_CU⟩'] += (CU * current_det_sign)
        self.light_cumulants['⟨E_T⟩'] += ((k + CU + CV) * current_det_sign)
        self.light_cumulants['⟨m_z^2⟩'] += (total_mz_squared(G_down, G_up) * current_det_sign)

        return

    def signs_avg(self, array, signs):
        return np.mean(np.array(array) * signs)


    def signs_std(self, array, signs):
        return (np.std(np.array(array) * signs) / np.mean(signs) - \
                np.std(signs) * np.mean(np.array(array) * signs) / np.mean(signs) ** 2) / np.sqrt(len(signs))

    def write_light_observables(self, config, n_sweep):
        signs = np.array(self.light_signs_history)

        self.global_average_sign = (self.global_average_sign * self.global_average_sign_measurements + \
                np.sum(signs)) / (self.global_average_sign_measurements + len(signs))
        self.global_average_sign_measurements += len(signs)

        num_data = []
        for _, val in self.light_cumulants.items():
            num_data.append((val / self.global_average_sign / self.n_cumulants).real)
            num_data.append((val / self.global_average_sign / self.n_cumulants).imag)

        data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), self.global_average_sign.real, self.global_average_sign.imag,
                np.mean(self.light_signs_history).real, np.mean(self.light_signs_history).imag] + num_data
        self.log_file.write(("{:d} " + "{:.6f} " * (len(data) - 1) + '\n').format(n_sweep, *data[1:]))

        if n_sweep % 100 == 0:
            self.log_file.flush()

        self.global_average_sign = (self.global_average_sign * self.global_average_sign_measurements + \
                np.sum(signs)) / (self.global_average_sign_measurements + len(signs))
        self.global_average_sign_measurements += len(signs)
        self.refresh_light_logs()  # TODO: keep sign-averaged observables and accumulate them forever
        return

    def measure_green_functions(self, phi, current_det_sign):
        t = time()
        phi.copy_to_GPU()


        phi.refresh_all_decompositions()
        phi.refresh_G_functions()
        identity = np.eye(phi.Bdim, dtype=np.complex128)


        ### equaltime ###
        GFs_up_equaltime = []
        GFs_down_equaltime = []
        for time_slice in range(phi.config.Nt):
            if time_slice in phi.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
                index = np.where(phi.refresh_checkpoints == time_slice)[0][0]
                phi.append_new_decomposition(phi.refresh_checkpoints[index - 1], time_slice)
                phi.refresh_G_functions()

            GFs_up_equaltime.append(phi.make_symmetric_displacement(phi.current_G_function_up, valley=+1))
            GFs_down_equaltime.append(phi.make_symmetric_displacement(phi.current_G_function_down, valley=-1))
            phi.wrap_up(time_slice)

        #GFs_up_naive = phi.get_G_tau_tau_naive(+1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(-GFs_up_equaltime[i] + identity - GFs_up_naive[i])))


        #GFs_down_naive = phi.get_G_tau_tau_naive(-1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(-GFs_down_equaltime[i] + identity - GFs_down_naive[i])))

        # contains G_tt
        GFs_equaltime = np.array([\
                np.kron(gf_up, np.array([[1, 0], [0, 0]])) + \
                np.kron(gf_down, np.array([[0, 0], [0, 1]])) \
                for gf_up, gf_down in zip(GFs_up_equaltime, GFs_down_equaltime)])
        self.GF_stored_equaltime[self.cur_buffer_size, ...] = GFs_equaltime



        ### forwards ###
        phi.current_G_function_up = phi.get_G_no_optimisation(+1, -1)[0]
        phi.current_G_function_down = phi.get_G_no_optimisation(-1, -1)[0]
        print('G_0_noopt takes', time() - t); t = time()

        GFs_up = np.array(phi.get_nonequal_time_GFs(+1.0, phi.current_G_function_up))
        GFs_down = np.array(phi.get_nonequal_time_GFs(-1.0, phi.current_G_function_down))
        print('G_nonequal_time takes', time() - t); t = time()


        #GFs_up_naive = phi.get_G_tau_0_naive(+1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(GFs_up[i] - GFs_up_naive[i])))


        #GFs_down_naive = phi.get_G_tau_0_naive(-1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(GFs_down[i] - GFs_down_naive[i])))


        GFs = np.array([\
                np.kron(gf_up, np.array([[1, 0], [0, 0]])) + \
                np.kron(gf_down, np.array([[0, 0], [0, 1]])) \
                for gf_up, gf_down in zip(GFs_up, GFs_down)])
        print('kron takes', time() - t); t = time()

        phi.copy_to_CPU()
        self.GF_stored[self.cur_buffer_size, ...] = GFs
        
        print('obtaining of non-equal Gfs takes', time() - t)





        ### backwards ###
        GFs_up = np.array(phi.get_nonequal_time_GFs_inverted(+1.0, phi.current_G_function_up))
        GFs_down = np.array(phi.get_nonequal_time_GFs_inverted(-1.0, phi.current_G_function_down))
        print('G_nonequal_time takes', time() - t); t = time()


        #GFs_up_naive = phi.get_G_0_tau_naive(+1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(GFs_up[i] - GFs_up_naive[i])))


        #GFs_down_naive = phi.get_G_0_tau_naive(-1)
        #for i in range(phi.config.Nt):
        #    print(np.sum(np.abs(GFs_down[i] - GFs_down_naive[i])))


        GFs = np.array([\
                np.kron(gf_up, np.array([[1, 0], [0, 0]])) + \
                np.kron(gf_down, np.array([[0, 0], [0, 1]])) \
                for gf_up, gf_down in zip(GFs_up, GFs_down)])
        print('kron takes', time() - t); t = time()

        phi.copy_to_CPU()
        self.GF_stored_backwards[self.cur_buffer_size, ...] = GFs
        print('obtaining of non-equal Gfs takes', time() - t)





        self.cur_buffer_size += 1
        if self.cur_buffer_size == self.max_buffer_size:
            self.refresh_gfs_buffer()
        return


    def refresh_gfs_buffer(self):
        if self.cur_buffer_size == 0:
            return

        self.GF_sum += self.GF_stored[:self.cur_buffer_size, ...].sum(axis = 0)
        self.GF_sum_backwards += self.GF_stored_backwards[:self.cur_buffer_size, ...].sum(axis = 0)
        self.GF_sum_equaltime += self.GF_stored_equaltime[:self.cur_buffer_size, ...].sum(axis = 0)

        self.num_chi_samples += self.cur_buffer_size

        print('current buffer size = {:d}'.format(self.cur_buffer_size))
        t = time()


        shape = self.GF_stored[:self.cur_buffer_size, ...].shape
        new_shape = (shape[0], shape[2], shape[3])

        G_prepared_00 = np.asfortranarray(np.repeat(self.GF_stored_equaltime[:self.cur_buffer_size, 0:1, ...], self.config.Nt, axis=1))
        G_prepared_tt = np.asfortranarray(self.GF_stored_equaltime[:self.cur_buffer_size, ...])

        ### G_prepared-- list 0 ...tau, G_prepared_backwards -- similarly
        G_prepared = np.asfortranarray(self.GF_stored[:self.cur_buffer_size, ...])
        G_prepared_backwards = np.asfortranarray(self.GF_stored_backwards[:self.cur_buffer_size, ...])
        t = time()

        arrays_rolled = [np.roll(G_prepared_tt, axis=1, shift=dt) for dt in range(G_prepared_tt.shape[1])]

        dX_ijkl_collinear, dX_ijkl_anticollinear = measure_gfs_correlatorX(arrays_rolled, G_prepared, G_prepared_backwards, self.ijkl, self.config.Ls)
        self.X_ijkl_collinear += dX_ijkl_collinear
        self.X_ijkl_anticollinear += dX_ijkl_anticollinear

        self.SC_ijkl += measure_gfs_correlatorCS(G_prepared, self.ijkl, self.config.Ls)

        new_shape = (shape[0] * shape[1], shape[2], shape[3])

        dZ_ijkl_collinear, dZ_ijkl_anticollinear = measure_gfs_correlator_sametime(G_prepared[:, 0:1, ...], self.ijkl, self.config.Ls)
        self.Z_ijkl_collinear += dZ_ijkl_collinear
        self.Z_ijkl_anticollinear += dZ_ijkl_anticollinear

        self.cur_buffer_size = 0
        return


    def measure_heavy_observables(self, phi):
        self.log_file.flush()
        self.refresh_gfs_buffer()
        return



    def write_heavy_observables(self, phi, n_sweep):
        config = phi.config
        self.measure_heavy_observables(phi)

        self.refresh_heavy_logs()
        self.save_GF_data()
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
    G_function_up, G_function_down = phi.get_equal_time_GF()

    K_mean = phi_field.config.main_hopping * xp.einsum('ij,ji', G_function_up + G_function_down, A) / (phi_field.config.total_dof // 2)
    return K_mean

def double_occupancy(h_configuration, K, config):
    G_function_up, G_function_down = phi.get_equal_time_GF()

    return xp.trace(G_function_up * G_function_down) / (config.n_sublattices * config.n_orbitals * config.Ls ** 2)

def SzSz_onsite(phi_field):
    G_function_up, G_function_down = phi.get_equal_time_GF()

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
    G_function_up, G_function_down = phi.get_equal_time_GF()

    return (xp.einsum('i,j,ij', xp.diag(G_function_up) - xp.diag(G_function_down), xp.diag(G_function_up) - xp.diag(G_function_down), adj) - \
            xp.einsum('ij,ji,ij', G_function_up, G_function_up, adj) - xp.einsum('ij,ji,ij', G_function_down, G_function_down, adj)) / xp.sum(adj)

def n_up_n_down_correlator(phi, adj):
    G_function_up, G_function_down = phi.get_equal_time_GF()

    return phi.la.einsum('i,j,ij', phi.la.diag(G_function_up), phi.la.diag(G_function_down), adj) / phi.la.sum(adj)

def kinetic_energy(phi):
    G_function_up, G_function_down = phi.get_equal_time_GF()

    return phi.config.n_spins * (phi.la.einsum('ij,ij', phi.K_matrix_plus.T, G_function_up) + \
                                 phi.la.einsum('ij,ij', phi.K_matrix_minus.T, G_function_down)) / G_function_up.shape[0] / 2.

'''  # for U/V interaction
def Coloumb_energy(phi):
    G_function_up, G_function_down = phi.get_equal_time_GF()

    G_function = np.kron(G_function_up, np.array([[1, 0], [0, 0]])) + np.kron(G_function_down, np.array([[0, 0], [0, 1]]))

    energy_coloumb_U = 0.0 + 0.0j 
    energy_coloumb_V = 0.0 + 0.0j

    energy_coloumb_U = phi.config.U * phi.la.sum(phi.la.diag(G_function) * phi.la.diag(G_function)).item() / G_function.shape[0]

    if phi.config.n_orbitals == 1:
        return energy_coloumb_U

    orbital_1 = phi.la.arange(0, G_function.shape[0], 2)
    orbital_2 = phi.la.arange(1, G_function.shape[0], 2)
    energy_coloumb_U += phi.config.U * phi.la.einsum('i,i', (2 * phi.la.diag(G_function_up) - 1), (2 * phi.la.diag(G_function_down) - 1)).item() / G_function.shape[0]

    total_density = 2 * (phi.la.diag(G_function_up) + phi.la.diag(G_function_down))

    energy_coloumb_V += phi.config.V * np.einsum('i,ij,j', total_density, phi.connectivity, total_density) / G_function.shape[0] / 2.  # why all contribs imag? bullshit


    for GF in [G_function_up, G_function_down]:
        energy_coloumb_V -= 2 * phi.config.V * np.trace((GF * phi.connectivity).dot(GF * phi.connectivity)) / G_function.shape[0] / 2.

    return energy_coloumb_U, energy_coloumb_V
'''


def Coloumb_energy(phi):
    G_function_up, G_function_down = phi.get_equal_time_GF()

    energy = 0.0 + 0.0j
    for h in phi.hexagons:
        total_h_charge = np.sum(G_function_up[np.array(h), np.array(h)] + G_function_down[np.array(h), np.array(h)]) * phi.config.n_spins
        energy += phi.config.U / 9. / 2. * total_h_charge ** 2
        energy -= phi.config.U / 9. / 2. * np.sum(G_function_up[np.array(h), np.array(h)] ** 2 + \
                                                  G_function_down[np.array(h), np.array(h)] ** 2) * phi.config.n_spins

        for i in h:
            for j in h:
                if i == j:
                    continue
                energy -= phi.config.U / 9. / 2 * G_function_up[i, j] * G_function_up[j, i] * phi.config.n_spins
                energy -= phi.config.U / 9. / 2 * G_function_down[i, j] * G_function_down[j, i] * phi.config.n_spins

    return energy / G_function_up.shape[0] / 2., 0.0



@jit(nopython=True)
def measure_gfs_correlator_sametime(GF, ijkl, L):
    C_ijkl_collinear = np.zeros((len(ijkl)), dtype=np.complex128)
    C_ijkl_anticollinear = np.zeros((len(ijkl)), dtype=np.complex128)

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        for shift_x in range(1):  # FIXME FIXME FIXME
            for shift_y in range(1):  # FIXME FIXME FIXME
                ix, iy, io = (i // 4) // L, (i // 4) % L, i % 4
                jx, jy, jo = (j // 4) // L, (j // 4) % L, j % 4
                kx, ky, ko = (k // 4) // L, (k // 4) % L, k % 4
                lx, ly, lo = (l // 4) // L, (l // 4) % L, l % 4

                i_shift = io + ((ix + shift_x) % L) * 4 * L + ((iy + shift_y) % L) * 4
                j_shift = jo + ((jx + shift_x) % L) * 4 * L + ((jy + shift_y) % L) * 4
                k_shift = ko + ((kx + shift_x) % L) * 4 * L + ((ky + shift_y) % L) * 4
                l_shift = lo + ((lx + shift_x) % L) * 4 * L + ((ly + shift_y) % L) * 4


                A = np.sum(GF[:, 0, i_shift, j_shift] * GF[:, 0, l_shift, k_shift])
                B = np.sum(GF[:, 0, l_shift, j_shift] * GF[:, 0, i_shift, k_shift])
                C_ijkl_collinear[xi] += A - B
                C_ijkl_anticollinear[xi] += -B

    return C_ijkl_collinear / (L ** 2) / 4., C_ijkl_anticollinear / (L ** 2) / 4.


@jit(nopython=True)
def roll_last_axis(A, shift):
    A_rolled = np.empty(A.shape, dtype=np.complex64)

    for i in range(A.shape[0]):
        A_rolled[i] = np.roll(A[i], shift=shift)
    return A_rolled


@jit(nopython=True)
def measure_gfs_correlatorX(GF_tt, GF_forward, GF_backward, ijkl, L):
    C_ijkl_collinear = np.zeros((GF_forward.shape[1], len(ijkl)), dtype=np.complex64)
    C_ijkl_anticollinear = np.zeros((GF_forward.shape[1], len(ijkl)), dtype=np.complex64)

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        for shift_x in range(L):
            for shift_y in range(L):
                ix, iy, io = (i // 4) // L, (i // 4) % L, i % 4
                jx, jy, jo = (j // 4) // L, (j // 4) % L, j % 4
                kx, ky, ko = (k // 4) // L, (k // 4) % L, k % 4
                lx, ly, lo = (l // 4) // L, (l // 4) % L, l % 4

                i_shift = io + ((ix + shift_x) % L) * 4 * L + ((iy + shift_y) % L) * 4
                j_shift = jo + ((jx + shift_x) % L) * 4 * L + ((jy + shift_y) % L) * 4
                k_shift = ko + ((kx + shift_x) % L) * 4 * L + ((ky + shift_y) % L) * 4
                l_shift = lo + ((lx + shift_x) % L) * 4 * L + ((ly + shift_y) % L) * 4

                A = np.sum(GF_tt[:, :, i_shift, j_shift] * GF_00[:, :, l_shift, k_shift], axis = 0)
                B = np.sum(GF_forward[:, :, i_shift, k_shift] * GF_backward[:, :, l_shift, j_shift], axis = 0)

                C_ijkl_collinear[:, xi] += -B #A - B
                C_ijkl_anticollinear[:, xi] += -B

    #for dt in range(GF_tt.shape[1]):


    return C_ijkl_collinear / L ** 2, C_ijkl_anticollinear / L ** 2



@jit(nopython=True)
def measure_gfs_correlatorCS(GF, ijkl, L):
    C_ijkl = np.zeros((GF.shape[1], len(ijkl)), dtype=np.complex64)


    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        for shift_x in range(1):
            for shift_y in range(1):  # FIXME FIXME FIXME
                ix, iy, io = (i // 4) // L, (i // 4) % L, i % 4
                jx, jy, jo = (j // 4) // L, (j // 4) % L, j % 4
                kx, ky, ko = (k // 4) // L, (k // 4) % L, k % 4
                lx, ly, lo = (l // 4) // L, (l // 4) % L, l % 4

                i_shift = io + ((ix + shift_x) % L) * 4 * L + ((iy + shift_y) % L) * 4
                j_shift = jo + ((jx + shift_x) % L) * 4 * L + ((jy + shift_y) % L) * 4
                k_shift = ko + ((kx + shift_x) % L) * 4 * L + ((ky + shift_y) % L) * 4
                l_shift = lo + ((lx + shift_x) % L) * 4 * L + ((ly + shift_y) % L) * 4

                C_ijkl[:, xi] += np.sum(GF[:, :, j_shift, l_shift] * GF[:, :, i_shift, k_shift], axis = 0)

    return C_ijkl / (L ** 2) / 4.




'''
@jit(nopython=True)
def measure_gfs_correlatorZ(GF_forwards, GF_backwards, ijkl, L):
    C_ijkl = np.zeros((GF_forwards.shape[1], len(ijkl)), dtype=np.complex128)

    for xi in range(ijkl.shape[0]):
        i, j, k, l = ijkl[xi]
        for shift_x in range(L):
            for shift_y in range(L):
                ix, iy, io = (i // 4) // L, (i // 4) % L, i % 4
                jx, jy, jo = (j // 4) // L, (j // 4) % L, j % 4
                kx, ky, ko = (k // 4) // L, (k // 4) % L, k % 4
                lx, ly, lo = (l // 4) // L, (l // 4) % L, l % 4

                i_shift = io + ((ix + shift_x) % L) * 4 * L + ((iy + shift_y) % L) * 4
                j_shift = jo + ((jx + shift_x) % L) * 4 * L + ((jy + shift_y) % L) * 4
                k_shift = ko + ((kx + shift_x) % L) * 4 * L + ((ky + shift_y) % L) * 4
                l_shift = lo + ((lx + shift_x) % L) * 4 * L + ((ly + shift_y) % L) * 4

                C_ijkl[:, xi] += np.sum(GF_forwards[:, :, i_shift, k_shift] * GF_backwards[:, :, l_shift, j_shift], axis=0)

    return C_ijkl / (L ** 2) / 4.
'''

@jit(nopython=True)
def get_idxs_list(reduced_A):
    ijkl = [];

    for i in range(4):
        for k in range(reduced_A.shape[0]):
            for j in reduced_A[i, ...]:
                for l in reduced_A[k, ...]:
                    if (l % 2) + (i % 2) == (k % 2) + (j % 2):
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
def get_order_average(order_s1, order_s2, ijkl, X_s1s2_ijkl, ik_marking, Ls):
    corr = np.zeros(Ls * Ls * 16) + 0.0j

    for xi in range(len(ijkl)):
        i, j, k, l = ijkl[xi]
        corr[ik_marking[i, k]] += np.conj(order_s1[i, j]) * order_s2[k, l] * X_s1s2_ijkl[xi]
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
            index_spatial = dy * Ls + dx

            index_orbital = n_sublattices * n_sublattices * (oi * n_sublattices + si) + (oj * n_sublattices + sj)

            A[i, j] = index_spatial * n_orbitals ** 2 * n_sublattices ** 2 + index_orbital
    return A


def total_mz_squared(G_down, G_up):
    M = np.trace(G_up) - np.trace(G_down)
    vol = G_up.shape[0]
    return (M ** 2 + np.trace((np.eye(vol) - G_up).dot(G_up)) + \
            np.trace((np.eye(vol) - G_down).dot(G_down))) / vol ** 2
