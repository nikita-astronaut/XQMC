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
    def __init__(self, phi, local_workdir, local_workdir_heavy):
        self.config = phi.config 
        self.local_workdir = local_workdir
        self.local_workdir_heavy = local_workdir_heavy

        open_mode = 'a' if phi.config.start_type == 'presaved' else 'w'
        
        self.log_file = open(os.path.join(self.local_workdir, 'general_log.dat'), open_mode)
        self.gap_file = open(os.path.join(self.local_workdir, 'gap_log.dat'), open_mode)
        self.gf_file = open(os.path.join(self.local_workdir, 'gf_log.dat'), open_mode)

        self.C_ijkl_filename = os.path.join(self.local_workdir_heavy, 'C_ijkl')
        self.PHI_ijkl_filename = os.path.join(self.local_workdir_heavy, 'Phi_ijkl')
        self.G_up_sum_filename = os.path.join(self.local_workdir_heavy, 'G_up_sum')
        self.G_down_sum_filename = os.path.join(self.local_workdir_heavy, 'G_down_sum')

        self.Z_uu_ijkl_filename = os.path.join(self.local_workdir_heavy, 'Z_uu_ijkl')
        self.Z_dd_ijkl_filename = os.path.join(self.local_workdir_heavy, 'Z_dd_ijkl')

        self.X_uu_ijkl_filename = os.path.join(self.local_workdir_heavy, 'X_uu_ijkl')
        self.X_ud_ijkl_filename = os.path.join(self.local_workdir_heavy, 'X_ud_ijkl')
        self.X_du_ijkl_filename = os.path.join(self.local_workdir_heavy, 'X_du_ijkl')
        self.X_dd_ijkl_filename = os.path.join(self.local_workdir_heavy, 'X_dd_ijkl')

        self.chi_ijkl_total_filename = os.path.join(self.local_workdir_heavy, 'chi_ijkl_total')
        self.chi_ijkl_free_filename = os.path.join(self.local_workdir_heavy, 'chi_ijkl_free')


        self.sign_filename = os.path.join(self.local_workdir, 'sign')
        self.num_samples_filename = os.path.join(self.local_workdir, 'n_samples')


        self.refresh_light_logs()
        self.init_light_log_file()
        self.global_average_sign = 0.
        self.global_average_sign_measurements = 0

        # for Gap-Gap susceptibility
        self.reduced_A_gap = models.get_reduced_adjacency_matrix(self.config, \
            self.config.max_square_pairing_distance)
        self.ijkl = np.array(get_idxs_list(self.reduced_A_gap))
        #np.save('ijkl_{:d}.npy'.format(self.config.Ls), self.ijkl)
        #exit(-1)

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

        NN = models.get_adjacency_list(self.config, orbital_mod = False)[0][1]

        self.NN = phi.connectivity
        self.chiral_to_xy = np.kron(np.eye(self.config.total_dof // 2 // 2), np.array([[1., 1.0j], [1.0, -1.0j]]) / np.sqrt(2))
        self.O_pm_chiral = np.kron(NN, np.array([[0, 1], [0, 0]]))
        self.O_pm_xy = self.chiral_to_xy.conj().T.dot(self.O_pm_chiral).dot(self.chiral_to_xy)

        self.O_mm_chiral = np.kron(NN, np.array([[0, 0], [0, 1]]))
        self.O_mm_xy = self.chiral_to_xy.conj().T.dot(self.O_mm_chiral).dot(self.chiral_to_xy)

        self.violation_vals = []
        self.violation_signs = []

        ### for fourier transforms ###
        self.n_bands = self.config.n_sublattices * self.config.n_orbitals
        k_mesh = np.meshgrid(np.arange(self.config.Ls) / self.config.Ls, np.arange(self.config.Ls) / self.config.Ls)
        r_mesh = np.meshgrid(np.arange(self.config.Ls), np.arange(self.config.Ls))

        self.U_ft_space = np.exp(2. * np.pi * 1.0j * np.outer(k_mesh[0].flatten(), r_mesh[0].flatten()) + \
                                 2. * np.pi * 1.0j * np.outer(k_mesh[1].flatten(), r_mesh[1].flatten())) / self.config.Ls
        self.invert_momenta = np.zeros(self.config.Ls ** 2, dtype=np.int64)
        for k in range(self.config.Ls ** 2):
            kx = k // self.config.Ls
            ky = k % self.config.Ls
            q = ((-kx) % self.config.Ls) * self.config.Ls + ((-ky) % self.config.Ls)
            self.invert_momenta[k] = q
        assert np.allclose(self.invert_momenta[self.invert_momenta], np.arange(self.config.Ls ** 2))

        self.load_presaved_GF_data()
        self.refresh_heavy_logs()
        self.init_heavy_logs_files()


        self.init_light_cumulants()
        self.n_saved_times = 0
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
        if self.config.start_type == 'presaved' and (os.path.isfile(self.G_up_sum_filename + '.npy') \
                                                  or os.path.isfile(self.G_up_sum_filename + '_dump.npy')):
            try:
                self.GF_up_sum = np.load(self.G_up_sum_filename + '.npy')
                self.GF_down_sum = np.load(self.G_down_sum_filename + '.npy')
                self.C_ijkl = np.load(self.C_ijkl_filename + '.npy')
                self.PHI_ijkl = np.load(self.PHI_ijkl_filename + '.npy')
                self.Z_uu_ijkl = np.load(self.Z_uu_ijkl_filename + '.npy')
                self.Z_dd_ijkl = np.load(self.Z_dd_ijkl_filename + '.npy')
                self.X_uu_ijkl = np.load(self.X_uu_ijkl_filename + '.npy')
                self.X_ud_ijkl = np.load(self.X_ud_ijkl_filename + '.npy')
                self.X_du_ijkl = np.load(self.X_du_ijkl_filename + '.npy')
                self.X_dd_ijkl = np.load(self.X_dd_ijkl_filename + '.npy')
                self.chi_ijkl_total = np.load(self.chi_ijkl_total_filename + '.npy')
                self.chi_ijkl_free = np.load(self.chi_ijkl_free_filename + '.npy')

                self.num_chi_samples = np.load(self.num_samples_filename + '.npy')[0]
                self.total_sign = np.load(self.sign_filename + '.npy')[0]

                print('Successfully loaded saved GFs files from default location')
                loaded = True
            except Exception:
                print('Failed to load from default location: will try to load from dump')

                try:
                    self.GF_up_sum = np.load(self.G_up_sum_filename + '_dump.npy')
                    self.GF_down_sum = np.load(self.G_down_sum_filename + '_dump.npy')
                    self.C_ijkl = np.load(self.C_ijkl_filename + '_dump.npy')
                    self.PHI_ijkl = np.load(self.PHI_ijkl_filename + '_dump.npy')
                    self.Z_uu_ijkl = np.load(self.Z_uu_ijkl_filename + '_dump.npy')
                    self.Z_dd_ijkl = np.load(self.Z_dd_ijkl_filename + '_dump.npy')
                    self.X_uu_ijkl = np.load(self.X_uu_ijkl_filename + '_dump.npy')
                    self.X_ud_ijkl = np.load(self.X_ud_ijkl_filename + '_dump.npy')
                    self.X_du_ijkl = np.load(self.X_du_ijkl_filename + '_dump.npy')
                    self.X_dd_ijkl = np.load(self.X_dd_ijkl_filename + '_dump.npy')
                    self.chi_ijkl_total = np.load(self.chi_ijkl_total_filename + '_dump.npy')
                    self.chi_ijkl_free = np.load(self.chi_ijkl_free_filename + '_dump.npy')


                    self.num_chi_samples = np.load(self.num_samples_filename + '_dump.npy')[0]
                    self.total_sign = np.load(self.sign_filename + '_dump.npy')[0]

                    print('Successfully loaded saved GFs files from dump location')
                    loaded = True
                except Exception:
                    print('Failed to load from dump location: will start from scratch')

        if not loaded:
            print('Initialized GFs buffer from scratch')
            self.GF_up_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
            self.GF_down_sum = np.zeros((self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
            self.num_chi_samples = 0
            self.total_sign = 0.0
            self.C_ijkl = np.zeros(len(self.ijkl))
            self.PHI_ijkl = np.zeros(len(self.ijkl))

            self.Z_uu_ijkl = np.zeros(len(self.ijkl_order))
            self.Z_dd_ijkl = np.zeros(len(self.ijkl_order))

            self.X_uu_ijkl = np.zeros(len(self.ijkl_order))
            self.X_ud_ijkl = np.zeros(len(self.ijkl_order))
            self.X_du_ijkl = np.zeros(len(self.ijkl_order))
            self.X_dd_ijkl = np.zeros(len(self.ijkl_order))

            self.chi_ijkl_total = np.zeros((self.n_bands, self.n_bands, self.n_bands, self.n_bands, self.config.Ls ** 2, self.config.Ls ** 2))
            self.chi_ijkl_free = np.zeros((self.n_bands, self.n_bands, self.n_bands, self.n_bands, self.config.Ls ** 2, self.config.Ls ** 2))
        return

    def save_GF_data(self):
        ### save GF data ###
        addstring = '_dump.npy' if self.n_saved_times % 2 == 0 else '.npy'

        np.save(self.G_up_sum_filename + addstring, self.GF_up_sum)
        np.save(self.G_down_sum_filename + addstring, self.GF_down_sum)
        np.save(self.C_ijkl_filename + addstring, self.C_ijkl)
        np.save(self.PHI_ijkl_filename + addstring, self.PHI_ijkl)
        np.save(self.Z_uu_ijkl_filename + addstring, self.Z_uu_ijkl)
        np.save(self.Z_dd_ijkl_filename + addstring, self.Z_dd_ijkl)

        np.save(self.X_uu_ijkl_filename + addstring, self.X_uu_ijkl)
        np.save(self.X_ud_ijkl_filename + addstring, self.X_ud_ijkl)
        np.save(self.X_du_ijkl_filename + addstring, self.X_du_ijkl)
        np.save(self.X_dd_ijkl_filename + addstring, self.X_dd_ijkl)
        np.save(self.chi_ijkl_free_filename + addstring, self.chi_ijkl_free)
        np.save(self.chi_ijkl_total_filename + addstring, self.chi_ijkl_total)


        np.save(self.num_samples_filename + addstring, np.array([self.num_chi_samples]))
        np.save(self.sign_filename + addstring, np.array([self.total_sign]))

        self.n_saved_times += 1
        return

    def refresh_heavy_logs(self):
        self.gap_file.flush()

        ### buffer for efficient GF-measurements ###
        self.cur_buffer_size = 0; self.max_buffer_size = 20
        self.GF_up_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.GF_down_stored = np.zeros((self.max_buffer_size, self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

        self.gap_observables_list = OrderedDict()
        self.order_observables_list = OrderedDict()

        adj_list = self.config.adj_list[:self.config.n_adj_density]  # only largest distance

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

    def init_light_cumulants(self):
        self.light_cumulants = OrderedDict({
            '⟨Re density⟩' : 0.0, 
            '⟨Im density⟩' : 0.0,
            '⟨Re E_K⟩' : 0.0, 
            '⟨Im E_K⟩' : 0.0,
            '⟨Re E_CU⟩' : 0.0,
            '⟨Im E_CU⟩' : 0.0,
            '⟨Re E_CV⟩' : 0.0,
            '⟨Im E_CV⟩' : 0.0,
            '⟨Re E_T⟩' : 0.0,
            '⟨Im E_T⟩' : 0.0,
            '⟨c^dag_{+down}c_{-down}⟩_re' : 0.0,
            '⟨c^dag_{+down}c_{-down}⟩_im' : 0.0,
            '⟨c^dag_{-down}c_{-down}⟩_re' : 0.0,
            '⟨c^dag_{-down}c_{-down}⟩_im' : 0.0,
            '⟨m_z^2⟩' : 0.0,
            'non-hermicity': 0.0
        })
        self.n_cumulants = 0
        return


    def refresh_light_logs(self):
        # self.log_file.flush()
        self.light_observables_list = OrderedDict({
            '⟨Re density⟩' : [], 
            '⟨Im density⟩' : [],
            '⟨Re E_K⟩' : [], 
            '⟨Im E_K⟩' : [],
            '⟨Re E_CU⟩' : [],
            '⟨Im E_CU⟩' : [],
            '⟨Re E_CV⟩' : [],
            '⟨Im E_CV⟩' : [],
            '⟨Re E_T⟩' : [],
            '⟨Im E_T⟩' : [],
            '⟨c^dag_{+down}c_{-down}⟩_re' : [],
            '⟨c^dag_{+down}c_{-down}⟩_im' : [],
            '⟨c^dag_{-down}c_{-down}⟩_re' : [],
            '⟨c^dag_{-down}c_{-down}⟩_im' : [],
            '⟨m_z^2⟩' : [],
            'non-hermicity' : []
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
        print('# sweep ⟨r⟩ ⟨acc⟩ ⟨sign⟩ ⟨Re n⟩ ⟨Im n⟩ ⟨Re E_K⟩ ⟨Im E_K⟩ ⟨Re E_CU⟩ ⟨Im E_CU⟩ ⟨Re E_CV⟩ ⟨Im E_CV⟩ ⟨Re E_T⟩ ⟨Im E_T⟩ ⟨c^dag_{+down}c_{-down}⟩_re ⟨c^dag_{+down}c_{-down}⟩_im ⟨c^dag_{-down}c_{-down}⟩_re ⟨c^dag_{-down}c_{-down}⟩_im ⟨m_z^2⟩ non-hermicity imaginary')
        return
    '''
    def print_std_logs(self, n_sweep):
        print("{:d} {:.5f} {:.2f} {:.3f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(
            n_sweep, 
            np.mean(self.ratio_history),
            np.mean(self.acceptance_history),
            self.global_average_sign,
            np.mean(self.light_observables_list['⟨density⟩']),
            np.mean(self.light_observables_list['⟨E_K⟩']),
            np.mean(self.light_observables_list['⟨E_C⟩']),
            np.mean(self.light_observables_list['⟨E_T⟩']),
            np.mean(self.light_observables_list['⟨c^dag_{+down}c_{-down}⟩_re']),
            np.mean(self.light_observables_list['⟨c^dag_{+down}c_{-down}⟩_im']),
            np.mean(self.light_observables_list['⟨c^dag_{-down}c_{-down}⟩_re']),
            np.mean(self.light_observables_list['⟨c^dag_{-down}c_{-down}⟩_im']),
            np.mean(self.light_observables_list['⟨m_z^2⟩']),
        ))
        return
    '''
    def print_std_logs(self, n_sweep):
        if self.n_cumulants == 0 or self.global_average_sign == 0:
            return
        print("{:d} {:.5f} {:.2f} {:.3f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}".format(
            n_sweep, 
            np.mean(self.ratio_history),
            np.mean(self.acceptance_history),
            self.global_average_sign,
            self.light_cumulants['⟨Re density⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Im density⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Re E_K⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Im E_K⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Re E_CU⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Im E_CU⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Re E_CV⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Im E_CV⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Re E_T⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨Im E_T⟩'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨c^dag_{+down}c_{-down}⟩_re'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨c^dag_{+down}c_{-down}⟩_im'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨c^dag_{-down}c_{-down}⟩_re'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨c^dag_{-down}c_{-down}⟩_im'] / self.n_cumulants / self.global_average_sign,
            self.light_cumulants['⟨m_z^2⟩'] / self.n_cumulants / self.global_average_sign,
            #self.light_cumulants['non-hermicity'] / self.n_cumulants / self.global_average_sign
            self.gf_nonhermicity,
            self.gf_imaginary 
        ))
        return
    def measure_light_observables(self, phi, current_det_sign, n_sweep, print_gf = False):
        if print_gf:
            gf_print = phi.G_up_sum[::2, 0]
            for i in range(8):
                self.gf_file.write('{:.20f} '.format(gf_print[i] / phi.n_gf_measures))
            self.gf_file.write('\n')
            self.gf_file.flush()  # TODO: DEBUG
        self.light_signs_history.append(current_det_sign)  

        k = kinetic_energy(phi).item()
        CU, CV = Coloumb_energy(phi)
        density = total_density(phi).item()

        G_up, G_down = phi.get_equal_time_GF()

        self.light_observables_list['⟨Re density⟩'].append(density.real)
        self.light_observables_list['⟨Re E_K⟩'].append(k.real)
        self.light_observables_list['⟨Re E_CU⟩'].append(CU.real)
        self.light_observables_list['⟨Re E_CV⟩'].append(CV.real)
        self.light_observables_list['⟨Re E_T⟩'].append((k + CU + CV).real)
        
        self.light_observables_list['⟨Im density⟩'].append(density.imag)
        self.light_observables_list['⟨Im E_K⟩'].append(k.imag)
        self.light_observables_list['⟨Im E_CU⟩'].append(CU.imag)
        self.light_observables_list['⟨Im E_CV⟩'].append(CV.imag)
        self.light_observables_list['⟨Im E_T⟩'].append((k + CU + CV).imag)

        self.light_observables_list['⟨c^dag_{+down}c_{-down}⟩_re'].append(np.real(np.trace(G_down.dot(self.O_pm_xy))))
        self.light_observables_list['⟨c^dag_{+down}c_{-down}⟩_im'].append(np.imag(np.trace(G_down.dot(self.O_pm_xy))))
        self.light_observables_list['⟨c^dag_{-down}c_{-down}⟩_re'].append(np.real(np.trace(G_down.dot(self.O_mm_xy))))
        self.light_observables_list['⟨c^dag_{-down}c_{-down}⟩_im'].append(np.imag(np.trace(G_down.dot(self.O_mm_xy))))
        self.light_observables_list['⟨m_z^2⟩'].append(total_mz_squared(G_down, G_up).real)
        #self.light_observables_list['non-hermicity'].append(np.sum(np.abs(phi.G_up_sum.conj().T - phi.G_up_sum)) / phi.n_df_measures)
        self.gf_nonhermicity = np.sum(np.abs(phi.G_up_sum.conj().T - phi.G_up_sum)) / phi.n_gf_measures
        self.gf_imaginary = np.max(np.abs(phi.G_up_sum.imag)) / phi.n_gf_measures
        # print(self.gf_nonhermicity, self.gf_imaginary)

        self.n_cumulants += 1
        self.light_cumulants['⟨Re density⟩'] += (density * current_det_sign).real
        self.light_cumulants['⟨Re E_K⟩'] += (k * current_det_sign).real
        self.light_cumulants['⟨Re E_CU⟩'] += (CU * current_det_sign).real
        self.light_cumulants['⟨Re E_CV⟩'] += (CV * current_det_sign).real
        self.light_cumulants['⟨Re E_T⟩'] += ((k + CU + CV) * current_det_sign).real

        self.light_cumulants['⟨Im density⟩'] += (density * current_det_sign).imag
        self.light_cumulants['⟨Im E_K⟩'] += (k * current_det_sign).imag
        self.light_cumulants['⟨Im E_CU⟩'] += (CU * current_det_sign).imag
        self.light_cumulants['⟨Im E_CV⟩'] += (CV * current_det_sign).imag
        self.light_cumulants['⟨Im E_T⟩'] += ((k + CU + CV) * current_det_sign).imag


        self.light_cumulants['⟨c^dag_{+down}c_{-down}⟩_re'] += np.real(np.trace(G_down.dot(self.O_pm_xy))) * current_det_sign
        self.light_cumulants['⟨c^dag_{+down}c_{-down}⟩_im'] += np.imag(np.trace(G_down.dot(self.O_pm_xy))) * current_det_sign
        self.light_cumulants['⟨c^dag_{-down}c_{-down}⟩_re'] += np.real(np.trace(G_down.dot(self.O_mm_xy))) * current_det_sign
        self.light_cumulants['⟨c^dag_{-down}c_{-down}⟩_im'] += np.imag(np.trace(G_down.dot(self.O_mm_xy))) * current_det_sign
        self.light_cumulants['⟨m_z^2⟩'] += (total_mz_squared(G_down, G_up) * current_det_sign).real

        return

    def signs_avg(self, array, signs):
        return np.mean(np.array(array) * signs)


    def signs_std(self, array, signs):
        return (np.std(np.array(array) * signs) / np.mean(signs) - \
               np.std(signs) * np.mean(np.array(array) * signs) / np.mean(signs) ** 2) / np.sqrt(len(signs))

    def write_light_observables(self, config, n_sweep):
        signs = np.array(self.light_signs_history)

        #data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), np.mean(self.sign_history),
        #        np.mean(self.light_signs_history)] + [self.signs_avg(val, signs) for _, val in self.light_observables_list.items()]

        self.global_average_sign = (self.global_average_sign * self.global_average_sign_measurements + \
                                    np.sum(signs)) / (self.global_average_sign_measurements + len(signs))
        self.global_average_sign_measurements += len(signs)

        data = [n_sweep, np.mean(self.ratio_history), np.mean(self.acceptance_history), self.global_average_sign,
                np.mean(self.light_signs_history)] + [val / self.global_average_sign / self.n_cumulants for _, val in self.light_cumulants.items()]
        data[-1] = self.gf_nonhermicity
        data.append(self.gf_imaginary)
        # print(data)
        self.log_file.write(("{:d} " + "{:.6f} " * (len(data) - 1) + '\n').format(n_sweep, *data[1:]))

        if n_sweep % 100 == 0:
            self.log_file.flush()

        self.global_average_sign = (self.global_average_sign * self.global_average_sign_measurements + \
                                    np.sum(signs)) / (self.global_average_sign_measurements + len(signs))
        self.global_average_sign_measurements += len(signs)
        self.refresh_light_logs()  # TODO: keep sign-averaged observables and accumulate them forever
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
        signs = np.array(self.heavy_signs_history[-self.cur_buffer_size:])
        self.total_sign += np.sum(signs)


        shape = self.GF_up_stored[:self.cur_buffer_size, ...].shape
        new_shape = (shape[0] * shape[1], shape[2], shape[3])
        G_up_prepared = np.asfortranarray(np.einsum('ijkl,i->ijkl', self.GF_up_stored[:self.cur_buffer_size, ...], signs).reshape(new_shape))
        G_down_prepared = np.asfortranarray(self.GF_down_stored[:self.cur_buffer_size, ...].reshape(new_shape))

        t = time()
        self.C_ijkl += measure_gfs_correlator(G_up_prepared, G_down_prepared, self.ijkl) / 2.

        G_down_prepared = np.asfortranarray(np.einsum('ijkl,i->ijkl', self.GF_down_stored[:self.cur_buffer_size, ...], signs).reshape(new_shape))
        G_up_prepared = np.asfortranarray(self.GF_up_stored[:self.cur_buffer_size, ...].reshape(new_shape))

        self.C_ijkl += measure_gfs_correlator(G_down_prepared, G_up_prepared, self.ijkl) / 2.  # SU(2) symmetry to stabilyze the measurements

        print('C_ijkl take', time() - t)
        self.PHI_ijkl += measure_gfs_correlator(np.asfortranarray(np.einsum('ijkl,i->ijkl', \
                       self.GF_up_stored[:self.cur_buffer_size, 0:1, ...], signs).reshape((shape[0] * 1, shape[2], shape[3]))), \
            np.asfortranarray(self.GF_down_stored[:self.cur_buffer_size, 0:1, ...].reshape((shape[0] * 1, shape[2], shape[3]))), self.ijkl)

        t = time()
        self.Z_uu_ijkl += measure_Z_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)
        self.Z_dd_ijkl += measure_Z_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)

        print('Z_ss_ijkl take', time() - t)
        t = time()

        self.X_uu_ijkl += measure_X_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], \
                                               self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)
        
        self.X_ud_ijkl += measure_X_correlator(self.GF_up_stored[:self.cur_buffer_size, 0, ...], \
                                               self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)
        
        self.X_du_ijkl += measure_X_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], \
                                               self.GF_up_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)
        
        self.X_dd_ijkl += measure_X_correlator(self.GF_down_stored[:self.cur_buffer_size, 0, ...], \
                                               self.GF_down_stored[:self.cur_buffer_size, 0, ...], signs, self.ijkl_order)
        print('X_s1s2_ijkl take', time() - t)



        ### chi_ijkl_total ###
        t = time()
        shape = self.GF_up_stored[:self.cur_buffer_size, ...].shape
        new_shape = (shape[0] * shape[1], shape[2] // self.n_bands, self.n_bands, shape[3] // self.n_bands, self.n_bands)
        
        G_up_prepared = np.einsum('ijkl,i->ijkl', self.GF_up_stored[:self.cur_buffer_size, ...], signs).reshape(new_shape).transpose((2, 4, 0, 1, 3))
        G_down_prepared = self.GF_down_stored[:self.cur_buffer_size, ...].reshape(new_shape).transpose((2, 4, 0, 1, 3))

        G_up_prepared_ft = np.dot(np.dot(self.U_ft_space.conj().T, G_up_prepared), self.U_ft_space).transpose((1, 2, 3, 4, 0))
        G_down_prepared_ft = np.dot(np.dot(self.U_ft_space.conj().T, G_down_prepared), self.U_ft_space).transpose((1, 2, 3, 4, 0))

        G_down_prepared_ft = G_down_prepared_ft[:, :, :, self.invert_momenta, :]
        G_down_prepared_ft = G_down_prepared_ft[:, :, :, :, self.invert_momenta]

        self.chi_ijkl_total = self.config.dt * np.einsum('ikabc,jlabc->ijklbc', G_up_prepared_ft, G_down_prepared_ft) / self.total_sign
        print('chi_ijkl_total take', time() - t)


        self.GF_up_sum += np.einsum('ijkl,i->jkl', self.GF_up_stored[:self.cur_buffer_size, ...], signs)
        self.GF_down_sum += np.einsum('ijkl,i->jkl', self.GF_down_stored[:self.cur_buffer_size, ...], signs)
        self.cur_buffer_size = 0



        ### chi_ijkl_free ###
        t = time()
        shape = self.GF_up_sum.shape
        new_shape = (shape[0], shape[1] // self.n_bands, self.n_bands, shape[2] // self.n_bands, self.n_bands)

        G_up_prepared = self.GF_up_sum.reshape(new_shape).transpose((2, 4, 0, 1, 3))
        G_down_prepared = self.GF_down_sum.reshape(new_shape).transpose((4, 2, 0, 1, 3))

        G_up_prepared_ft = np.dot(np.dot(self.U_ft_space.conj().T, G_up_prepared), self.U_ft_space).transpose((1, 2, 3, 4, 0))
        G_down_prepared_ft = np.dot(np.dot(self.U_ft_space.conj().T, G_down_prepared), self.U_ft_space).transpose((1, 2, 3, 4, 0))

        G_down_prepared_ft = G_down_prepared_ft[:, :, :, self.invert_momenta, :]
        G_down_prepared_ft = G_down_prepared_ft[:, :, :, :, self.invert_momenta]

        self.chi_ijkl_free = self.config.dt * np.einsum('ikabc,jlabc->ijklbc', G_up_prepared_ft, G_down_prepared_ft) / self.total_sign ** 2

        print('chi_ijkl_free take', time() - t)
        return


    def measure_heavy_observables(self, phi):
        self.log_file.flush()
        print('refreshing gfs buffers...')
        self.refresh_gfs_buffer()
        t = time()
        mean_signs = np.mean(self.heavy_signs_history)
        mean_signs_global = self.total_sign / self.num_chi_samples

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
                    self.ijkl, self.C_ijkl, np.ones(shape = gap_alpha.shape)) / (self.num_chi_samples * mean_signs_global)
                free_chi = self.config.dt * np.sum([
                    np.trace(self.GF_up_sum[tau, ...].dot(gap_beta).dot(self.GF_down_sum[tau, ...].T).dot(gap_alpha.T.conj())) + \
                    np.trace(self.GF_down_sum[tau, ...].dot(gap_beta).dot(self.GF_up_sum[tau, ...].T).dot(gap_alpha.T.conj())) \
                    for tau in range(self.config.Nt)
                                                   ]) / ((self.num_chi_samples * mean_signs_global) ** 2) / 2.  
                # FIXME: is this symmetrisation valid for alpha != beta?


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
                self.ijkl, self.PHI_ijkl, self.distances_list) / (self.num_chi_samples * mean_signs_global)
            free_chi_n = np.trace((self.GF_up_sum[0, ...] * self.distances_list).dot(gap).dot(self.GF_down_sum[0, ...].T).dot(gap.T.conj())) \
                 / ((self.num_chi_samples * mean_signs_global) ** 2)

            corr_length_gap = total_chi_n - free_chi_n
            total_chi_d = get_gap_susceptibility(gap, gap, \
                self.ijkl, self.PHI_ijkl, np.ones(shape=gap.shape)) / (self.num_chi_samples * mean_signs_global)
            free_chi_d = np.trace((self.GF_up_sum[0, ...]).dot(gap).dot(self.GF_down_sum[0, ...].T).dot(gap.T.conj())) \
                 / ((self.num_chi_samples * mean_signs_global) ** 2)
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
            order_correlator = get_order_average(order_up, order_up, self.ijkl_order, self.X_uu_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average(order_up, order_down, self.ijkl_order, self.X_ud_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average(order_down, order_up, self.ijkl_order, self.X_du_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average(order_down, order_down, self.ijkl_order, self.X_dd_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average(order_up, order_up, self.ijkl_order, self.Z_uu_ijkl, self.ik_marking, self.config.Ls) + \
                               get_order_average(order_down, order_down, self.ijkl_order, self.Z_dd_ijkl, self.ik_marking, self.config.Ls)
            order_correlator = order_correlator.reshape((self.config.Ls, self.config.Ls, 4, 4))

            order_correlator /= (self.num_chi_samples * mean_signs_global * norm * N)
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
        name = os.path.join(self.local_workdir, 'chi_vertex_final.npy' if self.n_saved_times % 2 == 0 else 'chi_vertex_final_dump.npy')
        np.save(name, (chi_vertex + chi_vertex.conj().T) / 2.)

        name = os.path.join(self.local_workdir, 'chi_total_{:d}.npy'.format(n_sweep))
        np.save(name, (chi_total + chi_total.conj().T) / 2.)
        name = os.path.join(self.local_workdir, 'chi_total_final.npy' if self.n_saved_times % 2 == 0 else 'chi_total_final_dump.npy')
        np.save(name, (chi_total + chi_total.conj().T) / 2.)

        name = os.path.join(self.local_workdir, 'Sq0_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(Sq0))
        name = os.path.join(self.local_workdir, 'Sq0_final.npy' if self.n_saved_times % 2 == 0 else 'Sq0_final_dump.npy')
        np.save(name, np.array(Sq0))


        name = os.path.join(self.local_workdir, 'Pq0_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(Pq0))
        name = os.path.join(self.local_workdir, 'Pq0_final.npy' if self.n_saved_times % 2 == 0 else 'Pq0_final_dump.npy')
        np.save(name, np.array(Pq0))
 

        name = os.path.join(self.local_workdir, 'corr_lengths_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(corr_lengths))
        name = os.path.join(self.local_workdir, 'corr_lengths_final.npy' if self.n_saved_times % 2 == 0 else 'corr_lengths_final_dump.npy')
        np.save(name, np.array(corr_lengths))
 

        np.save(os.path.join(self.local_workdir, 'gap_names.npy'), np.array(self.config.pairings_list_names))


        orders = []
        for order_name in self.config.waves_list_names:
            orders.append(self.order_observables_list[order_name + '_order'])

        np.save(os.path.join(self.local_workdir, 'orders_names.npy'), np.array(self.config.waves_list_names))
        name = os.path.join(self.local_workdir, 'order_{:d}.npy'.format(n_sweep))
        np.save(name, np.array(orders))


        self.gap_file.write(("{:d} " + "{:.6f} " * (len(gap_data) - 1) + '\n').format(n_sweep, *gap_data[1:]))
        self.gap_file.flush()
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

    return phi.la.einsum('ij,ij', phi.K_matrix, G_function_up + G_function_down) / G_function_up.shape[0]


def Coloumb_energy(phi):
    G_function_up, G_function_down = phi.get_equal_time_GF()
    # o rint(np.trace(G_function_up), np.trace(G_function_down))
    #print(G_function_up)
    #print(G_function_up - G_function_up.conj().T)
    #exit(-1)
    #print(np.abs(G_function_up - G_function_up.conj().T))
    #print(phi.G_up_sum[np.arange(0, 16, 2), np.arange(0, 16, 2)].sum() / phi.n_gf_measures)
    #print(G_function_up)
    #for i in range(16):
    #    print(G_function_up[0, i], '-->', G_function_up[i, 0], (0, i))
    #    print(np.sum(np.abs(phi.G_up_sum - phi.G_up_sum.T.conj()) / phi.n_gf_measures))
    #print(np.abs(phi.G_down_sum - phi.G_down_sum.T.conj()) / phi.n_gf_measures)
    
    
    energy_coloumb_U = 0.0 + 0.0j; 
    energy_coloumb_V = 0.0 + 0.0j
    energy_coloumb_U = phi.config.U * phi.la.sum(phi.la.diag(G_function_up) * phi.la.diag(G_function_down)).item() / G_function_up.shape[0]
    # print(phi.la.diag(G_function_up) * phi.la.diag(G_function_down))

    #(phi.config.U / 2.) * phi.la.sum((phi.la.diag(G_function_up) + phi.la.diag(G_function_down) - 1.) ** 2).item() \
    #                 / G_function_up.shape[0]
    if phi.config.n_orbitals == 1:
        return energy_coloumb_U

    orbital_1 = phi.la.arange(0, G_function_up.shape[0], 2)
    orbital_2 = phi.la.arange(1, G_function_up.shape[0], 2)
    energy_coloumb_U += phi.config.U * phi.la.einsum('i,i', (phi.la.diag(G_function_up)[orbital_1] + phi.la.diag(G_function_down)[orbital_1] - 1),
                                                      (phi.la.diag(G_function_up)[orbital_2] + phi.la.diag(G_function_down)[orbital_2] - 1)).item() \
                                                      / G_function_up.shape[0]

    total_density = phi.la.diag(G_function_up)[orbital_1] + phi.la.diag(G_function_down)[orbital_1] + \
                    phi.la.diag(G_function_up)[orbital_2] + phi.la.diag(G_function_down)[orbital_2]

    #print(phi.la.diag(G_function_up)[orbital_1])
    #print(phi.la.diag(G_function_up)[orbital_2])
    # print(total_density)
    #energy_coloumb += phi.config.U / 2 * np.sum(total_density ** 2) / G_function_up.shape[0] # FIXME: coloumb energy purely imaginary (the GF diagonal is imaginary) -- what to do? the problem is that there is G^2! do not account for same-site terms
    #print(energy_coloumb)
    energy_coloumb_V += phi.config.V * np.einsum('i,ij,j', total_density, phi.connectivity, total_density) / G_function_up.shape[0] / 2.  # why all contribs imag? bullshit

    #print(total_density)
    #print(energy_coloumb_V)
    #for i in range(G_function_up.shape[0]):
    #    for j in range(G_function_up.shape[1]):
    #        if (i + j) % 2 == 1 and np.abs(G_function_down[i, j]) > 1e-7:
    #            print(i, j)
    #            exit(-1)


    G_up_orb1 = G_function_up[::2, :]; 
    G_up_orb1 = G_up_orb1[:, ::2]

    G_up_orb2 = G_function_up[1::2, :]; 
    G_up_orb2 = G_up_orb2[:, 1::2]
    
    G_down_orb1 = G_function_down[::2, :]; 
    G_down_orb1 = G_down_orb1[:, ::2]
    
    G_down_orb2 = G_function_down[1::2, :]; 
    G_down_orb2 = G_down_orb2[:, 1::2]

    # assert np.isclose(np.sum(np.abs(G_function_up - np.kron(G_up_orb1, np.diag([1, 0])) - np.kron(G_up_orb2, np.diag([0, 1])))), 0.0)

    for GF in [G_up_orb1, G_up_orb2, G_down_orb1, G_down_orb2]:
        energy_coloumb_V -= phi.config.V * np.trace((GF * phi.connectivity).dot(GF * phi.connectivity)) / G_function_up.shape[0] / 2.
        # energy_coloumb_V -= phi.config.V * np.einsum('ij,ji,ij', GF, GF, phi.connectivity) / G_function_up.shape[0] / 2.

    return energy_coloumb_U, energy_coloumb_V # + phi.config.U / 2. #phi.la.sum(phi.la.diag(G_function_up) ** 2).item() \
                          #   / G_function_up.shape[0] 
    # 



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
