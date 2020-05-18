import numpy as np
import pickle
import time
import scipy
import models
from copy import deepcopy
from numba import jit
import os
try:
    import cupy as cp
except ImportError:
    pass

class AuxiliaryFieldIntraorbital:
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir):
        self.gpu_avail = config.gpu
        self.la = np
        self.cpu = True

        self.config = config
        self.adj_list = config.adj_list
        self.conf_path = os.path.join(local_workdir, 'last_conf.npy')
        self._get_initial_field_configuration()

        self.K = K
        self.K_inverse = K_inverse
        self.K_matrix = K_matrix

        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self.current_lhs_SVD_up = []
        self.current_lhs_SVD_down = []

        self.current_G_function_up = []
        self.current_G_function_down = []
        self.copy_to_GPU()

        self.refresh_all_decompositions()
        
        self.refresh_G_functions()
        self.current_time_slice = 0
        self.log_det_up = 0
        self.sign_det_up = 0
        self.log_det_down = 0
        self.sign_det_down = 0

        self.refresh_checkpoints = [0]
        t = self.config.Nt % self.config.s_refresh
        if t == 0:
            self.refresh_checkpoints = []
        while t < self.config.Nt:
            self.refresh_checkpoints.append(t)
            t += self.config.s_refresh
        self.refresh_checkpoints = np.array(self.refresh_checkpoints)

        self.local_conf_combinations = [tuple([1]), tuple([-1])]
        return
    def get_gauge_factor_move(self, sp_index, time_slice, local_conf):
        return 1.

    def get_current_gauge_factor_log(self):
        return 0

    def propose_move(self, sp_index, time_slice, o_index):
        return -self.configuration[time_slice, sp_index, o_index], 1.0  # gauge factor is always 1

    def SVD(self, matrix):
        if self.cpu:
            return scipy.linalg.svd(matrix, lapack_driver='gesvd')  # this is VERY IMPORTANT
        return cp.linalg.svd(matrix)

    def refresh_G_functions(self):
        self.current_G_function_up, self.log_det_up, self.sign_det_up = self.get_current_G_function(+1, return_logdet = True)
        self.current_G_function_down, self.log_det_down, self.sign_det_down = self.get_current_G_function(-1, return_logdet = True)
        return
    
    def get_current_G_function(self, spin, return_logdet = False):
        if spin == +1:
            svd_rhs = self.partial_SVD_decompositions_up[-1]
            svd_lhs = self.current_lhs_SVD_up
        else:
            svd_rhs = self.partial_SVD_decompositions_down[-1]
            svd_lhs = self.current_lhs_SVD_down
        u1, s1, v1 = svd_lhs
        u2, s2, v2 = svd_rhs
        m = v1.dot(u2)
        middle_mat = (u1.T).dot(v2.T) + (self.la.diag(s1).dot(m)).dot(self.la.diag(s2))
        inv = self.la.linalg.inv(middle_mat)

        s, ld = np.linalg.slogdet(inv)
        sign = np.sign(np.linalg.slogdet(self.to_numpy(v2.T))[0] * np.linalg.slogdet(self.to_numpy(u1.T))[0] * s)
        if return_logdet:
            return v2.T.dot(inv).dot(u1.T), ld, sign
        return v2.T.dot(inv).dot(u1.T)

    def refresh_all_decompositions(self):
        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self._get_partial_SVD_decompositions(spin = +1)
        self._get_partial_SVD_decompositions(spin = -1)

        self.current_lhs_SVD_up = self.SVD(self.la.eye(self.config.total_dof // 2))
        self.current_lhs_SVD_down = self.SVD(self.la.eye(self.config.total_dof // 2))
        return

    def _product_svds(self, svd1, svd2):
        u1, s1, v1 = svd1
        u2, s2, v2 = svd2
        m = v1.dot(u2)
        middle_mat = self.la.diag(s1).dot(m).dot(self.la.diag(s2))
        um, sm, vm = self.SVD(middle_mat)
        return u1.dot(um), sm, vm.dot(v2)

    def append_new_decomposition(self, tmin, tmax):
        assert tmax - tmin == self.config.Nt % self.config.s_refresh or tmax - tmin == self.config.s_refresh
        lhs_change_up = self._get_partial_SVD_decomposition_range(+1, tmin, tmax)
        lhs_change_down = self._get_partial_SVD_decomposition_range(-1, tmin, tmax)

        self.current_lhs_SVD_up = self._product_svds(lhs_change_up, self.current_lhs_SVD_up)
        self.current_lhs_SVD_down = self._product_svds(lhs_change_down, self.current_lhs_SVD_down)

        del self.partial_SVD_decompositions_up[-1]
        del self.partial_SVD_decompositions_down[-1]
        self.current_time_slice = tmax
        return
    
    def _get_partial_SVD_decompositions(self, spin):
        M = self.la.eye(self.config.total_dof // 2)
        current_U = self.la.eye(self.config.total_dof // 2)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            M = M.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1 or nr == self.config.Nt - 1:
                u, s, v = self.SVD(M)  # this is a VERY tricky point
                
                assert self.la.linalg.norm(u.dot(self.la.diag(s)).dot(v) - M) / self.la.linalg.norm(M) < 1e-14
                current_U = current_U.dot(u)
                if spin == +1:
                    self.partial_SVD_decompositions_up.append((current_U, s, v))
                else:
                    self.partial_SVD_decompositions_down.append((current_U, s, v))
                M = self.la.diag(s).dot(v)
        return

    def _get_left_partial_SVD_decompositions(self, spin):
        decompositions = []
        M = self.la.eye(self.config.total_dof // 2)
        current_V = self.la.eye(self.config.total_dof // 2)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(slices):
            B = self.B_l(spin, slice_idx)
            M = B.dot(M)
            if nr % self.config.s_refresh == self.config.s_refresh - 1:
                u, s, v = self.SVD(M)
                
                current_V = v.dot(current_V)
                decompositions.append((u, s, current_V))
                M = u.dot(self.la.diag(s))
        return decompositions

    def _get_right_partial_SVD_decompositions(self, spin):
        decompositions = []
        M = self.la.eye(self.config.total_dof // 2)
        current_U = self.la.eye(self.config.total_dof // 2)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            M = M.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1:
                u, s, v = self.SVD(M)

                current_U = current_U.dot(u)
                decompositions.append((current_U, s, v))
                M = self.la.diag(s).dot(v)
        return decompositions

    def _get_partial_SVD_decomposition_range(self, spin, tmin, tmax):
        M = self.la.eye(self.config.total_dof // 2)
        
        for time_slice in range(tmin, tmax):
            M = self.B_l(spin, time_slice, inverse = False).dot(M)
        return self.SVD(M)

    def _load_configuration(self):
        return np.load(self.conf_path)

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = self.la.asarray(np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)
            return
        if self.config.start_type == 'hot':
            self.configuration = self.la.asarray(np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)
            return

        if os.path.isfile(self.conf_path):
            self.configuration = self.la.asarray(self._load_configuration())
            return
        self.configuration = self.la.asarray(np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)  # hot anyway

        return

    def save_configuration(self):
        return np.save(self.conf_path, self.configuration)

    def B_l(self, spin, l, inverse = False):
        if not inverse:
            V = self.la.diag(self.la.exp(spin * self.config.nu_U * self.configuration[l, ...]))
            return V.dot(self.K)
    
        V = self.la.diag(self.la.exp(-spin * self.config.nu_U * self.configuration[l, ...]))
        return self.K_inverse.dot(V)

    def compute_deltas(self, sp_index, time_slice, *args):
        self.Delta_up = self.get_delta(+1., sp_index, time_slice)
        self.Delta_down = self.get_delta(-1., sp_index, time_slice)
        return

    def get_delta(self, spin, sp_index, time_slice):  # sign change proposal is made at (time_slice, sp_index, o_index)
        return get_delta_intraorbital(self.configuration[time_slice, sp_index], spin, self.config.nu_U)

    def update_G_seq(self, sp_index, *args):
        self.current_G_function_up = _update_G_seq_intra(self.current_G_function_up, self.Delta_up, sp_index, self.config.total_dof)
        self.current_G_function_down = _update_G_seq_intra(self.current_G_function_down, self.Delta_down, sp_index, self.config.total_dof)
        return

    def update_field(self, sp_index, time_slice, new_conf):
        self.configuration[time_slice, sp_index] = new_conf[0]
        return

    def to_numpy(self, array):
        if self.cpu:
            return array
        return cp.asnumpy(array)


    def copy_to_CPU(self):
        if not self.gpu_avail:
            return self
        self.cpu = True
        self.current_G_function_up = cp.asnumpy(self.current_G_function_up)
        self.current_G_function_down = cp.asnumpy(self.current_G_function_down)
        self.K = cp.asnumpy(self.K)
        self.K_inverse = cp.asnumpy(self.K_inverse)
        self.la = np
        self.configuration = cp.asnumpy(self.configuration)
        return self

    def copy_to_GPU(self):
        if not self.gpu_avail:
            return self
        self.cpu = False
        self.current_G_function_up = cp.array(self.current_G_function_up)
        self.current_G_function_down = cp.array(self.current_G_function_down)
        self.K = cp.array(self.K)
        self.K_inverse = cp.array(self.K_inverse)
        self.la = cp
        self.configuration = cp.asarray(self.configuration)
        return self


    def wrap_up(self, time_slice):
        B_wrap_up = self.B_l(+1, time_slice, inverse = False)
        B_wrap_up_inverse = self.B_l(+1, time_slice, inverse = True)
        B_wrap_down = self.B_l(-1, time_slice, inverse = False)
        B_wrap_down_inverse = self.B_l(-1, time_slice, inverse = True)

        self.current_G_function_up = B_wrap_up.dot(self.current_G_function_up.dot(B_wrap_up_inverse))
        self.current_G_function_down = B_wrap_down.dot(self.current_G_function_down.dot(B_wrap_down_inverse))

        return

    def compute_B_chain(self, spin, tmax, tmin):
        if tmax == self.config.Nt:
            index_decomp = (tmax - tmin) // self.config.s_refresh
            tmax -= index_decomp * self.config.s_refresh
            
            if index_decomp > 0:
                current_U, s, v = self.right_decompositions[index_decomp - 1]
            else:
                current_U, s, v = self.SVD(self.la.eye(self.config.total_dof // 2))
            chain = self.la.diag(s).dot(v)

            for i in reversed(range(tmin, tmax)):
                chain = chain.dot(self.B_l(spin, i))

            u, s, v = self.SVD(chain)
            return current_U.dot(u), s, v
        index_decomp = (tmax - tmin) // self.config.s_refresh
        tmin += index_decomp * self.config.s_refresh
        if index_decomp > 0:
            u, s, current_V = self.left_decompositions[index_decomp - 1]
        else:
            u, s, current_V = self.SVD(self.la.eye(self.config.total_dof // 2))

        chain = u.dot(self.la.diag(s))

        for i in range(tmin, tmax):
            chain = self.B_l(spin, i).dot(chain)

        u, s, v = self.SVD(chain)
        return u, s, v.dot(current_V)

    def get_nonequal_time_GFs(self, spin, GF_0):
        current_GF = 1. * GF_0.copy()
        self.left_decompositions = self._get_left_partial_SVD_decompositions(spin)
        self.right_decompositions = self._get_right_partial_SVD_decompositions(spin)
        GFs = [1. * self.to_numpy(current_GF)]
        for tau in range(1, self.config.Nt):
            B = self.B_l(spin, tau - 1)
            
            if tau % self.config.s_refresh != 0:
                current_GF = B.dot(current_GF)  # just wrap-up / wrap-down
            else:  # recompute GF from scratch
                u1, s1, v1 = self.compute_B_chain(spin, tau, 0)  # tau - 1 | ... | 0
                u2, s2, v2 = self.compute_B_chain(spin, self.config.Nt, tau)  # Nt - 1 | ... | tau
                s1_min = 1.0 * s1; s1_max = 1.0 * s1
                s1_min[s1_min > 1.] = 1.
                s1_max[s1_max < 1.] = 1.

                s2_min = 1.0 * s2; s2_max = 1.0 * s2
                s2_min[s2_min > 1.] = 1.
                s2_max[s2_max < 1.] = 1.
                m = self.la.diag(s1_max ** -1).dot(u1.T).dot(v2.T).dot(self.la.diag(s2_max ** -1)) + \
                    self.la.diag(s1_min).dot(v1).dot(u2).dot(self.la.diag(s2_min))
                current_GF = (v2.T).dot(self.la.diag(s2_max ** -1)).dot(self.la.linalg.inv(m)).dot(self.la.diag(s1_min)).dot(v1)
            
            GFs.append(1.0 * self.to_numpy(current_GF))
        return np.array(GFs)

    ####### DEBUG ######
    def get_G_no_optimisation(self, spin, time_slice, return_udv = False):
        M = self.la.eye(self.config.total_dof // 2)
        current_U = self.la.eye(self.config.total_dof // 2)
        slices = list(range(time_slice + 1, self.config.Nt)) + list(range(0, time_slice + 1))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            M = M.dot(B)
            u, s, v = self.SVD(M)
            # print(self.la.sum(self.la.abs(u.dot(self.la.diag(s)).dot(v) - M)) / self.la.sum(self.la.abs(M)), 'discrepancy of SVD')
            current_U = current_U.dot(u)
            M = self.la.diag(s).dot(v)
        m = current_U.T.dot(v.T) + self.la.diag(s)
        um, sm, vm = self.SVD(m)

        if return_udv:
            return (vm.dot(v)).T, self.la.diag(sm ** -1), (current_U.dot(um)).T
        return ((vm.dot(v)).T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).T), self.la.sum(self.la.log(sm ** -1)), \
               np.sign(np.linalg.det(((vm.dot(v)).T)) * np.linalg.det((current_U.dot(um)).T))

    def get_assymetry_factor(self):
        G_up = self.get_G_no_optimisation(+1, 0)[0]
        G_down = self.get_G_no_optimisation(-1, 0)[0]
        sign_up, log_det_up = np.linalg.slogdet(G_up)
        sign_down, log_det_down = np.linalg.slogdet(G_down)
        s_factor_log = self.config.nu_U * self.la.sum(self.configuration[..., 1:3])  # in case of xy-yx pairings
        return log_det_up + s_factor_log - log_det_down, sign_up - sign_down
    ####### END DEBUG ######

    def get_current_conf(self, sp_index, time_slice):
        return tuple(self.configuration[time_slice, sp_index, ...])


class AuxiliaryFieldInterorbital(AuxiliaryFieldIntraorbital):
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir):
        super().__init__(config, K, K_inverse, K_matrix, local_workdir)
        self.local_conf_combinations = [
                                    (-1, -1, -1), (-1, -1, 1), \
                                    (-1, 1, -1), (-1, 1, 1), \
                                    (1, -1, -1), (1, -1, 1), \
                                    (1, 1, -1), (1, 1, 1),
                                ]
        return

    def _V_from_configuration(self, s, sign, spin):
        if spin > 0:
            V = self.config.nu_V * sign * np.array([-s[0], s[0]]) + \
                self.config.nu_U * sign * np.array([s[2], s[1]])
        else:
            V = self.config.nu_V * sign * np.array([-s[0], s[0]]) + \
                self.config.nu_U * sign * np.array([-s[2], -s[1]])
        return np.diag(np.exp(V))

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0
        elif self.config.start_type == 'hot':
            self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0
        else:
            if os.path.isfile(self.conf_path):
                self.configuration = self._load_configuration()
                print('Starting from a presaved field configuration', flush=True)
            else:
                self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0

        NtVolVol_shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2)
        self.V_up = np.zeros(shape = NtVolVol_shape); self.Vinv_up = np.zeros(shape = NtVolVol_shape)
        self.V_down = np.zeros(shape = NtVolVol_shape); self.Vinv_down = np.zeros(shape = NtVolVol_shape)
        

        for time_slice in range(self.config.Nt):
            for sp_index in range(self.config.total_dof // 2 // 2):
                sx = sp_index * 2
                sy = sp_index * 2 + 1
                self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], +1.0, +1.0)
                self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], -1.0, +1.0)

                self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], +1.0, -1)
                self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], -1.0, -1)
        return


    def update_field(self, sp_index, time_slice, new_conf):
        self.configuration[time_slice, sp_index, ...] = np.array(new_conf)
        sx = sp_index * 2
        sy = sp_index * 2 + 1
        self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration(new_conf, +1.0, +1.0, self.config.nu_U, self.config.nu_V)
        self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration(new_conf, -1.0, +1.0, self.config.nu_U, self.config.nu_V)
        self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration(new_conf, +1.0, -1.0, self.config.nu_U, self.config.nu_V)
        self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration(new_conf, -1.0, -1.0, self.config.nu_U, self.config.nu_V)
        return

    def B_l(self, spin, l, inverse = False):
        if not inverse:
            if spin > 0:
                return self.V_up[l, ...].dot(self.K)
            return self.V_down[l, ...].dot(self.K)

        if spin > 0:
            return self.K_inverse.dot(self.Vinv_up[l, ...])
        return self.K_inverse.dot(self.Vinv_down[l, ...])

    def compute_deltas(self, sp_index, time_slice, local_conf_proposed):
    	self.Delta_up = self.get_delta(+1., sp_index, time_slice, local_conf_proposed)
    	self.Delta_down = self.get_delta(-1., sp_index, time_slice, local_conf_proposed)
    	return


    def get_delta(self, spin, sp_index, time_slice, local_conf_proposed):  # sign change proposal is made at (time_slice, sp_index, o_index)
        return get_delta_interorbital(tuple(self.configuration[time_slice, sp_index, :]), \
                                      local_conf_proposed, spin, self.config.nu_U, self.config.nu_V)

    def update_G_seq(self, sp_index):
        self.current_G_function_up = _update_G_seq_inter(self.current_G_function_up, \
                                                         self.Delta_up, sp_index, self.config.total_dof)
        self.current_G_function_down = _update_G_seq_inter(self.current_G_function_down, \
                                                           self.Delta_down, sp_index, self.config.total_dof)
        return

    def copy_to_CPU(self):
        super().copy_to_CPU()
        if not self.gpu_avail:
            return self
        self.V_up = cp.asnumpy(self.V_up)
        self.Vinv_up = cp.asnumpy(self.Vinv_up)
        self.V_down = cp.asnumpy(self.V_down)
        self.Vinv_down = cp.asnumpy(self.Vinv_down)
        return

    def copy_to_GPU(self):
        super().copy_to_GPU()
        if not self.gpu_avail:
            return self
        self.V_up = cp.asarray(self.V_up)
        self.Vinv_up = cp.asarray(self.Vinv_up)
        self.V_down = cp.asarray(self.V_down)
        self.Vinv_down = cp.asarray(self.Vinv_down)
        self.configuration = cp.asnumpy(self.configuration)
        return

class AuxiliaryFieldInterorbitalAccurate(AuxiliaryFieldInterorbital):
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir):
        self.eta = {
            -2 : -np.sqrt(6 + 2 * np.sqrt(6)),
            +2 : +np.sqrt(6 + 2 * np.sqrt(6)),
            -1 : -np.sqrt(6 - 2 * np.sqrt(6)),
            +1 : +np.sqrt(6 - 2 * np.sqrt(6)),
        }

        self.gauge = {
            -2 : 1. - np.sqrt(6) / 3,
            2 : 1. - np.sqrt(6) / 3.,
            -1 : 1. + np.sqrt(6) / 3.,
            +1 : 1 + np.sqrt(6) / 3.
        }

        self.gauge_log = {
            -2 : np.log(1. - np.sqrt(6) / 3),
            2 : np.log(1. - np.sqrt(6) / 3),
            -1 : np.log(1. + np.sqrt(6) / 3),
            +1 : np.log(1 + np.sqrt(6) / 3)
        }

        super().__init__(config, K, K_inverse, K_matrix, local_workdir)
        self.local_conf_combinations = 
                                    [
                                        (-2, -1, -1), (-1, -1, -1), (1, -1, -1), (2, -1, -1), \
                                        (-2, -1, 1),  (-1, -1, 1),  (1, -1, 1),  (2, -1, 1), \
                                        (-2, 1, -1),  (-1, 1, -1),  (1, 1, -1),  (2, 1, -1), \
                                        (-2, 1, 1),   (-1, 1, 1),   (1, 1, 1),   (2, 1, 1)
                                    ]

        self.rnd = np.random.choice(np.array([-2, -1, 1, 2]), size=100000)
        self.rnd_idx = 0
        return

    def get_gauge_factor_move(self, sp_index, time_slice, local_conf):
        return self.gauge[local_conf[0]] / self.gauge[self.configuration[time_slice, sp_index, 0]]

    def get_current_gauge_factor_log(self):
        cf = self.configuration[..., 0].flatten()
        factor_logs = np.zeros(len(cf))
        factor_logs[cf == -2] = self.gauge_log[-2]
        factor_logs[cf == 2] = self.gauge_log[2]
        factor_logs[cf == 1] = self.gauge_log[1]
        factor_logs[cf == -1] = self.gauge_log[-1]

        return np.sum(factor_logs)

    def _V_from_configuration(self, s, sign, spin):
        if spin > 0:
            V = self.config.nu_V * self.eta[s[0]] * sign * np.array([1, -1]) + \
                self.config.nu_U * sign * np.array([s[2], s[1]])
        else:
            V = self.config.nu_V * self.eta[s[0]] * sign * np.array([1, -1]) + \
                self.config.nu_U * sign * np.array([-s[2], -s[1]])
        return np.diag(np.exp(V))

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0
        elif self.config.start_type == 'hot':
            self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0  # standard 2-valued aux Hubbard field
            self.configuration[..., 0] = np.random.choice(np.array([-2, -1, 1, 2]), \
                                                          size = (self.config.Nt, self.config.total_dof // 2 // 2))  # 4-valued F.F. Assaad field
        else:
            if os.path.isfile(self.conf_path):
                self.configuration = self._load_configuration()
                print('Starting from a presaved field configuration', flush=True)
            else:
                self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0
                self.configuration[..., 0] = np.random.choice(np.array([-2, -1, 1, 2]), \
                                                              size = (self.config.Nt, self.config.total_dof // 2 // 2))

        NtVolVol_shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2)
        self.V_up = np.zeros(shape = NtVolVol_shape); self.Vinv_up = np.zeros(shape = NtVolVol_shape)
        self.V_down = np.zeros(shape = NtVolVol_shape); self.Vinv_down = np.zeros(shape = NtVolVol_shape)

        for time_slice in range(self.config.Nt):
            for sp_index in range(self.config.total_dof // 2 // 2):
                sx = sp_index * 2
                sy = sp_index * 2 + 1
                self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], +1.0, +1.0)
                self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], -1.0, +1.0)

                self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], +1.0, -1)
                self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    self._V_from_configuration(self.configuration[time_slice, sp_index, :], -1.0, -1)
        return

    def update_field(self, sp_index, time_slice, new_conf):
        self.configuration[time_slice, sp_index, ...] = np.array(new_conf)
        sx = sp_index * 2
        sy = sp_index * 2 + 1
        self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate(new_conf, +1.0, +1.0, self.config.nu_U, self.config.nu_V)
        self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] =\
            _V_from_configuration_accurate(new_conf, -1.0, +1.0, self.config.nu_U, self.config.nu_V)
        self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate(new_conf, +1.0, -1.0, self.config.nu_U, self.config.nu_V)
        self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate(new_conf, -1.0, -1.0, self.config.nu_U, self.config.nu_V)

        return

    def compute_deltas(self, sp_index, time_slice, local_conf, local_conf_proposed):
        self.Delta_up = _get_delta_interorbital_accurate(local_conf, local_conf_proposed, +1, self.config.nu_U, self.config.nu_V)
        self.Delta_down = _get_delta_interorbital_accurate(local_conf, local_conf_proposed, -1, self.config.nu_U, self.config.nu_V)
        return

    def get_delta(self, spin, sp_index, time_slice, local_conf_proposed):  # sign change proposal is made at (time_slice, sp_index, o_index)
        return _get_delta_interorbital_accurate(tuple(self.configuration[time_slice, sp_index, :]), \
                                                local_conf_proposed, spin, self.config.nu_U, self.config.nu_V)

@jit(nopython = True)
def _get_delta_interorbital_accurate(local_conf, local_conf_proposed, spin, \
                                     nu_U, nu_V):  # sign change proposal is made at (time_slice, sp_index, o_index)
    local_V_inv = _V_from_configuration_accurate(local_conf, -1.0, spin, nu_U, nu_V)  # already stored in self.V or self.Vinv
    local_V_proposed = _V_from_configuration_accurate(local_conf_proposed, 1.0, spin, nu_U, nu_V)
    return np.diag(np.array([local_V_proposed[0, 0] * local_V_inv[0, 0] - 1, local_V_proposed[1, 1] * local_V_inv[1, 1] - 1])) # local_V_proposed.dot(local_V_inv) - np.eye(2)

@jit(nopython=True)
def _V_from_configuration_accurate(s, sign, spin, nu_U, nu_V):
    #eta = {
    #    -2 : -np.sqrt(6 + 2 * np.sqrt(6)),
    #    +2 : +np.sqrt(6 + 2 * np.sqrt(6)),
    #    -1 : -np.sqrt(6 - 2 * np.sqrt(6)),
    #    +1 : +np.sqrt(6 - 2 * np.sqrt(6)),
    #}

    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]
    if spin > 0:
        V = nu_V * eta[int(s[0]) + 2] * sign * np.array([1, -1]) + \
            nu_U * sign * np.array([s[2], s[1]])
    else:
        V = nu_V * eta[int(s[0]) + 2] * sign * np.array([1, -1]) + \
            nu_U * sign * np.array([-s[2], -s[1]])
    return np.diag(np.exp(V))


@jit(nopython=True)
def _V_from_configuration(s, sign, spin, nu_U, nu_V):
    if spin > 0:
        V = nu_V * sign * np.array([-s[0], s[0]]) + nu_U * sign * np.array([s[2], s[1]])
    else:
        V = nu_V * sign * np.array([-s[0], s[0]]) + nu_U * sign * np.array([-s[2], -s[1]])
    return np.diag(np.exp(V))


@jit(nopython = True)
def get_delta_interorbital(local_configuration, local_conf_proposed, spin, \
                           nu_U, nu_V):  # sign change proposal is made at (time_slice, sp_index, o_index)
    local_V_inv = _V_from_configuration(local_configuration, -1.0, spin, nu_U, nu_V)  # already stored in self.V or self.Vinv
    local_V_proposed = _V_from_configuration(local_conf_proposed, 1.0, spin, nu_U, nu_V)
    return local_V_proposed.dot(local_V_inv) - np.eye(2)

@jit(nopython=True)
def get_delta_intraorbital(s, spin, nu_U):
    return np.exp(-2 * spin * s * nu_U) - 1.

@jit(nopython=True)
def get_det_ratio_inter(sp_index, Delta, G):
    sx = sp_index * 2
    sy = sp_index * 2 + 1

    return np.linalg.det(np.eye(2) + Delta.dot(np.eye(2) - G[sx : sy + 1, sx : sy + 1]))

@jit(nopython=True)
def get_det_ratio_intra(sp_index, Delta, G):
    return 1. + Delta * (1. - G[sp_index, sp_index])

@jit(nopython=True)	
def _update_G_seq_inter(G, Delta, sp_index, total_dof):
    sx = sp_index * 2
    sy = sp_index * 2 + 1
    G_sliced_right = G[:, sx : sy + 1]
    G_sliced_left = G[sx:sy + 1, :]

    update_matrix = np.zeros((2, total_dof // 2))  # keep only two nontrivial rows here
    update_matrix[:, sx:sy + 1] = np.eye(2) + Delta
    update_matrix -= np.dot(Delta, np.ascontiguousarray(G_sliced_left))
    det = np.linalg.det(update_matrix[:, sx:sy + 1])

    inverse_update_matrix = np.zeros((2, total_dof // 2))  # keep only two nontrivial rows here

    inverse_update_matrix[0, :] = -(update_matrix[0, :] * update_matrix[1, sy] - \
                                    update_matrix[1, :] * update_matrix[0, sy]) / det  # my vectorized det :))
    inverse_update_matrix[1, :] = (update_matrix[0, :] * update_matrix[1, sx] - \
    	                           update_matrix[1, :] * update_matrix[0, sx]) / det

    inverse_update_matrix[0, sx] = update_matrix[1, sy] / det - 1
    inverse_update_matrix[1, sx] = -update_matrix[1, sx] / det
    inverse_update_matrix[0, sy] = -update_matrix[0, sy] / det
    inverse_update_matrix[1, sy] = update_matrix[0, sx] / det - 1

    G = G + np.dot(np.ascontiguousarray(G_sliced_right), inverse_update_matrix)
    return G

def _update_G_seq_intra(G, Delta, sp_index, total_dof):
    update_matrix = Delta * (np.eye(total_dof // 2) - G)[sp_index, :]
    update_matrix[sp_index] += 1.
    det_update_matrix = update_matrix[sp_index]
    update_matrix_inv = -update_matrix / det_update_matrix
    update_matrix_inv[sp_index] = 1. / det_update_matrix - 1.
    G = G + np.outer(G[:, sp_index], update_matrix_inv)

    return G
