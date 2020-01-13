import numpy as np
import pickle
import time
import scipy
import models
from copy import deepcopy

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass


class auxiliary_field_intraorbital:
    def __init__(self, config, K, K_inverse, K_matrix, gpu_avail):
        self.gpu_avail = gpu_avail
        self.la = np
        self.cpu = True

        self.config = config
        self.adj_list = models.get_adjacency_list(self.config)[0]
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
        return

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
        #print(svd_lhs)
        u2, s2, v2 = svd_rhs
        m = v1.dot(u2)
        middle_mat = (u1.T).dot(v2.T) + xp.diag(s1).dot(m).dot(xp.diag(s2))
        um, sm, vm = self.SVD(middle_mat)

        left = (vm.dot(v2)).T
        right = (u1.dot(um)).T
        # print(xp.sum(xp.abs(xp.linalg.inv(left) - left.T)))
        # print(xp.sum(xp.abs(xp.linalg.inv(right) - right.T)))
        if return_logdet:
            return left.dot((xp.diag(sm ** -1)).dot(right)), \
                   xp.sum(xp.log(sm ** -1)), \
                   xp.sign(xp.linalg.det(left) * xp.linalg.det(right))
        return left.dot(xp.diag(sm ** -1)).dot(right)

    def refresh_all_decompositions(self):
        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self._get_partial_SVD_decompositions(spin = +1)
        self._get_partial_SVD_decompositions(spin = -1)

        self.current_lhs_SVD_up = list(self.SVD(self.la.diag(self.la.ones(self.config.total_dof // 2))))
        self.current_lhs_SVD_down = list(self.SVD(self.la.diag(self.la.ones(self.config.total_dof // 2))));
        return

    def _product_svds(self, svd1, svd2):
        u1, s1, v1 = svd1
        u2, s2, v2 = svd2
        m = v1.dot(u2)
        middle_mat = xp.diag(s1).dot(m).dot(xp.diag(s2))
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
                # there are two possible lapack drivers: gesdd and gesvd. the former ises the divide-and-conquer approach and is faster,
                # while the latter performs QR decomposition and is MUCH MORE PRECISE. As small T and large U this is CRUCIAL
                # otherwise the SVD decomposition is a complete mess.
                # numpy uses gesdd and thus can not be osed in this calculation
                # cupy and scipy, on contrary, use gesvd and must be used as the GPU and CPU backends, respectively
                
                # print('refresh', nr)
                # print(xp.sum(xp.abs(xp.imag(u))), xp.sum(xp.abs(xp.imag(v))))
                # print(xp.allclose((u.dot(xp.diag(s))).dot(v), M, atol=1e-11))
                # print(xp.max(xp.abs(xp.eye(self.config.total_dof // 2) - (v.T).dot(xp.diag(s**-1).dot(u.T)).dot(M))), xp.max(M))
                
                current_U = current_U.dot(u)
                if spin == +1:
                    self.partial_SVD_decompositions_up.append((current_U, s, v))
                else:
                    self.partial_SVD_decompositions_down.append((current_U, s, v))
                M = self.la.diag(s).dot(v)
        return


    ####### DEBUG ######
    def get_G_no_optimisation(self, spin, time_slice):
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
        return ((vm.dot(v)).T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).T), self.la.sum(self.la.log(sm ** -1))

    def get_assymetry_factor(self):
        log_det_up, sign_up = self.get_current_G_function(+1, return_logdet = True)[1:]
        log_det_down, sign_down = self.get_current_G_function(-1, return_logdet = True)[1:]
        s_factor_log = self.config.nu_U * xp.sum(self.configuration[..., 0:2])  # in case of xy-yx pairings
        return log_det_up + s_factor_log - log_det_down, sign_up - sign_down

    def _get_partial_SVD_decomposition_range(self, spin, tmin, tmax):
        M = xp.eye(self.config.total_dof // 2)
        
        for time_slice in range(tmin, tmax):
            M = self.B_l(spin, time_slice, inverse = False).dot(M)
        return self.SVD(M)

    def _load_configuration(self, path):
        return np.load(start_type)

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = xp.asarray(np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)
            return
        if self.config.start_type == 'hot':
            self.configuration = xp.asarray(np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)
            return
        self.configuration = xp.asarray(self._load_configuration(start_type))
        return

    def save_configuration(self, path):
        return np.save(path, self.configuration)

    def B_l(self, spin, l, inverse = False):
        if not inverse:
            V = self.la.diag(self.la.exp(spin * self.config.nu_U * self.configuration[l, ...]))
            return V.dot(self.K)
    
        V = self.la.diag(self.la.exp(-spin * self.config.nu_U * self.configuration[l, ...]))
        return self.K_inverse.dot(V)

    def get_delta(self, spin, time_slice, sp_index):
        return self.la.exp(-2 * spin * self.configuration[time_slice, sp_index] * self.config.nu_U) - 1.

    def get_det_ratio(self, spin, sp_index, time_slice, *args):
        Delta = self.get_delta(spin, time_slice, sp_index)
        if spin == +1:
            G = self.current_G_function_up
        else:
            G = self.current_G_function_down
        return 1. + Delta * (1. - G[sp_index, sp_index])

    def update_G_seq(self, spin, sp_index, time_slice, *args):
        Delta = self.get_delta(spin, time_slice, sp_index)

        if spin == +1:
            G = self.current_G_function_up
        else:
            G = self.current_G_function_down
        update_matrix = Delta * (self.la.eye(self.config.total_dof // 2) - G)[sp_index, :]
        update_matrix[sp_index] += 1.
        det_update_matrix = update_matrix[sp_index]
        update_matrix_inv = -update_matrix / det_update_matrix
        update_matrix_inv[sp_index] = 1. / det_update_matrix - 1.
        G = G + self.la.einsum('i,k->ik', G[:, sp_index], update_matrix_inv)

        if spin == +1:
            self.current_G_function_up = G
        else:
            self.current_G_function_down = G
        
        return

    def update_field(self, sp_index, time_slice, *args):
        self.configuration[time_slice, sp_index] *= -1
        return

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


    def wrap_up(self, time_slice, gpu = False):
        B_wrap_up = self.B_l(+1, time_slice, inverse = False)
        B_wrap_up_inverse = self.B_l(+1, time_slice, inverse = True)
        B_wrap_down = self.B_l(-1, time_slice, inverse = False)
        B_wrap_down_inverse = self.B_l(-1, time_slice, inverse = True)

        self.current_G_function_up = B_wrap_up.dot(self.current_G_function_up.dot(B_wrap_up_inverse))
        self.current_G_function_down = B_wrap_down.dot(self.current_G_function_down.dot(B_wrap_down_inverse))

        return


class auxiliary_field_interorbital(auxiliary_field_intraorbital):
    def __init__(self, config, K, K_inverse, K_matrix, gpu_avail):
        super().__init__(config, K, K_inverse, K_matrix, gpu_avail)
        return

    def _V_from_configuration(self, s, sign, spin):
        if spin > 0:
            V = self.config.nu_V * sign * np.array([-s[0], s[0]]) + self.config.nu_U * sign * np.array([s[2], s[1]])
        else:
            V = self.config.nu_V * sign * np.array([-s[0], s[0]]) + self.config.nu_U * sign * np.array([-s[2], -s[1]])
        return np.diag(np.exp(V))

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0
        if self.config.start_type == 'hot':
            self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 3)) * 2. - 1.0

        self.V_up = np.zeros(shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.Vinv_up = np.zeros(shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.V_down = np.zeros(shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))
        self.Vinv_down = np.zeros(shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2))

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

        self.V_up = self.V_up
        self.Vinv_up = self.Vinv_up
        self.V_down = self.V_down
        self.Vinv_down = self.Vinv_down
        return

    def update_field(self, sp_index, time_slice, o_index):
        self.configuration[time_slice, sp_index, o_index] *= -1
        sx = sp_index * 2
        sy = sp_index * 2 + 1
        self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            self.la.asarray(self._V_from_configuration(self.configuration[time_slice, sp_index, ...], +1.0, +1.0))
        self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            self.la.asarray(self._V_from_configuration(self.configuration[time_slice, sp_index, ...], -1.0, +1.0))

        self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            self.la.asarray(self._V_from_configuration(self.configuration[time_slice, sp_index, ...], +1.0, -1.0))
        self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            self.la.asarray(self._V_from_configuration(self.configuration[time_slice, sp_index, ...], -1.0, -1.0))
        return

    def B_l(self, spin, l, inverse = False):
        if not inverse:
            if spin > 0:
                return self.V_up[l, ...].dot(self.K)
            return self.V_down[l, ...].dot(self.K)

        if spin > 0:
            return self.K_inverse.dot(self.Vinv_up[l, ...])
        return self.K_inverse.dot(self.Vinv_down[l, ...])

    def get_delta(self, spin, sp_index, time_slice, o_index):  # sign change proposal is made at (time_slice, sp_index, o_index)
        local_configuration = self.configuration[time_slice, sp_index, :]
        local_configuration_proposed = deepcopy(self.configuration[time_slice, sp_index, :])
        local_configuration_proposed[o_index] *= -1

        local_V = self._V_from_configuration(local_configuration, -1.0, spin)  # already stored in self.V or self.Vinv
        local_V_proposed = self._V_from_configuration(local_configuration_proposed, 1.0, spin)
        return local_V_proposed.dot(local_V) - np.eye(2)

    def get_det_ratio(self, spin, sp_index, time_slice, o_index):
        Delta = self.get_delta(spin, sp_index, time_slice, o_index)
        sx = sp_index * 2
        sy = sp_index * 2 + 1

        if spin == +1:
            self.Delta_up = self.la.asarray(Delta)
            G = self.current_G_function_up
        else:
            self.Delta_down = self.la.asarray(Delta)
            G = self.current_G_function_down
        if self.cpu:
            return np.linalg.det(np.eye(2) + Delta.dot(np.eye(2) - G[sx : sy + 1, sx : sy + 1]))
        return np.linalg.det(np.eye(2) + Delta.dot(np.eye(2) - cp.asnumpy(G[sx : sy + 1, sx : sy + 1])))

    def update_G_seq(self, spin, sp_index, time_slice, o_index):
        if spin == +1:
            Delta = self.Delta_up
            G = self.current_G_function_up
        else:
            Delta = self.Delta_down
            G = self.current_G_function_down

        sx = sp_index * 2
        sy = sp_index * 2 + 1

        update_matrix = self.la.zeros((2, self.config.total_dof // 2))  # keep only two nontrivial rows here
        update_matrix[:, sx:sy + 1] = self.la.eye(2) + Delta
        update_matrix -= self.la.einsum('ij,jk->ik', Delta, G[sx:sy + 1, :])
        det = self.la.linalg.det(update_matrix[:, sx : sy + 1])

        inverse_update_matrix = self.la.zeros((2, self.config.total_dof // 2))  # keep only two nontrivial rows here

        inverse_update_matrix[0, :] = -(update_matrix[0, :] * update_matrix[1, sy] - update_matrix[1, :] * update_matrix[0, sy]) / det  # my vectorized det :))
        inverse_update_matrix[1, :] = (update_matrix[0, :] * update_matrix[1, sx] - update_matrix[1, :] * update_matrix[0, sx]) / det

        inverse_update_matrix[0, sx] = update_matrix[1, sy] / det - 1
        inverse_update_matrix[1, sx] = -update_matrix[1, sx] / det
        inverse_update_matrix[0, sy] = -update_matrix[0, sy] / det
        inverse_update_matrix[1, sy] = update_matrix[0, sx] / det - 1

        G = G + self.la.einsum('ij,jk->ik', G[:, sx : sy + 1], inverse_update_matrix)
        if spin == +1:
            self.current_G_function_up = G
        else:
            self.current_G_function_down = G

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
