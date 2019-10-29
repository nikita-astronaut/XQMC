import numpy as np
import pickle
import time

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass


class auxiliary_field:
    def __init__(self, config, K, K_inverse):
        self.config = config
        self.configuration = self._get_initial_field_configuration()
        self.K = K
        self.K_inverse = K_inverse
        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self.current_lhs_SVD_up = []
        self.current_lhs_SVD_down = []

        self.refresh_all_decompositions()
        self.current_G_function_up = []
        self.current_G_function_down = []
        self.refresh_G_functions()
        self.current_time_slice = 0

        self.refresh_checkpoints = [0]
        t = self.config.Nt % self.config.s_refresh
        if t == 0:
            self.refresh_checkpoints = []
        while t < self.config.Nt:
            self.refresh_checkpoints.append(t)
            t += self.config.s_refresh
        self.refresh_checkpoints = np.array(self.refresh_checkpoints)
        print('checkpoints = ', self.refresh_checkpoints)
        return

    def refresh_G_functions(self):
        self.current_G_function_up = self.get_current_G_function(+1)
        self.current_G_function_down = self.get_current_G_function(-1)
        return

    def get_current_G_function(self, spin):
        if spin == +1:
            svd_rhs = self.partial_SVD_decompositions_up[-1]
            svd_lhs = self.current_lhs_SVD_up
        else:
            svd_rhs = self.partial_SVD_decompositions_down[-1]
            svd_lhs = self.current_lhs_SVD_down
        return self.inv_illcond(xp.diag(xp.ones(self.config.total_dof // 2)) + self._unwrap_svd(svd_lhs).dot(self._unwrap_svd(svd_rhs)))


    def refresh_all_decompositions(self):
        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self._get_partial_SVD_decompositions(spin = +1)
        self._get_partial_SVD_decompositions(spin = -1)

        self.current_lhs_SVD_up = [xp.diag(xp.ones(self.config.total_dof // 2)), xp.ones(self.config.total_dof // 2), xp.diag(xp.ones(self.config.total_dof // 2))]
        self.current_lhs_SVD_down = [xp.diag(xp.ones(self.config.total_dof // 2)), xp.ones(self.config.total_dof // 2), xp.diag(xp.ones(self.config.total_dof // 2))]
        return

    def _unwrap_svd(self, svd):
        return svd[0].dot(xp.diag(svd[1]).dot(svd[2]))

    def append_new_decomposition(self, tmin, tmax):
        assert tmax - tmin == self.config.Nt % self.config.s_refresh or tmax - tmin == self.config.s_refresh
        lhs_change_up = self._get_partial_SVD_decomposition_range(+1, tmin, tmax)
        lhs_change_down = self._get_partial_SVD_decomposition_range(-1, tmin, tmax)

        self.current_lhs_SVD_up = xp.linalg.svd(self._unwrap_svd(lhs_change_up).dot(self._unwrap_svd(self.current_lhs_SVD_up)))
        self.current_lhs_SVD_down = xp.linalg.svd(self._unwrap_svd(lhs_change_down).dot(self._unwrap_svd(self.current_lhs_SVD_down)))

        del self.partial_SVD_decompositions_up[-1]
        del self.partial_SVD_decompositions_down[-1]
        self.current_time_slice = tmax
        return

    def _get_partial_SVD_decompositions(self, spin):
        M = xp.diag(xp.ones(self.config.total_dof // 2))
        current_U = xp.diag(xp.ones(self.config.total_dof // 2))

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            M = M.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1 or nr == self.config.Nt - 1:
                print('write dec at nr = ', nr, ' index = ', slice_idx)
                u, s, v = xp.linalg.svd(M)
                current_U = current_U.dot(u)
                if spin == +1:
                    self.partial_SVD_decompositions_up.append((current_U, s, v))
                else:
                    self.partial_SVD_decompositions_down.append((current_U, s, v))
                M = xp.diag(s).dot(v)
        return


    def _get_partial_SVD_decomposition_range(self, spin, tmin, tmax):
        M = xp.diag(xp.ones(self.config.total_dof // 2))
        
        for time_slice in range(tmin, tmax):
            M = self.B_l(spin, time_slice, inverse = False).dot(M)
        return xp.linalg.svd(M)


    def inv_illcond(self, A):
        u, s, v = xp.linalg.svd(A)
        Ainv = xp.dot(v.transpose(), xp.dot(xp.diag(s**-1), u.transpose()))
        return Ainv

    def _load_configuration(self, path):
        return np.load(start_type)

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            return xp.asarray(np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)
        if self.config.start_type == 'hot':
            return xp.asarray(np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2)) * 2. - 1.0)

        return xp.asarray(self._load_configuration(start_type))

    def save_configuration(self, path):
        return np.save(path, self.configuration)

    def flip_random_spin(self):
        time_slice = np.random.randint(0, self.config.Nt)
        spatial_index = np.random.randint(0, self.config.total_dof // 2)
        self.configuration[time_slice, spatial_index] *= -1
        return

    def B_l(self, spin, l, inverse = False):
        if not inverse:
            V = xp.diag(xp.exp(spin * self.config.nu * self.configuration[l, ...]))
            return V.dot(self.K)
    
        V = xp.diag(xp.exp(-spin * self.config.nu * self.configuration[l, ...]))
        return self.K_inverse.dot(V)

    def get_det_ratio(self, spin, sp_index, time_slice):
        Delta = np.exp(2 * spin * self.configuration[time_slice, sp_index] * self.config.nu) - 1.
        if spin == +1:
            G = self.current_G_function_up
        else:
            G = self.current_G_function_down
        return 1. + Delta * (1. - G[sp_index, sp_index])

    def update_G_seq(self, spin, sp_index, time_slice):
        Delta = np.exp(2 * spin * self.configuration[time_slice, sp_index] * self.config.nu) - 1.
        if spin == +1:
            G = self.current_G_function_up
        else:
            G = self.current_G_function_down

        update_matrix = Delta * (xp.diag(xp.ones(self.config.total_dof // 2)) - G)[sp_index, :]
        update_matrix[sp_index] += 1.
        det_update_matrix = update_matrix[sp_index]
        update_matrix_inv = -update_matrix / det_update_matrix
        update_matrix_inv[sp_index] = 1. / det_update_matrix - 1.
        G = G + xp.einsum('i,k->ik', G[:, sp_index], update_matrix_inv)

        if spin == +1:
            self.current_G_function_up = G
        else:
            self.current_G_function_down = G
        
        return

    def wrap_up(self, time_slice):
        B_wrap_up = self.B_l(+1, time_slice, inverse = False)
        B_wrap_up_inverse = self.B_l(+1, time_slice, inverse = True)
        B_wrap_down = self.B_l(-1, time_slice, inverse = False)
        B_wrap_down_inverse = self.B_l(-1, time_slice, inverse = True)

        self.current_G_function_up = B_wrap_up.dot(self.current_G_function_up.dot(B_wrap_up_inverse))
        self.current_G_function_down = B_wrap_down.dot(self.current_G_function_down.dot(B_wrap_down_inverse))

        return

    def get_log_det(self):
        sign_det_up, log_det_up = xp.linalg.slogdet(self.current_G_function_up)
        sign_det_down, log_det_down = xp.linalg.slogdet(self.current_G_function_down)

        return -np.real(log_det_up + log_det_down), sign_det_up * sign_det_down

    ####### DEBUG ######
    def get_G_no_optimisation(self, spin, time_slice):
        M = xp.diag(xp.ones(self.config.total_dof // 2))

        slices = list(range(time_slice + 1, self.config.Nt)) + list(range(0, time_slice + 1))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            M = M.dot(B)
        return self.inv_illcond(xp.diag(xp.ones(self.config.total_dof // 2)) + M)


def get_det_partial_matrices(M_up_partial, B_up_l, M_down_partial, B_down_l, identity):
    sign_det_up, log_det_up = xp.linalg.slogdet(identity + B_up_l.dot(M_up_partial))
    sign_det_down, log_det_down = xp.linalg.slogdet(identity + B_down_l.dot(M_down_partial))
    return np.real(log_det_up + log_det_down), sign_det_up * sign_det_down
