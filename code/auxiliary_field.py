import numpy as np
import pickle
from time import time
import scipy
import models
from copy import deepcopy
from numba import jit
import os
from mpmath import mp


try:
    import cupy as cp
except ImportError:
    pass

class AuxiliaryFieldIntraorbital:
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        self.gpu_avail = config.gpu
        self.la = np
        self.cpu = True
        self.total_SVD_time = 0.0

        self.config = config
        self.adj_list = config.adj_list
        self.conf_path = os.path.join(local_workdir, 'last_conf')
        self._get_initial_field_configuration()

        self.K = K
        self.K_inverse = K_inverse
        self.K_matrix = K_matrix
        self.K_half_inverse = K_half_inverse
        self.K_half = K_half

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

        self.n_times_saved = 0

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

    def SVD(self, matrix, mode = 'right_to_left'):
        def checkSVDDone(D, threshold):
            N = len(D)

            if N < 2 or threshold < 0.:
                return True, 0
            S0t = D[0] * threshold
            start = 1
            while start < N:
                if D[start] < S0t:
                    break
                start += 1
            if start >= N - 1:
                return True, N - 1
            return False, start

        def qr_positive(V):
            Q, R = np.linalg.qr(V)
            nc = Q.shape[1]

            for c in range(nc):
                if R[c, c] == 0.:
                    continue
                phase = R[c, c] / np.abs(R[c, c])
                R[c, c:] *= np.conj(phase)
                Q[:, c] *= phase
            return Q, R

        def svd_recursive(M, threshold = 1e-2, north_pass = 2):
            rho = -np.dot(M, M.conj().T)
            D, U = np.linalg.eigh(rho)

            Nd = len(D)

            V = np.dot(M.conj().T, U)# M'*U
            V, R = qr_positive(V)

            D = np.diag(R) * 1.

            done, start = checkSVDDone(D, threshold)
            if done:
                return U, D, V
            
            u = U[:, start:]
            v = V[:, start:]
            b = u.conj().T.dot(np.dot(M, v))  # a square matrix still

            bu,bd,bv = svd_recursive(b,
                          threshold=threshold,
                          north_pass=north_pass)

            U[:, start:] = np.dot(U[:, start:], bu)
            V[:, start:] = np.dot(V[:, start:], bv)
            D[start:] = bd

            return U, D, V


        if self.cpu:
            #return scipy.linalg.svd(matrix, lapack_driver='gesvd') 
            if mode == 'right_to_left':
                t = time()
                U, D, V = svd_recursive(matrix.conj().T)
                self.total_SVD_time += time() - t
                return V, D, U.conj().T
            else:
                t = time()
                U, D, V = svd_recursive(matrix)
                self.total_SVD_time += time() - t
                return U, D, V.conj().T

            _, Unew = np.linalg.eigh(matrix @ matrix.conj().T)
            Vnew, Rnew = np.linalg.qr(matrix.conj().T @ Unew)
            assert np.allclose(matrix.conj().T @ Unew, np.dot(Vnew, Rnew))
            ##print(np.diag(Rnew), 'awqweqwewqe')
            print(np.sum(np.abs(Unew.dot(np.diag(np.diag(Rnew))).dot(Vnew.conj().T) - matrix)))
            #print(np.diag(Rnew))
            assert np.allclose(Unew.dot(np.diag(np.diag(Rnew))).dot(Vnew.conj().T), matrix)
            return Unew, np.diag(Rnew), Vnew.conj().T

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

        #print(s1, 'decomposition S1')
        #print(s2,' decomposition S2')


        s1_min = s1.copy(); s1_min[s1_min > 1.] = 1.
        s2_min = s2.copy(); s2_min[s2_min > 1.] = 1.
        s1_max = s1.copy(); s1_max[s1_max < 1.] = 1.
        s2_max = s2.copy(); s2_max[s2_max < 1.] = 1.

        middle_mat = np.diag(s1_max ** -1) @ u1.conj().T @ v2.conj().T @ np.diag(s2_max ** -1) + np.diag(s1_min) @ v1 @ u2 @ np.diag(s2_min)


        #middle_mat = (u1.conj().T).dot(v2.conj().T) + (self.la.diag(s1).dot(m)).dot(self.la.diag(s2))  # are these u1.T / v2.T correct incase of imag code? I think no

        U, D, V = self.SVD(middle_mat)
        inv = V.conj().T.dot(np.diag(D ** -1)).dot(U.conj().T)
        #inv = self.la.linalg.inv(middle_mat)

        #assert np.allclose(v2.conj().T.dot(inv).dot(u1.conj().T), \
        #                   np.linalg.inv(np.eye(len(s1), dtype=np.complex128) + u1.dot(self.la.diag(s1)).dot(v1).dot(u2).dot(self.la.diag(s2)).dot(v2)))

        
        res = v2.conj().T @ np.diag(s2_max ** -1) @ inv @ np.diag(s1_max ** -1) @ u1.conj().T
        
        '''
        if spin == 1.0:
            current_tau = self.config.Nt * self.config.dt
            B = u1.dot(np.diag(s1)).dot(v1).dot(u2).dot(np.diag(s2)).dot(v2)
            energies, states = np.linalg.eigh(self.K_matrix_plus)
            states = states.T.conj()
            assert np.allclose(self.K_matrix_plus, self.K_matrix_plus.conj().T)
            assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), self.K_matrix_plus)
            correct_Bstring = np.einsum('i,ij,ik->jk', np.exp(current_tau * energies), states.conj(), states)

            print('G without optimization obtaining: B chain precision:')
            d_B = self.SVD(B)[1]
            d_corr = self.SVD(correct_Bstring)[1]
            print(np.abs((d_B - d_corr) / d_corr), 'eigenvalues discrepancy in B chain from two parts')


            energies, states = np.linalg.eigh(self.K_matrix_plus)
            states = states.T.conj()
            assert np.allclose(self.K_matrix_plus, self.K_matrix_plus.conj().T)
            assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), self.K_matrix_plus)
            correct_string = np.einsum('i,ij,ik->jk', 1. / (1. + np.exp(current_tau * energies)), states.conj(), states)
            print('get_current_G_function ERROR: ', np.linalg.norm(correct_string - res) / np.linalg.norm(correct_string))

            svd_check = self.SVD(correct_string)[1]
            svd_res = self.SVD(res)[1]

            print(np.abs((svd_res - svd_check) / svd_check), 'eigenvalues discrepancy')

            idx = np.where(np.abs((svd_res - svd_check) / svd_check) > 1e-2)[0][0]
            print(svd_check, 'eigenvalues themselves', svd_check[idx])
        
        '''
        phase, ld = np.linalg.slogdet(res)
        #U, D, V = self.SVD(res)
        #ld = self.la.sum(self.la.log(D)) + np.linalg.slogdet(U)[1] + np.linalg.slogdet(V)[1]
        #phase = np.linalg.slogdet(res)[0]
 
        #phase, ld = np.linalg.slogdet(res)
        #ph_check, log_check = np.linalg.slogdet(correct_string)
        #U, D, V = self.SVD(correct_string)
        #log_check = self.la.sum(self.la.log(D)) + np.linalg.slogdet(U)[1] + np.linalg.slogdet(V)[1]
        #ph_check = np.linalg.slogdet(U.dot(V))[0]



        #print('phase check opt =', ph_check - phase, ph_check, phase)
        #print('log check opt =', log_check - ld, log_check, ld)
        if return_logdet:
            return res, ld, phase
        return res

    def refresh_all_decompositions(self):
        self.partial_SVD_decompositions_up = []
        self.partial_SVD_decompositions_down = []
        self._get_partial_SVD_decompositions(spin = +1)
        self._get_partial_SVD_decompositions(spin = -1)

        self.current_lhs_SVD_up = self.SVD(self.la.eye(self.Bdim, dtype=np.complex128))
        self.current_lhs_SVD_down = self.SVD(self.la.eye(self.Bdim, dtype=np.complex128))
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

        U, D, V = self.current_lhs_SVD_up
        u, d, v = self.SVD(np.dot(np.dot(lhs_change_up, U), np.diag(D)), mode='left_to_right')
        self.current_lhs_SVD_up = u, d, v.dot(V)
        #self.current_lhs_SVD_up = self._product_svds(lhs_change_up, self.current_lhs_SVD_up)

        U, D, V = self.current_lhs_SVD_down
        u, d, v = self.SVD(np.dot(np.dot(lhs_change_down, U), np.diag(D)), mode='left_to_right')
        self.current_lhs_SVD_down = u, d, v.dot(V)
        # self.current_lhs_SVD_down = self._product_svds(lhs_change_down, self.current_lhs_SVD_down)

        del self.partial_SVD_decompositions_up[-1]
        del self.partial_SVD_decompositions_down[-1]
        self.current_time_slice = tmax
        return
    
    def _get_partial_SVD_decompositions(self, spin):  # redo in the inverse order like in paper?
        M = self.la.eye(self.Bdim, dtype=np.complex128)
        current_U = self.la.eye(self.Bdim, dtype=np.complex128)
        current_V = self.la.eye(self.Bdim, dtype=np.complex128)
        current_D = self.la.ones(self.Bdim, dtype=np.complex128)
        buff = self.la.eye(self.Bdim, dtype=np.complex128)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(reversed(slices)):
            #print('partial', slice_idx)
            B = self.B_l(spin, slice_idx)
            buff = buff.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1 or nr == self.config.Nt - 1:
                # u, current_D, current_V = self.SVD(np.dot(buff, current_U).dot(self.la.diag(current_D)))
                u, current_D, current_V = self.SVD(self.la.diag(current_D).dot(np.dot(current_V, buff)))
                #assert np.allclose(np.linalg.inv(u), u.conj().T)
                #assert np.allclose(np.linalg.inv(v), v.conj().T)
                #assert self.la.linalg.norm(u.dot(self.la.diag(s)).dot(v) - M) / self.la.linalg.norm(M) < 1e-13  # FIXME
                current_U = current_U.dot(u)
                #current_V = v.dot(current_V)

                if spin == +1:
                    self.partial_SVD_decompositions_up.append((current_U * 1., current_D * 1., current_V * 1.))
                    '''
                    current_tau = (nr + 1) * self.config.dt
                    energies, states = np.linalg.eigh(self.K_matrix_plus)
                    states = states.T.conj()
                    assert np.allclose(self.K_matrix_plus, self.K_matrix_plus.conj().T)
                    assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), self.K_matrix_plus)
                    correct_string_d = self.SVD(np.einsum('i,ij,ik->jk', np.exp(current_tau * energies), states.conj(), states))[1]
                    correct_string = np.einsum('i,ij,ik->jk', np.exp(current_tau * energies), states.conj(), states)
                    string = np.dot(current_U, self.la.diag(current_D)).dot(current_V)
                    print((current_D - correct_string_d) / correct_string_d, nr, 'eigenvalues on step of obtaining partial svd decompositions')
                    #print(correct_string)
                    print(self.la.linalg.norm(string - correct_string) / self.la.linalg.norm(correct_string), 'string discrepancy in NORM', nr, flush=True)
                    '''
                else:
                    self.partial_SVD_decompositions_down.append((current_U * 1., current_D * 1., current_V * 1.))


                buff = self.la.eye(self.Bdim, dtype=np.complex128)
                #M = self.la.diag(s).dot(v)
        return

    def _get_left_partial_SVD_decompositions(self, spin):
        decompositions = []
        current_V = self.la.eye(self.Bdim, dtype=np.complex128)
        current_U = self.la.eye(self.Bdim, dtype=np.complex128)
        current_D = self.la.ones(self.Bdim, dtype=np.complex128)
        buff = self.la.eye(self.Bdim, dtype=np.complex128)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(slices):
            B = self.B_l(spin, slice_idx)
            buff = B.dot(buff)
            if nr % self.config.s_refresh == self.config.s_refresh - 1:
                current_U, current_D, v = self.SVD(np.dot(buff, current_U).dot(self.la.diag(current_D)), mode='left_to_right')
                
                current_V = v.dot(current_V)
                decompositions.append((current_U, current_D, current_V))
                buff = self.la.eye(self.Bdim, dtype=np.complex128)
        return decompositions

    def _get_right_partial_SVD_decompositions(self, spin):
        decompositions = []
        current_V = self.la.eye(self.Bdim, dtype=np.complex128)
        current_U = self.la.eye(self.Bdim, dtype=np.complex128)
        current_D = self.la.ones(self.Bdim, dtype=np.complex128)
        buff = self.la.eye(self.Bdim, dtype=np.complex128)

        slices = list(range(0, self.config.Nt))
        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            buff = buff.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1:
                u, current_D, current_V = self.SVD(self.la.diag(current_D).dot(np.dot(current_V, buff)), mode='right_to_left')

                current_U = current_U.dot(u)
                decompositions.append((current_U, current_D, current_V))
                buff = self.la.eye(self.Bdim, dtype=np.complex128)


        return decompositions

    def _get_partial_SVD_decomposition_range(self, spin, tmin, tmax): # FIXME later (only nonequal)
        M = self.la.eye(self.Bdim, dtype=np.complex128)
        
        for time_slice in range(tmin, tmax):
            M = self.B_l(spin, time_slice, inverse = False).dot(M)
        return M

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
        addstring = '_dump' if self.n_times_saved % 2 == 1 else ''
        self.n_times_saved += 1
        return np.save(self.conf_path + addstring, self.configuration)

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

    def make_symmetric_displacement(self, M, valley):
        if valley > 0:
            return self.K_plus_half.dot(M).dot(self.K_plus_half_inverse)
        return self.K_minus_half.dot(M).dot(self.K_minus_half_inverse)

    def get_equal_time_GF(self):
        # phase = np.exp(1.0j * np.imag(self.get_current_gauge_factor_log() / 2))
        self.G_up_sum += self.make_symmetric_displacement(self.current_G_function_up, valley= +1)# / phase
        self.G_down_sum += self.make_symmetric_displacement(self.current_G_function_down, valley = -1)# / phase
        self.n_gf_measures += 1
        return self.make_symmetric_displacement(self.current_G_function_up, valley = +1), \
               self.make_symmetric_displacement(self.current_G_function_down, valley = -1)

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

        self.K_half = cp.asnumpy(self.K_half)
        self.K_half_inverse = cp.asnumpy(self.K_half_inverse)

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
        self.K_half = cp.array(self.K_half)
        self.K_half_inverse = cp.array(self.K_half_inverse)

        self.la = cp
        self.configuration = cp.asarray(self.configuration)
        return self


    def wrap_up(self, time_slice):
        B_wrap_up = self.B_l(+1, time_slice, inverse = False)
        B_wrap_up_inverse = self.B_l(+1, time_slice, inverse = True)
        B_wrap_down = self.B_l(-1, time_slice, inverse = False)
        B_wrap_down_inverse = self.B_l(-1, time_slice, inverse = True)


        # assert np.allclose(np.linalg.inv(B_wrap_up), B_wrap_up_inverse)
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
                current_U, s, v = self.SVD(self.la.eye(self.Bdim, dtype=np.complex128))
            chain = self.la.eye(self.Bdim, dtype=np.complex128)

            for i in reversed(range(tmin, tmax)):
                chain = chain.dot(self.B_l(spin, i))

            u, s, v = self.SVD(self.la.diag(s).dot(np.dot(v, chain)), mode='right_to_left')
            return current_U.dot(u), s, v


        index_decomp = (tmax - tmin) // self.config.s_refresh
        tmin += index_decomp * self.config.s_refresh
        if index_decomp > 0:
            u, s, current_V = self.left_decompositions[index_decomp - 1]
        else:
            u, s, current_V = self.SVD(self.la.eye(self.Bdim // 2, dtype=np.complex128))

        chain = self.la.eye(self.Bdim, dtype=np.complex128)

        for i in range(tmin, tmax):
            chain = self.B_l(spin, i).dot(chain)

        u, s, v = self.SVD(np.dot(chain, u).dot(self.la.diag(s)), mode='left_to_right')
        return u, s, v.dot(current_V)

    def get_nonequal_time_GFs(self, spin, GF_0):
        current_GF = 1. * GF_0.copy()
        self.left_decompositions = self._get_left_partial_SVD_decompositions(spin)
        self.right_decompositions = self._get_right_partial_SVD_decompositions(spin)
        GFs = [1. * self.make_symmetric_displacement(self.to_numpy(current_GF), valley = spin)]
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
                m = self.la.diag(s1_max ** -1).dot(u1.T.conj()).dot(v2.T.conj()).dot(self.la.diag(s2_max ** -1)) + \
                    self.la.diag(s1_min).dot(v1).dot(u2).dot(self.la.diag(s2_min))
                current_GF = (v2.T.conj()).dot(self.la.diag(s2_max ** -1)).dot(self.la.linalg.inv(m)).dot(self.la.diag(s1_min)).dot(v1)

            GFs.append(self.make_symmetric_displacement(1.0 * self.to_numpy(current_GF), valley = spin))
        return np.array(GFs)

    ####### DEBUG ######
    def get_G_no_optimisation(self, spin, time_slice, return_udv = False):
        current_V = self.la.eye(self.Bdim, dtype=np.complex128)
        current_D = self.la.ones(self.Bdim, dtype=np.complex128)
        current_U = self.la.eye(self.Bdim, dtype=np.complex128)
        buff = self.la.eye(self.Bdim, dtype=np.complex128)

        slices = list(range(time_slice + 1, self.config.Nt)) + list(range(0, time_slice + 1))

        for nr, slice_idx in enumerate(reversed(slices)):
            B = self.B_l(spin, slice_idx)
            buff = buff.dot(B)
            if nr % self.config.s_refresh == self.config.s_refresh - 1 or nr == self.config.Nt - 1:
                u, current_D, current_V = self.SVD(self.la.diag(current_D).dot(np.dot(current_V, buff)))

                #print(self.la.sum(self.la.abs(u.dot(self.la.diag(s)).dot(v) - M)) / self.la.sum(self.la.abs(M)), 'discrepancy of SVD')
                # assert self.la.linalg.norm(u.dot(self.la.diag(s)).dot(v) - M) / self.la.linalg.norm(M) < 1e-13
                current_U = current_U.dot(u)
                buff = self.la.eye(self.Bdim, dtype=np.complex128)

        v = current_V
        s = current_D

        '''
        if spin == 1.0:
        
            current_tau = self.config.Nt * self.config.dt
            energies, states = np.linalg.eigh(self.K_matrix_plus)
            states = states.T.conj()
            assert np.allclose(self.K_matrix_plus, self.K_matrix_plus.conj().T)
            assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), self.K_matrix_plus)
            correct_string_B = np.einsum('i,ij,ik->jk', np.exp(current_tau * energies), states.conj(), states)
            ub, db, vb = self.SVD(correct_string_B)
            #string_B_svd = ub.dot(self.la.diag(db)).dot(vb)
            #db_after = self.SVD(string_B_svd)[1]
            #print((db_after - db) / db, '!!!')

            print(self.la.linalg.norm(current_U.dot(self.la.diag(current_D)).dot(current_V) - correct_string_B) / self.la.linalg.norm(correct_string_B), 'GF_noopt discrepancy before inversion', nr, flush=True)
            sing_true = self.SVD(correct_string_B)[1]
            #print((current_D - sing_true) / sing_true)
        '''

        m = current_U.conj().T.dot(v.conj().T) + self.la.diag(s)
        # print(s, 'sing values nonoptimized')
        um, sm, vm = self.SVD(m)
        #print(self.la.linalg.norm(um.dot(self.la.diag(sm)).dot(vm) - m) / self.la.linalg.norm(m), 'svd of M discrepancy')
        #assert np.allclose(((vm.dot(v)).conj().T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).conj().T) ,\
        #                   np.linalg.inv(self.la.eye(self.config.total_dof // 2, dtype=np.complex128) + current_U.dot(self.la.diag(s)).dot(v)))

        if return_udv:
            return (vm.dot(v)).conj().T, self.la.diag(sm ** -1), (current_U.dot(um)).conj().T
            #return (vm.dot(v)).conj().T, self.la.diag(sm ** -1), (current_U.dot(current_U)).conj()
        '''
        #res = ((vm.dot(v)).conj().T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).conj().T)
        sm_max = current_D.copy(); sm_max[sm_max < 1.] = 1.
        sm_min = current_D.copy(); sm_min[sm_min > 1.] = 1.

        print(np.max(current_D), np.min(current_D), 'range of singular values')

        M = current_V.conj().T.dot(self.la.diag(sm_max ** -1)) + current_U.dot(self.la.diag(sm_min))
        um, sm, vm = self.SVD(M)
        #res = current_V.conj().T.dot(self.la.diag(sm_max ** -1)).dot(vm.conj().T).dot(self.la.diag(sm ** -1)).dot(um.conj().T)

        A = current_V.conj().T.dot(self.la.diag(sm_max ** -1)) + current_U.dot(self.la.diag(sm_min))
        assert np.allclose(np.eye(self.Bdim), A.dot(self.la.linalg.inv(A)))
        res = current_V.conj().T.dot(self.la.diag(sm_max ** -1)).dot(self.la.linalg.inv(current_V.conj().T.dot(self.la.diag(sm_max ** -1)) + current_U.dot(self.la.diag(sm_min))))
        # res = self.la.linalg.inv(np.eye(self.Bdim) + current_U.dot(self.la.diag(current_D)).dot(current_V))
        '''

        res = ((vm.dot(v)).conj().T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).conj().T)
        '''
        if spin == 1.0:
            current_tau = self.config.Nt * self.config.dt
            energies, states = np.linalg.eigh(self.K_matrix_plus)
            states = states.T.conj()
            assert np.allclose(self.K_matrix_plus, self.K_matrix_plus.conj().T)
            assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), self.K_matrix_plus)
            correct_string = np.einsum('i,ij,ik->jk', 1. / (1. + np.exp(current_tau * energies)), states.conj(), states)
            # inv_B = self.la.linalg.inv(self.la.eye(self.Bdim) + correct_string_B)
            print(self.la.linalg.norm(res - correct_string) / self.la.linalg.norm(correct_string), 'GF_noopt final discrepancy', flush=True)
            # print(self.la.linalg.norm(inv_B - correct_string) / self.la.linalg.norm(correct_string), 'correct strings discrepancy', nr, flush=True)


            d_res = self.SVD(res)[1]
            d_correct = self.SVD(correct_string)[1]
            idx = np.where(np.abs((d_res - d_correct) / d_res) > 1e-2)[0][0]

            print((d_res - d_correct) / d_correct, 'discrepancy noopt', d_correct[idx])
        '''
        
        return res, self.la.sum(self.la.log(sm ** -1)), np.linalg.slogdet(res)[0] #res / np.abs(res)
        #return ((v.dot(vm)).conj()).dot(self.la.diag(sm ** -1)).dot((um.dot(current_U)).conj()), self.la.sum(self.la.log(sm ** -1)), \
        #        np.sign(np.linalg.det(((v.dot(vm)).conj())) * np.linalg.det((um.dot(current_U)).conj()))

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
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        super().__init__(config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse)
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
            loaded = False
            if os.path.isfile(self.conf_path):
                try:
                    self.configuration = np.load(self.conf_path)
                    print('Starting from a presaved field configuration', flush=True)
                    loaded = True
                except Exception:
                    print('Failed during loading of configuration from default location: try from dump')

                    try:
                        self.configuration = np.load(self.conf_path + '_dump')
                        print('Starting from a presaved field configuration in dump', flush=True)
                        loaded = True
                    except Exception:
                        print('Failed during loading of configuration from dump location: initialize from scratch')

            if not loaded:
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
        #print('inverse =', inverse)
        #assert np.allclose(np.linalg.inv(self.K), self.K_inverse)
        #print(self.V_up[l, ...], self.Vinv_up[l, ...])
        #assert np.allclose(np.linalg.inv(self.V_up[l, ...]), self.Vinv_up[l, ...])

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
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
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

        super().__init__(config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse)
        self.local_conf_combinations = \
                                    [
                                        (-2, -1, -1), (-1, -1, -1), (1, -1, -1), (2, -1, -1), \
                                        (-2, -1, 1),  (-1, -1, 1),  (1, -1, 1),  (2, -1, 1), \
                                        (-2, 1, -1),  (-1, 1, -1),  (1, 1, -1),  (2, 1, -1), \
                                        (-2, 1, 1),   (-1, 1, 1),   (1, 1, 1),   (2, 1, 1)
                                    ]

        self.rnd = np.random.choice(np.array([-2, -1, 1, 2]), size=100000)
        self.rnd_idx = 0
        return

    def get_gauge_factor_move(self, sp_index, time_slice, local_conf_old, local_conf):
        return self.gauge[local_conf[0]] / self.gauge[local_conf_old[0]]

    def get_current_gauge_factor_log(self):
        cf = self.configuration[..., 0].flatten()
        factor_logs = np.zeros(len(cf), dtype=np.complex128)
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
            loaded = False
            if os.path.isfile(self.conf_path + '.npy'):
                try:
                    self.configuration = np.load(self.conf_path + '.npy')
                    print('Starting from a presaved field configuration', flush=True)
                    loaded = True
                except Exception:
                    print('Failed during loading of configuration from default location: try from dump')

                    try:
                        self.configuration = np.load(self.conf_path + '_dump.npy')
                        print('Starting from a presaved field configuration in dump', flush=True)
                        loaded = True
                    except Exception:
                        print('Failed during loading of configuration from dump location: initialize from scratch')
            if not loaded:
                print('Random initial configuration')
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



class AuxiliaryFieldInterorbitalAccurateImag(AuxiliaryFieldInterorbitalAccurate):
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        super().__init__(config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse)
        
        self.gauge = {
            -2 : (1. - np.sqrt(6) / 3 ) * np.exp(-2.0j * self.config.nu_V * self.eta[-2]),
            +2 : (1. - np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_V * self.eta[+2]),
            -1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_V * self.eta[-1]),
            +1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_V * self.eta[+1])
        }

        self.gauge_log = {
            -2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.nu_V * self.eta[-2],
            +2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.nu_V * self.eta[+2],
            -1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.nu_V * self.eta[-1],
            +1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.nu_V * self.eta[+1]
        }

        self.local_conf_combinations = [[-2], [-1], [1], [2]]
        return

    def _V_from_configuration(self, s, sign, spin):
        return np.diag(np.exp(1.0j * self.config.nu_V * self.eta[s[0]] * np.ones(2)))

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            self.configuration = np.random.randint(0, 1, size = (self.config.Nt, self.config.total_dof // 2 // 2, 1)) * 2. - 1.0
        elif self.config.start_type == 'hot':
            self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 1)) * 2. - 1.0  # standard 2-valued aux Hubbard field
            self.configuration[..., 0] = np.random.choice(np.array([-2, -1, 1, 2]), \
                                                          size = (self.config.Nt, self.config.total_dof // 2 // 2))  # 4-valued F.F. Assaad field
        else:
            loaded = False
            if os.path.isfile(self.conf_path + '.npy'):
                try:
                    self.configuration = np.load(self.conf_path + '.npy')
                    print('Starting from a presaved field configuration', flush=True)
                    loaded = True
                except Exception:
                    print('Failed during loading of configuration from default location: try from dump')

                    try:
                        self.configuration = np.load(self.conf_path + '_dump.npy')
                        print('Starting from a presaved field configuration in dump', flush=True)
                        loaded = True
                    except Exception:
                        print('Failed during loading of configuration from dump location: initialize from scratch')
            if not loaded:
                print('Random initial configuration')
                self.configuration = np.random.randint(0, 2, size = (self.config.Nt, self.config.total_dof // 2 // 2, 1)) * 2. - 1.0
                self.configuration[..., 0] = np.random.choice(np.array([-2, -1, 1, 2]), \
                                                              size = (self.config.Nt, self.config.total_dof // 2 // 2))

        NtVolVol_shape = (self.config.Nt, self.config.total_dof // 2, self.config.total_dof // 2)
        self.V_up = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); self.Vinv_up = np.zeros(shape = NtVolVol_shape, dtype=np.complex128)
        self.V_down = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); self.Vinv_down = np.zeros(shape = NtVolVol_shape, dtype=np.complex128)

        for time_slice in range(self.config.Nt):
            for sp_index in range(self.config.total_dof // 2 // 2):
                sx = sp_index * 2
                sy = sp_index * 2 + 1
                self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    _V_from_configuration_accurate_imag(self.configuration[time_slice, sp_index, :], +1.0, +1.0, self.config.nu_V)
                self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] = \
                    _V_from_configuration_accurate_imag(self.configuration[time_slice, sp_index, :], -1.0, +1.0, self.config.nu_V)

                self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    _V_from_configuration_accurate_imag(self.configuration[time_slice, sp_index, :], +1.0, -1.0, self.config.nu_V)
                self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
                    _V_from_configuration_accurate_imag(self.configuration[time_slice, sp_index, :], -1.0, -1.0, self.config.nu_V)
        return

    def update_field(self, sp_index, time_slice, new_conf):
        # print('update field right new way', flush=True)
        self.configuration[time_slice, sp_index, ...] = np.array(new_conf)
        sx = sp_index * 2
        sy = sp_index * 2 + 1
        self.V_up[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate_imag(new_conf, +1.0, +1.0, self.config.nu_V)
        self.Vinv_up[time_slice, sx : sy + 1, sx : sy + 1] =\
            _V_from_configuration_accurate_imag(new_conf, -1.0, +1.0, self.config.nu_V)
        self.V_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate_imag(new_conf, +1.0, -1.0, self.config.nu_V)
        self.Vinv_down[time_slice, sx : sy + 1, sx : sy + 1] = \
            _V_from_configuration_accurate_imag(new_conf, -1.0, -1.0, self.config.nu_V)

        return

    def compute_deltas(self, sp_index, time_slice, local_conf, local_conf_proposed):
        self.Delta_up = _get_delta_interorbital_accurate_imag(local_conf, local_conf_proposed, +1, self.config.nu_V)
        self.Delta_down = _get_delta_interorbital_accurate_imag(local_conf, local_conf_proposed, -1, self.config.nu_V)
        return

    def get_delta(self, spin, sp_index, time_slice, local_conf_proposed):  # sign change proposal is made at (time_slice, sp_index, o_index)
        return _get_delta_interorbital_accurate_imag(tuple(self.configuration[time_slice, sp_index, :]), \
                                                local_conf_proposed, spin, self.config.nu_V)


class AuxiliaryFieldInterorbitalAccurateImagNN(AuxiliaryFieldInterorbitalAccurate):
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        self.conf_path_eta = os.path.join(local_workdir, 'last_conf_eta')
        self.conf_path_xi = os.path.join(local_workdir, 'last_conf_xi')

        self.K_plus = K[::2, :]
        self.K_plus = self.K_plus[:, ::2]
        self.K_minus = K[1::2, :]
        self.K_minus = self.K_minus[:, 1::2]

        self.K_plus_inverse = K_inverse[::2, :]
        self.K_plus_inverse = self.K_plus_inverse[:, ::2]
        self.K_minus_inverse = K_inverse[1::2, :]
        self.K_minus_inverse = self.K_minus_inverse[:, 1::2]

        self.K_plus_half = K_half[::2, :]
        self.K_plus_half = self.K_plus_half[:, ::2]
        self.K_minus_half = K_half[1::2, :]
        self.K_minus_half = self.K_minus_half[:, 1::2]

        self.K_plus_half_inverse = K_half_inverse[::2, :]
        self.K_plus_half_inverse = self.K_plus_half_inverse[:, ::2]

        self.K_minus_half_inverse = K_half_inverse[1::2, :]
        self.K_minus_half_inverse = self.K_minus_half_inverse[:, 1::2]

        self.K_matrix_plus = K_matrix[::2, :]
        self.K_matrix_plus = self.K_matrix_plus[:, ::2]
        self.K_matrix_minus = K_matrix[1::2, :]
        self.K_matrix_minus = self.K_matrix_minus[:, 1::2]

        assert np.allclose(np.linalg.inv(self.K_minus), self.K_minus_inverse)
        assert np.allclose(np.linalg.inv(self.K_plus), self.K_plus_inverse)

        K_oneband = K_matrix[np.arange(0, K_matrix.shape[0], 2), :];
        K_oneband = K_oneband[:, np.arange(0, K_matrix.shape[0], 2)];

        self.connectivity = (K_oneband == K_oneband.real.max()).astype(np.float64)
        assert np.allclose(self.connectivity, self.connectivity.T)
        self.n_bonds = int(np.sum(self.connectivity) / 2.)

        self.bonds = []
        self.bonds_by_site = []

        for i in range(self.connectivity.shape[0]):
            adjacent_sites = np.where(self.connectivity[i, :] == 1.0)[0]
            for j in adjacent_sites:
                if j < i:
                    continue
                self.bonds.append((i, j))

        assert self.n_bonds == len(self.bonds)

        for i in range(self.connectivity.shape[0]):
            self.bonds_by_site.append([])
            adjacent_sites = np.where(self.connectivity[i, :] == 1.0)[0]
            for j in adjacent_sites:
                if (i, j) in self.bonds:
                    self.bonds_by_site[i].append(self.bonds.index((i, j)))
                else:
                    self.bonds_by_site[i].append(self.bonds.index((j, i)))
            assert len(self.bonds_by_site[i]) == 3

        assert self.n_bonds == K_oneband.shape[0] // 2 * 3
     
        print(self.bonds_by_site)
        print(self.bonds)
        self.Bdim = config.total_dof // 2 // 2


        super().__init__(config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse)

        self.gauge_site = {
            -2 : (1. - np.sqrt(6) / 3 ) * np.exp(-2.0j * self.config.nu_U * self.eta[-2]),
            +2 : (1. - np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_U * self.eta[+2]),
            -1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_U * self.eta[-1]),
            +1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.nu_U * self.eta[+1])
        }

        self.gauge_bond = {
            -2 : (1. - np.sqrt(6) / 3 ) * np.exp(-4.0j * self.config.nu_V * self.eta[-2]),
            +2 : (1. - np.sqrt(6) / 3.) * np.exp(-4.0j * self.config.nu_V * self.eta[+2]),
            -1 : (1. + np.sqrt(6) / 3.) * np.exp(-4.0j * self.config.nu_V * self.eta[-1]),
            +1 : (1. + np.sqrt(6) / 3.) * np.exp(-4.0j * self.config.nu_V * self.eta[+1])
        }


        self.gauge_site_log = {
            -2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.nu_U * self.eta[-2],
            +2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.nu_U * self.eta[+2],
            -1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.nu_U * self.eta[-1],
            +1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.nu_U * self.eta[+1]
        }

        self.gauge_bond_log = {
            -2 : np.log(1. - np.sqrt(6) / 3) - 4.0j * self.config.nu_V * self.eta[-2],
            +2 : np.log(1. - np.sqrt(6) / 3) - 4.0j * self.config.nu_V * self.eta[+2],
            -1 : np.log(1. + np.sqrt(6) / 3) - 4.0j * self.config.nu_V * self.eta[-1],
            +1 : np.log(1. + np.sqrt(6) / 3) - 4.0j * self.config.nu_V * self.eta[+1]
        }

        self.local_conf_combinations = [[-2], [-1], [1], [2]]

        self.G_up_sum = np.zeros((self.config.total_dof // 2 // 2, self.config.total_dof // 2 // 2), dtype=np.complex128)
        self.G_down_sum = np.zeros((self.config.total_dof // 2 // 2, self.config.total_dof // 2 // 2), dtype=np.complex128)
        self.n_gf_measures = 0

        return

    def get_gauge_factor_move_eta(self, sp_index, time_slice, local_conf_old, local_conf):
        return self.gauge_site[local_conf[0]] / self.gauge_site[local_conf_old[0]]

    def get_gauge_factor_move_xi(self, bond_index, time_slice, local_conf_old, local_conf):
        return self.gauge_bond[local_conf] / self.gauge_bond[local_conf_old]

    def get_current_gauge_factor_log(self):
        cf = self.eta_sites[..., 0].flatten()
        factor_logs_eta = np.zeros(len(cf), dtype=np.complex128)
        factor_logs_eta[cf == -2] = self.gauge_site_log[-2]
        factor_logs_eta[cf == 2] = self.gauge_site_log[2]
        factor_logs_eta[cf == 1] = self.gauge_site_log[1]
        factor_logs_eta[cf == -1] = self.gauge_site_log[-1]

        cf = self.xi_bonds.flatten()
        factor_logs_xi = np.zeros(len(cf), dtype=np.complex128)
        factor_logs_xi[cf == -2] = self.gauge_bond_log[-2]
        factor_logs_xi[cf == 2] = self.gauge_bond_log[2]
        factor_logs_xi[cf == 1] = self.gauge_bond_log[1]
        factor_logs_xi[cf == -1] = self.gauge_bond_log[-1]

        return np.sum(factor_logs_eta) + np.sum(factor_logs_xi)

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            exit(-1)
        elif self.config.start_type == 'hot':
            self.eta_sites = np.random.choice(np.array([-2, -1, 1, 2]), \
                                              size = (self.config.Nt, self.config.total_dof // 2 // 2, 1))  # 4-valued F.F. Assaad field for on-site
            self.xi_bonds = np.random.choice(np.array([-2, -1, 1, 2]), \
                                              size = (self.config.Nt, self.n_bonds))  # 4-valued F.F. Assaad field for bonds
            ## Note: last 3 components are only defined on the A-sublattice

        else:
            loaded = False
            if os.path.isfile(self.conf_path_eta + '.npy') and os.path.isfile(self.conf_path_xi + '.npy'):
                try:
                    self.eta_sites = np.load(self.conf_path_eta + '.npy')
                    self.xi_bonds = np.load(self.conf_path_xi + '.npy')
                    print('Starting from a presaved field configuration', flush=True)
                    loaded = True
                except Exception:
                    print('Failed during loading of configuration from default location: try from dump')

                    try:
                        self.eta_sites = np.load(self.conf_path_eta + '_dump.npy')
                        self.xi_bonds = np.load(self.conf_path_xi + '_dump.npy')

                        print('Starting from a presaved field configuration in dump', flush=True)
                        loaded = True
                    except Exception:
                        print('Failed during loading of configuration from dump location: initialize from scratch')
            if not loaded:
                print('Random initial configuration')
                self.eta_sites = np.random.choice(np.array([-2, -1, 1, 2]), \
                                              size = (self.config.Nt, self.config.total_dof // 2 // 2, 1))  # 4-valued F.F. Assaad field for on-site
                self.xi_bonds = np.random.choice(np.array([-2, -1, 1, 2]), \
                                              size = (self.config.Nt, self.n_bonds))  # 4-valued F.F. Assaad field for bonds


        NtVolVol_shape = (self.config.Nt, self.config.total_dof // 2 // 2, self.config.total_dof // 2 // 2)
        self.V = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 

        for time_slice in range(self.config.Nt):
            for sp_index in range(self.config.total_dof // 2 // 2):
                bonds = self.bonds_by_site[sp_index]
                xi_variables = np.array([self.xi_bonds[time_slice, b] for b in bonds]) 

                self.V[time_slice, sp_index, sp_index] = \
                    _V_from_configuration_onesite_accurate_imag(self.eta_sites[time_slice, sp_index], xi_variables, +1.0, +1.0, self.config.nu_U, self.config.nu_V)
        return
    
    def save_configuration(self):
        addstring = '_dump' if self.n_times_saved % 2 == 1 else ''
        self.n_times_saved += 1
        np.save(self.conf_path_xi + addstring, self.xi_bonds)
        return np.save(self.conf_path_eta + addstring, self.eta_sites)


    def update_eta_site_field(self, sp_index, time_slice, new_conf): ## TODO: this update can be made faster
        '''
            we update site-variable, which affects only 2 d.o.f. and use `_V_from_configuration_accurate_imag` standard function
        '''
        self.eta_sites[time_slice, sp_index, ...] = np.array(new_conf)
   
        bonds = self.bonds_by_site[sp_index]
        xi_variables = np.array([self.xi_bonds[time_slice, b] for b in bonds])

        self.V[time_slice, sp_index, sp_index] = \
            _V_from_configuration_onesite_accurate_imag(self.eta_sites[time_slice, sp_index], xi_variables, +1.0, +1.0, self.config.nu_U, self.config.nu_V)

        return


    def update_xi_bond_field(self, bond_index, time_slice, new_conf):
        '''
            we update bond-variable, which affects 2 sites and 4 d.o.f., thus use `_V_from_configuration_onesite_accurate_imag`
        '''
        self.xi_bonds[time_slice, bond_index] = new_conf

        sp_index1, sp_index2 = self.bonds[bond_index]

        bonds1 = self.bonds_by_site[sp_index1]
        xi_variables1 = np.array([self.xi_bonds[time_slice, b] for b in bonds1])  # all xi variables entering the site 1, including the one that has been changed (this is suboptimal)

        self.V[time_slice, sp_index1, sp_index1] = \
            _V_from_configuration_onesite_accurate_imag(self.eta_sites[time_slice, sp_index1], \
             xi_variables1, +1.0, +1.0, self.config.nu_U, self.config.nu_V)

        bonds2 = self.bonds_by_site[sp_index2]
        xi_variables2 = np.array([self.xi_bonds[time_slice, b] for b in bonds2])  # all xi variables entering the site 1, including the one that has been changed (this is suboptimal)

        self.V[time_slice, sp_index2, sp_index2] = \
            _V_from_configuration_onesite_accurate_imag(self.eta_sites[time_slice, sp_index2], \
             xi_variables2, +1.0, +1.0, self.config.nu_U, self.config.nu_V)

        return


    def compute_deltas_eta(self, sp_index, time_slice, local_conf, local_conf_proposed):
        '''
            deltas for site-update of the eta-field (use the standard `_get_delta_interorbital_accurate_imag`)
        '''
        self.Delta = _get_delta_interorbital_accurate_imag(local_conf, local_conf_proposed, +1, self.config.nu_U)  # delta is the same for both valleys [since V is also the same]
        return

    def compute_deltas_xi(self, sp_index, time_slice, local_conf, local_conf_proposed):
        '''
            deltas for bond-update of the xi-field (use the standard `_get_delta_interorbital_twosite_accurate_imag`)
        '''
        self.Delta = _get_delta_interorbital_twosite_accurate_imag(local_conf, local_conf_proposed, +1, self.config.nu_V)  # here as well the same Delta
        return

    def get_current_eta(self, sp_index, time_slice):
        return self.eta_sites[time_slice, sp_index, ...]

    def get_current_xi(self, bond_index, time_slice):
        return self.xi_bonds[time_slice, bond_index]


    def update_G_seq_eta(self, sp_index):
        self.current_G_function_up = _update_G_seq_intra(self.current_G_function_up, self.Delta, sp_index, self.config.total_dof // 2)
        self.current_G_function_down = _update_G_seq_intra(self.current_G_function_down, self.Delta, sp_index, self.config.total_dof // 2)
        return

    def update_G_seq_xi(self, bond_index):
        sp_index1, sp_index2 = self.bonds[bond_index]
        self.current_G_function_up = _update_G_seq_inter(self.current_G_function_up, self.Delta, sp_index1, sp_index2, self.config.total_dof // 2)
        self.current_G_function_down = _update_G_seq_inter(self.current_G_function_down, self.Delta, sp_index1, sp_index2, self.config.total_dof // 2)
        return

    def get_equal_time_GF(self):
        self.G_up_sum += self.make_symmetric_displacement(self.current_G_function_up, valley = +1)
        self.G_down_sum += self.make_symmetric_displacement(self.current_G_function_up, valley = -1)
        self.n_gf_measures += 1

        return self.make_symmetric_displacement(self.current_G_function_up, valley = +1), \
               self.make_symmetric_displacement(self.current_G_function_down, valley = -1)

    def wrap_up(self, time_slice):
        B_wrap_up = self.B_l(+1, time_slice, inverse = False)
        B_wrap_up_inverse = self.B_l(+1, time_slice, inverse = True)

        B_wrap_down = self.B_l(-1, time_slice, inverse = False)
        B_wrap_down_inverse = self.B_l(-1, time_slice, inverse = True)

        self.current_G_function_up = B_wrap_up.dot(self.current_G_function_up.dot(B_wrap_up_inverse))
        self.current_G_function_down = B_wrap_down.dot(self.current_G_function_down.dot(B_wrap_down_inverse))
        return

    def B_l(self, spin, l, inverse = False):  # spin = valley
        if not inverse:
            if spin > 0:
                return self.V[l, ...].dot(self.K_plus)
            return self.V[l, ...].dot(self.K_minus)

        if spin > 0:
            return self.K_plus_inverse.dot(np.conj(self.V[l, ...]))
        return self.K_minus_inverse.dot(np.conj(self.V[l, ...]))  # in case of imaginary HST, inversion = c.c.

@jit(nopython = True)
def _get_delta_interorbital_accurate(local_conf, local_conf_proposed, spin, \
                                     nu_U, nu_V):  # sign change proposal is made at (time_slice, sp_index, o_index)
    local_V_inv = _V_from_configuration_accurate(local_conf, -1.0, spin, nu_U, nu_V)  # already stored in self.V or self.Vinv
    local_V_proposed = _V_from_configuration_accurate(local_conf_proposed, 1.0, spin, nu_U, nu_V)
    return np.diag(np.array([local_V_proposed[0, 0] * local_V_inv[0, 0] - 1, local_V_proposed[1, 1] * local_V_inv[1, 1] - 1])) # local_V_proposed.dot(local_V_inv) - np.eye(2)


@jit(nopython = True)
def _get_delta_interorbital_accurate_imag(local_conf, local_conf_proposed, spin, nu_U):
    local_V_inv = _V_from_configuration_accurate_imag(local_conf, -1.0, spin, nu_U)  # already stored in self.V or self.Vinv
    local_V_proposed = _V_from_configuration_accurate_imag(local_conf_proposed, 1.0, spin, nu_U)
    return local_V_proposed * local_V_inv - 1.


@jit(nopython = True)
def _get_delta_interorbital_twosite_accurate_imag(local_conf, local_conf_proposed, spin, nu_V):
    local_V_inv = _V_from_configuration_twosite_accurate_imag(local_conf, -1.0, spin, nu_V)
    local_V_proposed = _V_from_configuration_twosite_accurate_imag(local_conf_proposed, 1.0, spin, nu_V)
    return local_V_proposed * local_V_inv - np.eye(2)

@jit(nopython=True)
def _V_from_configuration_accurate(s, sign, spin, nu_U, nu_V):
    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, \
           +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]

    if spin > 0:
        V = nu_V * eta[int(s[0]) + 2] * sign * np.array([1, -1]) + \
            nu_U * sign * np.array([s[2], s[1]])
    else:
        V = nu_V * eta[int(s[0]) + 2] * sign * np.array([1, -1]) + \
            nu_U * sign * np.array([-s[2], -s[1]])
    return np.diag(np.exp(V))

@jit(nopython=True)
def _V_from_configuration_accurate_imag(s, sign, spin, nu_V):
    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, \
           +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]

    return np.exp(1.0j * nu_V * eta[int(s[0]) + 2] * sign)

@jit(nopython=True)
def _V_from_configuration_onesite_accurate_imag(eta_site, xi_bond, sign, spin, nu_U, nu_V):  # used for initialization!
    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, \
           +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]
    return np.exp(1.0j * (nu_U * eta[int(eta_site[0]) + 2] + \
                                  nu_V * (eta[int(xi_bond[0]) + 2] + \
                                          eta[int(xi_bond[1]) + 2] + \
                                          eta[int(xi_bond[2]) + 2]) \
                                         ) * sign)  # bond-variable is the same for both sites


@jit(nopython=True)
def _V_from_configuration_twosite_accurate_imag(s, sign, spin, nu_V):  # !!! valid in this form ONLY for update (computation of Delta)
    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, \
           +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]
    return np.diag(np.exp(1.0j * nu_V * eta[int(s) + 2] * sign * np.ones(2)))  # bond-variable is the same for both sites


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
def get_det_ratio_inter(sp_index1, sp_index2, Delta, G):
    idxs = np.array([sp_index1, sp_index2], dtype=np.int64)
    G_slice = G[idxs, :]
    G_slice = G_slice[:, idxs]

    return np.linalg.det(np.eye(2, dtype=np.complex128) + np.dot(Delta, np.eye(2, dtype=np.complex128) - G_slice))

@jit(nopython=True)
def get_det_ratio_inter_bond(sp_index1, sp_index2, Delta, G):
    sx1 = sp_index1 * 2
    sy1 = sp_index1 * 2 + 1
    sx2 = sp_index2 * 2
    sy2 = sp_index2 * 2 + 1

    idxs = np.array([sx1, sy1, sx2, sy2], dtype=np.int64)

    G_cut = np.zeros((4, 4), dtype=np.complex128)
    for ii, i in enumerate(idxs):
        for jj, j in enumerate(idxs):
            G_cut[ii, jj] = G[i, j]

    return np.linalg.det(np.eye(4, dtype=np.complex128) + Delta.dot(np.eye(4, dtype=np.complex128) - G_cut))

@jit(nopython=True)
def get_det_ratio_intra(sp_index, Delta, G):
    return 1. + Delta * (1. - G[sp_index, sp_index])

'''
@jit(nopython=True)
def _update_G_seq_inter(G, Delta, sp_index1, sp_index2, total_dof):
    idxs = np.array([sp_index1, sp_index2], dtype=np.int64)
    G_sliced_right = G[:, idxs]
    G_sliced_left = G[idxs, :]

    update_matrix = np.zeros((2, total_dof // 2), dtype=np.complex128)  # keep only two nontrivial rows here
    update_matrix[:, idxs] = np.eye(2, dtype=np.complex128) + Delta
    update_matrix -= np.dot(Delta, np.ascontiguousarray(G_sliced_left))
    det = np.linalg.det(update_matrix[:, idxs])

    inverse_update_matrix = np.zeros((2, total_dof // 2), dtype=np.complex128)  # keep only two nontrivial rows here

    inverse_update_matrix[0, :] = -(update_matrix[0, :] * update_matrix[1, sp_index2] - \
                                    update_matrix[1, :] * update_matrix[0, sp_index2]) / det  # my vectorized det :))
    inverse_update_matrix[1, :] = (update_matrix[0, :] * update_matrix[1, sp_index1] - \
                                   update_matrix[1, :] * update_matrix[0, sp_index1]) / det

    inverse_update_matrix[0, sp_index2] = update_matrix[1, sp_index2] / det - 1
    inverse_update_matrix[1, sp_index1] = -update_matrix[1, sp_index1] / det
    inverse_update_matrix[0, sp_index2] = -update_matrix[0, sp_index2] / det
    inverse_update_matrix[1, sp_index1] = update_matrix[0, sp_index1] / det - 1

    G = G + np.dot(np.ascontiguousarray(G_sliced_right), inverse_update_matrix)
    return G
'''

@jit(nopython=True)
def _update_G_seq_inter(G, Delta, sp_index1, sp_index2, total_dof):
    U = np.zeros((total_dof // 2, 2), dtype=np.complex128)
    U[sp_index1, 0] = Delta[0, 0]
    U[sp_index2, 1] = Delta[1, 1]

    G_sliced_left = G[np.array([sp_index1, sp_index2], dtype=np.int64), :]
    V = G_sliced_left

    V[0, sp_index1] -= 1
    V[1, sp_index2] -= 1
    V = np.ascontiguousarray(V)
    U = np.ascontiguousarray(U)
    G = np.ascontiguousarray(G)

    GU = G.dot(U)
    Zinv = np.linalg.inv(np.eye(2, dtype=np.complex128) - V.dot(U)).dot(V)
    Zinv = np.ascontiguousarray(Zinv)

    #print(np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G), (np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G)) ** -1, 'det computed from update')
    return G + GU.dot(Zinv)

'''
@jit(nopython=True)	
def _update_G_seq_inter(G, Delta, sp_index, total_dof):
    sx = sp_index * 2
    sy = sp_index * 2 + 1
    G_sliced_right = G[:, sx : sy + 1]
    G_sliced_left = G[sx:sy + 1, :]

    update_matrix = np.zeros((2, total_dof // 2), dtype=np.complex128)  # keep only two nontrivial rows here
    update_matrix[:, sx:sy + 1] = np.eye(2, dtype=np.complex128) + Delta
    update_matrix -= np.dot(Delta, np.ascontiguousarray(G_sliced_left))
    det = np.linalg.det(update_matrix[:, sx:sy + 1])

    inverse_update_matrix = np.zeros((2, total_dof // 2), dtype=np.complex128)  # keep only two nontrivial rows here

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
'''


@jit(nopython=True)
def _update_G_seq_inter_twosite(G, Delta, sp_index1, sp_index2, total_dof):
    sx1 = sp_index1 * 2
    sy1 = sp_index1 * 2 + 1

    sx2 = sp_index2 * 2
    sy2 = sp_index2 * 2 + 1

    U = np.zeros((total_dof // 2, 4), dtype=np.complex128)
    U[sx1, 0] = Delta[0, 0]
    U[sy1, 1] = Delta[1, 1]
    U[sx2, 2] = Delta[2, 2]
    U[sy2, 3] = Delta[3, 3]

    G_sliced_left = G[np.array([sx1, sy1, sx2, sy2], dtype=np.int64), :]
    V = G_sliced_left

    V[0, sx1] -= 1
    V[1, sy1] -= 1
    V[2, sx2] -= 1
    V[3, sy2] -= 1
    V = np.ascontiguousarray(V)
    U = np.ascontiguousarray(U)
    G = np.ascontiguousarray(G)

    GU = G.dot(U)
    Zinv = np.linalg.inv(np.eye(4, dtype=np.complex128) - V.dot(U)).dot(V)
    Zinv = np.ascontiguousarray(Zinv)

    #print(np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G), (np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G)) ** -1, 'det computed from update')
    return G + GU.dot(Zinv)


@jit(nopython=True)
def _update_G_seq_intra(G, Delta, sp_index, total_dof):
    update_matrix = -G[sp_index, :]
    update_matrix[sp_index] = 1. + update_matrix[sp_index]
    update_matrix *= Delta
    #update_matrix2 = Delta * (np.eye(total_dof // 2) - G)[sp_index, :]
    #assert np.allclose(update_matrix, update_matrix2)

    update_matrix[sp_index] += 1.
    det_update_matrix = update_matrix[sp_index]
    update_matrix_inv = -update_matrix / det_update_matrix
    update_matrix_inv[sp_index] = 1. / det_update_matrix - 1.
    G = G + np.outer(np.ascontiguousarray(G[:, sp_index]), np.ascontiguousarray(update_matrix_inv))

    return G


'''
@jit(nopython=True)
def _update_G_seq_intra(G, Delta, sp_index, total_dof):
    U = np.zeros((total_dof // 2, 1), dtype=np.complex128)
    U[sp_index, 0] = Delta

    G_sliced_left = G[sp_index:sp_index+1, :]
    V = G_sliced_left

    V[0, sp_index] -= 1
    V = np.ascontiguousarray(V)
    U = np.ascontiguousarray(U)
    G = np.ascontiguousarray(G)

    GU = G.dot(U)
    Zinv = np.linalg.inv(np.eye(1, dtype=np.complex128) - V.dot(U)).dot(V)
    Zinv = np.ascontiguousarray(Zinv)

    #print(np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G), (np.linalg.det(G + GU.dot(Zinv)) / np.linalg.det(G)) ** -1, 'det computed from update')
    return G + GU.dot(Zinv)
'''
