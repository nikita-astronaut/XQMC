import numpy as np
import pickle
from time import time
import scipy
import models
from copy import deepcopy
from numba import jit
from numba.typed import List

import os
from scipy.sparse.csgraph import connected_components
#import torch


try:
    import cupy as cp
except ImportError:
    pass

class AuxiliaryFieldIntraorbital:
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        self.gpu_avail = config.gpu
        self.exponentiated = np.ones(config.Nt, dtype=bool)
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

    def SVD(self, matrix, mode='right_to_left'):
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

        def svd_recursive(M, threshold = 1e-3):
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

            bu, bd, bv = svd_recursive(b,
                          threshold=threshold)

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

    def get_nonequal_time_GFs_inverted(self, spin, GF_0):
        identity = np.eye(self.Bdim, dtype=np.complex128)
        current_GF = 1. * GF_0.copy() - identity
        self.left_decompositions = self._get_left_partial_SVD_decompositions(spin)
        self.right_decompositions = self._get_right_partial_SVD_decompositions(spin)
        GFs = [1. * self.make_symmetric_displacement(self.to_numpy(current_GF), valley = spin)]

        for tau in range(1, self.config.Nt):
            B = self.B_l(spin, tau - 1, inverse=True)

            if tau % self.config.s_refresh != 0:
                current_GF = current_GF @ B  # just wrap-up
            else:  # recompute GF from scratch
                u1, s1, v1 = self.compute_B_chain(spin, tau, 0)  # tau - 1 | ... | 0
                u2, s2, v2 = self.compute_B_chain(spin, self.config.Nt, tau)  # Nt - 1 | ... | tau
                s1_min = 1.0 * s1; s1_max = 1.0 * s1
                s1_min[s1_min > 1.] = 1.
                s1_max[s1_max < 1.] = 1.

                s2_min = 1.0 * s2; s2_max = 1.0 * s2
                s2_min[s2_min > 1.] = 1.
                s2_max[s2_max < 1.] = 1.
                m = self.la.diag(s2_max ** -1).dot(u2.T.conj()).dot(v1.T.conj()).dot(self.la.diag(s1_max ** -1)) + \
                    self.la.diag(s2_min).dot(v2).dot(u1).dot(self.la.diag(s1_min))
                current_GF = -(v1.T.conj()).dot(self.la.diag(s1_max ** -1)).dot(self.la.linalg.inv(m)).dot(self.la.diag(s2_min)).dot(v2)

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
                current_U = current_U.dot(u)
                buff = self.la.eye(self.Bdim, dtype=np.complex128)

        v = current_V
        s = current_D

        m = current_U.conj().T.dot(v.conj().T) + self.la.diag(s)
        um, sm, vm = self.SVD(m)

        if return_udv:
            return (vm.dot(v)).conj().T, self.la.diag(sm ** -1), (current_U.dot(um)).conj().T
        res = ((vm.dot(v)).conj().T).dot(self.la.diag(sm ** -1)).dot((current_U.dot(um)).conj().T)
        
        return res, self.la.sum(self.la.log(sm ** -1)), np.linalg.slogdet(res)[0] #res / np.abs(res)

    def get_G_tau_0_naive(self, spin):
        G0 = np.eye(self.Bdim, dtype=np.complex128)
        for time_slice in np.arange(self.config.Nt):
            B = self.B_l(spin, time_slice)
            G0 = B.dot(G0)

        G0 = np.linalg.inv(np.eye(self.Bdim, dtype=np.complex128) + G0)

        GFs = [self.make_symmetric_displacement(G0, valley = spin)]
        for t in range(0, self.config.Nt - 1):
            B = self.B_l(spin, t)
            G0 = B.dot(G0)
            GFs.append(self.make_symmetric_displacement(G0, valley = spin))

        return GFs

    def get_G_tau_tau_naive(self, spin):
        G0 = np.eye(self.Bdim, dtype=np.complex128)
        for time_slice in np.arange(self.config.Nt):
            B = self.B_l(spin, time_slice)
            G0 = B.dot(G0)

        G0 = np.linalg.inv(np.eye(self.Bdim, dtype=np.complex128) + G0)

        GFs = [self.make_symmetric_displacement(G0, valley = spin)]
        for t in range(0, self.config.Nt - 1):
            B = self.B_l(spin, t)
            Binv = self.B_l(spin, t, inverse=True)
            G0 = B.dot(G0).dot(Binv)
            GFs.append(self.make_symmetric_displacement(G0, valley = spin))

        return GFs

    def get_G_0_tau_naive(self, spin):
        G0 = np.eye(self.Bdim, dtype=np.complex128)
        for time_slice in np.arange(self.config.Nt):
            B = self.B_l(spin, time_slice)
            G0 = B.dot(G0)

        G0 = np.linalg.inv(np.eye(self.Bdim, dtype=np.complex128) + G0) - np.eye(self.Bdim, dtype=np.complex128)

        GFs = [self.make_symmetric_displacement(G0, valley = spin)]
        for t in range(0, self.config.Nt - 1):
            B = self.B_l(spin, t, inverse=True)
            G0 = G0.dot(B)
            GFs.append(self.make_symmetric_displacement(G0, valley = spin))

        return GFs



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

        '''
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
        '''
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

        self.local_conf_combinations = [[-2], [-1], [1], [2]]  # FIXME FIXME FIXME
        #self.local_conf_combinations = [[-1], [1]]

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
        assert self.exponentiated[l]

        if not inverse:
            if spin > 0:
                #return self.K_plus_half @ self.V_plus_exp[l, ...] @ self.K_plus_half
                return self.V_plus_exp[l, ...] @ self.K_plus
            #return self.K_minus_half @ self.V_minus_exp[l, ...] @ self.K_minus_half
            return self.V_minus_exp[l, ...] @ self.K_minus


        if spin > 0:
            #return self.K_plus_half_inverse @ self.V_plus_exp_inv[l, ...] @ self.K_plus_half_inverse
            return self.K_plus_inverse @ self.V_plus_exp_inv[l, ...]
        #return self.K_minus_half_inverse @ self.V_minus_exp_inv[l, ...] @ self.K_minus_half_inverse
        return self.K_minus_inverse @ self.V_minus_exp_inv[l, ...]


class AuxiliaryFieldInterorbitalAccurateCluster(AuxiliaryFieldInterorbitalAccurateImagNN):
    def __init__(self, config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse):
        self.config = config
        self.Bdim = config.total_dof // 2 // 2

        self.G_up_sum = np.zeros((self.Bdim, self.Bdim), dtype=np.complex128)
        self.G_down_sum = np.zeros((self.Bdim, self.Bdim), dtype=np.complex128)
        self.n_gf_measures = 0

        self.conf_path = os.path.join(local_workdir, 'last_conf')

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

        #self.connectivity = (K_oneband == K_oneband.real.max()).astype(np.float64)
        self.connectivity = (np.abs(K_oneband + 0.331) < 1e-5).astype(np.float64)
        assert np.allclose(self.connectivity, self.connectivity.T)
        self.n_hexagons = config.Ls ** 2

        self.hexagons = []  # hexagon -- typle of 6 sites
        self.hexagons_by_site = []  # list of 3 hexagons that include this site into list


        self.ratios = []
        self.n_bonds = []
        self.sites = []
        E = np.array([np.array([1. / 2., np.sqrt(3) / 2.]), np.array([1., 0])])
        sl = (E[0] + E[1]) / 3.
        for i in range(config.Ls ** 2):
            x, y = i % config.Ls, i // config.Ls
            self.sites.append(E[0] * x + E[1] * y)
            self.sites.append(self.sites[-1] + sl)

        @jit(nopython=True)
        def get_bc_distance(site1, site2, Lx, Ly):
            E = np.zeros((2, 2))
            E[0] = np.array([1. / 2., np.sqrt(3) / 2.])
            E[1] = np.array([1., 0])


            copies = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    shift = dx * E[0] * Lx + dy * E[1] * Ly

                    copies.append(site1 + shift)
            return np.sqrt(np.min(np.array([np.sum((site2 - copy) ** 2) for copy in copies]))), \
                   site2 - copies[np.argmin(np.array([np.sum((site2 - copy) ** 2) for copy in copies]))]

        ### check that sites connectivity graph agrees with the one obtained within old procedure ###
        for idx1, site1 in enumerate(self.sites):
            for idx2, site2 in enumerate(self.sites):
                #print(idx1, idx2)
                if np.isclose(get_bc_distance(site1, site2, config.Ls, config.Ls)[0], 1. / np.sqrt(3)):
                    assert self.connectivity[idx1, idx2] == 1.0
                if self.connectivity[idx1, idx2] == 1.0:
                    #print(get_bc_distance(site1, site2, config.Ls, config.Ls), site1, site2)
                    assert np.isclose(get_bc_distance(site1, site2, config.Ls, config.Ls)[0], 1. / np.sqrt(3))
        

        hex_centers = []
        self.hexagons = []
        self.hexagons_by_site = [[] for _ in range(len(self.sites))]
        hex_idx = 0
        for site in self.sites[::2]:
            hex_centers.append(site + np.array([1, -1]) * (E[0] + E[1]) / 3.)
            hexagon = []
            for idx, site in enumerate(self.sites):
                if np.isclose(get_bc_distance(site, hex_centers[-1], config.Ls, config.Ls)[0], 1. / np.sqrt(3)):
                    hexagon.append(idx)
                    self.hexagons_by_site[idx].append(hex_idx)
            assert len(hexagon) == 6
            hex_idx += 1
            self.hexagons.append(deepcopy(hexagon))

        for m in self.hexagons_by_site:
            assert len(m) == 3

        for idx, hexagon in enumerate(self.hexagons):
            hexagon_rearranged = [hexagon[0]]
            for i in range(1, 6):

                NNs = np.where(self.connectivity[:, hexagon_rearranged[i - 1]] == 1.0)[0]
                for nn in NNs:
                    A = get_bc_distance(self.sites[hexagon_rearranged[i - 1]], hex_centers[idx], config.Ls, config.Ls)[1]
                    B = get_bc_distance(self.sites[nn], hex_centers[idx], config.Ls, config.Ls)[1]
                    if nn in hexagon and nn not in hexagon_rearranged and np.cross(A, B) > 0:
                        hexagon_rearranged.append(nn)
                        break
            self.hexagons[idx] = deepcopy(hexagon_rearranged)

        self.meta_eye = []
        for _ in range(config.Nt):
            self.meta_eye.append(np.eye(len(self.K_plus)))
        self.meta_eye = np.array(self.meta_eye)

        # check that everything is alright
        for idx, hexagon in enumerate(self.hexagons):
            for i in range(len(hexagon)):
                assert self.connectivity[hexagon[i], hexagon[(i + 1) % 6]] == 1.0

        ### build neighboring hexagons ###
        # each hexagon hex_0 contains 6 elements #
        # for bond hex_0[i] to hex_0[(i + 1) % 6] we will pack to neighboring_hexagons[i] index of hexagon also containing this bond #

        self.neighboring_hexagons = [[] for _ in range(config.Ls ** 2)]
        for idx, hexagon_0 in enumerate(self.hexagons):
            for i in range(6):
                a, b = hexagon_0[i], hexagon_0[(i + 1) % 6]

                a_hexs = self.hexagons_by_site[a]
                b_hexs = self.hexagons_by_site[b]

                for hex_to in a_hexs:
                    if hex_to in b_hexs and hex_to != idx:
                        self.neighboring_hexagons[idx].append(hex_to)
                        break
        print(self.hexagons)
        print(self.neighboring_hexagons)
        ###


        if self.config.n_site_fields == 2:
            self.eta = {
                -1 : -np.sqrt(2),
                +1 : +np.sqrt(2),
            }
        elif self.config.n_site_fields == 4:
            self.eta = {
                -2 : -np.sqrt(6 + 2 * np.sqrt(6)),
                +2 : +np.sqrt(6 + 2 * np.sqrt(6)),
                -1 : -np.sqrt(6 - 2 * np.sqrt(6)),
                +1 : +np.sqrt(6 - 2 * np.sqrt(6)),
            }
        else:
            exit(-1)

        if self.config.n_site_fields == 4:
            self.gauge_hexagon = {
                -2 : (1. - np.sqrt(6) / 3 ) * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[-2]),
                +2 : (1. - np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[+2]),
                -1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[-1]),
                +1 : (1. + np.sqrt(6) / 3.) * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[+1])
            }

            self.gauge_hexagon_log = {
                -2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[-2],
                +2 : np.log(1. - np.sqrt(6) / 3) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[+2],
                -1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[-1],
                +1 : np.log(1. + np.sqrt(6) / 3) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[+1]
            }
            self.etajit = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0., +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]
            self.local_conf_combinations = [[-2], [-1], [1], [2]]
        elif self.config.n_site_fields == 2:
            self.gauge_hexagon = {
                -1 : 0.5 * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[-1]),
                +1 : 0.5 * np.exp(-2.0j * self.config.n_spins * self.config.nu_U * self.eta[+1])
            }

            self.gauge_hexagon_log = {
                -2 : 0,
                -1 : np.log(0.5) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[-1],
                +1 : np.log(0.5) - 2.0j * self.config.n_spins * self.config.nu_U * self.eta[+1],
                +2 : 0
            }

            self.etajit = [0., -np.sqrt(2), 0., +np.sqrt(2), 0.]
            self.local_conf_combinations = [[-1], [1]]
        else:
            exit(-1)
        #super().__init__(config, K, K_inverse, K_matrix, local_workdir, K_half, K_half_inverse)


        


        self.gpu_avail = config.gpu
        self.exponentiated = np.ones(config.Nt, dtype=bool)
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










        self.local_c_dict = {}
        self.nonzero_idxs = None
        print(self.hexagons_by_site)
        print(self.hexagons)

        #self.connections = []
        #for site in range(self.connectivity.shape[0]):
        #    self.connections.append(np.where((self.connectivity + np.eye(self.connectivity.shape[0]))[site] > 0)[0])
        #print(self.connections)
        return

    def get_gauge_factor_move_hex(self, local_conf_old, local_conf):
        return self.gauge_hexagon[local_conf] / self.gauge_hexagon[local_conf_old]

    def get_current_gauge_factor_log_hex(self):
        cf = self.hex.flatten()
        factor_logs = np.zeros(len(cf), dtype=np.complex128)
        factor_logs[cf == -2] = self.gauge_hexagon_log[-2]
        factor_logs[cf == 2] = self.gauge_hexagon_log[2]
        factor_logs[cf == 1] = self.gauge_hexagon_log[1]
        factor_logs[cf == -1] = self.gauge_hexagon_log[-1]

        return np.sum(factor_logs)

    def _get_initial_field_configuration(self):
        if self.config.start_type == 'cold':
            exit(-1)
        elif self.config.start_type == 'hot':
            self.hex = np.random.choice(np.array([x[0] for x in self.local_conf_combinations]), \
                                              size = (self.config.Nt, self.n_hexagons))

        else:
            loaded = False
            if os.path.isfile(self.conf_path + '.npy'):
                try:
                    self.hex = np.load(self.conf_path + '.npy')
                    print('Starting from a presaved field configuration', flush=True)
                    loaded = True
                except Exception:
                    print('Failed during loading of configuration from default location: try from dump')

                    try:
                        self.hex = np.load(self.conf_path + '_dump.npy')

                        print('Starting from a presaved field configuration in dump', flush=True)
                        loaded = True
                    except Exception:
                        print('Failed during loading of configuration from dump location: initialize from scratch')
            if not loaded:
                print('Random initial configuration')
                self.hex = np.random.choice(np.array([x[0] for x in self.local_conf_combinations]), \
                                              size = (self.config.Nt, self.n_hexagons))


        NtVolVol_shape = (self.config.Nt, 2 * self.config.Ls ** 2, 2 * self.config.Ls ** 2)
        self.V_plus = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 
        self.V_minus = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 

        self.V_plus_exp = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 
        self.V_minus_exp = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 

        self.V_plus_exp_inv = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 
        self.V_minus_exp_inv = np.zeros(shape = NtVolVol_shape, dtype=np.complex128); 


        for time_slice in range(self.config.Nt):
            for hex_index in range(self.config.Ls ** 2):
                block_plus, block_minus = _V_from_configuration_onesite_accurate_imag_hex(
                        self.hex[time_slice, hex_index], self.config.nu_U, self.config.alpha, self.etajit
                    )
                
                self.V_plus = _assign_6x6(self.V_plus, time_slice, self.hexagons[hex_index], block_plus)
                self.V_minus = _assign_6x6(self.V_minus, time_slice, self.hexagons[hex_index], block_minus)
                self.exponentiated[time_slice] = False

        self.exponentiate_V()
        return
    
    def exponentiate_V(self):
        t = time()
        for time_slice in range(self.config.Nt):
            if self.exponentiated[time_slice]:
                continue
            self.V_plus_exp[time_slice] = scipy.linalg.expm(self.V_plus[time_slice])
            self.V_minus_exp[time_slice] = scipy.linalg.expm(self.V_minus[time_slice])
            self.V_plus_exp_inv[time_slice] = scipy.linalg.expm(-self.V_plus[time_slice])
            self.V_minus_exp_inv[time_slice] = scipy.linalg.expm(-self.V_minus[time_slice])
            self.exponentiated[time_slice] = True

        return

    def save_configuration(self):
        addstring = '_dump' if self.n_times_saved % 2 == 1 else ''
        self.n_times_saved += 1
        return np.save(self.conf_path + addstring, self.hex)


    def update_hex_field(self, hex_index, time_slice, old_conf, new_conf):
        '''
            we update hex-variable, which affects 6 sites
        '''
        self.hex[time_slice, hex_index] = new_conf

        block_plus_old, block_minus_old = _V_from_configuration_onesite_accurate_imag_hex(
                old_conf, self.config.nu_U, self.config.alpha, self.etajit
            )

        block_plus, block_minus = _V_from_configuration_onesite_accurate_imag_hex(
                new_conf, self.config.nu_U, self.config.alpha, self.etajit
            )

        self.V_plus = _assign_6x6(self.V_plus, time_slice, self.hexagons[hex_index], block_plus - block_plus_old)
        self.V_minus = _assign_6x6(self.V_minus, time_slice, self.hexagons[hex_index], block_minus - block_minus_old)
        self.exponentiated[time_slice] = False
        return

    def prepare_current_Z(self, time_slice):
        #self.Z_plus = scipy.sparse.csr_matrix(self.V_plus[time_slice])
        #self.Z_minus = scipy.sparse.csr_matrix(self.V_minus[time_slice])

        return



    def compute_deltas_hex(self, sp_index, time_slice, local_conf, local_conf_combinations):
        '''
            deltas for bond-update of the xi-field (use the standard `_get_delta_interorbital_twosite_accurate_imag`)
        '''

        #if local_conf == local_conf_proposed:
        #    self.Delta_plus, self.Delta_minus, self.support = np.zeros((6, 6), dtype=np.complex128), np.zeros((6, 6), dtype=np.complex128), np.array(self.hexagons[sp_index])
        #    return

        #self.Delta_plus, self.Delta_minus, self.support = self._get_delta_interorbital_twosite_accurate_imag_hex(
        #        self.Z_plus, self.Z_minus, local_conf, local_conf_proposed, sp_index, self.config.nu_U, self.config.alpha
        #    )

        #return 


        if not np.isclose(self.config.alpha, 0.):
            deltas = []
            V_plus_test = self.V_plus[time_slice] * 1.

            Vs = np.zeros((len(local_conf_combinations), V_plus_test.shape[0], V_plus_test.shape[1]), dtype=np.complex128)
            #Qs = np.zeros((4, V_plus_test.shape[0], V_plus_test.shape[1]), dtype=np.complex128)
            Vs[0] = -V_plus_test


            idx = 1
            for local_conf_proposed in local_conf_combinations:
                if local_conf_proposed[0] == local_conf:
                    continue
                V_plus_testx = _assign_6x6_not(V_plus_test.copy() * 1., self.hexagons[sp_index], \
                                              _V_from_configuration_onesite_accurate_imag_hex(local_conf_proposed[0], self.config.nu_U, self.config.alpha, self.etajit)[0] - \
                                              _V_from_configuration_onesite_accurate_imag_hex(local_conf, self.config.nu_U, self.config.alpha, self.etajit)[0])
                Vs[idx] = V_plus_testx.copy() * 1.0
                idx += 1



            #t = time()            

            '''
            t = time()
            Vbig = np.zeros((4 * Vs.shape[1], Vs.shape[2]* 4), dtype=np.complex128)
            for i in range(4):
                Vbig[i * Vs.shape[1]:(i + 1) * Vs.shape[1], i * Vs.shape[1]:(i + 1) * Vs.shape[1]] = Vs[i]
            Vbig = scipy.linalg.expm(Vbig)
            print(Vbig.shape)
            exps = np.array([Vbig[i * Vs.shape[1]:(i + 1) * Vs.shape[1], \
                                  i * Vs.shape[1]:(i + 1) * Vs.shape[1]]] for i in range(4))
            print(exps.shape)
            print(time() - t, 'new')

            t = time()
            '''



            self.n_bonds.append(np.sum(np.abs(Vs[0] - np.diag(np.diag(Vs[0]))) > 1e-3) / Vs.shape[2] / 3.)
            
            n_clusters, arr = connected_components(np.abs(Vs[0]) > 1e-3)
            excl = np.array(excluded_sites(n_clusters, arr, self.hexagons[sp_index]), dtype=np.int64)
            active_sites = np.sort(np.setdiff1d(np.arange(Vs[0].shape[1]), excl, assume_unique=True))
            Vs_selected = np.ascontiguousarray(Vs[:, active_sites, :][:, :, active_sites])

            self.ratios.append(len(active_sites) ** 3 / Vs.shape[2] ** 3)
            
            exps = np.array([scipy.linalg.expm(x) for x in Vs_selected])
                        


            '''
            if self.previous_V is None:
                exps = np.array([scipy.linalg.expm(x) for x in Vs])
            else:
                exps = np.concatenate([self.previous_V[np.newaxis, ...], np.array([scipy.linalg.expm(x) for x in Vs[1:]])], axis=0)
            '''
            
            
            
            #t1 = time() - t

            #t = time()
            #exps = np.array([_fast_matrix_exp(x) for x in Vs])
            #print(time() - t, 'takes exponent fast')
            #t2 = time() - t
            #self.ratios.append(t1 / t2)
            #print(np.mean(self.ratios))

            #print(np.linalg.norm(exps - exps_cc))
            # print('after exp', np.sum(np.abs(exps[1]) > 1e-3) / exps[1].shape[1] ** 2)

            '''
            print(time() - t, 'vectorized')
            t = time()
            exps = np.array([scipy.linalg.expm(x) for x in Vs])
            print(time() - t, 'standard')
            '''

            
            inv_exps = np.linalg.inv(exps).conj()
            prod_plus = exps @ exps[0]
            prod_minus = inv_exps @ inv_exps[0]


            idx = 1
            for local_conf_proposed in local_conf_combinations:
                if local_conf_proposed[0] == local_conf:
                    deltas.append((np.eye(2) * 0. + 0.0j, np.eye(2) * 0. + 0.0j, np.arange(2)))
                    continue

                
                '''
                delta_plus = (prod_plus[idx] - np.eye(V_plus_test.shape[0]))
                delta_minus = (prod_minus[idx] - np.eye(V_plus_test.shape[0]))
                support_row = np.where(np.sum(np.abs(delta_plus), axis=0) > 1e-8)[0]
                deltas.append((delta_plus[support_row][:, support_row].copy() * 1., \
                               delta_minus[support_row][:, support_row].copy() * 1., \
                               support_row, inv_exps[idx].conj()))


                '''
                delta_plus = (prod_plus[idx] - np.eye(len(active_sites)))
                delta_minus = (prod_minus[idx] - np.eye(len(active_sites)))
                support_row = active_sites
                

                deltas.append((delta_plus, delta_minus, support_row, inv_exps[idx].conj()))
                
                idx += 1
                

            return deltas

        support = np.array(self.hexagons[sp_index]).astype(np.int64)
        deltas = []
        for local_conf_proposed in local_conf_combinations:
            dV = _V_from_configuration_onesite_accurate_imag_hex(local_conf_proposed[0], self.config.nu_U, self.config.alpha, self.etajit)[0] - \
                 _V_from_configuration_onesite_accurate_imag_hex(local_conf, self.config.nu_U, self.config.alpha, self.etajit)[0]
            delta = np.diag(np.exp(np.diag(dV))) - np.eye(6)

            deltas.append((delta.copy() * 1., delta.copy() * 1., support))
        return deltas

        

        


    def get_current_hex(self, hex_index, time_slice):
        return self.hex[time_slice, hex_index]


    def update_G_seq_hex(self, hex_index):
        idxs = self.support#np.array(self.hexagons[hex_index], dtype=np.int64)
        self.current_G_function_up = _update_G_seq_inter_hex(self.current_G_function_up, self.Delta_plus, idxs, self.config.total_dof // 2)
        self.current_G_function_down = _update_G_seq_inter_hex(self.current_G_function_down, self.Delta_minus, idxs, self.config.total_dof // 2)
        return

    def _get_delta_interorbital_twosite_accurate_imag_hex(self, full_plus, full_minus, local_conf, local_conf_proposed, hex_idx, nu_U, alpha):
        Vp, expVp, Vm, expVm = self.local_configurations_dict(local_conf, local_conf_proposed, hex_idx, nu_U, alpha)

        #t = time()
        deltap, support = _get_valley_delta(Vp.todense(), expVp.todense(), full_plus.todense())
        #print('time', time() - t)
        deltam, _ = _get_valley_delta(Vm.todense(), expVm.todense(), full_minus.todense())

        return deltap, deltam, support

    def local_configurations_dict(self, conf_from, conf_to, hex_idx, nu_U, alpha):
        if (conf_from, conf_to, hex_idx) in self.local_c_dict:
            return self.local_c_dict[(conf_from, conf_to, hex_idx)]

        local_V_inv_plus, local_V_inv_minus = _V_from_configuration_onesite_accurate_imag_hex(conf_from, nu_U, alpha, self.etajit)
        local_V_proposed_plus, local_V_proposed_minus = _V_from_configuration_onesite_accurate_imag_hex(conf_to, nu_U, alpha, self.etajit)

        hex_0 = self.hexagons[hex_idx]
        rows, cols = np.meshgrid(hex_0, hex_0)
        rows, cols = rows.flatten(), cols.flatten()

        nonzero = np.abs((local_V_proposed_plus - local_V_inv_plus).flatten()) > 1e-10


        self.local_c_dict[(conf_from, conf_to, hex_idx)] = (scipy.sparse.csr_matrix(((local_V_proposed_plus - local_V_inv_plus).flatten()[nonzero], (cols[nonzero], rows[nonzero])), shape=self.V_plus[0].shape), \
                                                            scipy.sparse.csr_matrix(((scipy.linalg.expm(local_V_proposed_plus - local_V_inv_plus) - np.eye(6)).flatten(), (cols, rows)), shape=self.V_plus[0].shape), \
                                                            scipy.sparse.csr_matrix(((local_V_proposed_minus - local_V_inv_minus).flatten()[nonzero], (cols[nonzero], rows[nonzero])), shape=self.V_minus[0].shape), \
                                                            scipy.sparse.csr_matrix(((scipy.linalg.expm(local_V_proposed_minus - local_V_inv_minus) - np.eye(6)).flatten(), (cols, rows)), shape=self.V_minus[0].shape)\
                                                            )

        return self.local_c_dict[(conf_from, conf_to, hex_idx)]



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
           


#@jit(nopython=True)
def _get_valley_delta(Q, ApBexp_full, Z):
    Delta_full = ApBexp_full #scipy.sparse.csr_matrix(Z.shape, dtype=np.complex128)
    #comm = Z.dot(Q) - Z.dot(Q)
    #commQ2 = comm + Q.dot(Q)
    #Delta_full = ApBexp_full + comm / 2. + 1. / 6. * (Z.dot(commQ2) - commQ2.dot(Z))# (Z.dot(comm) - comm.dot(Z) + Z.dot(Q.dot(Q)) - Q.dot(Q).dot(Z))

    
    X = Q
    Qp = Q
    fact = 1.
    for n in range(1, 5):
        fact *= n
        Delta_full += X / fact - Qp / fact
        X = Z.dot(X) - X.dot(Z) + Q.dot(X)
        Qp = Qp.dot(Q)


    #Delta_full = ApBexp_full + comm / 2. + 1. / 6. * (Z.dot(comm) - comm.dot(Z) + Z.dot(Q.dot(Q)) - Q.dot(Q).dot(Z))
    
    support_cols = np.where(np.abs(Delta_full).sum(axis = 1) > 1e-10)[0]
    #support_rows = np.where(np.abs(Delta_full.todense()).sum(axis = 0) > 1e-10)[1]

    #assert np.allclose(support_rows, support_cols)

    ret = Delta_full[support_cols][:, support_cols]
    return ret, support_cols


@jit(nopython=True, boundscheck=False)
def _compute_support(indptr):
    support = List()
    i = 0
    for r in range(len(indptr) - 1):
        if indptr[r + 1] - indptr[r] > 0:
            support.append(r)
    return np.asarray(support)

@jit(nopython=True, boundscheck=True)
def _construct_delta(elements, indices, indptr):
    support = _compute_support(indptr)
    dim = len(support)
    Delta = np.zeros((dim, dim), dtype=elements.dtype)
    for i in range(dim):
        r = support[i]
        cs = np.searchsorted(support, indices[indptr[r] : indptr[r + 1]])
        for offset, c in enumerate(cs):
            Delta[i, c] = elements[indptr[r] + offset]
    return Delta, support
'''
def _get_valley_delta(ApB_full, ApBexp_full, Z):
    commutator = Z @ ApB_full - ApB_full @ Z + (Z @ ApB_full + ApB_full @ Z) @ (ApB_full / 6 - Z / 3)
    Delta = commutator + ApBexp_full
    return _construct_delta(Delta.data, Delta.indices, Delta.indptr)
'''

'''


@jit(nopython=True)
def _get_valley_delta(ApB, ApBexp, Z, hex_0, nonzero_idxs):
    ZAmB = Z * 0.0; AmBZ = Z * 0.0;
    for i in range(Z.shape[0]):
        for kk, k in enumerate(hex_0):
            for jj, j in enumerate(hex_0):
                ZAmB[i, k] += Z[i, j] * ApB[jj, kk]
    for k in range(Z.shape[1]):
        for ii, i in enumerate(hex_0):
            for jj, j in enumerate(hex_0):
                AmBZ[i, k] += ApB[ii, jj] * Z[j, k]

    Delta_full = _assign_6x6_not(0.5 * (ZAmB - AmBZ), hex_0, ApBexp - np.eye(6))
    support_cols = np.where(np.abs(Delta_full).sum(axis = 1) != 0)[0]

    return Delta_full[support_cols][:, support_cols], support_cols

'''

@jit(nopython=True)
def _assign_6x6(array, t, indexes, small_array):
    for ii, i in enumerate(indexes):
        for jj, j in enumerate(indexes):
            array[t, i, j] += small_array[ii, jj]
    return array


@jit(nopython=True)
def _assign_6x6_not(array, indexes, small_array):
    for ii, i in enumerate(indexes):
        for jj, j in enumerate(indexes):
            array[i, j] += small_array[ii, jj]
    return array


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
def _V_from_configuration_onesite_accurate_imag_hex(configuration, nu_U, alpha, eta):  # used for initialization!
    V_hex_plus = np.zeros((6, 6), dtype=np.complex128)
    V_hex_minus = np.zeros((6, 6), dtype=np.complex128)

    ### work out onsite part ###
    for idx in range(6):
        V_hex_plus[idx, idx] += 1.0j / 3.
        V_hex_minus[idx, idx] += 1.0j / 3.

    ### work out bond part ###
    for b_idx in range(6):
        V_hex_plus[b_idx, (b_idx + 1) % 6] += -alpha
        V_hex_plus[(b_idx + 1) % 6, b_idx] -= -alpha

        V_hex_minus[b_idx, (b_idx + 1) % 6] -= -alpha
        V_hex_minus[(b_idx + 1) % 6, b_idx] += -alpha

    return V_hex_plus * nu_U * eta[int(configuration) + 2], V_hex_minus * nu_U * eta[int(configuration) + 2]


@jit(nopython=True)
def _V_from_configuration_twosite_accurate_imag_hex(s, sign, spin, nu_U):  # !!! valid in this form ONLY for update (computation of Delta)
    eta = [-np.sqrt(6 + 2 * np.sqrt(6)), -np.sqrt(6 - 2 * np.sqrt(6)), 0, \
           +np.sqrt(6 - 2 * np.sqrt(6)), np.sqrt(6 + 2 * np.sqrt(6))]
    return np.diag(np.exp(1.0j * nu_U * eta[int(s) + 2] * sign * np.ones(6) / 3.))  # bond-variable is the same for both sites




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
def get_det_ratio_inter_hex(idxs, Delta, G):
    G_slice = G[idxs, :]
    G_slice = G_slice[:, idxs]

    return np.linalg.det(np.eye(len(idxs), dtype=np.complex128) + np.dot(Delta, np.eye(len(idxs), dtype=np.complex128) - G_slice))

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

    return G + GU.dot(Zinv)


@jit(nopython=True)
def _update_G_seq_inter_hex(G, Delta, idxs, total_dof):
    U = np.zeros((total_dof // 2, len(idxs)), dtype=np.complex128)
    for i in range(len(idxs)):
        U[idxs[i]] = Delta[i]
    G_sliced_left = G[idxs, :] * 1.0
    V = G_sliced_left

    for i in range(len(idxs)):
        V[i, idxs[i]] -= 1
    V = np.ascontiguousarray(V)
    U = np.ascontiguousarray(U)
    G = np.ascontiguousarray(G)

    GU = G.dot(U)
    Zinv = np.linalg.inv(np.eye(len(idxs), dtype=np.complex128) - V.dot(U)).dot(V)
    Zinv = np.ascontiguousarray(Zinv)

    return G + GU.dot(Zinv)


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

def _vector_matrix_exp(mat):
    result = mat * 0
    current_mat = mat * 1.
    fact = 1
    for n in range(1, 15):
        fact = fact * n
        result += current_mat / fact

        current_mat = np.einsum('ijk,ikl->ijl', current_mat, mat)

    return result



import scipy.sparse.linalg

def _fast_matrix_exp(A):
    h = scipy.sparse.linalg.matfuncs._ExpmPadeHelper(A, structure=None, use_exact_onenorm=(A.shape[0] < 200))
    eta_1 = max(h.d4_loose, h.d6_loose)
    U, V = h.pade3()
    return scipy.sparse.linalg.matfuncs._solve_P_Q(U, V, structure=None)


def exponent_non_excluded(V, active_sites):
    return scipy.linalg.expm(V[:, active_sites][active_sites])


@jit(nopython=True)
def excluded_sites(n_clusters, arr, hexagon):
    groups = [[-1] for _ in range(n_clusters)]
    for site, cluster in enumerate(arr):
        groups[cluster].append(site)

    excluded_sites = [-1]
    for group in groups:
        has_overlap = False
        for site in hexagon:
            if site in group:
                has_overlap = True
                break
        if not has_overlap:
            excluded_sites += group[1:]
    return excluded_sites[1:]
