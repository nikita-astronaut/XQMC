import numpy as np
import pickle
import time
import cupy as cp

def get_initial_field_configuration(config):
    if config.start_type == 'cold':
        return np.random.randint(0, 1, size = (config.Nt, 4 * config.Ls * config.Ls)) * 2. - 1.0
    if config.start_type == 'hot':
        return np.random.randint(0, 2, size = (config.Nt, 4 * config.Ls * config.Ls)) * 2. - 1.0

    return pickle.load(open(start_type, 'rb'))

def flip_random_spin(h_configuration):
    time_slice = np.random.randint(0, Nt)
    spatial_index = np.random.randint(0, 2 * Ls * 2 * Ls)
    h_configuration[time_slice, spatial_index] *= -1
    return h_configuration

def get_det(h_configuration, K, config):
    t = time.time()
    M_up = cp.diag(cp.ones(4 * config.Ls ** 2))
    M_down = cp.diag(cp.ones(4 * config.Ls ** 2))

    for slice_idx in range(config.Nt):
        V_up = cp.diag(cp.exp(config.nu * h_configuration[slice_idx, ...]))
        V_down = cp.diag(cp.exp(-config.nu * h_configuration[slice_idx, ...]))
        M_up = M_up.dot(K)
        M_down = M_down.dot(K)
        M_up = M_up.dot(V_up)
        M_down = M_down.dot(V_down)

    M_up = cp.diag(cp.ones((2 * Ls) ** 2)) + M_up
    M_down = cp.diag(cp.ones((2 * Ls) ** 2)) + M_down

    # print('construction of M matrixes took ' + str(time.time() - t))
    t = time.time()

    sign_det_up, log_det_up = cp.linalg.slogdet(M_up)
    sign_det_down, log_det_down = cp.linalg.slogdet(M_down)

    # s = cp.sum(h_configuration)
    # log_factor = -config.nu * s
    # print('eh/symmetry breaking log = ', log_det_down + np.log(sign_det_down) - log_factor - log_det_up - np.log(sign_det_up))

    return np.real(log_det_up + log_det_down), sign_det_up * sign_det_down