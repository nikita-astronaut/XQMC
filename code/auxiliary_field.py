import numpy as np
import pickle
import time

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def load_configuration(path):
    return np.load(start_type)

def get_initial_field_configuration(config):
    if config.start_type == 'cold':
        return np.random.randint(0, 1, size = (config.Nt, 4 * config.Ls * config.Ls)) * 2. - 1.0
    if config.start_type == 'hot':
        return np.random.randint(0, 2, size = (config.Nt, 4 * config.Ls * config.Ls)) * 2. - 1.0

    return load_configuration(start_type)

def save_configuration(configuration, path):
    return np.save(path, configuration)

def flip_random_spin(h_configuration, config):
    time_slice = np.random.randint(0, config.Nt)
    spatial_index = np.random.randint(0, 4 * config.Ls ** 2)
    h_configuration[time_slice, spatial_index] *= -1
    return h_configuration

def fermionic_matrix(h_configuration, K, spin, config):
    M = xp.diag(xp.ones(4 * config.Ls ** 2))
    for slice_idx in range(config.Nt):
        V = xp.diag(xp.exp(spin * config.nu * h_configuration[slice_idx, ...]))
        M = M.dot(K)
        M = M.dot(V)
    return xp.diag(xp.ones((2 * config.Ls) ** 2)) + M

def get_det(h_configuration, K, config):
    t = time.time()
    M_up = fermionic_matrix(h_configuration, K, +1.0, config)
    M_down = fermionic_matrix(h_configuration, K, -1.0, config)

    # print('construction of M matrixes took ' + str(time.time() - t))
    t = time.time()

    sign_det, log_det = xp.linalg.slogdet(M_up.dot(M_down))
    # print('diagonalization of M matrixes took ' + str(time.time() - t))
    # s = xp.sum(h_configuration)
    # log_factor = -config.nu * s
    # print('eh/symmetry breaking log = ', log_det_down + np.log(sign_det_down) - log_factor - log_det_up - np.log(sign_det_up))

    return np.real(log_det), sign_det

def get_green_function(h_configuration, K, spin, config):
    return xp.linalg.inv(fermionic_matrix(h_configuration, K, spin, config))
