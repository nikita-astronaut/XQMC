import numpy as np
import pickle
import time

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def inv_illcond(A):
    u, s, v = xp.linalg.svd(A)
    Ainv = xp.dot(v.transpose(), xp.dot(xp.diag(s**-1), u.transpose()))
    return Ainv

def load_configuration(path):
    return np.load(start_type)

def get_initial_field_configuration(config):
    if config.start_type == 'cold':
        return np.random.randint(0, 1, size = (config.Nt, config.n_orbitals * config.n_sublattices * config.Ls ** 2)) * 2. - 1.0
    if config.start_type == 'hot':
        return np.random.randint(0, 2, size = (config.Nt, config.n_orbitals * config.n_sublattices * config.Ls ** 2)) * 2. - 1.0

    return load_configuration(start_type)

def save_configuration(configuration, path):
    return np.save(path, configuration)

def flip_random_spin(h_configuration, config, current_n_flips):
    for _ in range(current_n_flips):
        time_slice = np.random.randint(0, config.Nt)
        spatial_index = np.random.randint(0, config.n_orbitals * config.n_sublattices * config.Ls ** 2)
        h_configuration[time_slice, spatial_index] *= -1
    return h_configuration

def B_l(h_configuration, spin, l, K, config):
    V = xp.diag(xp.exp(spin * config.nu * h_configuration[l, ...]))
    return V.dot(K)


def fermionic_matrix(h_configuration, K, spin, config, time = 0, return_Bl = False):
    M = xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2))
    current_V = xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2))

    slices = list(range(time + 1, config.Nt)) + list(range(0, time))
    if not return_Bl:
        slices = list(range(time + 1, config.Nt)) + list(range(0, time + 1))

    for nr, slice_idx in enumerate(slices):
        B = B_l(h_configuration, spin, slice_idx, K, config)
        M = B.dot(M)
        if nr % 15 == 14:
            u, s, v = xp.linalg.svd(M)
            current_V = v.dot(current_V)
            M = u.dot(xp.diag(s))
    M = M.dot(current_V)

    if not return_Bl:
        return xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2)) + M
    return M, B_l(h_configuration, spin, time, K, config)

def get_det(M_up, M_down):
    sign_det_up, log_det_up = xp.linalg.slogdet(M_up)
    sign_det_down, log_det_down = xp.linalg.slogdet(M_down)

    return np.real(log_det_up + log_det_down), sign_det_up * sign_det_down

def get_det_partial_matrices(M_up_partial, B_up_l, M_down_partial, B_down_l, identity):
    sign_det_up, log_det_up = xp.linalg.slogdet(identity + B_up_l.dot(M_up_partial))
    sign_det_down, log_det_down = xp.linalg.slogdet(identity + B_down_l.dot(M_down_partial))
    return np.real(log_det_up + log_det_down), sign_det_up * sign_det_down

def get_green_function(h_configuration, K, spin, config):
    # print("K matrix condition number = " + str(np.linalg.cond(xp.asnumpy(fermionic_matrix(h_configuration, K, spin, config)))))
    return inv_illcond(fermionic_matrix(h_configuration, K, spin, config))
    
