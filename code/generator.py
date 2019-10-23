import numpy as np

xp = np  # by default the code is executed on the CPU
cp = np

try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    print('No CuPy found in the system, running on a CPU')


import scipy.linalg
from copy import deepcopy
import scipy.sparse as scs
import time
import auxiliary_field
import observables
from config_generator import simulation_parameters
accept_history = []
sign_history = []
ratio_history = []

config = simulation_parameters()
S_AF_history = []
SzSz_history = []
def print_greetings(config):
    print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
          "lattice = {:d}^2 x {:d}".format(config.start_type, 1.0 / config.dt / config.Nt, config.mu, config.Ls, config.Nt))
    print("# iteration current_flips N_swipes <log(ratio)> d<log(ratio)> <acceptance> <sign> d<sign> <density> <S_AF> <K> <Sz(0)Sz(0)> <Sz(0)Sz(1)> <Sz(0)Sz(2)> <Sz(0)Sz(3)> <Sz(0)Sz(4)> <Sz(0)Sz(5)> <n_up(0) n_down(0)> <n_up(0) n_down(1)> <n_up(0) n_down(2)> <n_up(0) n_down(3)> <n_up(0) n_down(4)>")
    return

def print_generator_log(n_sweep, h_field, K_matrix, K, config):
    global accept_history, sign_history, ratio_history
    if n_sweep % config.n_print_frequency != 0:
        return
    n_print = np.min([n_sweep * config.total_dof // 2, config.n_smoothing])
    n_history = np.min([n_print, len(ratio_history)])
    # S_AF_history.append(cp.asnumpy(observables.staggered_magnetisation(h_field, K, config)))
    print("{:d}, {:d}, {:.3f} +/- {:.3f}, {:.3f}, {:.3f} +/- {:.3f}, {:.9f}".format(n_sweep, current_n_flips, \
        np.mean(ratio_history[-n_history:]), np.std(ratio_history[-n_history:]), \
        np.mean(accept_history[n_sweep-n_print:n_sweep]), \
        np.mean(sign_history[-n_print:]), np.std(sign_history[-n_print:]), \
        cp.asnumpy(observables.total_density(h_field, K, config)), \
        cp.asnumpy(observables.kinetic_energy(h_field, K, K_matrix, config)), \
        cp.asnumpy(observables.SzSz_onsite(h_field, K, config))), flush = True)
    return


def get_det_ratio(proposed_conf, spin, G, sp_index, config):
    Delta = np.exp(2 * spin * proposed_conf * config.nu) - 1.
    return 1. + Delta * (1. - G[sp_index, sp_index])

def update_G_seq(proposed_conf, spin, G, sp_index, config):
    Delta = np.exp(2 * spin * proposed_conf * config.nu) - 1.
    update_matrix = xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2))
    update_matrix_inv = xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2))

    update_matrix[sp_index, :] += Delta * (xp.diag(xp.ones(config.n_orbitals * config.n_sublattices * config.Ls ** 2)) - G)[sp_index, :]
    det_update_matrix = update_matrix[sp_index, sp_index]
    update_matrix_inv[sp_index, :] = -update_matrix[sp_index, :] / det_update_matrix
    update_matrix_inv[sp_index, sp_index] = 1. / det_update_matrix

    result = deepcopy(G)
    result += xp.einsum('i,k->ik', G[:, sp_index], update_matrix_inv[sp_index, :])

    return result


def perform_sweep(configuration, config, K_operator):
    for time_slice in range(config.Nt):
        t = time.time()
        # M_up_partial, B_time_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice, return_Bl = True)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        # M_down_partial, B_time_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice, return_Bl = True)

        M_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        M_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice)        

        G_up_seq = auxiliary_field.inv_illcond(M_up)
        G_down_seq = auxiliary_field.inv_illcond(M_down)

        print('construction of M matrixes took ' + str(time.time() - t))
        t = time.time()
        current_det_log, current_det_sign = auxiliary_field.get_det(M_up, M_down)

        for sp_index in range(M_up.shape[0]):
            sign_history.append(current_det_sign)
            configuration[time_slice, sp_index] *= -1

            ratio = get_det_ratio(configuration[time_slice, sp_index], +1, G_up_seq, sp_index, config) * get_det_ratio(configuration[time_slice, sp_index], -1, G_down_seq, sp_index, config)

            # B_up_new = auxiliary_field.B_l(configuration, +1, time_slice, K_operator, config)
            # B_down_new = auxiliary_field.B_l(configuration, -1, time_slice, K_operator, config)
            # print('B_matrixes making took ' + str(time.time() - t))
            # t = time.time()
            # new_det_log, sign_new_det = auxiliary_field.get_det_partial_matrices(M_up_partial, B_up_new, M_down_partial, B_down_new, identity)
            # print('det computation took ' + str(time.time() - t))
            # t = time.time()
            # ratio = np.min([1, np.exp(new_det_log - current_det_log)])
            # print(np.exp(new_det_log - current_det_log) / ratio_improved - 1.)
            lamb = np.random.uniform(0, 1)
            if lamb < ratio:
                current_det_log += np.log(np.abs(ratio))
                ratio_history.append(np.log(np.abs(ratio)))
                current_det_sign *= np.sign(ratio)
                accept_history.append(+current_n_flips)
                G_up_seq = update_G_seq(configuration[time_slice, sp_index], +1, G_up_seq, sp_index, config)
                G_down_seq = update_G_seq(configuration[time_slice, sp_index], -1, G_down_seq, sp_index, config)
            else:
                configuration[time_slice, sp_index] *= -1  # roll back

        M_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        M_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice)        

        G_up = auxiliary_field.inv_illcond(M_up)
        G_down = auxiliary_field.inv_illcond(M_down)

        print('final discrepancy after sweep = ', np.sum(np.abs(G_up - G_up_seq)) / np.sum(np.abs(G_up)), np.sum(np.abs(G_down - G_down_seq)) / np.sum(np.abs(G_down)))
        

        print('on-slice sweep took ' + str(time.time() - t))
        t = time.time()
    return configuration


if __name__ == "__main__":
    print_greetings(config)

    current_n_flips = 1
    n_flipped = 0
    K_matrix = config.model(config.Ls, config.mu)
    K_operator = xp.asarray(scipy.linalg.expm(config.dt * K_matrix))

    current_field = xp.asarray(auxiliary_field.get_initial_field_configuration(config))

    for n_sweep in range(config.n_sweeps):
        current_field = perform_sweep(current_field, config, K_operator)

        print_generator_log(n_sweep, current_field, K_matrix, K_operator, config)