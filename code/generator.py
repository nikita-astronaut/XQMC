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


def perform_sweep(phi_field):
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()

    for time_slice in range(phi_field.config.Nt):
        t = time.time()
        # M_up_partial, B_time_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice, return_Bl = True)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        # M_down_partial, B_time_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice, return_Bl = True)
        if time_slice % phi_field.config.s_refresh == phi_field.config.Nt % phi_field.config.s_refresh:  # every s-th configuration we refresh the Green function
            phi_field.append_new_decomposition(tmin, tmax)
            phi_field.refresh_G_functions()
            current_det_log, current_det_sign = phi_field.get_log_det()
        else:  # wrapping up 
            print('simple wrap-up')
            phi_field.wrap_up(time_slice)
        print('going to next time slice took ' + str(time.time() - t))
        t = time.time()

        for sp_index in range(phi_field.config.total_dof // 2):
            sign_history.append(current_det_sign)
            phi_field.configuration[time_slice, sp_index] *= -1

            ratio = phi_field.get_det_ratio(+1, sp_index, time_slice) * phi_field.get_det_ratio(-1, sp_index, time_slice)

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
                phi_field.update_G_seq(+1, sp_index, time_slice)
                phi_field.update_G_seq(-1, sp_index, time_slice)


                G_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                G_down_check = phi_field.get_G_no_optimisation(-1, time_slice)
                print('discrepancy after one flip step = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))
            else:
                phi_field.configuration[time_slice, sp_index] *= -1  # roll back

        # M_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        # M_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice)        

        G_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
        G_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

        print('final discrepancy after sweep = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))

        print('on-slice sweep took ' + str(time.time() - t))
        t = time.time()
    return phi_field


if __name__ == "__main__":
    print_greetings(config)

    current_n_flips = 1
    n_flipped = 0
    K_matrix = config.model(config.Ls, config.mu)
    K_operator = xp.asarray(scipy.linalg.expm(config.dt * K_matrix))
    K_operator_inverse = xp.asarray(scipy.linalg.expm(-config.dt * K_matrix))

    phi_field = auxiliary_field.auxiliary_field(config, K_operator, K_operator_inverse)

    for n_sweep in range(config.n_sweeps):
        current_field = perform_sweep(phi_field)

        print_generator_log(n_sweep, current_field, K_matrix, K_operator, config)