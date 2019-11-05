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
densities = []
def print_greetings(config):
    print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
          "lattice = {:d}^2 x {:d}".format(config.start_type, 1.0 / config.dt / config.Nt, config.mu, config.Ls, config.Nt))
    print("# iteration current_flips N_swipes <log(ratio)> d<log(ratio)> <acceptance> <sign> d<sign> <density> <S_AF> <K> <Sz(0)Sz(0)> <Sz(0)Sz(1)> <Sz(0)Sz(2)> <Sz(0)Sz(3)> <Sz(0)Sz(4)> <Sz(0)Sz(5)> <n_up(0) n_down(0)> <n_up(0) n_down(1)> <n_up(0) n_down(2)> <n_up(0) n_down(3)> <n_up(0) n_down(4)>")
    return

def print_generator_log(n_sweep, phi_field, K_matrix):
    global accept_history, sign_history, ratio_history, densities, S_AF_history, SzSz_history
    if n_sweep % config.n_print_frequency != 0:
        return
    n_print = np.min([n_sweep * config.total_dof // 2 * config.Nt, config.n_smoothing])
    n_history = np.min([n_print, len(ratio_history)])
    n_print_AF = np.min([len(S_AF_history), 500])
    # S_AF_history.append(cp.asnumpy(observables.staggered_magnetisation(phi_field)))
    print("{:d}, {:d}, {:.9f} +/- {:.9f}, {:.3f}, {:.3f} +/- {:.3f}, {:.9f} +/- {:.9f}, {:.9f} +/- {:.9f}, {:.9f} +/- {:.9f}".format(n_sweep, current_n_flips, \
        np.mean(ratio_history[-n_history:]), np.std(ratio_history[-n_history:]) / np.sqrt(len(ratio_history[-n_history:])), \
        np.mean(accept_history), \
        np.mean(sign_history[-n_print:]), np.std(sign_history[-n_print:]) / np.sqrt(len(sign_history[-n_print:])), \
        np.mean(densities[-n_print:]), np.std(densities[-n_print:]) / np.sqrt(len(densities[-n_print:])),
        np.mean(S_AF_history[-n_print_AF:]), np.std(S_AF_history[-n_print_AF:]) / np.sqrt(len(S_AF_history[-n_print_AF:])),
        np.mean(SzSz_history[-n_print_AF:]), np.std(SzSz_history[-n_print_AF:]) / np.sqrt(len(SzSz_history[-n_print_AF:]))), flush = True)
    return


def perform_sweep(phi_field, K_matrix, orbits, switch):
    global accept_history, densities, S_AF_history, SzSz_history
    if switch:
        phi_field.copy_to_GPU()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()
    if switch:
        phi_field.copy_to_CPU()
    # print('assymetry = ', phi_field.get_assymetry_factor())


    for time_slice in range(phi_field.config.Nt):
        S_AF_history.append(cp.asnumpy(observables.staggered_magnetisation(phi_field)))
        SzSz_history.append(cp.asnumpy(observables.SzSz_n_neighbor(phi_field, K_matrix, 1)))
        if time_slice == 0:
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            # print('refreshed')
            # print('assymetry = ', phi_field.get_assymetry_factor(), time_slice)
        #t = time.time()
        # M_up_partial, B_time_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice, return_Bl = True)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        # M_down_partial, B_time_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice, return_Bl = True)
        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            if switch:
                phi_field.copy_to_GPU()
            t = time.time()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            # print('recomputation', phi_field.refresh_checkpoints[index - 1], time_slice)
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)
            phi_field.refresh_G_functions()
            if switch:
                phi_field.copy_to_CPU()
            # print('before', current_det_sign)
            current_det_log, current_det_sign = -phi_field.log_det_up -phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            # print('F computation from scratch took ' + str(time.time() - t))
            # print('after', current_det_sign)
            # print('refreshed')
            # print('assymetry = ', phi_field.get_assymetry_factor(), time_slice)
        # print('simple wrap-up')
        
        phi_field.wrap_up(time_slice)
        # G_up_check = phi_field.get_G_no_optimisation(+1, time_slice)[0]
        # G_down_check = phi_field.get_G_no_optimisation(-1, time_slice)[0]
        # print('discrepancy BEFORE step = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), \
        #                                      np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))
        #t = time.time()

        for sp_index in range(phi_field.config.total_dof // 2 * orbits):
            site_idx = sp_index // orbits ** 2
            o1 = sp_index % orbits
            o2 = (sp_index // orbits) % orbits

            t = time.time()

            sign_history.append(current_det_sign)
            #t = time.time()
            ratio = phi_field.get_det_ratio(+1, site_idx, time_slice, o1, o2) * \
                    phi_field.get_det_ratio(-1, site_idx, time_slice, o1, o2)
            #print('ratio of G took ' + str(time.time() - t))
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

            if lamb < np.min([1, np.abs(ratio)]):
                current_det_log += np.log(np.abs(ratio))
                # print(current_det_log)
                ratio_history.append(np.log(np.abs(ratio)))
                # print(np.mean(ratio_history))
                current_det_sign *= np.sign(ratio)
                accept_history.append(+1)
                #t = time.time()
                phi_field.update_G_seq(+1, site_idx, time_slice, o1, o2)
                phi_field.update_G_seq(-1, site_idx, time_slice, o1, o2)
                #print('update G took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                #t = time.time()
                phi_field.update_field(site_idx, time_slice, o1, o2)
                #print('update V took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                #print('update G took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                # print(current_det_sign)

                # G_up_check = phi_field.get_G_no_optimisation(+1, time_slice)[0]
                # G_down_check = phi_field.get_G_no_optimisation(-1, time_slice)[0]
                # print('discrepancy after one flip step = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))
            else:
                accept_history.append(0)
                ratio_history.append(0)
            #print('one slip on slice took ' + str(time.time() - t))
            t = time.time()
            densities.append((observables.total_density(phi_field)).item())
            # print('density ' + str(time.time() - t))
        # M_up = auxiliary_field.fermionic_matrix(configuration, K_operator, +1.0, config, time = time_slice)  # returns the product B_{l - 1} B_{l - 2}... B_0 B_{n - 1} ... B_{l + 1} and B_l
        # M_down = auxiliary_field.fermionic_matrix(configuration, K_operator, -1.0, config, time = time_slice)        

        # G_up_check = phi_field.get_G_no_optimisation(+1, time_slice)[0]
        # G_down_check = phi_field.get_G_no_optimisation(-1, time_slice)[0]

        # print('final discrepancy after sweep = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))

        # print('on-slice sweep took ' + str(time.time() - t))
        #t = time.time()
    return phi_field


if __name__ == "__main__":
    print_greetings(config)

    current_n_flips = 1
    n_flipped = 0
    K_matrix = config.model(config.Ls, config.mu)
    K_operator = xp.asarray(scipy.linalg.expm(config.dt * K_matrix))
    K_operator_inverse = xp.asarray(scipy.linalg.expm(-config.dt * K_matrix))
    phi_field = config.field(config, K_operator, K_operator_inverse)
    for n_sweep in range(config.n_sweeps):
        t = time.time()
        accept_history = []
        current_field = perform_sweep(phi_field, K_matrix, config.n_orbitals, switch = True)
        print('sweep took ' + str(time.time() - t))
        print_generator_log(n_sweep, current_field, K_matrix)
