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
import observables as obs_methods
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

def print_generator_log(n_sweep, phi_field):
    global accept_history, sign_history, ratio_history
    if n_sweep % config.n_print_frequency != 0:
        return
    n_print = np.min([n_sweep * config.total_dof // 2 * config.Nt, config.n_smoothing])
    n_history = np.min([n_print, len(ratio_history)])
    
    print("{:d}, {:d}, {:.9f} +/- {:.9f}, {:.3f}, {:.3f} +/- {:.3f}".format(n_sweep, current_n_flips, \
        np.mean(ratio_history[-n_history:]), np.std(ratio_history[-n_history:]) / np.sqrt(len(ratio_history[-n_history:])), \
        np.mean(accept_history), \
        np.mean(sign_history[-n_print:]), np.std(sign_history[-n_print:]) / np.sqrt(len(sign_history[-n_print:]))), flush = True)
    return


def perform_sweep(phi_field, switch, observables_log, n_sweep):
    global accept_history
    if switch:
        phi_field.copy_to_GPU()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()
    if switch:
        phi_field.copy_to_CPU()
    # G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, -1)
    # G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, -1)
    # la = phi_field.la
    # print('discrepancy BEFORE even loop = ', la.sum(la.abs(phi_field.current_G_function_up - G_up_check)) / la.sum(la.abs(G_up_check)), \
    #                                          la.sum(la.abs(phi_field.current_G_function_down - G_down_check)) / la.sum(la.abs(G_down_check)))

    observables = []

    for time_slice in range(phi_field.config.Nt):
        if time_slice == 0:
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            # print('refreshed')
            # print('assymetry = ', phi_field.get_assymetry_factor(), time_slice)
        #t = time.time()
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
            # print('assymetry = ', phi_field.get_assymetry_factor(), time_slice)
        # t = time.time()
        phi_field.wrap_up(time_slice)
        # print('wrap-up took ' + str(time.time() - t))
        #phi_field.copy_to_GPU()
        # G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
        # G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, time_slice)
        # print('discrepancy BEFORE step = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check)), \
        #                                      np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check)))
        # print(current_det_log + det_log_up_check + det_log_down_check)
        # t = time.time()

        for sp_index in range(phi_field.config.total_dof // 4 * 3):
            site_idx = sp_index // 3
            o_index = sp_index % 3

            # t = time.time()

            sign_history.append(current_det_sign)
            # t = time.time()
            ratio = phi_field.get_det_ratio(+1, site_idx, time_slice, o_index) * \
                    phi_field.get_det_ratio(-1, site_idx, time_slice, o_index)
            # print('ratio of G took ' + str(time.time() - t))
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
                # print(np.log(np.abs(ratio)))
                # print(np.mean(ratio_history))
                current_det_sign *= np.sign(ratio)
                accept_history.append(+1)
                # t = time.time()
                phi_field.update_G_seq(+1, site_idx, time_slice, o_index)
                phi_field.update_G_seq(-1, site_idx, time_slice, o_index)
                # print('update G took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                # t = time.time()
                phi_field.update_field(site_idx, time_slice, o_index)
                # print('update V took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                #print('update G took ' + str(time.time() - t), phi_field.config.total_dof // 2 * orbits)
                # print(current_det_sign)

                # G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                # G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

                # print('final discrepancy after step = ', np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / 
                #                                          np.sum(np.abs(G_up_check)), 
                #                                          np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / 
                #                                         np.sum(np.abs(G_down_check)))
                ##print(current_det_log + det_log_up_check + det_log_down_check)
            else:
                accept_history.append(0)
                ratio_history.append(0)
            # print('one slip on slice took ' + str(time.time() - t))
            # t = time.time()
        obs, names = obs_methods.compute_all_observables(phi_field)
        observables.append(np.array(obs))
    if n_sweep == 0:
        for obs_name in names:
            observables_log.write(" ⟨" + obs_name + "⟩ ⟨d" + obs_name + "⟩")
        observables_log.write('\n')
    observables = np.array(observables)
    observables = np.concatenate([observables.mean(axis = 0)[:, np.newaxis], observables.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)

    observables_log.write(("{:4d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} " + " {:.5e}" * len(observables) + "\n").format(n_sweep, np.mean(ratio_history),
                            np.std(ratio_history) / np.sqrt(len(ratio_history)),
                            np.mean(accept_history),
                            np.mean(sign_history), np.std(sign_history) / np.sqrt(len(sign_history)),
                            *observables))
    observables_log.flush()

    return phi_field


if __name__ == "__main__":
    print_greetings(config)

    current_n_flips = 1
    n_flipped = 0
    K_matrix = config.model(config, config.mu)
    K_operator = scipy.linalg.expm(config.dt * K_matrix)
    K_operator_inverse = scipy.linalg.expm(-config.dt * K_matrix)
    phi_field = config.field(config, K_operator, K_operator_inverse)
    phi_field.copy_to_GPU()

    observables_log = open(config.observables_log_name + '_U_' + str(config.U) + '.dat', 'w')
    observables_log.write("⟨step⟩ ⟨ratio⟩ ⟨dratio⟩ ⟨acceptance⟩ ⟨sign⟩ ⟨dsign⟩ ")

    for n_sweep in range(config.n_sweeps):
        t = time.time()
        accept_history = []
        current_field = perform_sweep(phi_field, True, observables_log, n_sweep)
        print('sweep took ' + str(time.time() - t))
        print_generator_log(n_sweep, current_field)
