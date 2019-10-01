import numpy as np

xp = np  # by default the code is executed on the CPU
gpu = False
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
    gpu = True
except ImportError:
    print('No CuPy found in the system, running on a CPU')


import scipy.linalg
import scipy.sparse as scs
import time
import auxiliary_field
import observables
from config_generator import simulation_parameters

config = simulation_parameters()

def print_greetings(config):
    print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
          "lattice = {:d}^2 x {:d}".format(config.start_type, 1.0 / config.dt / config.Nt, config.mu, config.Ls, config.Nt))
    print("# iteration <log(ratio)> d<log(ratio)> <acceptance> <sign> d<sign> <density>")
    return

def print_generator_log(generator_iteration, h_field, K, config, accept_history, sign_history, ratio_history):
    if generator_iteration % config.n_print_frequency != 0:
        return
    n_print = np.min([generator_iteration, config.n_smoothing])
    print("{:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(generator_iteration, np.mean(ratio_history[-n_print:]), np.std(ratio_history[-n_print:]), \
                                                                         np.mean(accept_history[-n_print:]), np.mean(sign_history[-n_print:]), np.std(sign_history[-n_print:]), \
                                                                         observables.total_density(h_field, K, config)),
                                                                         observables.staggered_magnetisation(h_field, K, config))
    return

if __name__ == "__main__":
    print_greetings(config)
    accept_history = []
    sign_history = []
    ratio_history = []

    K_matrix = config.model(config.Ls, config.mu)
    K_operator = xp.asarray(scipy.linalg.expm(config.dt * K_matrix))

    current_field = xp.asarray(auxiliary_field.get_initial_field_configuration(config))
    current_det_log, current_det_sign = auxiliary_field.get_det(current_field, K_operator, config)

    for generator_iteration in range(config.n_generator):
        sign_history.append(current_det_sign)
        proposed_field = auxiliary_field.flip_random_spin(1.0 * current_field, config)  # 1.0 -- to ensure the copy instead of passing by pointer
        proposed_det_log, proposed_det_sign = auxiliary_field.get_det(proposed_field, K_operator, config)

        ratio = np.min([1, np.exp(proposed_det_log - current_det_log)])
        ratio_history.append(proposed_det_log - current_det_log)
        lamb = np.random.uniform(0, 1)
        if lamb < ratio:
            current_field = 1.0 * proposed_field  # ensure copy here
            current_det_log = proposed_det_log
            current_det_sign = proposed_det_sign
            accept_history.append(+1)
        else:
            accept_history.append(0)

        print_generator_log(generator_iteration, current_field, K_operator, config, accept_history, sign_history, ratio_history)