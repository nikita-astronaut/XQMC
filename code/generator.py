import numpy as np

xp = np  # by default the code is executed on the CPU
gpu = False
cp = np

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
S_AF_history = []
SzSz_history = []
def print_greetings(config):
    print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
          "lattice = {:d}^2 x {:d}".format(config.start_type, 1.0 / config.dt / config.Nt, config.mu, config.Ls, config.Nt))
    print("# iteration current_flips N_swipes <log(ratio)> d<log(ratio)> <acceptance> <sign> d<sign> <density> <S_AF> <K> <Sz(0)Sz(0)> <Sz(0)Sz(1)> <Sz(0)Sz(2)> <Sz(0)Sz(3)> <Sz(0)Sz(4)> <Sz(0)Sz(5)> <n_up(0) n_down(0)> <n_up(0) n_down(1)> <n_up(0) n_down(2)> <n_up(0) n_down(3)> <n_up(0) n_down(4)>")
    return

def print_generator_log(generator_iteration, h_field, K_matrix, K, config, accept_history, sign_history, ratio_history, current_n_flips, n_flipped):
    if generator_iteration % config.n_print_frequency != 0:
        return
    n_print = np.min([generator_iteration, config.n_smoothing])
    n_history = np.min([n_print, len(ratio_history)])
    S_AF_history.append(cp.asnumpy(observables.staggered_magnetisation(h_field, K, config)))
    print("{:d}, {:d}, {:.3f}, {:.3f} +/- {:.3f}, {:.3f}, {:.3f} +/- {:.3f}, {:.5f}, {:.7f} +/- {:.7f}, {:.12f}, {:.12f}, {:.12f}, {:.7f}, {:.7f}, {:.7f}, {:.7f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(generator_iteration, current_n_flips, \
        n_flipped * 1.0 / config.Ls ** 2 / config.Nt / 2 / config.n_orbitals, \
        np.mean(ratio_history[-n_history:]), np.std(ratio_history[-n_history:]), \
        np.mean(accept_history[generator_iteration-n_print:generator_iteration]), \
        np.mean(sign_history[-n_print:]), np.std(sign_history[-n_print:]), \
        cp.asnumpy(observables.total_density(h_field, K, config)), \
        np.array(S_AF_history[-n_print:]).mean(), np.array(S_AF_history[-n_print:]).std() / np.sqrt(len(S_AF_history[-n_print:])), \
        cp.asnumpy(observables.kinetic_energy(h_field, K, K_matrix, config)), \
        cp.asnumpy(observables.SzSz_onsite(h_field, K, config)), \
        cp.asnumpy(observables.SzSz_n_neighbor(h_field, K, K_matrix, config, 1)),
        cp.asnumpy(observables.SzSz_n_neighbor(h_field, K, K_matrix, config, 2)),
        cp.asnumpy(observables.SzSz_n_neighbor(h_field, K, K_matrix, config, 3)),
        cp.asnumpy(observables.SzSz_n_neighbor(h_field, K, K_matrix, config, 4)),
        cp.asnumpy(observables.SzSz_n_neighbor(h_field, K, K_matrix, config, 5)),
        cp.asnumpy(observables.double_occupancy(h_field, K, config)), \
        cp.asnumpy(observables.double_occupancy_n_neighbor(h_field, K, K_matrix, config, 1)),
        cp.asnumpy(observables.double_occupancy_n_neighbor(h_field, K, K_matrix, config, 2)),
        cp.asnumpy(observables.double_occupancy_n_neighbor(h_field, K, K_matrix, config, 3)),
        cp.asnumpy(observables.double_occupancy_n_neighbor(h_field, K, K_matrix, config, 4))), flush = True)
    return

if __name__ == "__main__":
    print_greetings(config)
    accept_history = np.zeros(config.n_generator)
    sign_history = []
    ratio_history = []

    current_n_flips = 1
    n_flipped = 0
    K_matrix = config.model(config.Ls, config.mu)
    K_operator = xp.asarray(scipy.linalg.expm(config.dt * K_matrix))

    current_field = xp.asarray(auxiliary_field.get_initial_field_configuration(config))
    current_det_log, current_det_sign = auxiliary_field.get_det(current_field, K_operator, config)

    for generator_iteration in range(config.n_generator):
        sign_history.append(current_det_sign)
        proposed_field = auxiliary_field.flip_random_spin(1.0 * current_field, config, current_n_flips)  # 1.0 -- to ensure the copy instead of passing by pointer
        proposed_det_log, proposed_det_sign = auxiliary_field.get_det(proposed_field, K_operator, config)
        ratio = np.min([1, np.exp(proposed_det_log - current_det_log)])
        # ratio_history.append(proposed_det_log - current_det_log)
        lamb = np.random.uniform(0, 1)
        if lamb < ratio:
            ratio_history.append(proposed_det_log - current_det_log)
            current_field = 1.0 * proposed_field  # ensure copy here
            current_det_log = proposed_det_log
            current_det_sign = proposed_det_sign
            accept_history[generator_iteration] = +current_n_flips
            n_flipped += current_n_flips

        #if generator_iteration % 1000 == 0:
        #    if accept_history[generator_iteration-1000:generator_iteration].mean() > 0.35:
        #        current_n_flips += 1
        #    if accept_history[generator_iteration-1000:generator_iteration].mean() < 0.25 and current_n_flips > 1:
        #        current_n_flips -= 1
        print_generator_log(generator_iteration, current_field, K_matrix, K_operator, config, accept_history, sign_history, ratio_history, current_n_flips, n_flipped)
