import numpy as np
from wavefunction_vmc import wavefunction_singlet

def compare_derivatives_numerically(wf_1, wf_2, der_idx, dt):
    der_numerically = 2 * (np.abs(wf_2.current_ampl) - np.abs(wf_1.current_ampl)) / dt / (np.abs(wf_1.current_ampl) + np.abs(wf_2.current_ampl))
    der_analytically = 0.5 * wf_1.get_O()[der_idx] + 0.5 * wf_2.get_O()[der_idx]

    result = np.abs(der_numerically - der_analytically.real) < 1e-5
    if not result:
        print('Warning! The numerical derivative w.r. to one of the parameters did not match the analytical expression! :', der_numerically, der_analytically)
    return result


def perform_numerical_derivative_check(config):
    dt = 1e-6

    print('chemical potential derivative check...')
    np.random.seed(11)
    wf_1 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters - dt / 2], config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    np.random.seed(11)
    wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters + dt / 2], config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 
    der_idx = 2
    print(compare_derivatives_numerically(wf_1, wf_2, 0, dt))

    print('Pairings derivative check...')
    for gap_idx in range(len(config.initial_gap_parameters)):
        delta = np.zeros(len(config.initial_gap_parameters))
        delta[gap_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], config.initial_gap_parameters - dt / 2 * delta, config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], config.initial_gap_parameters + dt / 2 * delta, config.initial_jastrow_parameters, False, None) 
        der_idx = 2
        print(compare_derivatives_numerically(wf_1, wf_2, gap_idx + 1, dt))

    print('Jastrow derivative check...')
    for jastrow_idx in range(len(config.initial_jastrow_parameters)):
        delta = np.zeros(len(config.initial_jastrow_parameters))
        delta[jastrow_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], config.initial_gap_parameters, config.initial_jastrow_parameters - dt / 2 * delta, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], config.initial_gap_parameters, config.initial_jastrow_parameters + dt / 2 * delta, False, None) 
        der_idx = 2
        print(compare_derivatives_numerically(wf_1, wf_2, jastrow_idx + 1 + len(config.initial_gap_parameters), dt))
    return

    def perform_double_move_check(config):
        wf = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters - dt / 2], config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)