import numpy as np
from wavefunction_vmc import wavefunction_singlet, get_wf_ratio, get_Jastrow_ratio, get_det_ratio, get_wf_ratio_double_exchange
from copy import deepcopy

def compare_derivatives_numerically(wf_1, wf_2, der_idx, dt):
    der_numerically = 2 * (np.abs(wf_2.current_ampl) - np.abs(wf_1.current_ampl)) / dt / (np.abs(wf_1.current_ampl) + np.abs(wf_2.current_ampl))
    der_analytically = 0.5 * wf_1.get_O()[der_idx] + 0.5 * wf_2.get_O()[der_idx]

    if np.abs(der_analytically) < 1e-8 and np.abs(der_numerically) < 1e-8:
        return True

    result = np.abs(der_numerically - der_analytically.real) / np.abs(der_numerically) < 1e-5
    if not result:
        print('Warning! The numerical derivative w.r. to one of the parameters did not match the analytical expression! :', der_numerically, der_analytically)
    return result

def perform_explicit_factors_check(config):
    # np.random.seed(14)
    wf = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], \
                              config.initial_sdw_parameters, config.initial_cdw_parameters,
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)

    delta = wf.Jastrow - wf.Jastrow.T
    success = True
    print('Testing the Jastrow matrix is symmetric')
    if np.sum(np.abs(delta)) < 1e-10:
        print('Passed')
    else:
        print('Failed:', np.sum(np.abs(delta)))

    det_initial = wf.get_cur_det()
    Jastrow_initial = wf.get_cur_Jastrow_factor()

    acc = False
    ddet = 1.
    dJastrow = 1.
    while not acc:
        state = deepcopy((wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy))
        acc, ddet, dJastrow, moved_site, empty_site = wf.perform_MC_step()
        # ddet = get_det_ratio(*state, moved_site, empty_site)
        # dJastrow = get_Jastrow_ratio(wf.Jastrow, wf.occupancy, wf.state, moved_site, empty_site)
    wf.perform_explicit_GF_update()
    det_final = wf.get_cur_det()
    Jastrow_final = wf.get_cur_Jastrow_factor()

    print('Testing the det(U_ini) / det(U_fin) ratio')
    if np.abs(ddet - det_final / det_initial) / np.abs(ddet) < 1e-11:
        print('Passed:', ddet, det_final / det_initial)
    else:
        print('Failed:', ddet, det_final / det_initial)
        success = False

    print('Testing the Jastrow(U_ini) / Jastrow(U_fin) ratio')
    if np.abs(dJastrow - Jastrow_final / Jastrow_initial) / np.abs(dJastrow) < 1e-11:
        print('Passed:', dJastrow, Jastrow_final / Jastrow_initial)
    else:
        print('Failed:', dJastrow, Jastrow_final / Jastrow_initial)
        success = False
    return success


def perform_numerical_derivative_check(config):
    dt = 1e-6
    success = True

    print('chemical potential derivative check...')
    np.random.seed(11)
    wf_1 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters - dt / 2], 
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    np.random.seed(11)
    wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters + dt / 2], 
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 
    der_idx = 2
    if compare_derivatives_numerically(wf_1, wf_2, 0, dt):
        print('Passed')
    else:
        print('Failed!')
        success = False

    print('SDW derivative check...')
    n_passed = 0
    for sdw_idx in range(len(config.initial_sdw_parameters)):
        delta = np.zeros(len(config.initial_sdw_parameters))
        delta[sdw_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            [config.initial_mu_parameters], 
            config.initial_sdw_parameters - dt / 2 * delta, config.initial_cdw_parameters,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], 
            config.initial_sdw_parameters + dt / 2 * delta, config.initial_cdw_parameters,
            config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 
        der_idx = 2
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, sdw_idx + 1, dt))

    if n_passed == len(config.initial_sdw_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False

    print('CDW derivative check...')
    n_passed = 0
    for cdw_idx in range(len(config.initial_cdw_parameters)):
        delta = np.zeros(len(config.initial_cdw_parameters))
        delta[cdw_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters - dt / 2 * delta,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters + dt / 2 * delta,
            config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 
        der_idx = 2
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, cdw_idx + 1 + 
                          len(config.initial_sdw_parameters), dt))

    if n_passed == len(config.initial_cdw_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False

    print('Pairings derivative check...')
    n_passed = 0
    for gap_idx in range(len(config.initial_gap_parameters)):
        delta = np.zeros(len(config.initial_gap_parameters))
        delta[gap_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters - dt / 2 * delta, config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters + dt / 2 * delta, config.initial_jastrow_parameters, False, None) 
        der_idx = 2
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, gap_idx + 1 + 
                          len(config.initial_sdw_parameters) + len(config.initial_cdw_parameters), dt))
    if n_passed == len(config.initial_gap_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False

    print('Jastrow derivative check...')
    n_passed = 0
    for jastrow_idx in range(len(config.initial_jastrow_parameters)):
        delta = np.zeros(len(config.initial_jastrow_parameters))
        delta[jastrow_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters - dt / 2 * delta, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], 
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters, config.initial_jastrow_parameters + dt / 2 * delta, False, None) 
        der_idx = 2
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, jastrow_idx + 1 + 
                          len(config.initial_gap_parameters) + len(config.initial_sdw_parameters) + 
                          len(config.initial_cdw_parameters), dt))

    if n_passed == len(config.initial_jastrow_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False

    return success

def perform_single_move_check(config):
    success = True
    print('Testing simple moves ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i, j = np.random.randint(0, 2 * L, size = 2)

        initial_ampl = wf.current_ampl
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
        ratio_fast = get_wf_ratio(*state, i, j)
        
        acc = wf.perform_MC_step((i, j), enforce = True)[0]
        if not acc:
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl

        if (np.abs(final_ampl / initial_ampl - ratio_fast) < 1e-11):
            n_agreed += 1
        else:
            print('single move check ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩ failed:', final_ampl / initial_ampl, ratio_fast)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success


def perform_onsite_gf_is_density_check(config):
    success = True
    print('Testing ⟨x|d^{\\dag}_i d_i|Ф⟩ / ⟨x|Ф⟩ = n_i')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters], \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i = np.random.randint(0, 2 * L)

        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
        gf = get_wf_ratio(*state, i, i)

        density = float(wf.place_in_string[i] > -1)
        
        if np.abs(density - gf) < 1e-11:
            n_agreed += 1
        else:
            print('Testing ⟨x|d^{\\dag}_i d_i|Ф⟩ / ⟨x|Ф⟩ = n_i failed:', density, gf, i)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success



def perform_double_move_check(config):
    success = True
    print('Testing double moves ⟨x|d_{j + L} d^{\\dag}_i d_k d^{\\dag}_{l + L}|Ф⟩ / ⟨x|Ф⟩')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, [config.initial_mu_parameters],
                              config.initial_sdw_parameters, config.initial_cdw_parameters,
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i, j, k, l = np.random.randint(0, L, size = 4)
        if i == k:  # I am tired :)) this case is already considered everywhere
            continue
        fillings = wf.place_in_string[j + L] > -1, wf.place_in_string[k] > -1

        initial_ampl = wf.current_ampl
        state = deepcopy((wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy))
        ratio_fast = get_wf_ratio_double_exchange(*state, i, j, k, l)
        
        W_ik_0 = get_wf_ratio(*state, i, k)
        acc = wf.perform_MC_step(proposed_move = (i, k), enforce = True)[0]

        if not acc:
            continue
        wf.perform_explicit_GF_update()
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

        middle_ampl = wf.current_ampl

        W_lj_upd = get_wf_ratio(*state, l + L, j + L)

        if j == l:
            if wf.place_in_string[j + L] > -1:
                ratio_check = 0
            else:
                ratio_check = W_ik_0
        else:
            ratio_check = -W_lj_upd * W_ik_0

        acc = wf.perform_MC_step(proposed_move = (l + L, j + L), enforce = True)[0]
        if not acc:
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl

        ratio_straight = -final_ampl / initial_ampl
        if j == l:
            if wf.place_in_string[j + L] > -1:
                ratio_straight = 0
            else:
                ratio_straight = middle_ampl / initial_ampl

        if (np.abs(ratio_fast - ratio_straight) / np.abs(ratio_fast) < 1e-11) and (np.abs(ratio_fast - ratio_check) / np.abs(ratio_fast) < 1e-11):
            n_agreed += 1
        else:
            print('double move check ⟨x|d_{j + L} d^{\\dag}_i d_k d^{\\dag}_{l + L}|Ф⟩ / ⟨x|Ф⟩ failed:', ratio_fast, ratio_straight, ratio_check, i, j, k, l)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success

def perform_all_tests(config):
    success = True
    success = success and perform_explicit_factors_check(config)
    success = success and perform_double_move_check(config)
    success = success and perform_single_move_check(config)
    success = success and perform_numerical_derivative_check(config)
    success = success and perform_onsite_gf_is_density_check(config)

    return success