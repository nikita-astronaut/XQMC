import numpy as np
from wavefunction_vmc import wavefunction_singlet, get_wf_ratio, get_det_ratio, get_wf_ratio_double_exchange
from copy import deepcopy

def compare_derivatives_numerically(wf_1, wf_2, der_idx, dt):
    der_numerically = 2 * (np.abs(wf_2.current_ampl) - np.abs(wf_1.current_ampl)) / dt / (np.abs(wf_1.current_ampl) + np.abs(wf_2.current_ampl))
    der_analytically = 0.5 * wf_1.get_O()[der_idx] + 0.5 * wf_2.get_O()[der_idx]

    if np.abs(der_analytically) < 1e-6 and np.abs(der_numerically) < 1e-6:
        return True

    result = np.isclose(der_numerically, der_analytically.real, rtol=1e-5, atol=1e-5)
    if not result:
        print('Warning! The numerical derivative w.r. to one of the parameters did not match the analytical expression! :', der_numerically, der_analytically)
    else:
        print('Passed: {:.5f} / {:.5f}'.format(der_numerically, der_analytically.real)) 

    return result

def test_explicit_factors_check(config):
    # np.random.seed(14)
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, \
                              config.initial_fugacity_parameters, \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)

    delta = np.sum(np.abs(wf.Jastrow - wf.Jastrow.T))
    success = True
    print('Testing the Jastrow matrix is symmetric')
    if np.isclose(delta, 0.0, rtol=1e-11, atol=1e-11):
        print('Passed')
    else:
        print('Failed:', np.sum(np.abs(delta)))


    print('Testing det and jastrow factors')
    for _ in range(100):
        det_initial = wf.get_cur_det()
        Jastrow_initial = wf.get_cur_Jastrow_factor()
        
        acc = False
        ddet = 1.
        dJastrow = 1.
        while not acc:
            state = deepcopy((wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy))
            acc, ddet, dJastrow, moved_site, empty_site = wf.perform_MC_step()
        wf.perform_explicit_GF_update()
        det_final = wf.get_cur_det()
        Jastrow_final = wf.get_cur_Jastrow_factor()
        if not np.isclose(ddet, det_final / det_initial, atol = 1e-10, rtol = 1e-10):
            print('Det ratio failed:', ddet, det_final / det_initial, moved_site, empty_site)
            success = False

        if not np.isclose(dJastrow, Jastrow_final / Jastrow_initial, rtol=1e-10, atol=1e-10):
            print('Jastrow ratio failed:', dJastrow, Jastrow_final / Jastrow_initial, moved_site, empty_site)
            success = False

    if success:
        print('Passed')

    return success


def test_numerical_derivative_check(config):
    dt = 1e-6
    success = True
    der_shift = 0

    print('chemical potential derivative check...')
    np.random.seed(11)
    wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters - dt / 2, 
        config.initial_fugacity_parameters, \
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    np.random.seed(11)
    wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters + dt / 2, 
        config.initial_fugacity_parameters, \
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 

    if compare_derivatives_numerically(wf_1, wf_2, der_shift, dt):
        print('Passed')
    else:
        print('Failed!')
        success = False

    der_shift += 1
    print('fugacity derivative check...')
    np.random.seed(11)
    wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
        config.initial_fugacity_parameters - dt / 2, \
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    np.random.seed(11)
    wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
        config.initial_fugacity_parameters + dt / 2, \
        config.initial_sdw_parameters, config.initial_cdw_parameters,
        config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 

    if compare_derivatives_numerically(wf_1, wf_2, der_shift, dt):
        print('Passed')
    else:
        print('Failed!')
        success = False
    der_shift += 1

    print('SDW derivative check...')
    n_passed = 0
    for sdw_idx in range(len(config.initial_sdw_parameters)):
        delta = np.zeros(len(config.initial_sdw_parameters))
        delta[sdw_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            config.initial_mu_parameters, config.initial_fugacity_parameters, \
            config.initial_sdw_parameters - dt / 2 * delta, config.initial_cdw_parameters,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
            config.initial_fugacity_parameters, \
            config.initial_sdw_parameters + dt / 2 * delta, config.initial_cdw_parameters,
            config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 

        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, sdw_idx + der_shift, dt))

    if n_passed == len(config.initial_sdw_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False
    der_shift += len(config.initial_sdw_parameters)

    print('CDW derivative check...')
    n_passed = 0
    for cdw_idx in range(len(config.initial_cdw_parameters)):
        delta = np.zeros(len(config.initial_cdw_parameters))
        delta[cdw_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            config.initial_mu_parameters, config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters - dt / 2 * delta,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
            config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters + dt / 2 * delta,
            config.initial_gap_parameters, config.initial_jastrow_parameters, False, None) 

        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, cdw_idx + der_shift, dt))

    if n_passed == len(config.initial_cdw_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False
    der_shift += len(config.initial_cdw_parameters)


    print('Pairings derivative check...')
    n_passed = 0
    for gap_idx in range(len(config.initial_gap_parameters)):
        delta = np.zeros(len(config.initial_gap_parameters))
        delta[gap_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
            config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters - dt / 2 * delta, config.initial_jastrow_parameters, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
            config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters + dt / 2 * delta, config.initial_jastrow_parameters, False, None) 

        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, gap_idx + der_shift, dt))
    if n_passed == len(config.initial_gap_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False
    der_shift += len(config.initial_gap_parameters)

    print('Jastrow derivative check...')
    n_passed = 0
    for jastrow_idx in range(len(config.initial_jastrow_parameters)):
        delta = np.zeros(len(config.initial_jastrow_parameters))
        delta[jastrow_idx] = 1
        np.random.seed(11)
        wf_1 = wavefunction_singlet(config, config.pairings_list, 
            config.initial_mu_parameters, config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters, 
            config.initial_jastrow_parameters - dt / 2 * delta, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, 
            config.initial_fugacity_parameters, \
            config.initial_sdw_parameters, config.initial_cdw_parameters,
            config.initial_gap_parameters, config.initial_jastrow_parameters + dt / 2 * delta, False, None) 

        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, jastrow_idx + der_shift, dt))

    if n_passed == len(config.initial_jastrow_parameters):
        print('Passed')
    else:
        print('Failed!')
        success = False

    return success

def test_single_move_check(config):
    success = True
    print('Testing simple moves ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, \
                              config.initial_fugacity_parameters, \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i, j = np.random.randint(0, 2 * L, size = 2)

        initial_ampl = wf.current_ampl
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
        ratio_fast = get_wf_ratio(*state, wf.var_f, i, j)
        
        acc = wf.perform_MC_step((i, j), enforce = True)[0]
        if not acc:
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl

        if np.isclose(final_ampl / initial_ampl, ratio_fast):
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


def test_onsite_gf_is_density_check(config):
    success = True
    print('Testing ⟨x|d^{\\dag}_i d_i|Ф⟩ / ⟨x|Ф⟩ = n_i')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, \
                              config.initial_fugacity_parameters, \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i = np.random.randint(0, 2 * L)

        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
        gf = get_wf_ratio(*state, wf.var_f, i, i)

        density = float(wf.place_in_string[i] > -1)
        
        if np.isclose(density, gf, atol = 1e-11, rtol = 1e-11):
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

def test_all_jastrow_factors_add_to_one(config):
    print('Testing all Jastrow correlations included only once')
    factors = config.adjacency_list
    success = True

    result = np.zeros(factors[0][0].shape)
    for A in factors:
        result += A[0]

    if np.allclose(result, np.ones(A[0].shape)):
        print('Passed')
    else:
        print('Failed')
        success = False

    return success

def test_double_move_check(config):
    success = True
    print('Testing double moves ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_mu_parameters, \
                              config.initial_fugacity_parameters, \
                              config.initial_sdw_parameters, config.initial_cdw_parameters, \
                              config.initial_gap_parameters, config.initial_jastrow_parameters, False, None)
    while n_agreed < 5:
        L = len(wf.state) // 2
        i, j, k, l = np.random.randint(0, 2 * L, size = 4)
        if i == j or i == l or k == l or k == j:
            continue  # the degenerate cases are considered separately (directly by density operator)

        initial_ampl = wf.current_ampl
        state = deepcopy((wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy))
        ratio_fast = get_wf_ratio_double_exchange(*state, wf.var_f, i, j, k, l)
        
        W_ij_0 = get_wf_ratio(*state, wf.var_f, i, j)
        acc = wf.perform_MC_step(proposed_move = (i, j), enforce = True)[0]

        if not acc:
            continue
        wf.perform_explicit_GF_update()
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

        middle_ampl = wf.current_ampl

        W_kl_upd = get_wf_ratio(*state, wf.var_f, k, l)
        ratio_check = W_kl_upd * W_ij_0

        acc = wf.perform_MC_step(proposed_move = (k, l), enforce = True)[0]
        if not acc:
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl

        ratio_straight = final_ampl / initial_ampl

        if np.allclose([ratio_fast, ratio_fast], [ratio_straight, ratio_check], atol = 1e-11, rtol = 1e-11):
            n_agreed += 1
        else:
            print('double move check ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩ failed:', ratio_fast / ratio_straight, ratio_straight, ratio_check, i, j, k, l)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success

def perform_all_tests(config):
    success = True
    success = success and test_all_jastrow_factors_add_to_one(config)
    success = success and test_explicit_factors_check(config)
    success = success and test_double_move_check(config)
    success = success and test_single_move_check(config)
    success = success and test_onsite_gf_is_density_check(config)
    success = success and test_numerical_derivative_check(config)
    return success
