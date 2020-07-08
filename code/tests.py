import numpy as np
from wavefunction_vmc import wavefunction_singlet, get_wf_ratio, get_det_ratio, get_wf_ratio_double_exchange
from copy import deepcopy
from time import time
import models

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
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    for MC_step in range(config.MC_thermalisation):
        wf.perform_MC_step()
    wf.perform_explicit_GF_update()

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
        if not np.isclose(ddet, det_final / det_initial, atol = 1e-8, rtol = 1e-8):
            print('Det ratio failed:', ddet, det_final / det_initial, moved_site, empty_site)
            success = False

        if not np.isclose(dJastrow, Jastrow_final / Jastrow_initial, rtol=1e-10, atol=1e-10):
            print('Jastrow ratio failed:', dJastrow, Jastrow_final / Jastrow_initial, moved_site, empty_site)
            success = False

    if success:
        print('Passed')

    return success


def test_particle_hole(config):
    success = True
    parameters = config.initial_parameters
    parameters[:config.layout[0]] *= 0  # set mu_BCS = 0, otherwise no ph-symmetry
    #parameters *= 0.

    print('Particle-hole symmetry of the hamiltonian with twist check...') 
    wf_ph = wavefunction_singlet(config, config.pairings_list, parameters, False, None, particle_hole=False)
    HMF = wf_ph.T
    spectrum, _ = np.linalg.eigh(HMF)
    if np.allclose(spectrum, -spectrum[::-1]):
        print('Passed')
    else:
        success = False
        print('Failed')

    print('Particle-hole symmetry of the wave function check...')
    n_passed = 0
    
    for _ in range(200):
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)

        config.twist = [np.exp(2.0j * np.pi * 0.1904 * 1e-3), np.exp(2.0j * np.pi * (0.1904 + 0.10) * 1e-3)]
        wf_ph = wavefunction_singlet(config, config.pairings_list, parameters, False, None, particle_hole=False)

        np.random.seed(seed)
        #config.twist = [np.exp(-2.0j * np.pi * 0.1904), np.exp(-2.0j * np.pi * (0.1904 + 0.10))]
        wf_hp = wavefunction_singlet(config, config.pairings_list, parameters, False, None, particle_hole=True)

        n_passed += float(np.isclose(wf_ph.current_ampl, wf_hp.current_ampl))
        if not (np.abs(np.abs(wf_ph.current_ampl / wf_hp.current_ampl) - 1.0) < 1e-8):
            print('Failed', wf_ph.current_ampl / wf_hp.current_ampl)
            print('Failed', wf_ph.current_det / wf_hp.current_det)
        else:
            print('Passed', np.angle(wf_ph.current_ampl / wf_hp.current_ampl))
    if n_passed == 200:
        print('Passed')
    else:
        print('Failed!')
        success = False
    return success


def test_numerical_derivative_check(config):
    print(config.twist)
    dt = 1e-7
    success = True
    der_shift = 0

    print('chemical potentials derivative check...')
    n_passed = 0
    for mu_idx in range(config.layout[0]):
        np.random.seed(11)

        delta = np.zeros(len(config.initial_parameters)); delta[der_shift] += 1
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters - delta * dt / 2, False, None)

        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters + delta * dt / 2, False, None)
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, der_shift, dt))
        der_shift += 1

    if n_passed == config.layout[0]:
        print('Passed')
    else:
        print('Failed!')
        success = False


    print('fugacity derivative check...')
    if not config.PN_projection:
        np.random.seed(11)
        delta = np.zeros(len(config.initial_parameters)); delta[der_shift] += 1
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters - delta * dt / 2, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters + delta * dt / 2, False, None)

        if compare_derivatives_numerically(wf_1, wf_2, der_shift, dt):
            print('Passed')
        else:
            print('Failed!')
            success = False

    der_shift += config.layout[1]

    print('waves derivative check...')
    n_passed = 0
    for waves_idx in range(config.layout[2]):
        np.random.seed(11)
        delta = np.zeros(len(config.initial_parameters)); delta[der_shift] += 1
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters - delta * dt / 2, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters + delta * dt / 2, False, None)
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, der_shift, dt))
        der_shift += 1
        

    if n_passed == config.layout[2]:
        print('Passed')
    else:
        print('Failed!')
        success = False


    print('Pairings derivative check...')
    n_passed = 0
    for gap_idx in range(config.layout[3]):
        np.random.seed(11)
        delta = np.zeros(len(config.initial_parameters)); delta[der_shift] += 1
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters - delta * dt / 2, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters + delta * dt / 2, False, None)
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, der_shift, dt))
        der_shift += 1

    if n_passed == config.layout[3]:
        print('Passed')
    else:
        print('Failed!')
        success = False


    print('Jastrow derivative check...')
    n_passed = 0
    for jastrow_idx in range(config.layout[4]):
        np.random.seed(11)
        delta = np.zeros(len(config.initial_parameters)); delta[der_shift] += 1
        wf_1 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters - delta * dt / 2, False, None)
        np.random.seed(11)
        wf_2 = wavefunction_singlet(config, config.pairings_list, config.initial_parameters + delta * dt / 2, False, None)
        n_passed += float(compare_derivatives_numerically(wf_1, wf_2, der_shift, dt))
        der_shift += 1

    if n_passed == config.layout[4]:
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
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    #for MC_step in range(config.MC_thermalisation):
    #    wf.perform_MC_step()
    #wf.perform_explicit_GF_update()

    for _ in range(200):
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        L = len(wf.state) // 2
        i, j = np.random.randint(0, 2 * L, size = 2)

        initial_ampl = wf.current_ampl
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
        ratio_fast = get_wf_ratio(*state, wf.var_f, i, j)
        
        acc = wf.perform_MC_step((i, j), enforce = False)[0]
        if not acc:
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl
        final_ampl_solid = wf.get_cur_Jastrow_factor() * wf.get_cur_det()

        if np.isclose(final_ampl / initial_ampl, ratio_fast) and np.isclose(final_ampl_solid, final_ampl):
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

def test_delayed_updates_check(config):
    success = True
    print('Testing delayed updates')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    for _ in range(200):
        initial_ampl = wf.current_ampl
        for step in range(10):
            wf.perform_MC_step()
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl
        final_ampl_solid = wf.get_cur_Jastrow_factor() * wf.get_cur_det()

        if np.isclose(final_ampl_solid, final_ampl):
            n_agreed += 1
        else:
            print('Delayed updates test failed:', final_ampl, final_ampl_solid)
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
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
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

def test_all_jastrow_factors_included_only_once(config):
    print('Testing all Jastrow correlations included only once')
    factors = config.adjacency_list
    success = True

    result = np.zeros(factors[0][0].shape)
    for A in factors:
        result += A[0]

    if np.sum(result > 1) == 0:
        print('Passed', np.unique(result))
    else:
        print('Failed')
        success = False

    return success

def test_double_move_check(config):
    success = True
    print('Testing double moves ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩')
    n_agreed = 0
    n_failed = 0
    # wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    while n_agreed < 100:
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        L = len(wf.state) // 2
        i, j, k, l = np.random.randint(0, 2 * L, size = 4)
        #if i == j or i == l or k == l or k == j:
        #    continue  # the degenerate cases are considered separately (directly by density operator)

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


def test_double_move_commutation_check(config):
    success = True
    print('Testing fast double updates have correct commutation properties...')
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

    while n_agreed < 1000:
        L = len(wf.state) // 2
        i, j, k, l = np.random.randint(0, 2 * L, size = 4)
        if len(np.unique([i, j, k, l])) < 4:
            continue
        ratio_fast_ijkl = get_wf_ratio_double_exchange(*state, wf.var_f, i, j, k, l)
        ratio_fast_ilkj = get_wf_ratio_double_exchange(*state, wf.var_f, i, l, k, j)
        ratio_fast_kjil = get_wf_ratio_double_exchange(*state, wf.var_f, k, j, i, l)
        ratio_fast_lkij = get_wf_ratio_double_exchange(*state, wf.var_f, k, l, i, j)

        if np.allclose([ratio_fast_ilkj, ratio_fast_kjil, ratio_fast_lkij], \
                       [-ratio_fast_ijkl, -ratio_fast_ijkl, ratio_fast_ijkl], atol = 1e-11, rtol = 1e-11):
            n_agreed += 1
            # print('double move check permutation ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩ fine:', \
            #        ratio_fast_ijkl, ratio_fast_ilkj, ratio_fast_kjil, ratio_fast_lkij, i, j, k, l)
        else:
            print('double move check permutation ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩ failed:', \
                   ratio_fast_ijkl, ratio_fast_ilkj, ratio_fast_kjil, ratio_fast_lkij, i, j, k, l)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success

def test_BC_twist(config):
    print('Testing BC twist with twist ' + str(config.twist))
    for name, gap in zip(config.pairings_list_names, config.pairings_list_unwrapped):
        assert np.allclose(gap, models.apply_TBC(config, config.twist, models.apply_TBC(config, config.twist, deepcopy(gap), inverse = False), inverse=True))
        print('Passed {:s}'.format(name))
    return True


def perform_all_tests(config):
    success = True
    success = success and test_particle_hole(config)
    success = success and test_BC_twist(config)
    success = success and test_all_jastrow_factors_included_only_once(config)
    success = success and test_explicit_factors_check(config)
    success = success and test_double_move_commutation_check(config)
    success = success and test_single_move_check(config)
    success = success and test_delayed_updates_check(config)
    success = success and test_onsite_gf_is_density_check(config)
    success = success and test_numerical_derivative_check(config)
    # success = success and test_double_move_check(config)
    return success
