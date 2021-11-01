import numpy as np
from wavefunction_vmc import wavefunction_singlet, get_wf_ratio, get_det_ratio, get_wf_ratio_double_exchange
from copy import deepcopy
from time import time
import models
from visualisation import K_FT
from opt_parameters import pairings
from opt_parameters import jastrow
from numba import jit

def compare_derivatives_numerically(wf_1, wf_2, der_idx, dt):
    der_numerically = 2 * (np.abs(wf_2.current_ampl) - np.abs(wf_1.current_ampl)) / dt / (np.abs(wf_1.current_ampl) + np.abs(wf_2.current_ampl))
    der_analytically = 0.5 * wf_1.get_O()[der_idx] + 0.5 * wf_2.get_O()[der_idx]

    if np.abs(der_analytically) < 1e-6 and np.abs(der_numerically) < 1e-6:
        return True

    result = np.isclose(der_numerically, der_analytically.real, rtol=1e-5, atol=1e-5)
    if not result:
        print('Warning! The numerical derivative w.r. to one of the parameters did not match the analytical expression! :', der_numerically, der_analytically)
    else:
        print('Passed real: {:.5f} / {:.5f}'.format(der_numerically, der_analytically.real)) 


    U_1 = wf_1.U_matrix
    U_2 = wf_2.U_matrix
    extra_phase = np.angle(np.linalg.det(np.dot(U_1.T.conj(), U_2)))
    der_numerically_k = np.argmin([np.abs((np.angle(wf_2.current_ampl) - extra_phase - np.angle(wf_1.current_ampl) + 2 * np.pi * k) / dt) for k in range(-1, 2)])

    der_numerically = (np.angle(wf_2.current_ampl) - extra_phase - np.angle(wf_1.current_ampl) + 2 * np.pi * (-1 + der_numerically_k)) / dt
    der_analytically = 0.5 * wf_1.get_O()[der_idx] + 0.5 * wf_2.get_O()[der_idx]

    if np.abs(der_analytically) < 1e-6 and np.abs(der_numerically) < 1e-6:
        return True


    result = np.isclose(der_numerically, der_analytically.imag, rtol=1e-5, atol=1e-5)
    if not result:
        print('Warning! The numerical derivative w.r. to one of the parameters did not match the analytical expression! :', der_numerically, der_analytically)
    else:
        print('Passed imag: {:.5f} / {:.5f}'.format(der_numerically, der_analytically.imag))

    return result

def test_explicit_factors_check(config):
    # np.random.seed(14)
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    for MC_step in range(config.MC_thermalisation):
        wf.perform_MC_step()
    wf.perform_explicit_GF_update()

    delta = np.sum(np.abs(wf.Jastrow - wf.Jastrow.T))
    success = True
    print('Testing the Jastrow matrix is symmetric', flush=True)
    if np.isclose(delta, 0.0, rtol=1e-11, atol=1e-11):
        print('Passed')
    else:
        print('Failed:', np.sum(np.abs(delta)))


    print('Testing det and jastrow factors', flush=True)
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

    print('Particle-hole symmetry of the hamiltonian with twist check...', flush=True) 
    wf_ph = wavefunction_singlet(config, config.pairings_list, parameters, False, None, particle_hole=False)
    HMF = wf_ph.T
    spectrum, _ = np.linalg.eigh(HMF)
    if np.allclose(spectrum, -spectrum[::-1]):
        print('Passed')
    else:
        success = False
        print('Failed')

    print('Particle-hole symmetry of the wave function check...', flush=True)
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

    print('chemical potentials derivative check...', flush=True)
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

    '''
    print('fugacity derivative check...', flush=True)
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
    '''

    print('hoppings derivative check...', flush=True)
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


    print('Pairings derivative check...', flush=True)
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


    print('Jastrow derivative check...', flush=True)
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
    print('Testing simple moves ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩', flush=True)
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    #for MC_step in range(config.MC_thermalisation):
    #    wf.perform_MC_step()
    #wf.perform_explicit_GF_update()

    for _ in range(20):
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
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


def test_gf_means_correct(config):
    success = True
    print('Testing Greens function ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩', flush=True)
    n_agreed = 0
    n_failed = 0

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



def test_gf_symmetry(config):
    print('Testing WF symmetry ⟨x|S|Ф⟩ = U ⟨x|Ф⟩', flush=True)
    n_agreed = 0
    n_failed = 0

    C3z = np.argmax(np.abs(pairings.C3z_symmetry_map_chiral), axis = 0)
    C2y = np.argmax(np.abs(pairings.C2y_symmetry_map_chiral), axis = 0)
    Tx = np.argmax(np.abs(pairings.Tx_symmetry_map), axis = 0)
    Ty = np.argmax(np.abs(pairings.Ty_symmetry_map), axis = 0)


    TRS = np.concatenate([np.array([2 * i + 1, 2 * i]) for i in range(config.total_dof // 2)])
    PHS = np.concatenate([np.arange(config.total_dof // 2, config.total_dof), np.arange(0, config.total_dof // 2)], axis = 0)

    C3z = np.concatenate([C3z, C3z + config.total_dof // 2])
    C2y = np.concatenate([C2y, C2y + config.total_dof // 2])
    Tx = np.concatenate([Tx, Tx + config.total_dof // 2])
    Ty = np.concatenate([Ty, Ty + config.total_dof // 2])

    parameters = config.initial_parameters.copy()
    parameters[0] = 0;  # to ensure particle-hole symmetry
    for _ in range(20):
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        ampl = wf.get_cur_det() * wf.get_cur_Jastrow_factor()
        occ_sites, _, _ = wf.get_state()
        conf = np.zeros(config.total_dof); conf[occ_sites] = 1
        
        assert len(C3z) == len(conf)
        conf_new = conf[C3z]
        occ_new = np.where(conf_new > 0)[0]
        assert len(occ_new) == len(occ_sites)

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string))
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed C3z', ampl / ampl_new)
        else:
            n_failed += 1
            print('failed C3z!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())
            #exit(-1)

        conf_new = conf[C2y]
        occ_new = np.where(conf_new > 0)[0]

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string))
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed C2y', ampl / ampl_new)
        else:
            n_failed += 1
            print('failed C2y!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())


        conf_new = conf[Tx]
        occ_new = np.where(conf_new > 0)[0]

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string))
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed Tx', ampl / ampl_new)
        else:
            n_failed += 1
            print('failed Tx!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())



        conf_new = conf[Ty]
        occ_new = np.where(conf_new > 0)[0]

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string))
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed Ty', ampl / ampl_new)
        else:
            n_failed += 1
            print('failed Ty!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())




        conf_new = conf[TRS]
        occ_new = np.where(conf_new > 0)[0]

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string), trs_test = True)
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed TRS', ampl / ampl_new)
        else:
            n_failed += 1
            print('failed TRS!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())

        conf_new = conf[PHS]
        occ_new = np.where(conf_new > 0)[0]

        place_in_string = (np.zeros(config.total_dof) - 1).astype(np.int64)
        place_in_string[occ_new] = np.arange(len(occ_new))
        empty_sites = np.arange(config.total_dof); empty_sites[occ_new] = -1; empty_sites = set(empty_sites[empty_sites > -0.5])

        wf_transformed = wavefunction_singlet(config, config.pairings_list, \
                                              config.initial_parameters, True, (occ_new, empty_sites, place_in_string), ph_test = True)
        ampl_new = wf_transformed.get_cur_det() * wf_transformed.get_cur_Jastrow_factor()

        if np.isclose(np.abs(ampl / ampl_new), 1):
            n_agreed += 1
            print('passed PHS', ampl / ampl_new, ampl)
        else:
            n_failed += 1
            print('failed PHS!', ampl, ampl_new, ampl / ampl_new, \
                                 wf.get_cur_det() / wf_transformed.get_cur_det(), \
                                 wf.get_cur_Jastrow_factor() / wf_transformed.get_cur_Jastrow_factor())
        print('\n\n\n\n')



    return n_failed == 0


def test_chain_moves(config):
    success = True
    print('Testing chain of moves \\prod_{move} ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩', flush=True)
    n_agreed = 0
    n_failed = 0

    for _ in range(20):
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        ratio_acc = 1. + 0.0j
        initial_ampl = wf.current_ampl

        for move in range(300):
            L = len(wf.state) // 2
            #i, j = np.random.randint(0, 2 * L, size = 2)

            state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

            acc, det_ratio, j_ratio, i, j = wf.perform_MC_step()
            #print(ratio_acc, det_ratio, j_ratio)
            ratio_acc *= (det_ratio * j_ratio)
            #if i > L and j < L:
            #    print('non-conserving move')

        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl
        final_ampl_solid = wf.get_cur_Jastrow_factor() * wf.get_cur_det()

        if np.isclose(final_ampl / initial_ampl, ratio_acc) and np.isclose(final_ampl_solid, final_ampl):
            n_agreed += 1

        else:
            print('chain ⟨x|d^{\\dag}_i d_k|Ф⟩ / ⟨x|Ф⟩ failed:', final_ampl / initial_ampl, ratio_acc)
            n_failed += 1
            success = False
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success


def test_chiral_gap_preserves_something(config):
    success = True
    print('Testing that some elements of H_MF are zero', flush=True)
    n_agreed = 0
    n_failed = 0 

    for _ in range(20):
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        i, j = np.random.randint(0, config.total_dof // 2, size = 2)
        j += config.total_dof // 2
        acc, det_ratio, j_ratio, i, j = wf.perform_MC_step(proposed_move = (i, j), enforce=True)
        if not acc:
            continue
        if config.enforce_valley_orbitals:
            if np.abs(det_ratio) > 1e-10 and (i + j) % 2 == 1:
                n_failed += 1
                print(i, j, det_ratio, flush=True)
    
        if not config.enforce_valley_orbitals:
            if np.abs(det_ratio) > 1e-10 and (i + j) % 2 == 0:
                n_failed += 1
                print(i, j, det_ratio)
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)
        success = False
    # exit(-1)
    return success


def test_delayed_updates_check(config):
    success = True
    print('Testing delayed updates', flush=True)
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    for _ in range(200):
        current_ampl = wf.current_ampl
        for step in range(1000):
            acc, detr, jastrr = wf.perform_MC_step()[:3]
            current_ampl *= detr * jastrr
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl
        final_ampl_solid = wf.get_cur_Jastrow_factor() * wf.get_cur_det()

        if np.isclose(final_ampl_solid, final_ampl) and np.isclose(current_ampl, final_ampl):
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
    print('Testing ⟨x|d^{\\dag}_i d_i|Ф⟩ / ⟨x|Ф⟩ = n_i', flush=True)
    n_agreed = 0
    n_failed = 0
    wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)

    while n_agreed < 50:
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
    print('Testing all Jastrow correlations included only once', flush=True)
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
    print('Testing double moves ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩', flush=True)
    n_agreed = 0
    n_failed = 0
    # wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
    while n_agreed < 20:
        print('try', n_agreed)
        wf = wavefunction_singlet(config, config.pairings_list, config.initial_parameters, False, None)
        L = len(wf.state) // 2
        i, j, k, l = np.random.randint(0, 2 * L, size = 4)
        print(i, j, k, l)
        #if i == j or i == l or k == l or k == j:
        #    continue  # the degenerate cases are considered separately (directly by density operator)

        initial_ampl = wf.current_ampl
        state = deepcopy((wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy))
        ratio_fast = get_wf_ratio_double_exchange(*state, wf.var_f, i, j, k, l)
        
        W_ij_0 = get_wf_ratio(*state, wf.var_f, i, j)
        acc = wf.perform_MC_step(proposed_move = (i, j))[0]

        if not acc:
            print('failed first acc')
            continue
        wf.perform_explicit_GF_update()
        state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)

        middle_ampl = wf.current_ampl

        W_kl_upd = get_wf_ratio(*state, wf.var_f, k, l)
        ratio_check = W_kl_upd * W_ij_0

        acc = wf.perform_MC_step(proposed_move = (k, l))[0]
        if not acc:
            print('failed 2nd acc')
            continue
        wf.perform_explicit_GF_update()
        final_ampl = wf.current_ampl

        ratio_straight = final_ampl / initial_ampl

        if np.allclose([ratio_fast, ratio_fast], [ratio_straight, ratio_check], atol = 1e-11, rtol = 1e-11):
            n_agreed += 1
            print('success', i, j, k, l)
        else:
            print('double move check ⟨x|d^{\\dag}_i d_j d^{\\dag}_k d_l|Ф⟩ / ⟨x|Ф⟩ failed:', ratio_fast / ratio_straight, ratio_straight, ratio_check, i, j, k, l)
            n_failed += 1
            success = False
            exit(-1)
    if n_failed == 0:
        print('Passed')
    else:
        print('Failed on samples:', n_failed)

    return success


def test_double_move_commutation_check(config):
    success = True
    print('Testing fast double updates have correct commutation properties...', flush=True)
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
    twist = np.exp(2.0j * np.pi * np.random.uniform(0, 1, size=2))
    print('Testing BC twist is invertable, with twist ' + str(twist), flush=True)
    for name, gap in zip(config.pairings_list_names, config.pairings_list_unwrapped):
        assert np.allclose(gap, models.apply_TBC(config, twist, models.apply_TBC(config, twist, deepcopy(gap), inverse = False), inverse=True))
        print('Passed {:s}'.format(name))

    twist_2 = np.exp(2.0j * np.pi * np.random.uniform(0, 1, size=2))
    print('Testing BC twist is U(1) representation, with twists ' + str(twist) + ' ' + str(twist_2), flush=True)
    for name, gap in zip(config.pairings_list_names, config.pairings_list_unwrapped):
        assert np.allclose(models.apply_TBC(config, twist_2, models.apply_TBC(config, twist, deepcopy(gap), inverse = False), inverse=False), \
                models.apply_TBC(config, twist, models.apply_TBC(config, twist_2, deepcopy(gap), inverse = False), inverse=False))
        print('Passed {:s}'.format(name))
    print('Passed', flush=True)
    return True

@jit(nopython=True)
def get_fft(N, nbands, twist):

    W = np.zeros((N ** 2, N ** 2), dtype=np.complex128)
    for kx in range(N):
        for ky in range(N):
            for x in range(N):
                for y in range(N):
                    W[x * N + y, kx * N + ky] = np.exp((2.0j * np.pi / N * kx - 2.0j * np.pi * twist[0] / N) * x + \
                                                       (2.0j * np.pi / N * ky - 2.0j * np.pi * twist[1] / N) * y)
    return np.kron(W, np.eye(nbands))

from copy import deepcopy

def test_FT_BC_twist(config):
    K0 = config.K_0

    nbands = config.n_orbitals * config.n_sublattices

    for _ in range(100):
        #twist = np.array([0.5, 0.5]) #
        twist = np.random.uniform(0, 1, size=2) 
        fft_plus = get_fft(config.Ls, nbands // 2, twist)
        fft_minus = get_fft(config.Ls, nbands // 2, np.array([-twist[0], -twist[1]]))
        
        K0_twisted = models.apply_TBC(config, np.exp(1.0j * 2.0 * np.pi * twist), deepcopy(K0), inverse = False)
        K0_twisted_plus = K0_twisted[np.arange(0, K0_twisted.shape[0], 2), :]; K0_twisted_plus = K0_twisted_plus[:, np.arange(0, K0_twisted.shape[0], 2)]
        K0_twisted_minus = K0_twisted[np.arange(1, K0_twisted.shape[0], 2), :]; K0_twisted_minus = K0_twisted_minus[:, np.arange(1, K0_twisted.shape[0], 2)]

        K0_twisted_plus = fft_plus.conj().T.dot(K0_twisted_plus.dot(fft_plus))
        K0_twisted_minus = fft_minus.conj().T.dot(K0_twisted_minus.dot(fft_minus))

        K0_check = K0_twisted_plus.copy()


        for i in range(K0.shape[0] // nbands):
            K0_check[i * nbands // 2:i * nbands // 2 + nbands // 2,i * nbands // 2:i * nbands // 2 + nbands // 2] = 0.0
        if not np.isclose(np.sum(np.abs(K0_check)), 0.0):
            print('plus valley twist BC test failed with twist', twist)
            exit(-1)
            return False


        K0_check = K0_twisted_minus.copy()
        for i in range(K0.shape[0] // nbands):
            K0_check[i * nbands // 2:i * nbands // 2 + nbands // 2,i * nbands // 2:i * nbands // 2 + nbands // 2] = 0.0
        if not np.isclose(np.sum(np.abs(K0_check)), 0.0):
            print('minus valley twist BC test failed with twist', twist)
            exit(-1)
            return False

    print('Passed', flush=True)
    return True

def test_simple_Koshino_Jastrow_equaldistant_allaccounting(config):
    print('Testing Jastrow uniqueness...')
    jastrows = jastrow.jastrow_Koshino_simple
    all_distances = config.all_distances

    for J in jastrows:
        j, name = J

        unique_distances = np.unique(np.around(all_distances[j > 0], decimals = 5))
        if len(unique_distances) > 1:
            exit(-1)
            return False

        val = unique_distances[0]

        if val in np.unique(np.around(all_distances * (1. - 2 * j), decimals = 5)) and val > 0:
            print(val, np.unique(np.around(all_distances * (1. - 2 * j), decimals = 5)))
            exit(-1)
            return False
    print('Passed', flush=True)
    return True

def perform_all_tests(config):
    success = True
    success = success and test_double_move_check(config)
    success = success and test_numerical_derivative_check(config)
    success = success and test_gf_symmetry(config)
    success = success and test_simple_Koshino_Jastrow_equaldistant_allaccounting(config)
    success = success and test_FT_BC_twist(config)
    success = success and test_BC_twist(config)
    success = success and test_chiral_gap_preserves_something(config)
    success = success and test_chain_moves(config)
    success = success and test_single_move_check(config)
    success = success and test_delayed_updates_check(config)
    #success = success and test_particle_hole(config)
    success = success and test_all_jastrow_factors_included_only_once(config)
    success = success and test_explicit_factors_check(config)
    success = success and test_double_move_commutation_check(config)
    
    success = success and test_onsite_gf_is_density_check(config)
    
    return success
