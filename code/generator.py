from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
import os
import itertools

import scipy.linalg
from copy import deepcopy
import scipy.sparse as scs
from time import time
import auxiliary_field
import observables as obs_methods
import config_generator as cv_module
import sys
import os
import importlib
import psutil
import models

# np.random.seed(0)
# <<Borrowed>> from Tom
def import_config(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the module from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the module from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module

config_dqmc_file = import_config(sys.argv[1])
config_dqmc_import = config_dqmc_file.simulation_parameters(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), rank)

config = cv_module.simulation_parameters(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), rank)
config.__dict__ = config_dqmc_import.__dict__.copy()

# print_model_summary(config_vmc)

def perform_sweep(phi_field, observables, n_sweep, switch = True):
    if phi_field.config.n_orbitals == 1:
        n_fields = 1
        sp_index_range = phi_field.config.total_dof // 2 // phi_field.config.n_orbitals * n_fields
    elif phi_field.config.n_orbitals == 2:
        n_fields = 1
        sp_index_range = phi_field.config.total_dof // 2 // phi_field.config.n_orbitals * n_fields
    else:
        raise NotImplementedError()

    lambdas = np.random.uniform(0, 1, size = phi_field.config.Nt * sp_index_range)
    if switch:
        phi_field.copy_to_GPU()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()
    GF_checked = False

    for time_slice in range(phi_field.config.Nt):
        if time_slice == 0:
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, 1. / phi_field.sign_det_up / phi_field.sign_det_down
            current_det_sign = current_det_sign.item()
            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
            need_check = True
        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            if switch:
                phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)


            phi_field.refresh_G_functions()

            current_det_sign_before = current_det_sign * 1.0
            current_det_log_before = current_det_log * 1.0
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, 1. / phi_field.sign_det_up / phi_field.sign_det_down
            current_det_sign = current_det_sign.item()
            if np.abs(current_det_sign_before - current_det_sign) > 1e-3:  # refresh of Green's function must preserve sign (robust)
                print('Warning!!! Refresh did not preserve the det phase:', current_det_sign_before / current_det_sign, time_slice)
            if np.abs(current_det_log_before - current_det_log) > 1e-10:  # refresh of Green's function must preserve sign (robust)
                print('Warning!!! Refresh did not preserve the det log:', current_det_sign_before, current_det_sign, time_slice)

            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
            need_check = True
        # assert np.allclose(phi_field.get_G_no_optimisation(+1, time_slice)[0], phi_field.current_G_function_up)
        phi_field.wrap_up(time_slice)
        if switch:
            phi_field.copy_to_CPU()


        # assert np.allclose(phi_field.get_G_no_optimisation(+1, time_slice)[0], phi_field.current_G_function_up)
        for sp_index in range(sp_index_range // n_fields):
            site_idx = sp_index

            local_det_factors = []
            local_gauge_factors = []
            local_conf_old = phi_field.get_current_conf(site_idx, time_slice)


            for local_conf in phi_field.local_conf_combinations:
                gauge_ratio = phi_field.get_gauge_factor_move(site_idx, time_slice, local_conf_old, local_conf)

                phi_field.compute_deltas(site_idx, time_slice, local_conf_old, local_conf)
                if phi_field.config.n_orbitals > 1:
                    det_ratio = auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                                auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-16
                else:
                    det_ratio = auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                                auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-16

                local_det_factors.append(det_ratio)
                local_gauge_factors.append(gauge_ratio)
                

            probas = np.array(local_det_factors) * np.array(local_gauge_factors)
            assert np.allclose(probas.real, probas)

            probas = np.abs(probas)

            idx = np.random.choice(np.arange(len(local_det_factors)), \
                                   p = probas / np.sum(probas))

            new_conf = phi_field.local_conf_combinations[idx]
            assert probas[idx] > 0

            current_det_log += np.log(np.abs(local_det_factors[idx]))
            current_gauge_factor_log += np.log(local_gauge_factors[idx])

            current_det_sign *= local_det_factors[idx] / np.abs(local_det_factors[idx])
            
            ratio = np.log(np.abs(local_det_factors[idx]))
            accepted = (new_conf != local_conf_old)


            if accepted:
                phi_field.compute_deltas(site_idx, time_slice, local_conf_old, new_conf); phi_field.update_G_seq(site_idx)
                phi_field.update_field(site_idx, time_slice, new_conf)

            if True:#True:#False:#True: #need_check:
                G_up_check, det_log_up_check, phase_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                G_down_check, det_log_down_check, phase_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

                d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                print(np.linalg.norm(phi_field.current_G_function_up - G_up_check) / np.linalg.norm(G_up_check))
                print(np.linalg.norm(phi_field.current_G_function_down - G_down_check) / np.linalg.norm(G_down_check))
                GF_checked = True

                if np.abs(d_gf_up) > 1e-8 or np.abs(d_gf_down) > 1e-8:
                    print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                else:
                    print('test passed')
                print('log |det| discrepancy:', current_det_log + det_log_up_check + det_log_down_check)
                print('Gauge factor log discrepancy:', current_gauge_factor_log - phi_field.get_current_gauge_factor_log())
                print(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_up_check)
                print(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_down_check)
                print(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log())) / phase_up_check / phase_down_check)
                print('phase det discrepancy:', phase_up_check * phase_down_check * current_det_sign)
                print(phase_up_check)
                need_check = False

            observables.update_history(ratio, accepted, 1) # np.real(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_up_check))
        t = time()
        observables.measure_light_observables(phi_field, 1, n_sweep)
        print('measurement of light observables takes ', time() - t)
    
    if n_sweep >= phi_field.config.thermalization:
        t = time()
        observables.measure_green_functions(phi_field, 1.0)  # ??
        print('measurement of green functions takes ', time() - t)
        process = psutil.Process(os.getpid())
        print('using memory', process.memory_info().rss)
    return phi_field, observables




def perform_sweep_longrange(phi_field, observables, n_sweep, switch = True):
    eta_index_range = phi_field.config.total_dof // 2 // 2
    xi_index_range = phi_field.config.total_dof // 2 // 2 // 2 * 3
    assert xi_index_range == phi_field.n_bonds

    if switch:
        phi_field.copy_to_GPU()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()

    for time_slice in range(phi_field.config.Nt):
        if time_slice == 0:
            current_det_log, current_det_sign = -2 * phi_field.log_det_up - 2 * phi_field.log_det_down, 1. / phi_field.sign_det_up ** 2 / phi_field.sign_det_down ** 2
            current_det_sign = current_det_sign.item()
            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
            need_check_eta = True
            need_check_xi = True

        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            #t = time()
            if switch:
                    phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)

            G_up, G_down = phi_field.current_G_function_up * 1.0, phi_field.current_G_function_down * 1.0
            phi_field.refresh_G_functions()

            if np.log(np.linalg.norm(G_up - phi_field.current_G_function_up) / np.linalg.norm(G_up)) > -10:
                print('Warning! During refresh there is a big log discrepancy in up:', np.log(np.linalg.norm(G_up - phi_field.current_G_function_up) / np.linalg.norm(G_up)))

            # print(np.max(np.abs(G_down - phi_field.current_G_function_down)))
            current_det_sign_before = current_det_sign * 1.0
            current_det_log_before = current_det_log * 1.0
            current_det_log, current_det_sign = -phi_field.log_det_up * 2 - phi_field.log_det_down * 2, 1. / phi_field.sign_det_up ** 2 / phi_field.sign_det_down ** 2
            current_det_sign = current_det_sign.item()

            #if np.abs(current_det_sign_before - current_det_sign) > 1e-6:  # refresh of Green's function must preserve sign (robust)
            #    print('Warning!!! Refresh did not preserve the det phase:', current_det_sign_before / current_det_sign, time_slice)
            #if np.abs(current_det_log_before - current_det_log) > 1e-6:  # refresh of Green's function must preserve sign (robust)
            #    print('Warning!!! Refresh did not preserve the det log:', current_det_log_before / current_det_log, current_det_log_before, current_det_log, time_slice)

            #G_up_check, det_log_up_check, phase_up_check = phi_field.get_G_no_optimisation(+1, time_slice) # FIXME debug
            #G_up_current = phi_field.current_G_function_up

            #print('GF discrepancy: opt vs nonopt', np.linalg.norm(G_up_current - G_up_check) / np.linalg.norm(G_up_check))

            #if np.abs(det_log_up_check - phi_field.log_det_up) > 1e-6:  # refresh of Green's function must preserve sign (robust)
            #    print('Warning!!! Refresh did not preserve the det log noopt vs opt:', det_log_up_check, phi_field.log_det_up, time_slice)

            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
            need_check_xi = True
            need_check_eta = True
            #print('refresh: ', time() - t); t =time()
        # assert np.allclose(phi_field.get_G_no_optimisation(+1, time_slice)[0], phi_field.current_G_function_up)
        #t = time()
        phi_field.wrap_up(time_slice)
        #print('wrap up', time() - t)
        if switch:
            phi_field.copy_to_CPU()

        ### DEBUG ###

        GFs_up = np.array(phi_field.get_nonequal_time_GFs(+1.0, phi_field.current_G_function_up))
        GFs_up_naive = np.array(phi_field.get_G_tau_0_naive(+1.0))

        for gf, gf_naive in zip(GFs_up, GFs_up_naive):
            print(np.sum(np.abs(gf - gf_naive)))
        '''
        for t in range(len(GFs_up)):
            beta = phi_field.config.Nt * phi_field.config.dt
            current_tau = t * phi_field.config.dt
            energies, states = np.linalg.eigh(phi_field.K_matrix_plus)
            states = states.T.conj()
            assert np.allclose(phi_field.K_matrix_plus, phi_field.K_matrix_plus.conj().T)
            assert np.allclose(np.einsum('i,ij,ik->jk', energies, states.conj(), states), phi_field.K_matrix_plus)
            correct_string = np.einsum('i,ij,ik->jk', np.exp(current_tau * energies) / (1. + np.exp(beta * energies)), states.conj(), states)

            print(t, np.linalg.norm(GFs_up[t] - correct_string) / np.linalg.norm(correct_string))
        '''
        #energies, states = np.linalg.eigh(phi_field.K_matrix_plus)
        #beta = phi_field.config.Nt * phi_field.config.dt
        #correct_energy = np.sum(energies / (1. + np.exp(beta * energies)))

        #print('correct energy at beta = {:.10f} = {:.10f}'.format(beta, correct_energy * 4. / 4 / phi_field.config.Ls ** 2))


        #### eta-site field update ####
        # assert np.allclose(phi_field.get_G_no_optimisation(+1, time_slice)[0], phi_field.current_G_function_up)
        for site_idx in range(eta_index_range):
            local_det_factors = []
            local_gauge_factors = []
            local_conf_old = phi_field.get_current_eta(site_idx, time_slice)

            for local_conf in phi_field.local_conf_combinations:
                gauge_ratio = phi_field.get_gauge_factor_move_eta(site_idx, time_slice, local_conf_old, local_conf)
                local_gauge_factors.append(gauge_ratio)


            deltas = phi_field.compute_deltas_eta(site_idx, time_slice, local_conf_old, phi_field.local_conf_combinations)
            #t = time()
            #phi_field.compute_deltas_eta(site_idx, time_slice, local_conf_old, local_conf)
            #print('deltas eta', time() - t)

            for delta in deltas:
                #t = time()
                det_ratio = auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta, phi_field.current_G_function_up) ** 2 * \
                            auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta, phi_field.current_G_function_down) ** 2
                #print('det ratio eta', time() - t)


                local_det_factors.append(det_ratio)

            probas = np.array(local_det_factors) * np.array(local_gauge_factors)
            assert np.allclose(probas.real, probas)
            assert np.all(probas.real + 1e-12 > 0)
            assert np.isclose(probas, 1.0, atol=1e-8).any()


            probas = np.abs(probas)

            idx = np.random.choice(np.arange(len(local_det_factors)), p = probas / np.sum(probas))

            new_conf = phi_field.local_conf_combinations[idx]
            assert probas[idx] > 0

            current_det_log += np.log(np.abs(local_det_factors[idx]))
            # print(current_det_log)
            current_gauge_factor_log += np.log(local_gauge_factors[idx])

            current_det_sign *= local_det_factors[idx] / np.abs(local_det_factors[idx])

            ratio = np.log(np.abs(local_det_factors[idx]))
            accepted = (new_conf[0] != local_conf_old[0])
            
            if accepted:
                #t = time()
                phi_field.compute_deltas_eta(site_idx, time_slice, local_conf_old, new_conf); 
                #print('deltas', time() - t); t = time()

                phi_field.update_G_seq_eta(site_idx);
                #print('G_update eta', time() - t); t = time()

                phi_field.update_eta_site_field(site_idx, time_slice, new_conf)
                #print('update field: ', time() - t)

            if False:#need_check_eta:
                G_up_check, det_log_up_check, phase_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                G_down_check, det_log_down_check, phase_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

                d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                print(np.linalg.norm(phi_field.current_G_function_up - G_up_check) / np.linalg.norm(G_up_check))
                print(np.linalg.norm(phi_field.current_G_function_down - G_down_check) / np.linalg.norm(G_down_check))
                GF_checked = True

                if np.abs(d_gf_up) > 1e-8 or np.abs(d_gf_down) > 1e-8:
                    print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                else:
                    print('test of eta site update passed')
                print('log |det| discrepancy:', current_det_log + det_log_up_check * 2 + det_log_down_check * 2)
                print('Gauge factor log discrepancy:', current_gauge_factor_log - phi_field.get_current_gauge_factor_log())
                print('phase det discrepancy:', phase_up_check ** 2 * phase_down_check ** 2 * current_det_sign)
                print(phase_up_check)
                need_check_eta = False

            observables.update_history(ratio, accepted, 1) # np.real(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_up_check))



        #### xi-bond field update ####
        for bond_idx in range(xi_index_range):
            local_det_factors = []
            local_gauge_factors = []
            local_conf_old = phi_field.get_current_xi(bond_idx, time_slice)

            for local_conf in phi_field.local_conf_combinations:
                gauge_ratio = phi_field.get_gauge_factor_move_xi(bond_idx, time_slice, local_conf_old, local_conf[0])

                #t = time()
                phi_field.compute_deltas_xi(bond_idx, time_slice, local_conf_old, local_conf[0])
                #print('deltas xi:', time() - t)

                sp_index1, sp_index2 = phi_field.bonds[bond_idx]

                #t = time()
                det_ratio = auxiliary_field.get_det_ratio_inter(sp_index1, sp_index2, phi_field.Delta, phi_field.current_G_function_up) ** 2 * \
                            auxiliary_field.get_det_ratio_inter(sp_index1, sp_index2, phi_field.Delta, phi_field.current_G_function_down) ** 2
                #print('det ratio xi:', time() - t)
                local_det_factors.append(det_ratio)

                local_gauge_factors.append(gauge_ratio)

            probas = np.array(local_det_factors) * np.array(local_gauge_factors)
            assert np.allclose(probas.real, probas)

            probas = np.abs(probas)

            idx = np.random.choice(np.arange(len(local_det_factors)), p = probas / np.sum(probas))

            new_conf = phi_field.local_conf_combinations[idx]
            assert probas[idx] > 0

            current_det_log += np.log(np.abs(local_det_factors[idx]))
            current_gauge_factor_log += np.log(local_gauge_factors[idx])

            current_det_sign *= local_det_factors[idx] / np.abs(local_det_factors[idx])

            ratio = np.log(np.abs(local_det_factors[idx]))
            accepted = (new_conf[0] != local_conf_old)

            if accepted:
                #t = time()
                phi_field.compute_deltas_xi(bond_idx, time_slice, local_conf_old, new_conf[0]); 
                #print('deltas xi', time() - t); t = time()
                phi_field.update_G_seq_xi(bond_idx)
                #print('update G xi', time() - t); t = time()
                phi_field.update_xi_bond_field(bond_idx, time_slice, new_conf[0])
                #print('update xi bond', time() - t)

            if True:# need_check_xi:
                G_up_check, det_log_up_check, phase_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                G_down_check, det_log_down_check, phase_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

                d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                print(np.linalg.norm(phi_field.current_G_function_up - G_up_check) / np.linalg.norm(G_up_check))
                print(np.linalg.norm(phi_field.current_G_function_down - G_down_check) / np.linalg.norm(G_down_check))
                GF_checked = True

                if np.abs(d_gf_up) > 1e-8 or np.abs(d_gf_down) > 1e-8:
                    print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                else:
                    print('test of xi bond update passed')
                print('log |det| discrepancy:', current_det_log + 2 * det_log_up_check + 2 * det_log_down_check)
                print('Gauge factor log discrepancy:', current_gauge_factor_log - phi_field.get_current_gauge_factor_log())
                print('phase det discrepancy:', phase_up_check ** 2 * phase_down_check ** 2 * current_det_sign)
                print(phase_up_check)
                need_check_xi = False

            observables.update_history(ratio, accepted, 1) # np.real(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_up_check))
        
        observables.measure_light_observables(phi_field, 1, n_sweep, print_gf = (time_slice == phi_field.config.Nt - 1))

    if n_sweep >= phi_field.config.thermalization:
        t = time()
        observables.measure_green_functions(phi_field, current_det_sign)
        print('measurement of green functions takes ', time() - t)
        process = psutil.Process(os.getpid())
        print('using memory', process.memory_info().rss)
    return phi_field, observables










def perform_sweep_cluster(phi_field, observables, n_sweep, n_spins, switch = True):
    hex_index_range = phi_field.config.Ls ** 2
    assert hex_index_range == phi_field.n_hexagons

    total_exp = 0.0
    total_delta = 0.0
    total_G_upd = 0.0
    total_ratio = 0.0
    total_refresh = 0.0
    total_wrap = 0.0

    if switch:
        phi_field.copy_to_GPU()
    phi_field.exponentiate_V()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()

    for time_slice in range(phi_field.config.Nt):
        if time_slice == 0:
            t = time()
            current_det_log, current_det_sign = -n_spins * phi_field.log_det_up - n_spins * phi_field.log_det_down, 1. / phi_field.sign_det_up ** n_spins / phi_field.sign_det_down ** n_spins
            current_det_sign = current_det_sign.item()
            current_gauge_factor_log = phi_field.get_current_gauge_factor_log_hex()

            current_sign = np.exp(1.0j * np.imag(current_gauge_factor_log)) * current_det_sign #np.exp(current_gauge_factor_log) * current_det_sign
            total_refresh += time() - t

        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            tr = time()
            if switch:
                phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            t = time()
            phi_field.exponentiate_V()
            total_exp += time() - t
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)

            G_up, G_down = phi_field.current_G_function_up * 1.0, phi_field.current_G_function_down * 1.0
            phi_field.refresh_G_functions()

            if np.log(np.max(np.abs(G_up - phi_field.current_G_function_up))) > -10:
                print('Warning! During refresh there is a big log discrepancy in up:', np.log(np.max(np.abs(G_up - phi_field.current_G_function_up))))

            # print(np.max(np.abs(G_down - phi_field.current_G_function_down)))
            current_det_sign_before = current_det_sign * 1.0
            current_det_log_before = current_det_log * 1.0
            current_det_log, current_det_sign = -phi_field.log_det_up * n_spins - phi_field.log_det_down * n_spins, 1. / phi_field.sign_det_up ** n_spins / phi_field.sign_det_down ** n_spins
            current_det_sign = current_det_sign.item()

            current_gauge_factor_log = phi_field.get_current_gauge_factor_log_hex()

            current_sign = np.exp(1.0j * np.imag(current_gauge_factor_log)) * current_det_sign
            need_check = True
            total_refresh += time() - t
        
        if switch:
            phi_field.copy_to_CPU()
        t = time()
        phi_field.wrap_up(time_slice)
        observables.measure_light_observables(phi_field, current_sign, n_sweep)

        total_wrap += time() - t


        #### xi-bond field update ####
        for hex_idx in range(hex_index_range):
            local_det_factors = []
            local_gauge_factors = []
            local_conf_old = phi_field.get_current_hex(hex_idx, time_slice)

            deltas_plus, deltas_minus, supports = [], [], []
            phi_field.prepare_current_Z(time_slice)
            for local_conf in phi_field.local_conf_combinations:
                gauge_ratio = phi_field.get_gauge_factor_move_hex(local_conf_old, local_conf[0])
                local_gauge_factors.append(gauge_ratio)


            
            t = time()
            deltas = phi_field.compute_deltas_hex(hex_idx, time_slice, local_conf_old, phi_field.local_conf_combinations)
            total_delta += time() - t

            t = time()
            for delta in deltas:
                det_ratio = auxiliary_field.get_det_ratio_inter_hex(delta[2], delta[0], phi_field.current_G_function_up) ** n_spins * \
                            auxiliary_field.get_det_ratio_inter_hex(delta[2], delta[1], phi_field.current_G_function_down) ** n_spins
                local_det_factors.append(det_ratio)
            total_ratio += time() - t

            probas = np.array(local_det_factors) * np.array(local_gauge_factors)

            signs = probas / np.abs(probas)
            probas = np.abs(probas)

            idx = np.random.choice(np.arange(len(local_det_factors)), p = probas / np.sum(probas))

            new_conf = phi_field.local_conf_combinations[idx]
            assert probas[idx] > 0
            current_sign *= signs[idx]

            current_det_log += np.log(np.abs(local_det_factors[idx]))
            current_gauge_factor_log += np.log(local_gauge_factors[idx])

            current_det_sign *= local_det_factors[idx] / np.abs(local_det_factors[idx])
            ratio = np.log(np.abs(local_det_factors[idx]))
            accepted = (new_conf[0] != local_conf_old)

            if accepted:
                t = time()
                phi_field.Delta_plus = deltas[idx][0]
                phi_field.Delta_minus = deltas[idx][1]
                phi_field.support = deltas[idx][2]
                total_delta += time() - t
                t = time()
                phi_field.update_G_seq_hex(hex_idx)
                total_G_upd += time() - t
                phi_field.update_hex_field(hex_idx, time_slice, local_conf_old, new_conf[0])


            if False:# need_check_xi:
                phi_field.exponentiate_V()
                G_up_check, det_log_up_check, phase_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                G_down_check, det_log_down_check, phase_down_check = phi_field.get_G_no_optimisation(-1, time_slice)

                d_gf_up = np.linalg.norm(phi_field.current_G_function_up - G_up_check) / np.linalg.norm(G_up_check)
                GF_checked = True

                if np.abs(d_gf_up) > 1e-8:
                    print('\033[91m Warning: GF test failed! \033[0m', d_gf_up)
                else:
                    print('test of hex cluster update passed')
                print('log |det| discrepancy:', current_det_log + n_spins * det_log_up_check + n_spins * det_log_down_check)
                print('Gauge factor log discrepancy:', current_gauge_factor_log - phi_field.get_current_gauge_factor_log_hex())
                print('phase det discrepancy:', phase_up_check ** n_spins * phase_down_check ** n_spins * current_det_sign)
                print(phase_up_check)
                need_check = False

            observables.update_history(ratio, accepted, current_sign) # np.real(np.exp(1.0j * np.imag(phi_field.get_current_gauge_factor_log() / 2)) / phase_up_check))

    if n_sweep >= phi_field.config.thermalization:
        t = time()
        phi_field.exponentiate_V()
        observables.measure_green_functions(phi_field, current_det_sign)
        print('measurement of green functions takes ', time() - t)
        process = psutil.Process(os.getpid())
        print('using memory', process.memory_info().rss)
    #print('EXP TOTAL', total_exp)
    #print('DELTA TOTAL', total_delta)
    #print('UPD TOTAL', total_G_upd)
    #print('RATIO TOTAL', total_ratio)
    #print('TOTAL WRAP', total_wrap)
    #print('TOTAL REFRESH', total_refresh)
    return phi_field, observables

















def retrieve_last_n_sweep(local_workdir):
    try:
        general_log = open(os.path.join(local_workdir, 'last_n_sweep.dat'), 'r')
    except:
        return 0
    for line in general_log:
        pass
    try:
        return int(line.split(' ')[0])
    except:
        pass
    return 0

### DEBUG ###

Kim = np.array([[0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.097j, 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   -0.097j, 0.   +0.j   , 0.   +0.j   ],
       [0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.097j, 0.   +0.j   ],
       [0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   -0.097j],
       [0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   -0.097j, 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.331+0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.097j, 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.331+0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   -0.097j, 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.331+0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ],
       [0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.097j,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.   +0.j   , 0.   +0.j   , 0.   +0.j   ,
        0.   +0.j   , 0.331+0.j   , 0.   +0.j   , 0.   +0.j   ]])



if __name__ == "__main__":
    U_list = deepcopy(config.U); V_list = deepcopy(config.V); mu_list = deepcopy(config.mu); Nt_list = deepcopy(config.Nt);

    for U, V, mu, Nt in zip(U_list, V_list, mu_list, Nt_list):
        config.U = U; config.V = V; config.mu = mu; config.Nt = Nt
        n_copy = config.n_copy
        #config.nu_V = np.sqrt(V * config.dt / 2)  #np.arccosh(np.exp(V / 2. * config.dt))  # this is almost sqrt(V t)
        #config.nu_U = np.arccosh(np.exp((U / 2. + V / 2.) * config.dt))
        #assert V == U
        
        config.nu_U = np.sqrt(config.dt / 2 * U)
        config.nu_V = 0.#np.sqrt(config.dt / 2 * V)

        K_matrix = config.model(config, 0.0)[0]
        K_matrix -= np.eye(K_matrix.shape[0]) * config.mu

        ### application of real TBCs ###
        real_twists = [[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]]
        twist = real_twists[0] #[(rank + config.offset) % len(real_twists)]  # each rank knows its twist
        K_matrix = models.xy_to_chiral(K_matrix, 'K_matrix', config, config.chiral_basis)
        #np.save('K_matrix.npy', (K_matrix))
        #exit(-1)

        #K_matrix += 1.0j * Kim.imag  # FIXME FIXME FIXME
        #assert np.allclose(K_matrix.imag, K_matrix * 0.)
        assert np.allclose(K_matrix, K_matrix.conj().T)

        K_matrix = -K_matrix  # this is to agree with ED, this is just unitary transform in terms of particle-hole transformation


        print(repr(K_matrix))

        energies = np.linalg.eigh(K_matrix)[0]
        print(energies)
        print(np.sum(energies[energies < 0]) / 16.)

        K_matrix = models.apply_TBC(config, twist, deepcopy(K_matrix), inverse = False)
        config.pairings_list_unwrapped = [models.apply_TBC(config, twist, deepcopy(gap), inverse = False) for gap in config.pairings_list_unwrapped]

        ### creating precomputed exponents ###
        K_operator = scipy.linalg.expm(config.dt * K_matrix)
        K_operator_inverse = scipy.linalg.expm(-config.dt * K_matrix)
        assert np.allclose(np.linalg.inv(K_operator), K_operator_inverse)
        K_operator_half = scipy.linalg.expm(0.5 * config.dt * K_matrix)
        K_operator_half_inverse = scipy.linalg.expm(-0.5 * config.dt * K_matrix)

        local_workdir = config.workdir #,#os.path.join(config.workdir, 'U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_c_{:d}'.format(U, V, mu, int(Nt), rank + config.offset))
        local_workdir_heavy = config.workdir_heavy #,os.path.join(config.workdir_heavy, 'U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_c_{:d}'.format(U, V, mu, int(Nt), rank + config.offset))
        os.makedirs(local_workdir, exist_ok=True)
        os.makedirs(local_workdir_heavy, exist_ok=True)
        last_n_sweep_log = open(os.path.join(local_workdir, 'last_n_sweep.dat'), 'a')

        phi_field = config.field(config, K_operator, K_operator_inverse, \
                                 K_matrix, local_workdir, K_operator_half, K_operator_half_inverse)
        phi_field.copy_to_GPU()
        with open(os.path.join(local_workdir, 'config.py'), 'w') as target, open(sys.argv[1], 'r') as source:  # save config file to workdir (to remember!!)
            target.write(source.read())
        
        observables = obs_methods.Observables(phi_field, local_workdir, local_workdir_heavy)
        observables.print_greerings()

        for n_sweep in range(retrieve_last_n_sweep(local_workdir), config.n_sweeps):
            accept_history = []
            t = time()
            # phi_field, observables = perform_sweep(phi_field, observables, n_sweep)
            #phi_field, observables = perform_sweep_longrange(phi_field, observables, n_sweep)

            phi_field, observables = perform_sweep_cluster(phi_field, observables, n_sweep, config.n_spins)
            print('total sweep takes ', time() - t)
            print('total SVD time ', phi_field.total_SVD_time); phi_field.total_SVD_time = 0.0



            phi_field.save_configuration()
            observables.print_std_logs(n_sweep)
            if observables.n_cumulants > 0:
                observables.write_light_observables(phi_field.config, n_sweep)
            last_n_sweep_log.write(str(n_sweep) + '\n'); last_n_sweep_log.flush()
            if n_sweep > config.thermalization and n_sweep % config.n_print_frequency == 0:
                t = time()
                observables.write_heavy_observables(phi_field, n_sweep)
                print('measurement and writing of heavy observables took ', time() - t)
