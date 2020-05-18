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
config_dqmc_import = config_dqmc_file.simulation_parameters()

config = cv_module.simulation_parameters()
config.__dict__ = config_dqmc_import.__dict__.copy()

# print_model_summary(config_vmc)

def perform_sweep(phi_field, observables, n_sweep, switch = True):
    if phi_field.config.n_orbitals == 1:
        n_fields = 1
        sp_index_range = phi_field.config.total_dof // 2 // phi_field.config.n_orbitals * n_fields
    elif phi_field.config.n_orbitals == 2:
        n_fields = 3
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
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            current_det_sign = current_det_sign.item()
            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            if switch:
                phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)
            phi_field.refresh_G_functions()
            
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            current_det_sign = current_det_sign.item()
            current_gauge_factor_log = phi_field.get_current_gauge_factor_log()
        phi_field.wrap_up(time_slice)
        if switch:
            phi_field.copy_to_CPU()

        for sp_index in range(sp_index_range // n_fields):
            site_idx = sp_index

            local_det_factors = []
            local_gauge_factors = []
            local_conf_old = phi_field.get_current_conf(site_idx, time_slice)

            for local_conf in phi_field.local_conf_combinations:
                gauge_ratio = phi_field.get_gauge_factor_move(site_idx, time_slice, local_conf_old, local_conf)

                phi_field.compute_deltas(site_idx, time_slice, local_conf_old, local_conf)

                if n_fields > 1:
                    det_ratio = auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                                auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-11
                else:
                    det_ratio = auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                                auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-11
                local_det_factors.append(det_ratio)
                local_gauge_factors.append(gauge_ratio)

            probas = np.abs(np.array(local_det_factors) * np.array(local_gauge_factors))
            idx = np.random.choice(np.arange(len(local_det_factors)), \
                                   p = probas / np.sum(probas))
            new_conf = phi_field.local_conf_combinations[idx]

            current_det_log += np.log(np.abs(local_det_factors[idx]))
            current_gauge_factor_log += np.log(local_gauge_factors[idx])
            current_det_sign *= np.sign(local_det_factors[idx])

            ratio = np.log(np.abs(local_det_factors[idx]))
            accepted = (new_conf != local_conf_old)


            if accepted:
                phi_field.compute_deltas(site_idx, time_slice, local_conf_old, new_conf); phi_field.update_G_seq(site_idx)
                phi_field.update_field(site_idx, time_slice, new_conf)
            if False:
                G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, time_slice)[:2]
                G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, time_slice)[:2]

                d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                    
                GF_checked = True

                if np.abs(d_gf_up) > 1e-8 or np.abs(d_gf_down) > 1e-8:
                    print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                else:
                    print('test passed')
                print('Determinant discrepancy:', current_det_log + det_log_up_check + det_log_down_check)
                print('Gauge factor log discrepancy:', current_gauge_factor_log - phi_field.get_current_gauge_factor_log())
            observables.update_history(ratio, accepted, current_det_sign)
        observables.measure_light_observables(phi_field, current_det_sign.item(), n_sweep)
    
    if n_sweep >= phi_field.config.thermalization:
        t = time()
        observables.measure_green_functions(phi_field, current_det_sign.item())
        print('measurement of green functions takes ', time() - t)
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



if __name__ == "__main__":
    U_list = deepcopy(config.U); V_list = deepcopy(config.V); mu_list = deepcopy(config.mu); Nt_list = deepcopy(config.Nt);

    for U, V, mu, Nt in zip(U_list, V_list, mu_list, Nt_list):
        config.U = U; config.V = V; config.mu = mu; config.Nt = int(Nt);
        n_copy = config.n_copy
        config.nu_V = np.sqrt(V * config.dt / 2)  #np.arccosh(np.exp(V / 2. * config.dt))  # this is almost sqrt(V t)
        config.nu_U = np.arccosh(np.exp((U / 2. + V / 2.) * config.dt))
        assert V == 0 or V == U
        K_matrix = config.model(config, config.mu)[0].real
        K_operator = scipy.linalg.expm(config.dt * K_matrix).real
        K_operator_inverse = scipy.linalg.expm(-config.dt * K_matrix).real
        local_workdir = os.path.join(config.workdir, 'U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_c_{:d}'.format(U, V, mu, int(Nt), rank + config.offset))
        local_workdir_heavy = os.path.join(config.workdir_heavy, 'U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_c_{:d}'.format(U, V, mu, int(Nt), rank + config.offset))
        os.makedirs(local_workdir, exist_ok=True)
        os.makedirs(local_workdir_heavy, exist_ok=True)
        last_n_sweep_log = open(os.path.join(local_workdir, 'last_n_sweep.dat'), 'a')

        phi_field = config.field(config, K_operator, K_operator_inverse, \
                                 K_matrix, local_workdir)
        phi_field.copy_to_GPU()
        with open(os.path.join(local_workdir, 'config.py'), 'w') as target, open(sys.argv[1], 'r') as source:  # save config file to workdir (to remember!!)
            target.write(source.read())
        
        observables = obs_methods.Observables(phi_field, local_workdir, local_workdir_heavy)
        observables.print_greerings()

        for n_sweep in range(retrieve_last_n_sweep(local_workdir), config.n_sweeps):
            accept_history = []
            t = time()
            phi_field, observables = perform_sweep(phi_field, observables, n_sweep)
            print('total sweep takes ', time() - t)
            phi_field.save_configuration()
            observables.print_std_logs(n_sweep)
            observables.write_light_observables(phi_field.config, n_sweep)
            last_n_sweep_log.write(str(n_sweep) + '\n'); last_n_sweep_log.flush()
            if n_sweep > config.thermalization and n_sweep % config.n_print_frequency == 0:
                t = time()
                observables.write_heavy_observables(phi_field, n_sweep)
                print('measurement and writing of heavy observables took ', time() - t)
