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
        sp_index_range = phi_field.config.total_dof // 2
        n_fields = 1
    else:
        sp_index_range = phi_field.config.total_dof // 4 * 3
        n_fields = 3
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
        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            if switch:
                phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)
            phi_field.refresh_G_functions()
            
            current_det_log, current_det_sign = -phi_field.log_det_up -phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
            current_det_sign = current_det_sign.item()
        phi_field.wrap_up(time_slice)
        if switch:
            phi_field.copy_to_CPU()
        #if time_slice == 0:
        #    phi_field.current_G_function_up, phi_field.log_det_up, phi_field.sign_det_up = phi_field.get_G_no_optimisation(+1, 0)
        #    phi_field.current_G_function_down, phi_field.log_det_down, phi_field.sign_det_down = phi_field.get_G_no_optimisation(-1, 0)
        #    current_det_log, current_det_sign = -phi_field.log_det_up -phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
        #print('first measurement in loop {:d}'.format(time_slice))
        #observables.measure_light_observables(phi_field, current_det_sign)
        for sp_index in range(sp_index_range):
            site_idx = sp_index // n_fields
            o_index = sp_index % n_fields

            phi_field.compute_deltas(site_idx, time_slice, o_index)
            if n_fields > 1:
                ratio = auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                        auxiliary_field.get_det_ratio_inter(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-11
            else:
                ratio = auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_up, phi_field.current_G_function_up) * \
                        auxiliary_field.get_det_ratio_intra(site_idx, phi_field.Delta_down, phi_field.current_G_function_down) + 1e-11
            lamb = lambdas[sp_index + time_slice * sp_index_range]
            if lamb < np.min([1, np.abs(ratio)]):
                current_det_log += np.log(np.abs(ratio))

                current_det_sign *= np.sign(ratio)
                ratio = np.log(np.abs(ratio))
                # print(current_det_sign, ratio, np.sign(ratio))

                accepted = 1.0
                phi_field.update_G_seq(site_idx)
                phi_field.update_field(site_idx, time_slice, o_index)
                 
                if False:#not GF_checked:
                    G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, time_slice)[:2]
                    G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, time_slice)[:2]

                    d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                    d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                    
                    GF_checked = True

                    if np.abs(d_gf_up) > 1e-8 or np.abs(d_gf_down) > 1e-8:
                        print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                    # print(det_log_up_check + det_log_down_check + current_det_log, current_det_log)  # FIXME
            else:
                ratio = 0
                accepted = 0
            observables.update_history(ratio, accepted, current_det_sign)
        observables.measure_light_observables(phi_field, current_det_sign.item())
    
    if n_sweep >= phi_field.config.thermalization:
        t = time()
        observables.measure_green_functions(phi_field, current_det_sign.item())
        print('measurement of green functions takes ', time() - t)
    return phi_field, observables

def retrieve_last_n_sweep(config):
    try:
        general_log = open(os.path.join(config.local_workdir, 'general_log.dat'), 'r')
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
        
        config.nu_V = np.arccosh(np.exp(V / 2. * config.dt))
        config.nu_U = np.arccosh(np.exp((U / 2. + V / 2.) * config.dt))

        K_matrix = config.model(config, config.mu)[0].real
        K_operator = scipy.linalg.expm(config.dt * K_matrix).real
        K_operator_inverse = scipy.linalg.expm(-config.dt * K_matrix).real
        local_workdir = os.path.join(config.workdir, 'U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}'.format(U, V, mu, int(Nt)))
        os.makedirs(local_workdir, exist_ok=True)

        phi_field = config.field(config, K_operator, K_operator_inverse, \
                                 K_matrix, local_workdir)
        phi_field.copy_to_GPU()
        with open(os.path.join(local_workdir, 'config.py'), 'w') as target, open(sys.argv[1], 'r') as source:  # save config file to workdir (to remember!!)
            target.write(source.read())
        
        observables = obs_methods.Observables(phi_field, local_workdir)
        observables.print_greerings()

        for n_sweep in range(retrieve_last_n_sweep(config), config.n_sweeps):
            accept_history = []
            t = time()
            phi_field, observables = perform_sweep(phi_field, observables, n_sweep)
            print('total sweep takes ', time() - t)
            phi_field.save_configuration()
            observables.print_std_logs(n_sweep)
            observables.write_light_observables(phi_field.config, n_sweep)

            if n_sweep > config.thermalization and n_sweep % config.n_print_frequency == 0:
                t = time()
                observables.write_heavy_observables(phi_field, n_sweep)
                print('measurement and writing of heavy observables took ', time() - t)
