import numpy as np
import os
import itertools

xp = np  # by default the code is executed on the CPU
cp = np

gpu_avail = False

try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
    gpu_avail = True
except ImportError:
    print('No CuPy found in the system, running on a CPU')

# gpu_avail = False

import scipy.linalg
from copy import deepcopy
import scipy.sparse as scs
from time import time
import auxiliary_field
import observables as obs_methods
from config_generator import simulation_parameters


accept_history = []
sign_history = []
ratio_history = []

config = simulation_parameters()
def print_greetings(config):
    print("# Starting simulations using {} starting configuration, T = {:3f} meV, mu = {:3f} meV, "
          "lattice = {:d}^2 x {:d}".format(config.start_type, 1.0 / config.dt / config.Nt, config.mu, config.Ls, config.Nt))
    print('# sweep ⟨r⟩ d⟨r⟩ ⟨acc⟩ d⟨acc⟩ ⟨sign⟩ d⟨sign⟩ ⟨n⟩ d⟨n⟩ ⟨E_K⟩ d⟨E_K⟩ ⟨E_C⟩ d⟨E_C⟩ ⟨E_T⟩ d⟨E_T⟩')
    return

def perform_sweep(phi_field, n_sweep, switch = True):
    global accept_history, sign_history, ratio_history
    if switch:
        phi_field.copy_to_GPU()
    phi_field.refresh_all_decompositions()
    phi_field.refresh_G_functions()

    GF_checked = False
    observables_light = []; obs_signs_light = []; names_light = []
    observables_heavy = []; obs_signs_heavy = []; names_heavy = []
    

    for time_slice in range(phi_field.config.Nt):
        if time_slice == 0:
            current_det_log, current_det_sign = -phi_field.log_det_up - phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down

        if time_slice in phi_field.refresh_checkpoints and time_slice > 0:  # every s-th configuration we refresh the Green function
            if switch:
                phi_field.copy_to_GPU()
            index = np.where(phi_field.refresh_checkpoints == time_slice)[0][0]
            phi_field.append_new_decomposition(phi_field.refresh_checkpoints[index - 1], time_slice)
            phi_field.refresh_G_functions()
            
            current_det_log, current_det_sign = -phi_field.log_det_up -phi_field.log_det_down, phi_field.sign_det_up * phi_field.sign_det_down
        phi_field.wrap_up(time_slice)
        if switch:
            phi_field.copy_to_CPU()

        if phi_field.config.n_orbitals == 1:
            sp_index_range = phi_field.config.total_dof // 2
            n_fields = 1
        else:
            sp_index_range = phi_field.config.total_dof // 4 * 3
            n_fields = 3
        for sp_index in range(sp_index_range):
            site_idx = sp_index // n_fields
            o_index = sp_index % n_fields


            sign_history.append(current_det_sign)

            ratio = phi_field.get_det_ratio(+1, site_idx, time_slice, o_index) * \
                    phi_field.get_det_ratio(-1, site_idx, time_slice, o_index)

            lamb = np.random.uniform(0, 1)

            if lamb < np.min([1, np.abs(ratio)]):
                current_det_log += np.log(np.abs(ratio))

                ratio_history.append(np.log(np.abs(ratio)))

                current_det_sign *= np.sign(ratio)
                accept_history.append(+1)

                phi_field.update_G_seq(+1, site_idx, time_slice, o_index)
                phi_field.update_G_seq(-1, site_idx, time_slice, o_index)

                phi_field.update_field(site_idx, time_slice, o_index)

                 
                if not GF_checked:
                    G_up_check, det_log_up_check = phi_field.get_G_no_optimisation(+1, time_slice)
                    G_down_check, det_log_down_check = phi_field.get_G_no_optimisation(-1, time_slice)
                    d_gf_up = np.sum(np.abs(phi_field.current_G_function_up - G_up_check)) / np.sum(np.abs(G_up_check))
                    d_gf_down = np.sum(np.abs(phi_field.current_G_function_down - G_down_check)) / np.sum(np.abs(G_down_check))
                    GF_checked = True

                    if np.abs(d_gf_up) < 1e-8 and np.abs(d_gf_down) < 1e-8:
                        print('\033[92m GF test passed successfully \033[0m')
                    else:
                        print('\033[91m Warning: GF test failed! \033[0m', d_gf_up, d_gf_down)
                
            else:
                accept_history.append(0)
                ratio_history.append(0)
        if switch:
            phi_field.copy_to_GPU()


        ### light observables ### (calculated always during calculator and generator stages)
        obs, names_light = obs_methods.compute_light_observables(phi_field)
        observables_light.append(np.array(obs) * current_det_sign.item())  # the sign is included into observables (reweighting)
        obs_signs_light.append(current_det_sign.item())

        ### heavy observables ### (calculated only during the generator stage)
        if n_sweep >= phi_field.config.thermalization:
            obs, names_heavy = obs_methods.compute_heavy_observables(phi_field)
            observables_heavy.append(np.array(obs) * current_det_sign.item())  # the sign is included into observables (reweighting)
            obs_signs_heavy.append(current_det_sign.item())

        if switch:
            phi_field.copy_to_CPU()
    cut = phi_field.config.n_smoothing


    ### light observables ### (calculated always during calculator and generator stages)
    observables_light = np.array(observables_light) / np.mean(obs_signs_light)
    observables_light = np.concatenate([observables_light.mean(axis = 0)[:, np.newaxis], 
                                        observables_light.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)

    obs_light_extra = [np.mean(ratio_history[-cut:]), np.std(ratio_history[-cut:]) / np.sqrt(cut),
                       np.mean(accept_history[-cut:]), np.std(accept_history[-cut:]) / np.sqrt(cut),
                       np.mean(sign_history[-cut:]), np.std(sign_history[-cut:]) / np.sqrt(cut), 
                       np.mean(obs_signs_light), np.std(obs_signs_light) / np.sqrt(len(obs_signs_light))]

    observables_light = np.concatenate([np.array(obs_light_extra), observables_light], axis = 0)
    names_light = ['⟨ratio⟩', '⟨acc⟩', '⟨sign_gen⟩', '⟨sign_obs_l⟩'] + names_light

    ### heavy observables ### (calculated only during the generator stage)
    if n_sweep >= phi_field.config.thermalization:
        observables_heavy = np.array(observables_heavy) / np.mean(obs_signs_heavy)
        observables_heavy = np.concatenate([observables_heavy.mean(axis = 0)[:, np.newaxis], 
                                            observables_heavy.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)

        observables_heavy = np.concatenate([np.array([np.mean(obs_signs_heavy), 
                                np.std(obs_signs_heavy) / np.sqrt(len(obs_signs_heavy))]), observables_heavy], axis = 0)
        names_heavy = ['⟨sign_obs_h⟩'] + names_heavy
    return phi_field, (observables_light, names_light), (observables_heavy, names_heavy)


if __name__ == "__main__":
    print_greetings(config)

    K_matrix = config.model(config, config.mu)[0]
    K_operator = scipy.linalg.expm(config.dt * K_matrix)
    K_operator_inverse = scipy.linalg.expm(-config.dt * K_matrix)

    U_list = deepcopy(config.U); V_list = deepcopy(config.V)

    for U, V in zip(U_list, V_list):
        config.U = U; config.V = V;
        phi_field = config.field(config, K_operator, K_operator_inverse, K_matrix, gpu_avail)
        phi_field.copy_to_GPU()

        local_workdir = os.path.join(config.workdir, 'U_{:.2f}_V_{:.2f}'.format(U, V))  # add here all parameters that are being iterated
        os.makedirs(local_workdir, exist_ok=True)

        obs_files = []
        log_file = open(os.path.join(local_workdir, 'general_log.dat'), 'w')

        for n_sweep in range(config.n_sweeps):
            accept_history = []
            current_field, light, heavy = perform_sweep(phi_field, n_sweep)

            obs_l, names_l = light; obs_h, names_h = heavy

            ### light logging ###
            if n_sweep == 0:
                log_file.write('step ' + ('{:s} d{:s} ' * len(names_l)).format(*[x for pair in zip(names_l, names_l) for x in pair])); log_file.write('\n')
            log_file.write(('{:d} ' + '{:.5f} ' * len(obs_l)).format(n_sweep, *obs_l)); log_file.write('\n')
            print(('{:d} ' + '{:.5f} ' * len(obs_l)).format(n_sweep, *obs_l))
            log_file.flush()


            ### heavy logging ###
            if n_sweep == config.thermalization:
                for obs_name in names_h:
                    obs_files.append(open(os.path.join(local_workdir, obs_name + '.dat'), 'w'))
                
                obs_files[-1].write('step sign_obs dsign_obs ')
                for adj in current_field.adj_list:
                    obs_files[-1].write("f({:.5e}/{:d}/{:d}) df({:.5e}/{:d}/{:d}) ".format(adj[3], \
                                        adj[1], adj[2], adj[3], adj[1], adj[2])); obs_files[-1].write('\n')
                

            data_per_name = len(current_field.adj_list) * 2  # mean and std
            offset = 2
            for n, file in enumerate(obs_files):
                data = obs_h[:offset]; file.write(('{:d} ' + '{:.6e} ' * offset).format(n_sweep, *data))  # for sign
                data = obs_h[offset + data_per_name * n : offset + data_per_name * (n + 1)]
                file.write(("{:.6e} " * len(data)).format(*data)); file.write('\n')
                file.flush()
        log_file.close()
        [file.close() for file in obs_files]
