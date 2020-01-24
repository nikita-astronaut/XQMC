import numpy as np
import time
import sys
import os
from wavefunction_vmc import wavefunction_singlet
import pairings
from joblib import Parallel, delayed
import psutil
from time import time
import visualisation
import tests
import observables_vmc
from copy import deepcopy
import os
import pickle
import config_vmc as cv_module

def perform_transition_analysis(Es, U_vecs, current_labels, config):
    if len(U_vecs) < 2:
        return current_labels

    A = np.abs(np.einsum('ij,jk->ik', np.conj(U_vecs[-2]).T, U_vecs[-1]))

    new_labels = np.zeros(A.shape[0])
    for i in range(len(new_labels)):
        new_labels[i] = current_labels[np.argsort(A[:, i])[-1]]
        A[np.argsort(A[:, i])[-1], i] = 0.0
    # print('remainings:', [np.sum(np.abs(A[:, j]) ** 2) for j in range(A.shape[1])])
    return new_labels.astype(np.int64)

def remove_singularity(S):
    for i in range(S.shape[0]):
        if S[i, i] < 1e-4:
            S[i, :] = 0.0
            S[:, i] = 0.0
            S[i, i] = 1.0
    return S

def save_parameters(mu_BCS, fugacity, sdw, cdw, gap, jastrow, local_workdir, step_no):
    params_dict = {'mu' : mu_BCS, 'fugacity' : fugacity, 'sdw' : sdw, 'cdw' : cdw, \
                   'gap' : gap, 'jastrow' : jastrow, 'step_no' : step_no}
    return pickle.dump(params_dict, open(os.path.join(local_workdir, 'last_opt_params.p'), "wb"))

def load_parameters(filename):
    params_dict = pickle.load(open(filename, "rb"))
    return params_dict['mu'], params_dict['fugacity'], params_dict['sdw'], \
           params_dict['cdw'], params_dict['gap'], \
           params_dict['jastrow'], params_dict['step_no']

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

config_vmc_file = import_config(sys.argv[1])
config_vmc_import = config_vmc_file.MC_parameters()

config_vmc = cv_module.MC_parameters()
config_vmc.__dict__ = config_vmc_import.__dict__.copy()


os.makedirs(config_vmc.workdir, exist_ok=True)
with open(os.path.join(config_vmc.workdir, 'config.py'), 'w') as target, \
     open(sys.argv[1], 'r') as source:  # save config file to workdir (to remember!!)
    target.write(source.read())


if config_vmc.visualisation:
    visualisation.plot_fermi_surface(config_vmc)
    visualisation.plot_all_pairings(config_vmc)
    visualisation.plot_all_Jastrow(config_vmc)

if config_vmc.tests:
    config_tests = deepcopy(config_vmc)
    config_tests.U = config_tests.U[0]; config_tests.V = config_tests.V[0];
    if tests.perform_all_tests(config_tests):
        print('\033[92m All tests passed successfully \033[0m')
    else:
        print('\033[91m Warning: some of the tests failed! \033[0m')

n_cpus_max = psutil.cpu_count(logical = True) 
print('max available CPUs:', n_cpus_max)
n_cpus = config_vmc.n_cpus
if config_vmc.n_cpus == -1:
    n_cpus = n_cpus_max
print('performing simulation at', n_cpus, 'CPUs')
config_vmc.MC_chain = config_vmc.MC_chain // n_cpus  # the MC_chain contains the total required number of samples

def get_MC_chain_result(n_iter, config_vmc, pairings_list, opt_parameters, twist, final_state = False):
    config_vmc.twist = tuple(twist)
    hamiltonian = config_vmc.hamiltonian(config_vmc)  # the Hubbard Hamiltonian will be initialized with the 


    if final_state == False:
        wf = wavefunction_singlet(config_vmc, pairings_list, *opt_parameters, False, None)
    else:
        wf = wavefunction_singlet(config_vmc, pairings_list, *opt_parameters, True, final_state)

    if not wf.with_previous_state or n_iter < 30:  # for first iterations we thermalize anyway (because everything is varying too fast)
        for MC_step in range(config_vmc.MC_thermalisation):
            wf.perform_MC_step()
    else:
        for MC_step in range(config_vmc.MC_thermalisation // 4):  # else thermalize a little bit
            wf.perform_MC_step()


    energies = []
    Os = []
    acceptance = []
    densities = []
    t_energies = 0
    t_steps = 0
    t_forces = 0
    t_observables = 0
    t_update = 0
    observables = []
    names = []

    for MC_step in range(int(config_vmc.MC_chain * (config_vmc.opt_parameters[2] ** n_iter))):
        if MC_step % config_vmc.correlation == 0:
            t = time()
            wf.perform_explicit_GF_update()
            t_update += time() - t

            t = time()
            energies.append(hamiltonian(wf))
            densities.append(wf.total_density())
            t_energies += time() - t

            t = time()
            Os.append(wf.get_O())
            t_forces += time() - t

        t = time()
        if MC_step % config_vmc.observables_frequency == 0 and n_iter > config_vmc.thermalization \
            and (n_iter - config_vmc.thermalization) % config_vmc.obs_calc_frequency == 1:
            obs, names = observables_vmc.compute_observables(wf)
            observables.append(obs)
        t_observables += time() - t

        t = time()
        acceptance.append(wf.perform_MC_step()[0])
        t_steps += time() - t

    print(t_update, t_observables, t_energies, t_forces, t_steps, wf.update, wf.wf, twist)
    return energies, Os, acceptance, wf.get_state(), observables, names, wf.U_full, wf.E, densities

pairings_list = config_vmc.pairings_list
pairings_names = config_vmc.pairings_list_names

U_list = deepcopy(config_vmc.U)
V_list = deepcopy(config_vmc.V)
J_list = deepcopy(config_vmc.J)
mu_list = deepcopy(config_vmc.mu)

for U, V, J, mu in zip(U_list, V_list, J_list, mu_list):
    local_workdir = os.path.join(config_vmc.workdir, 'U_{:.2f}_V_{:.2f}_J_{:.2f}_mu_{:.2f}'.format(U, V, J, mu))  # add here all parameters that are being iterated
    os.makedirs(local_workdir, exist_ok=True)

    obs_files = []
    loaded_from_external = False
    if config_vmc.load_parameters:
        if config_vmc.load_parameters_path is not None:
            loaded_from_external = True
            filename = config_vmc.load_parameters_path
        else:
            filename = os.path.join(local_workdir, 'last_opt_params.p')
        mu_parameter, fugacity_parameter, sdw_parameter, cdw_parameter, gap_parameters, \
            jastrow_parameters, last_step = load_parameters(filename)
    else:
        gap_parameters = config_vmc.initial_gap_parameters
        jastrow_parameters = config_vmc.initial_jastrow_parameters
        mu_parameter = config_vmc.initial_mu_parameters
        sdw_parameter = config_vmc.initial_sdw_parameters
        cdw_parameter = config_vmc.initial_cdw_parameters
        fugacity_parameter = config_vmc.initial_fugacity_parameter
        last_step = 0

    config_vmc.U = U
    config_vmc.V = V
    config_vmc.J = J
    config_vmc.mu = mu

    H = config_vmc.hamiltonian(config_vmc)
 
    log_file = open(os.path.join(local_workdir, 'general_log.dat'), 'a+')

    final_states = [False] * n_cpus


    ### write log header only if we start from some random parameters ###
    if last_step == 0 or loaded_from_external:
        log_file.write("⟨opt_step⟩ ⟨energy⟩ ⟨denergy⟩ ⟨n⟩ ⟨dn⟩ ⟨variance⟩ ⟨acceptance⟩ ⟨force⟩ ⟨force_SR⟩ ⟨gap⟩")
        for gap_name in pairings_names:
            log_file.write(" ⟨" + gap_name + "⟩")
        for jastrow in config_vmc.adjacency_list:
            log_file.write(" ⟨jastrow_" + str(jastrow[1]) + '-' + \
                            str(jastrow[2]) + '_' + str(round(jastrow[3], 2)) + "⟩")

        for i in range(len(sdw_parameter)):
            log_file.write(" ⟨sdw_" + str(i % config_vmc.n_orbitals) + '_' + \
                            str((i // config_vmc.n_orbitals) % config_vmc.n_sublattices) + "⟩")
        for i in range(len(cdw_parameter)):
            log_file.write(" ⟨cdw_" + str(i % config_vmc.n_orbitals) + '_' + \
                            str((i // config_vmc.n_orbitals) % config_vmc.n_sublattices) + "⟩")

        log_file.write(' ⟨mu_BCS⟩ ⟨fugacity⟩\n')

    ### keeping track of levels occupations ###
    Es = []
    U_vecs = []
    initial_state_idx = np.arange(config_vmc.total_dof)  # enumerates the number of states with respect to adiabatic evolution of the initial states (threads)
    current_selected_states = np.arange(config_vmc.total_dof // 2)  # labels of the threads that are now in the min-level set [better they do not change...]


    ### generate twists once and for all (Sandro's suggestion) ###
    twists = [1., 1.]
    if config_vmc.BC_twist:
        twists = [np.exp(1.0j * np.random.uniform(0, 1, size = 2) * np.pi * 2) for _ in range(n_cpus)]  # np.exp(i \theta_x), np.exp(i \theta_y) for spin--up
    else:
        twists = [[1., 1.] for _ in range(n_cpus)]

    force_SR_abs_history = [10]
    force_abs_history = [100000]
    for n_step in range(last_step, last_step + config_vmc.optimisation_steps):
        results = Parallel(n_jobs=n_cpus)(delayed(get_MC_chain_result)(n_step - last_step, deepcopy(config_vmc), pairings_list, \
            (mu_parameter, fugacity_parameter, sdw_parameter, cdw_parameter, gap_parameters, jastrow_parameters), \
             twist = twists[i], final_state = final_states[i]) for i in range(n_cpus))
        gap = np.min([-results[i][7][np.argsort(results[i][7])[config_vmc.total_dof // 2 - 1]] + \
                       results[i][7][np.argsort(results[i][7])[config_vmc.total_dof // 2]] for i in range(n_cpus)])

        vol = config_vmc.total_dof // 2
        energies = [np.array(x[0]) for x in results]  # collection of all energy sets obtained from different threads
        variances = [np.real((np.mean(np.abs(energies_theta) ** 2) - np.mean(energies_theta) ** 2) / vol) for energies_theta in energies]
        mean_variance = np.mean(variances)

        Os = [np.array(x[1]) for x in results]

        acceptance = np.mean(np.concatenate([np.array(x[2]) for x in results], axis = 0))
        final_states = [x[3] for x in results]
        densities = np.concatenate([np.array(x[8]) for x in results], axis = 0)

        print('estimating gradient on ', len(energies), 'samples', flush = True)
        print('\033[93m <E> / t / vol = ' + str(np.mean(energies) / vol) + '+/-' + str(np.std(energies) / np.sqrt(len(energies)) / vol) + '\033[0m', flush = True)
        print('\033[93m <n> / vol = ' + str(np.mean(densities) / vol) + '+/-' + str(np.std(densities) / np.sqrt(len(densities)) / vol) + '\033[0m', flush = True)
        print('\033[93m σ^2 / t / vol = ' + str(mean_variance) + '\033[0m', flush = True)
        print('\033[92m acceptance =' + str(acceptance) + '\033[0m', flush = True)


        Os_mean = [np.mean(Os_theta, axis = 0) for Os_theta in Os]
        forces = np.array([-2 * (np.einsum('i,ik->k', energies_theta.conj(), Os_theta) / len(energies_theta) - np.mean(energies_theta.conj()) * Os_mean_theta).real for \
                           energies_theta, Os_theta, Os_mean_theta in zip(energies, Os, Os_mean)])  # all forces calculated independently for every twist angle
        forces = np.mean(forces, axis = 0)  # after calculation of the force independently for every twist angle, we average over all forces

        ### SR STEP ###
        Os_mean = [np.repeat(Os_mean_theta[np.newaxis, ...], len(Os_theta), axis = 0) for Os_mean_theta, Os_theta in zip(Os_mean, Os)]
        S_cov = [(np.einsum('nk,nl->kl', (Os_theta - Os_mean_theta).conj(), (Os_theta - Os_mean_theta)) / Os_theta.shape[0]).real \
                 for Os_mean_theta, Os_theta in zip(Os_mean, Os)]  # SR_matrix is computed independently for every twist angle theta

        S_cov = np.array([remove_singularity(S_cov_theta) for S_cov_theta in S_cov])
        S_cov = np.mean(S_cov, axis = 0)

        forces_pc = forces / np.sqrt(np.abs(np.diag(S_cov)))  # below (6.52)
        S_cov_pc = np.einsum('i,ij,j->ij', 1.0 / np.sqrt(np.abs(np.diag(S_cov))), S_cov, 1.0 / np.sqrt(np.abs(np.diag(S_cov))))  
        # (6.51, scale-invariant regularization)
        S_cov_pc += config_vmc.opt_parameters[0] * np.eye(S_cov_pc.shape[0])  # (6.54)
        S_cov_pc_inv = np.linalg.inv(S_cov_pc)

        step_pc = S_cov_pc_inv.dot(forces_pc)  # (6.52)
        step = step_pc / np.sqrt(np.abs(np.diag(S_cov)))

        print('\033[94m |f| = {:.4e}, |f_SR| = {:.4e} \033[0m'.format(np.sqrt(np.sum(forces ** 2)), \
                                                                      np.sqrt(np.sum(step ** 2))))
        step_abs = np.sqrt(np.sum(step ** 2))
        force_abs = np.sqrt(np.sum(forces ** 2))
        clip_length = np.min([10, len(force_abs_history)])
        if step_abs > 2. * np.median(force_SR_abs_history[-clip_length:]) or force_abs > 2. * np.median(force_abs_history[-clip_length:]):
            print('Warning! The force is too high -- this iteration will NOT be performed')
            step = 0.0 * step
        else:
            force_SR_abs_history.append(step_abs)
            force_abs_history.append(force_abs)
            step = config_vmc.opt_parameters[1] * step 

        offset = 0
        mu_parameter += step[offset]; offset += 1
        fugacity_parameter += step[offset]; offset += 1
        sdw_parameter += step[offset:offset + len(sdw_parameter)]; offset += len(sdw_parameter)
        cdw_parameter += step[offset:offset + len(cdw_parameter)]; offset += len(cdw_parameter)
        gap_parameters += step[offset:offset + len(gap_parameters)]; offset += len(gap_parameters)
        jastrow_parameters += step[offset:]
        save_parameters(mu_parameter, fugacity_parameter, sdw_parameter, 
                        cdw_parameter, gap_parameters, jastrow_parameters,
                        local_workdir, n_step)


        print('\033[91m mu_BCS = ' + str(mu_parameter) + 'fugacity = ' + str(fugacity_parameter) + \
              ' pairings =' + str(gap_parameters) + \
              ', Jastrow =' + str(jastrow_parameters) + \
              ', SDW/CDW = ' + str([sdw_parameter, cdw_parameter]) + '\033[0m', flush = True)
        log_file.write(("{:d} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.3e} {:.3e} {:.3e} {:.7e}" + " {:.7e} " * len(step) + "\n").format(n_step, \
                        np.mean(energies).real / vol, np.std(energies).real / np.sqrt(len(energies)) / vol, \
                        np.mean(densities).real / vol, np.std(densities).real / np.sqrt(len(densities)) / vol, \
                        mean_variance, acceptance, np.sqrt(np.sum(forces ** 2)), np.sqrt(np.sum(step ** 2)),
                        gap, *gap_parameters, *jastrow_parameters, *sdw_parameter, *cdw_parameter, mu_parameter, fugacity_parameter))
        log_file.flush()

        ### END SR STEP ###


        observables = np.concatenate([np.array(x[4]) for x in results], axis = 0)
        observables_names = results[0][5]
        if len(observables_names) == 0:
            continue

        if n_step == config_vmc.thermalization + 1:
            for obs_name in observables_names:
                obs_files.append(open(os.path.join(local_workdir, obs_name + '.dat'), 'a+'))
                
                if 'density' in obs_name:
                    adj_list = config_vmc.adjacency_list[:config_vmc.n_adj_density]  # on-site and nn
                else:
                    adj_list = config_vmc.adjacency_list[-config_vmc.n_adj_pairings:]  # only largest distance


                for adj in adj_list:
                    obs_files[-1].write("f({:.5e}/{:d}/{:d}) df({:.5e}/{:d}/{:d}) ".format(adj[3], \
                                        adj[1], adj[2], adj[3], adj[1], adj[2]))
                obs_files[-1].write('\n')

        observables = np.concatenate([observables.mean(axis = 0)[:, np.newaxis],\
                      observables.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)
        
        ### to files writing ###
        data_per_name_pairings = current_field.config.n_adj_pairings * 2  # mean and std
        data_per_name_densities = current_field.config.n_adj_density * 2  # mean and std
        current_written = 0
        for file in obs_files:
            file.write(('{:d} ').format(n_step))  # for sign and epoch no
            data_size = data_per_name_densities if 'density' in file.name else data_per_name_pairings

            data = obs_h[current_written:current_written + data_size]
            current_written += data_size
            file.write(("{:.6e} " * len(data)).format(*data)); file.write('\n')
            file.flush()
    log_file.close()
    [file.close() for file in obs_files]
