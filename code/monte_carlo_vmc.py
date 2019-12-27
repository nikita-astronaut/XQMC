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

def remove_singularity(S):
    for i in range(S.shape[0]):
        if S[i, i] < 1e-4:
            S[i, :] = 0.0
            S[:, i] = 0.0
            S[i, i] = 1.0
    return S

# <<Borrowed>> from Tom
def import_config(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the network from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module

config_vmc = import_config(sys.argv[1])
from config_vmc import MC_parameters as config_vmc
config_vmc = config_vmc()

if config_vmc.tests:
    if tests.perform_all_tests(config_vmc):
        print('\033[92m All tests passed successfully \033[0m')
    else:
        print('\033[91m Warning: some of the tests failed! \033[0m')

if config_vmc.visualisation:
    visualisation.plot_fermi_surface(config_vmc)
    visualisation.plot_all_pairings(config_vmc)
    visualisation.plot_Jastrow(config_vmc)

n_cpus_max = psutil.cpu_count(logical = True) 
print('max available CPUs:', n_cpus_max)
n_cpus = config_vmc.n_cpus
if config_vmc.n_cpus == -1:
    n_cpus = n_cpus_max
print('performing simulation at', n_cpus, 'CPUs')


def get_MC_chain_result(config_vmc, pairings_list, opt_parameters, final_state = False):
    hamiltonian = config_vmc.hamiltonian(config_vmc)
    if final_state == False:
        wf = wavefunction_singlet(config_vmc, pairings_list, *opt_parameters, False, None)
    else:
        wf = wavefunction_singlet(config_vmc, pairings_list, *opt_parameters, True, final_state)
    if not wf.with_previous_state:
        for MC_step in range(config_vmc.MC_chain * 2):
            wf.perform_MC_step()

    energies = []
    Os = []
    acceptance = []
    t_energies = 0
    t_steps = 0
    t_forces = 0
    t_observables = 0
    observables = []

    for MC_step in range(config_vmc.MC_chain):
        if MC_step % config_vmc.correlation == 0:
            wf.perform_explicit_GF_update()
            t = time()
            energies.append(hamiltonian(wf))
            t_energies += time() - t
            t = time()
            Os.append(wf.get_O())
            t_forces += time() - t

        t = time()
        if MC_step % config_vmc.observables_frequency == 0:
            obs, names = observables_vmc.compute_observables(wf)
            observables.append(obs)
        t_observables += time() - t

        t = time()
        acceptance.append(wf.perform_MC_step()[0])
        t_steps += time() - t

    print(t_observables, t_energies, t_steps, wf.update, wf.wf, t_forces)
    return energies, Os, acceptance, wf.get_state(), observables, names

pairings_list = config_vmc.pairings_list
pairings_names = config_vmc.pairings_list_names

U_list = deepcopy(config_vmc.U)
for U in U_list:
    gap_parameters = config_vmc.initial_gap_parameters
    jastrow_parameters = config_vmc.initial_jastrow_parameters
    mu_parameter = config_vmc.initial_mu_parameters

    config_vmc.U = U
    config_vmc.V = U
    H = config_vmc.hamiltonian(config_vmc)
 
    log_file = open(config_vmc.log_name + '_U_' + str(U) + '.dat', 'w')
    final_states = []

    log_file.write("⟨opt_step⟩ ⟨energy⟩ ⟨denergy⟩ ⟨acceptance⟩ ⟨force⟩")
    for gap_name in pairings_names:
        log_file.write(" ⟨" + gap_name + "⟩")
    for i in range(len(jastrow_parameters)):
        log_file.write(" ⟨jastrow_" + str(i) + "⟩")
    log_file.write(' ⟨mu_BCS⟩\n')

    observables_log = open(config_vmc.observables_log_name + '_U_' + str(U) + '.dat', 'w')
    observables_log.write("⟨opt_step⟩ ⟨energy⟩ ⟨denergy⟩ ⟨acceptance⟩")


    for n_step in range(config_vmc.optimisation_steps):
        if n_step == 0:
            results = Parallel(n_jobs=n_cpus)(delayed(get_MC_chain_result)(config_vmc, pairings_list, \
                                                                           (mu_parameter, gap_parameters, jastrow_parameters)) for i in range(n_cpus))
        else:
            results = Parallel(n_jobs=n_cpus)(delayed(get_MC_chain_result)(config_vmc, pairings_list, \
                                                                           (mu_parameter, gap_parameters, jastrow_parameters), \
                                                                           final_state = final_states[i]) for i in range(n_cpus))
        energies = np.concatenate([np.array(x[0]) for x in results], axis = 0)
        Os = np.concatenate([np.array(x[1]) for x in results], axis = 0)
        acceptance = np.mean(np.concatenate([np.array(x[2]) for x in results], axis = 0))
        final_states = [x[3] for x in results]

        observables = np.concatenate([np.array(x[4]) for x in results], axis = 0)
        observables_names = results[0][5]

        if n_step == 0:
            for obs_name in observables_names:
                observables_log.write(" ⟨" + obs_name + "⟩ ⟨d" + obs_name + "⟩")
            observables_log.write('\n')

        observables = np.concatenate([observables.mean(axis = 0)[:, np.newaxis], observables.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)
        vol = config_vmc.total_dof // 2

        Os_mean = np.mean(Os, axis = 0)
        forces = -2 * (np.einsum('i,ik->k', energies.conj(), Os) / len(energies) - np.mean(energies.conj()) * Os_mean).real

        print('estimating gradient on ', len(energies), 'samples', flush = True)
        print('\033[93m <E> / t / vol = ' + str(np.mean(energies) / vol) + '+/-' + str(np.std(energies) / np.sqrt(len(energies)) / vol) + '\033[0m', flush = True)
        print('\033[92m acceptance =' + str(acceptance) + '\033[0m', flush = True)
        print('\033[94m |forces_raw| = ' + str(np.sqrt(np.sum(forces ** 2))) + ' ' + str(forces) + '\033[0m', flush = True)


        Os_mean = np.repeat(Os_mean[np.newaxis, ...], len(Os), axis = 0)
        S_cov = (np.einsum('nk,nl->kl', (Os - Os_mean).conj(), (Os - Os_mean)) / Os.shape[0]).real
    
        S_cov = remove_singularity(S_cov)

        forces_pc = forces / np.sqrt(np.abs(np.diag(S_cov)))  # below (6.52)
        S_cov_pc = np.einsum('i,ij,j->ij', 1.0 / np.sqrt(np.abs(np.diag(S_cov))), S_cov, 1.0 / np.sqrt(np.abs(np.diag(S_cov))))  
        # (6.51, scale-invariant regularization)
        S_cov_pc += config_vmc.opt_parameters[0] * np.eye(S_cov_pc.shape[0])  # (6.54)
        S_cov_pc_inv = np.linalg.inv(S_cov_pc)

        step_pc = S_cov_pc_inv.dot(forces_pc)  # (6.52)
        step = step_pc / np.sqrt(np.abs(np.diag(S_cov)))

        if np.sqrt(np.sum(step ** 2)) / np.sqrt(np.sum(forces ** 2)) > 5:
            step = forces
            print('Too high enhancement due to S_cov matrix: switching to gradient descent for this step')

        print('\033[94m |forces_SR| = ' + str(np.sqrt(np.sum(step ** 2))) + ' ' + str(step) + '\033[0m', flush = True)
        step = config_vmc.opt_parameters[1] * step 

        mu_parameter += step[0]
        gap_parameters += step[1:1 + len(gap_parameters)]
        jastrow_parameters += step[1 + len(gap_parameters):]

        print('\033[91m mu = ' + str(mu_parameter) + ', pairings =' + str(gap_parameters) + ', Jastrow =' + str(jastrow_parameters) + '\033[0m', flush = True)
        log_file.write(("{:3d} {:.7e} {:.7e} {:.3e} {:.3e}" + " {:.7e}" * len(step) + "\n").format(n_step, np.mean(energies).real / vol,
                        np.std(energies).real / np.sqrt(len(energies)) / vol, acceptance, np.sqrt(np.sum(forces ** 2)),
                        *gap_parameters, *jastrow_parameters, mu_parameter))

        observables_log.write(("{:3d} {:.7e} {:.7e} {:.3e} " + " {:.5e}" * len(observables) + "\n").format(n_step, np.mean(energies).real / vol,
                            np.std(energies).real / np.sqrt(len(energies)) / vol, acceptance, 
                            *observables))
        log_file.flush()
        observables_log.flush()
    observables_log.close()
    log_file.close()
