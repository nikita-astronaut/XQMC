import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np
#np.set_printoptions(linewidth=np.inf)
import sys
#np.set_printoptions(threshold=sys.maxsize)
import time
import sys
from wavefunction_vmc import wavefunction_singlet
import opt_parameters.pairings
from joblib import Parallel, delayed, parallel_backend
import joblib
print('joblib version', joblib.__version__)
import psutil
from time import time
import visualisation
import tests
import observables_vmc
from copy import deepcopy
import os
import pickle
import config_vmc as cv_module
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

import models

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

def extract_MC_data(results, config_vmc, num_twists):
    gaps = [x[9] for x in results]
    gap = np.min(gaps)
    vol = config_vmc.total_dof // 2
    energies = [np.array(x[0]) for x in results]  # collection of all energy sets obtained from different threads
    variances = [np.real((np.mean(np.abs(energies_theta) ** 2) - np.mean(energies_theta) ** 2) / vol) for energies_theta in energies]
    mean_variance = np.mean(variances)
    Os = [np.array(x[1]) for x in results]

    acceptance = np.mean(np.concatenate([np.array(x[2]) for x in results], axis = 0))
    final_states = [x[3] for x in results]
    densities = np.concatenate([np.array(x[8]) for x in results], axis = 0)

    orbitals_in_use = [x[6] for x in results]
    occupied_numbers = [x[7] for x in results]
    return gaps, gap, energies, mean_variance, Os, acceptance, \
           final_states, densities, orbitals_in_use, occupied_numbers


def clip_forces(clips, step):
    for i in range(len(step)):
        if np.abs(step[i]) > clips[i]:
            step[i] = clips[i] * np.sign(step[i])
    return step



def make_SR_step(Os, energies, config_vmc, twists, gaps, n_iter, mask):
    def remove_singularity(S):
        for i in range(S.shape[0]):
            if S[i, i] < 1e-4:
                S[i, :] = 0.0
                S[:, i] = 0.0
                S[i, i] = 1.0
        return S

    Os = [np.einsum('ik,k->ik', Os_theta, mask) for Os_theta in Os]  # non-optimisable parameters do not exist at this point

    Os_mean = [np.mean(Os_theta, axis = 0) for Os_theta in Os]
    forces = np.array([-2 * (np.einsum('i,ik->k', energies_theta.conj() - np.mean(energies_theta.conj()), Os_theta) / len(energies_theta) ) for \
                       energies_theta, Os_theta in zip(energies, Os)])  # all forces calculated independently for every twist angle
    for i in range(forces.shape[0]):
        print(i, repr(forces[i]))
        #print(np.sort(np.abs(energies[i].conj() - np.mean(energies[i].conj())))))
        #print(np.sort(np.abs(Os[i][:, 0])))
    forces = np.mean(forces, axis = 0)  # after calculation of the force independently for every twist angle, we average over all forces

    print(forces, 'FORCES --- are they real?')
    forces = forces.real
    #forces = forces / 36. * config_vmc.Ls ** 2
    print('E_im = {:.10f} +/- {:.10f}'.format(np.mean(np.array(energies)), np.std(np.array(energies).flatten())))

    ### SR STEP ###
    Os_mean = [np.repeat(Os_mean_theta[np.newaxis, ...], len(Os_theta), axis = 0) for Os_mean_theta, Os_theta in zip(Os_mean, Os)]
    S_cov = [(np.einsum('nk,nl->kl', (Os_theta - Os_mean_theta).conj(), (Os_theta - Os_mean_theta)) / Os_theta.shape[0]).real \
             for Os_mean_theta, Os_theta in zip(Os_mean, Os)]  # SR_matrix is computed independently for every twist angle theta

    # S_cov = np.array([remove_singularity(S_cov_theta) for S_cov_theta in S_cov])
    S_cov = np.mean(S_cov, axis = 0)
    print(S_cov)

    eigvals, eigvecs = np.linalg.eigh(S_cov)
    #print('total_redundancies = ', np.sum(np.abs(eigvals) < 1e-6))
    print('S_cov eigvals = ', eigvals)
    print(np.diag(S_cov))
    for val, vec in zip(eigvals, eigvecs.T):
        if np.abs(val) < 1e-6:
            print('redundancy observed:', vec)
            for val, name in zip(vec, config_vmc.all_names):
                if np.abs(val) > 1e-1:
                    print(name, val)

    # https://journals.jps.jp/doi/pdf/10.1143/JPSJ.77.114701 (formula 71)
    # removal of diagonal singularities

    '''
    S_cov += config_vmc.opt_parameters[0] * np.diag(np.diag(S_cov))

    # https://journals.jps.jp/doi/10.1143/JPSJ.77.114701 (formula 76)
    # regularized inversion with truncation of redundant directions
    u, s, v = np.linalg.svd(S_cov)
    S_cov_inv = np.zeros(S_cov.shape)
    keep_lambdas = (s / s.max()) > config_vmc.opt_parameters[3]
    for lambda_idx in range(len(s)):
        if not keep_lambdas[lambda_idx]:
            continue
        S_cov_inv += (1. / s[lambda_idx]) * \
                      np.einsum('i,j->ij', v.T[:, lambda_idx], u.T[lambda_idx, :])
    step = S_cov_inv.dot(forces)
    '''
    #if n_iter < 20:
    #    S_cov_pc = S_cov + config_vmc.opt_parameters[0] * np.diag(np.diag(S_cov))
    #else:
    S_cov_pc = S_cov + config_vmc.opt_parameters[0] * np.diag(np.diag(S_cov))
    S_cov_pc += np.eye(S_cov.shape[0]) * 1e-3

    step = np.linalg.inv(S_cov_pc).dot(forces)
    #print('\033[94m |f| = {:.4e}, |f_SR| = {:.4e} \033[0m'.format(np.sqrt(np.sum(forces ** 2)), \
    #                                                              np.sqrt(np.sum(step ** 2))))
    #print(forces[-3], step[-3], 'forces of gap')
    print(forces)
    print(step)
    return step, forces


def write_initial_logs(log_file, force_file, force_SR_file, config_vmc):
    log_file.write("⟨opt_step⟩ ⟨ReE⟩ ⟨ImE⟩ ⟨denergy⟩ ⟨n⟩ ⟨dn⟩ ⟨variance⟩ ⟨acceptance⟩ ⟨force⟩ ⟨force_SR⟩ ⟨gap⟩ n_above_FS ")
    force_file.write("⟨opt_step⟩ ")
    force_SR_file.write("⟨opt_step⟩ ")

    for name in config_vmc.all_names:
        log_file.write(name + ' ')
        force_file.write(name + ' ')
        force_SR_file.write(name + ' ')
    
    log_file.write('\n')
    force_file.write('\n')
    force_SR_file.write('\n')

    return

def print_model_summary(config_vmc):
    print('Model geometry: {:d}x{:d}, {:d} sublattices, {:d} orbitals'.format(config_vmc.Ls, \
           config_vmc.Ls, config_vmc.n_sublattices, config_vmc.n_orbitals))
    print('epsilon_EM = {:2f}, '.format(config_vmc.epsilon))
    print('Monte Carlo chain length = {:d}, lr = {:.3f}, epsilon = {:.5f}'.format(\
        config_vmc.MC_chain, config_vmc.opt_parameters[1], config_vmc.opt_parameters[0])
    )
    print('Work with periodic BC' if not config_vmc.BC_twist \
     else 'Work with twisted BC, n_chains = {:d}'.format(config_vmc.n_chains))
    print('Mesh of k-points: {:s}'.format(config_vmc.twist_mesh))

    print('Work in Grand Canonical Approach' if not config_vmc.PN_projection \
     else 'Work in Canonical Approach at <n> = {:.2f}'.format(config_vmc.Ne / config_vmc.total_dof * 2))

    print('Gap parameters: ', config_vmc.pairings_list_names)
    print('Hopping parameters: ', [hopping for hopping in config_vmc.hopping_names])
    print('Jastrow parameters: ', [jastrow[-1] for jastrow in config_vmc.jastrows_list])

    print('mu_BCS initial guess {:.3f}'.format(config_vmc.initial_parameters[0]))

    print('Total number of optimized parameters: ', np.sum(config_vmc.layout))
    return

def write_intermediate_log(log_file, force_file, force_SR_file, n_step, vol, energies, densities, \
                           mean_variance, acceptance, forces, step, gap, \
                           n_above_FS, parameters):
    log_file.write(("{:d} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.3e} {:.3e} {:.3e} {:.7e} {:d}" + " {:.7e} " * len(step) + "\n").format(n_step, \
                    np.mean(energies).real / vol, np.mean(energies).imag / vol, np.std(energies).real / np.sqrt(len(energies)) / vol, \
                    np.mean(densities).real / vol, np.std(densities).real / np.sqrt(len(densities)) / vol, \
                    mean_variance, acceptance, np.sqrt(np.sum(forces ** 2)), np.sqrt(np.sum(step ** 2)),
                    gap, n_above_FS, *parameters))
    log_file.flush()

    force_file.write(("{:d}" + " {:.7e} " * len(step) + "\n").format(n_step, *forces))
    force_file.flush()

    force_SR_file.write(("{:d}" + " {:.7e} " * len(step) + "\n").format(n_step, *step))
    force_SR_file.flush()
    return

def write_gaps_log(gaps_file, gaps, step):
    gaps_file.write(("{:d}" + " {:.7e} " * len(gaps) + "\n").format(step, *gaps))
    gaps_file.flush()

    return

def create_obs_files(observables_names, config_vmc):
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
    return

def write_observables(n_step, obs_files, observables, config_vmc):
    observables = np.concatenate([observables.mean(axis = 0)[:, np.newaxis],\
                  observables.std(axis = 0)[:, np.newaxis]], axis = 1).reshape(-1)

    data_per_name_pairings = config_vmc.n_adj_pairings * 2 # mean and std
    data_per_name_densities = config_vmc.n_adj_density * 2  # mean and std
    current_written = 0
    for file in obs_files:
        file.write(('{:d} ').format(n_step))  # for sign and epoch no
        data_size = data_per_name_densities if 'density' in file.name else data_per_name_pairings

        data = obs_h[current_written:current_written + data_size]
        current_written += data_size
        file.write(("{:.6e} " * len(data)).format(*data)); file.write('\n')
        file.flush()
    return


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

def save_parameters(parameters, local_workdir, step_no):
    params_dict = {'parameters' : parameters, 'step_no' : step_no}
    return pickle.dump(params_dict, open(os.path.join(local_workdir, 'last_opt_params.p'), "wb"))

def load_parameters(filename):
    params_dict = pickle.load(open(filename, "rb"))
    return params_dict['parameters'], params_dict['step_no']


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


def get_MC_chain_result(n_iter, config_vmc, pairings_list, parameters, \
                        twists, final_states, orbitals_in_use, \
                        K_matrices_up, K_matrices_down, regs):
    res = []
    idx = 0
    for twist, final_state, o, K_up, K_down, reg in zip(twists, final_states, orbitals_in_use, \
                                                        K_matrices_up, K_matrices_down, regs):
        t = time()
        res.append(_get_MC_chain_result(n_iter, config_vmc, pairings_list, \
                                        parameters, twist, final_state, o, \
                                        K_up, K_down, reg, idx))
        print('one chain takes =', time() - t, flush = True)
        idx += 1
    return res


def _get_MC_chain_result(n_iter, config_vmc, pairings_list, \
                         parameters, twist, final_state = False, orbitals_in_use = None,
                         K_up = None, K_down = None, reg = None, twist_id = 0):
    config_vmc.twist = tuple(twist)
    t = time()
    hamiltonian = config_vmc.hamiltonian(config_vmc, K_up, K_down)  # the Hubbard Hamiltonian will be initialized with the 
    print('H init takes {:.10f}'.format(time() - t))

    # print(K_up, K_down, flush=True)
    ''' 
    if final_state == False:
        wf = wavefunction_singlet(config_vmc, pairings_list, parameters, False, None)
    else:
        wf = wavefunction_singlet(config_vmc, pairings_list, parameters, True, final_state)
    '''
    

    t = time()
    if final_state == False:
        wf = wavefunction_singlet(config_vmc, pairings_list, parameters, \
                              False, None, orbitals_in_use, \
                              False, K_up, K_down, reg)  # always start with bare configuration
    else:
        print('With presaved state')
        wf = wavefunction_singlet(config_vmc, pairings_list, parameters, \
                              True, final_state, orbitals_in_use, \
                              False, K_up, K_down, reg)
    print('WF Init takes {:.10f}'.format(time() - t))
    wf.perform_explicit_GF_update()
    print(hamiltonian(wf))
    print(wf.current_det)
    configs = []

    t_steps = 0
    t = time()
    n_steps = 0
    for MC_step in range(config_vmc.MC_thermalisation):
        n_steps += 1
        #t = time()
        acc = wf.perform_MC_step(demand_accept = False)[0]
        #print('MC step takes {:.10f}, accepted = {:b}'.format(time() - t, acc))

    t_steps += time() - t
    energies = []
    Os = []
    acceptance = []
    densities = []
    t_energies = 0
    t_forces = 0
    t_observables = 0
    t_update = 0
    observables = []
    names = []

    plus_densities = []
    minus_densities = []

    precision_factor = 1. if config_vmc.opt_raw > n_iter else 4.
    tc = time()
    for MC_step in range(int(precision_factor * config_vmc.MC_chain * (config_vmc.opt_parameters[2] ** n_iter))):
        n_steps += 1
        if MC_step % config_vmc.correlation == 0:
            t = time()
            wf.perform_explicit_GF_update()
            t_steps += time() - t

            t = time()
            energies.append(hamiltonian(wf))
            densities.append(wf.total_density())
            plus_densities.append(wf.total_plus_density())
            minus_densities.append(wf.total_minus_density())
            t_energies += time() - t
            #print('energies take {:.10f}'.format(time() - t))

            t = time()
            if config_vmc.generator_mode:  # forces only if necessary
                Os.append(wf.get_O())
            t_forces += time() - t
            #print('forces take {:.10f}'.format(time() - t))

        t = time()
        if MC_step % config_vmc.observables_frequency == 0 and n_iter > config_vmc.thermalization \
            and (n_iter - config_vmc.thermalization) % config_vmc.obs_calc_frequency == 1:
            obs, names = observables_vmc.compute_observables(wf)
            observables.append(obs)
        t_observables += time() - t

        #t = time()
        acceptance.append(wf.perform_MC_step(demand_accept = False)[0])
        #configs.append(wf.state.copy() * 1.0)
        #print('MC step take {:.10f}'.format(time() - t))
        t_steps += time() - t
    # print('t_chain = ', time() - tc)
    print(t_update, t_observables, t_energies, t_forces, t_steps, wf.update, wf.wf, twist)
    print(wf.t_jastrow / n_steps, wf.t_det / n_steps, wf.t_choose_site / n_steps, wf.t_overhead_after / n_steps, wf.t_gf_update / np.sum(acceptance), wf.t_ab / np.sum(acceptance))
    print(wf.t_jastrow, wf.t_det, wf.t_choose_site, wf.t_overhead_after, wf.t_gf_update, wf.t_ab, flush=True)
    #        self.t_jastrow = 0
    #    self.t_det = 0
    #    self.t_choose_site = 0
    #    self.t_overhead_after = 0
    #    self.t_gf_update = 0
    #    self.t_ab = 0
    #print('accepted = {:d}, rejected_filling = {:d}, rejected_factor = {:d}'.format(wf.accepted, wf.rejected_filled, wf.rejected_factor))
    
    '''
    U = wf.U_matrix
    plus_valley_particle = np.einsum('ij,ij->j', U[np.arange(0, config_vmc.total_dof // 2, 2), ...], \
                                                 U[np.arange(0, config_vmc.total_dof // 2, 2), ...].conj()).real
    plus_valley_hole = np.einsum('ij,ij->j', U[np.arange(config_vmc.total_dof // 2, config_vmc.total_dof, 2), ...], \
                                             U[np.arange(config_vmc.total_dof // 2, config_vmc.total_dof, 2), ...].conj()).real
    minus_valley_particle = np.einsum('ij,ij->j', U[np.arange(1, config_vmc.total_dof // 2, 2), ...], \
                                                  U[np.arange(1, config_vmc.total_dof // 2, 2), ...].conj()).real
    minus_valley_hole = np.einsum('ij,ij->j', U[np.arange(config_vmc.total_dof // 2 + 1, config_vmc.total_dof, 2), ...], \
                                              U[np.arange(config_vmc.total_dof // 2 + 1, config_vmc.total_dof, 2), ...].conj()).real
    thr = 0.99
    plus_valley_particle = plus_valley_particle > thr
    plus_valley_hole = plus_valley_hole > thr
    minus_valley_particle = minus_valley_particle > thr
    minus_valley_hole = minus_valley_hole > thr

    print(twist, np.mean(np.abs(wf.wf_ampls)), np.mean(acceptance), wf.gap, np.mean(densities), np.std(densities), \
          np.mean(plus_densities), np.std(plus_densities), np.mean(minus_densities), np.std(minus_densities), 'wave function ampl!', \
          np.mean(np.abs(np.abs(Os)[:, 0])), \
          np.sum(plus_valley_particle), np.sum(minus_valley_particle), np.sum(plus_valley_hole), np.sum(minus_valley_hole), \
          np.sum(plus_valley_particle + plus_valley_hole + minus_valley_particle + minus_valley_hole))
    print('!!!!!!!!!!!!!!!!', wf.accepted / (wf.accepted + wf.rejected_factor), np.mean(wf.ws), np.mean(np.array(wf.ws) ** 2))
    '''

    #configs = np.array(configs)
    #print('correlation: ')
    #for shift in range(1000):
    #    A = np.einsum('ij,ij', configs[np.arange(len(configs) - shift)], configs[shift + np.arange(len(configs) - shift)]) / len(np.arange(len(configs) - shift)) / len(configs[0]) * 2
    #    print('A[{:d}] = {:.10f}'.format(shift, A), flush=True)

    return energies, Os, acceptance, wf.get_state(), observables, \
           names, wf.U_matrix, wf.E, densities, wf.gap

def run_simulation(delta_reg, previous_params):
    config_vmc_file = import_config(sys.argv[1])
    # mu_BCS_fixed = - 1. /80 * rank # FIXME
    config_vmc_import = config_vmc_file.MC_parameters(int(sys.argv[2]), rank)

    config_vmc = cv_module.MC_parameters(int(sys.argv[2]), rank)
    config_vmc.__dict__ = config_vmc_import.__dict__.copy()

    print_model_summary(config_vmc)


    if previous_params is not None:
        config_vmc.initial_parameters = previous_params

    #config_vmc.initial_parameters[config_vmc.layout[0] + config_vmc.layout[1] + config_vmc.layout[2]:config_vmc.layout[0] + \
    #                          config_vmc.layout[1] + config_vmc.layout[2] + config_vmc.layout[3]] = delta_reg
    config_vmc.workdir = config_vmc.workdir + '/irrep_{:d}_delta_{:.3f}/'.format(rank, delta_reg)
    
    os.makedirs(config_vmc.workdir, exist_ok=True)
    with open(os.path.join(config_vmc.workdir, 'config.py'), 'w') as target, \
         open(sys.argv[1], 'r') as source:  # save config file to workdir (to remember!!)
        target.write(source.read())


    if config_vmc.visualisation:
        # config_vmc.twist = [np.exp(2.0j * np.pi * 0.1904), np.exp(2.0j * np.pi * (0.1904 + 0.1))]
        visualisation.plot_levels_evolution_mu(config_vmc)
        visualisation.plot_all_waves(config_vmc)
        visualisation.plot_DOS(config_vmc)
        visualisation.plot_fermi_surface(config_vmc)
        visualisation.plot_all_waves(config_vmc)
        visualisation.plot_all_Jastrow(config_vmc)
        visualisation.plot_MF_spectrum_profile(config_vmc)
        


    config_vmc.twist = [1, 1] #[np.exp(2.0j * np.pi * 0.1904), np.exp(2.0j * np.pi * (0.1904 + 0.10))]
    if config_vmc.tests:
        if rank == 0:
            if tests.perform_all_tests(config_vmc):
                print('\033[92m All tests passed successfully \033[0m', flush = True)
            else:
                print('\033[91m Warning: some of the tests failed! \033[0m', flush = True)
    comm.Barrier()

    n_cpus_max = psutil.cpu_count(logical = True) 
    print('max available CPUs:', n_cpus_max)
    n_cpus = config_vmc.n_cpus
    if config_vmc.n_cpus == -1:
        n_cpus = n_cpus_max
    print('performing simulation at', n_cpus, 'CPUs')


    ### generate twists once and for all (Sandro's suggestion) ###
 
    if config_vmc.twist_mesh == 'Baldereschi':
        print('Working with the Baldereschi mesh')
        if config_vmc.n_sublattices == 2:
            twists = [[np.exp(2.0j * np.pi * 0.1904), np.exp(2.0j * np.pi * (0.1904 + 0.1))] for _ in range(config_vmc.n_chains)]
        if config_vmc.n_sublattices == 1:
            twists = [[1., -1.] for _ in range(config_vmc.n_chains)] # FIXME
        twists_per_cpu = config_vmc.n_chains / n_cpus
    elif config_vmc.twist_mesh == 'PBC':
        twists = [[1., 1.] for _ in range(config_vmc.n_chains)]
        twists_per_cpu = config_vmc.n_chains / n_cpus
    elif config_vmc.twist_mesh == 'APBCy':
        twists = [[1., -1.] for _ in range(config_vmc.n_chains)]
        twists_per_cpu = config_vmc.n_chains / n_cpus
    elif config_vmc.twist_mesh == 'reals':
        assert config_vmc.n_chains == 4
        twists = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        twists_per_cpu = config_vmc.n_chains / n_cpus
        assert twists_per_cpu == 1
    elif config_vmc.twist_mesh == 'uniform':
        L = config_vmc.L_twists_uniform
        twists = []
        for i_x in range(L):
            for i_y in range(L):
                twists.append([np.exp(1.0j * np.pi * (-1. + 1. / L + 2. * i_x / L)), np.exp(1.0j * np.pi * (-1. + 1. / L + 2. * i_y / L))])
        twists_per_cpu = len(twists) // n_cpus
        if twists_per_cpu * n_cpus < len(twists):
            twists_per_cpu += 1
    else:
        print('Twist {:s} is not supported'.format(config_vmc.twist_mesh))
        exit(-1)
    print(twists)
    print('Number of twists: {:d}, number of chains {:d}, twists per cpu {:2f}'.format(len(twists), config_vmc.n_chains, twists_per_cpu))
    K_matrices_up = [models.apply_TBC(config_vmc, twist, deepcopy(config_vmc.K_0), inverse = False) for twist in twists]
    K_matrices_down = [models.apply_TBC(config_vmc, twist, deepcopy(config_vmc.K_0).T, inverse = True) for twist in twists]
    reg_terms = [models.apply_TBC(config_vmc, twist, deepcopy(config_vmc.reg_gap_term), inverse = False) * \
                 config_vmc.reg_gap_val for twist in twists]



    config_vmc.MC_chain = config_vmc.MC_chain // len(twists) # the MC_chain contains the total required number of samples

    pairings_list = config_vmc.pairings_list
    pairings_names = config_vmc.pairings_list_names


    # template = 'e_{:.2f}_Ne_{:d}'.format(config_vmc.epsilon, config_vmc.Ne) if config_vmc.PN_projection else \
    #            'e_{:.2f}_mu_{:.2f}'.format(config_vmc.epsilon, config_vmc.mu)

    local_workdir = config_vmc.workdir

    obs_files = []
    loaded_from_external = False
    if config_vmc.load_parameters:
        if config_vmc.load_parameters_path is not None:
            loaded_from_external = True
            filename = config_vmc.load_parameters_path
            parameters, last_step = load_parameters(filename)
        elif os.path.isfile(os.path.join(local_workdir, 'last_opt_params.p')):
            filename = os.path.join(local_workdir, 'last_opt_params.p')
            parameters, last_step = load_parameters(filename)
        else:
            parameters = config_vmc.initial_parameters
            last_step = 0
    else:
        parameters = config_vmc.initial_parameters
        last_step = 0
    # parameters[1] = 5e-3  # FIXME
    # parameters[0] = config_vmc.select_initial_muBCS(parameters = parameters) # FIXME: add flag for this (correct mu_BCS on relaunch) ??
    #if in_parameters is not None:
    #    parameters = in_parameters
 
    log_file = open(os.path.join(local_workdir, 'general_log.dat'), 'a+')
    force_file = open(os.path.join(local_workdir, 'force_log.dat'), 'a+')
    gaps_file = open(os.path.join(local_workdir, 'gaps_log.dat'), 'a+')
    force_SR_file = open(os.path.join(local_workdir, 'force_SR_log.dat'), 'a+')

    spectral_file = open(os.path.join(local_workdir, 'spectral_log.dat'), 'a+')
    final_states = [False] * len(twists)
    orbitals_in_use = [None] * len(twists)


    ### write log header only if we start from some random parameters ###
    if last_step == 0 or loaded_from_external:
        write_initial_logs(log_file, force_file, force_SR_file, config_vmc)

    #for n_step in range(last_step, config_vmc.optimisation_steps):
    n_step = last_step
    while n_step < config_vmc.optimisation_steps:
        t = time()
        
        if twists_per_cpu > 1:
            with parallel_backend("loky", inner_max_num_threads=1):
                results_batched = Parallel(n_jobs=n_cpus)(delayed(get_MC_chain_result)( \
                        n_step, \
                        deepcopy(config_vmc), \
                        pairings_list, \
                        parameters, \
                        twists = twists[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                        final_states = final_states[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                        orbitals_in_use = orbitals_in_use[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                        K_matrices_up = K_matrices_up[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                        K_matrices_down = K_matrices_down[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                        regs = reg_terms[i * twists_per_cpu:np.min([(i + 1) * twists_per_cpu, len(twists)])], \
                    ) for i in range(n_cpus))
                #for i in range(n_cpus):
                #    print(i * twists_per_cpu, np.min([(i + 1) * twists_per_cpu, len(twists)]))
                results = []
                for i, r in enumerate(results_batched):
                    results = results + r
                    print('obtained {:d} results from {:d} cpu'.format(len(r), i))
                print('obtained in total {:d} results'.format(len(results)))
                #print(len(twists))
        else:
            with parallel_backend("loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=config_vmc.n_chains)(delayed(_get_MC_chain_result)( \
                        n_step, \
                        deepcopy(config_vmc), \
                        pairings_list, \
                        parameters, \
                        twists[i], \
                        final_states[i], \
                        orbitals_in_use[i], \
                        K_matrices_up[i], \
                        K_matrices_down[i], \
                        reg_terms[i], \
                    ) for i in range(config_vmc.n_chains))
        print('MC chain generationof {:d} no {:d} took {:f}'.format(rank, n_step, time() - t))
        t = time() 

        ### print-out current energy levels ###
        E = results[0][7]
        spectral_file.write(str(n_step) + ' ' + ("{:.7f} " * len(E) + '\n').format(*E))
        spectral_file.flush()
        ### MC chains data extraction ###
        gaps, gap, energies, mean_variance, Os, acceptance, \
            final_states, densities, orbitals_in_use, occupied_numbers = \
            extract_MC_data(results, config_vmc, config_vmc.n_chains)
        energies_merged = np.concatenate(energies) 
        
        n_above_FS = len(np.setdiff1d(occupied_numbers[0], np.arange(config_vmc.total_dof // 2)))
        ### gradient step ###
        if config_vmc.generator_mode:  # evolve parameters only if it's necessary
            mask = np.ones(np.sum(config_vmc.layout))
            #if n_step < 50000:  # jastrows and mu_BCS have not converged yet
            #    mask = np.ones(np.sum(config_vmc.layout))
            #    # mask[-config_vmc.layout[4]:] = 1.
            #    # mask[:config_vmc.layout[0]] = 1.
            #    mask[config_vmc.layout[0] + config_vmc.layout[1] + config_vmc.layout[2]:config_vmc.layout[0] + \
            #         config_vmc.layout[1] + config_vmc.layout[2] + config_vmc.layout[3]] = 0.
            #mask[1] = 0.0  # fugacity is not optimized in the meantime

            # Os = [np.einsum('ik,k->ik', Os_theta, config_vmc.mask) for Os_theta in Os]

            step, forces = make_SR_step(Os, energies, config_vmc, twists, gaps, n_step, mask)
            
            write_intermediate_log(log_file, force_file, force_SR_file, n_step, config_vmc.total_dof // 2, energies, densities, \
                                   mean_variance, acceptance, forces, step, gap, n_above_FS, parameters)  # write parameters before step not to lose the initial values

            write_gaps_log(gaps_file, gaps, n_step)

            #if np.abs(gap) < 1e-4:  # if the gap is too small, SR will make gradient just 0
            #    step = forces
            #step = forces * config_vmc.opt_parameters[1]
            step = step * config_vmc.opt_parameters[1]
            #step = clip_forces(config_vmc.all_clips, step)

            parameters += step * mask  # lr better be ~0.01..0.1
            save_parameters(parameters, config_vmc.workdir, n_step)
        ### END SR STEP ###


        observables = np.concatenate([np.array(x[4]) for x in results], axis = 0)
        observables_names = results[0][5]
        n_step += 1
        if len(observables_names) == 0:
            continue

        if n_step == config_vmc.thermalization + 1:
            create_obs_files(observables_names, config_vmc)
        
        write_observables(n_step, obs_files, observables, config_vmc)
        if rank == 0:
            print('SR and logging {:d} took {:f}'.format(n_step, time() - t))

    log_file.close()
    force_file.close()
    force_SR_file.close()
    spectral_file.close()

    [file.close() for file in obs_files]
    return parameters

if __name__ == "__main__":
    previous_params = None
    for delta_reg in [0.000]:#np.linspace(5e-3, 5e-2, 10)[::-1]:
        previous_params = run_simulation(delta_reg = delta_reg, previous_params=previous_params)

