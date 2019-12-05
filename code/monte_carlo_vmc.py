import numpy as np
import time
import models_vmc
from wavefunction_vmc import wavefunction_singlet
from config_vmc import config as config_vmc
import pairings
from joblib import Parallel, delayed
import psutil
from time import time
n_cpus = psutil.cpu_count(logical = True) 
print('performing simulation at', n_cpus, 'threads')

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def get_MC_chain_result(hamiltonian, config_vmc, pairings_list, mu_parameter, gap_parameters, jastrow_parameters):
    t = time()
    wavefunction = wavefunction_singlet(config_vmc, pairings_list, mu_parameter, gap_parameters, jastrow_parameters)
    #print(time() - t)
    t = time()
    # print('doping =', 2 * wavefunction.n_particles / (config_vmc.total_dof // 2) - 1)
    # print('n_particles =', wavefunction.n_particles, 'n_holes =', wavefunction.n_holes)
    t = time()
    for MC_step in range(config_vmc.MC_thermalisation):
        wavefunction.perform_MC_step()
    #print(time() - t)
    #print('steps =', wavefunction.t_step, 'wfs =', wavefunction.t_wf,)
    t = time()
    t_steps = 0.0
    t_energies = 0.0
    t_Os = 0.0
    energies = []
    Os = []
    acceptance = []
    for MC_step in range(config_vmc.MC_chain):
        if MC_step % config_vmc.correlation == 0:
            t = time()
            energies.append(hamiltonian(wavefunction))
            t_energies += time() - t
            t = time()
            Os.append(wavefunction.get_O())
            t_Os += time() - t
        t = time()
        acceptance.append(wavefunction.perform_MC_step()[0])
        t_steps += time() - t
    #print(t_steps, t_Os, t_energies)
    return energies, Os, acceptance

pairings_list = config_vmc.pairings_list
gap_parameters = config_vmc.initial_gap_parameters
jastrow_parameters = config_vmc.initial_jastrow_parameters
mu_parameter = config_vmc.initial_mu_parameters  # chemical potential (mu)

H = config_vmc.hamiltonian(config_vmc)
opt = config_vmc.optimiser(config_vmc.opt_parameters)


while True:
    results = Parallel(n_jobs=n_cpus)(delayed(get_MC_chain_result)(config_vmc.hamiltonian(config_vmc), config_vmc, pairings_list, \
                                                                   mu_parameter, gap_parameters, jastrow_parameters) for i in range(n_cpus))
    
    energies = np.concatenate([np.array(x[0]) for x in results], axis = 0)
    Os = np.concatenate([np.array(x[1]) for x in results], axis = 0)
    acceptance = np.mean(np.concatenate([np.array(x[2]) for x in results], axis = 0))

    vol = config_vmc.total_dof // 2

    Os_mean = np.mean(Os, axis = 0)
    forces = -2 * (np.einsum('i,ik->k', energies.conj(), Os) / len(energies) - np.mean(energies) * Os_mean).real

    print('estimating gradient on ', len(energies), 'samples')
    print('\033[93m <E> / t / vol = ' + str(np.mean(energies) / vol) + '+/-' + str(np.std(energies) / np.sqrt(len(energies)) / vol) + '\033[0m')
    print('\033[92m acceptance =' + str(acceptance) + '\033[0m')
    print('\033[94m |forces| = ' + str(np.sqrt(np.sum(forces ** 2))) + ' ' + str(forces) + '\033[0m')

    S_cov = (np.einsum('nk,nl->kl', (Os - Os_mean[np.newaxis, :]), (Os - Os_mean[np.newaxis, :])) / Os.shape[0]).real
    if np.linalg.det(S_cov) != 0:
        forces = np.linalg.inv(S_cov).dot(forces)  # stochastic reconfiguration
    print('\033[94m |forces after SR| = ' + str(np.sqrt(np.sum(forces ** 2))) + ' ' + str(forces) + '\033[0m')
    print('\033[91m mu = ' + str(mu_parameter) + ', pairings =' + str(gap_parameters) + ', Jastrow =' + str(jastrow_parameters) + '\033[0m')
    step = opt.get_step(forces)
    # print(forces, step)

    mu_parameter += step[0]
    gap_parameters += step[1:2]
    jastrow_parameters += step[2:]


dt = 1e-6
np.random.seed(11)
wf_1 = wavefunction_singlet(config_vmc, pairings_list, [0.1 - dt / 2], [0.1], [0.0])
np.random.seed(11)
wf_2 = wavefunction_singlet(config_vmc, pairings_list, [0.1 + dt / 2], [0.1], [0.0])  
print(wf_1.E - wf_2.E)
while True:
    print(wf_1.get_O()[0], wf_2.get_O()[0])
    # print(np.sum(np.abs(wf_1.U_matrix - wf_2.U_matrix)), np.linalg.matrix_rank(np.concatenate([wf_1.U_matrix, wf_2.U_matrix], axis = 1)), np.concatenate([wf_1.U_matrix, wf_2.U_matrix], axis = 1).shape)
    
    # print(np.linalg.det((wf_1.U_matrix.conj().T).dot(wf_2.U_matrix)))
    # print(wf_1.occupied_sites - wf_2.occupied_sites)
    # print(np.linalg.slogdet(wf_1.U_tilde_matrix)[1] - np.linalg.slogdet(wf_2.U_tilde_matrix)[1], np.linalg.slogdet(wf_2.U_tilde_matrix)[1])
    print(2 * (wf_2.current_ampl - wf_1.current_ampl) / dt / (wf_1.current_ampl + wf_2.current_ampl) - 0.5 * wf_1.get_O()[0] - 0.5 * wf_2.get_O()[0], \
          2 * (-wf_2.current_ampl - wf_1.current_ampl) / dt / (wf_1.current_ampl - wf_2.current_ampl) - 0.5 * wf_1.get_O()[0] - 0.5 * wf_2.get_O()[0],  - 0.5 * wf_1.get_O()[0] - 0.5 * wf_2.get_O()[0])  # log derivative calculated explicitly
    a = np.random.randint(10000)
    np.random.seed(a)
    wf_1.perform_MC_step()
    np.random.seed(a)
    wf_2.perform_MC_step()
    # print(wf_1.get_O_pairing(0))  # log devirative calculated from O_k formula

# generate_MC_chain(wf)
