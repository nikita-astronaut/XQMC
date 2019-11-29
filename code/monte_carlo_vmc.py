import numpy as np
import time
import models_vmc
from wavefunction_vmc import wavefunction_singlet
from config_vmc import config as config_vmc
import pairings
from tqdm import tqdm

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def generate_MC_chain(hamiltinian, wavefunction):
    for MC_step in tqdm(range(config_vmc.MC_thermalisation)):
        wavefunction.perform_MC_step()

    energies = []
    Os = []
    for MC_step in tqdm(range(config_vmc.MC_chain)):
        if MC_step % config_vmc.correlation == 0:
            energies.append(hamiltinian(wavefunction))
            Os.append(wavefunction.get_O())
            # print(np.mean(energies))

        wavefunction.perform_MC_step()

    vol = wavefunction.config.total_dof // 2
    energies = np.array(energies)
    Os = np.array(Os)
    Os_mean = np.mean(Os, axis = 0)
    forces = -2 * (np.einsum('i,ik->k', energies.conj(), Os) / len(energies) - np.mean(energies) * Os_mean).real
    print('<E> / t / vol =', np.mean(energies) / vol, '+/-', np.std(energies) / np.sqrt(len(energies)) / vol)
    S_cov = (np.einsum('nk,nl->kl', (Os - Os_mean[np.newaxis, :]), (Os - Os_mean[np.newaxis, :])) / len(Os_mean)).real
    print('|forces| =', np.sqrt(np.sum(forces ** 2)))
    if True: # np.linalg.det(S_cov) == 0:
        return forces
    return np.linalg.inv(S_cov).dot(forces)

pairings_list = config_vmc.pairings_list
gap_parameters = config_vmc.initial_gap_parameters
jastrow_parameters = config_vmc.initial_jastrow_parameters
mu_parameter = config_vmc.mu  # chemical potential (mu)

H = config_vmc.hamiltonian(config_vmc)
opt = config_vmc.optimiser(config_vmc.opt_parameters)


while True:
    # np.random.seed(0)
    print(config_vmc.n_orbitals)
    wf = wavefunction_singlet(config_vmc, pairings_list, mu_parameter, gap_parameters, jastrow_parameters)
    forces = generate_MC_chain(H, wf)
    print(mu_parameter, gap_parameters, jastrow_parameters)
    step = opt.get_step(forces)
    print(forces, step)
    H.reset()
    mu_parameter += step[0]
    gap_parameters += step[1:2]
    jastrow_parameters += step[2:]

for dt in np.logspace(-7, -8, 100):
    np.random.seed(13)
    wf_1 = wavefunction_singlet(config_vmc, pairings_list, [1.0], [1.0])
    np.random.seed(13)
    wf_2 = wavefunction_singlet(config_vmc, pairings_list, [1.0 + dt], [1.0 + dt])
    print((wf_2.current_ampl - wf_1.current_ampl) / dt / (wf_1.current_ampl + wf_2.current_ampl) * 2., wf_1.get_O()[0] + wf_1.get_O()[1],
          (-wf_2.current_ampl - wf_1.current_ampl) / dt / (wf_1.current_ampl - wf_2.current_ampl))  # log derivative calculated explicitly
    # print(wf_1.get_O_pairing(0))  # log devirative calculated from O_k formula

# generate_MC_chain(wf)
