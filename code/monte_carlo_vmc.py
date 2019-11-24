import numpy as np
import time
import models_vmc
from wavefunction_vmc import wavefunction_singlet
from config_vmc import MC_parameters as config_vmc
import pairings
from tqdm import tqdm

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

def generate_MC_chain(hamiltinian, wavefunction):
    for MC_step in range(config_vmc.MC_thermalisation):
        wavefunction.perform_MC_step()

    energies = []
    Os = []
    for MC_step in range(config_vmc.MC_chain):
        if MC_step % config_vmc.correlation == 0:
            energies.append(hamiltinian(wavefunction))
            Os.append(wavefunction.get_O())
            # print(np.mean(energies))

        wavefunction.perform_MC_step()


    energies = np.array(energies)
    Os = np.array(Os)
    Os_mean = np.mean(Os, axis = 0)
    forces = -2 * (np.einsum('i,ik->k', energies.conj(), Os) / len(energies) - np.mean(energies) * Os_mean).real
    print('E_average =', np.mean(energies), '+/-', np.std(energies) / np.sqrt(len(energies)))
    S_cov = (np.einsum('nk,nl->kl', (Os - Os_mean[np.newaxis, :]), (Os - Os_mean[np.newaxis, :])) / len(Os_mean)).real
    print('|forces| =', np.sqrt(np.sum(forces ** 2)))
    if True:#np.linalg.det(S_cov) == 0:
        return forces
    return np.linalg.inv(S_cov).dot(forces)

config_vmc = config_vmc()
pairings_list = [pairings.on_site_pairings[0]]
parameters = np.array([0.5])


H = config_vmc.hamiltonian(config_vmc)
opt = config_vmc.optimiser(config_vmc.opt_parameters)
while True:
    # np.random.seed(0)
    wf = wavefunction_singlet(config_vmc, pairings_list, parameters)
    forces = generate_MC_chain(H, wf)
    print(parameters)
    step = opt.get_step(forces)
    print(forces, step)
    H.reset()
    parameters += step

for dt in np.logspace(-7, -8, 100):
    np.random.seed(10)
    wf_1 = wavefunction_singlet(config_vmc, pairings_list, [1.0 - dt])
    np.random.seed(10)
    wf_2 = wavefunction_singlet(config_vmc, pairings_list, [1.0 + dt])
    print((wf_2.current_det - wf_1.current_det) / dt / (wf_1.current_det + wf_2.current_det), wf_1.get_O_pairing(0),
          (-wf_2.current_det - wf_1.current_det) / dt / (wf_1.current_det - wf_2.current_det), wf_1.get_O_pairing(0))  # log derivative calculated explicitly
    # print(wf_1.get_O_pairing(0))  # log devirative calculated from O_k formula

# generate_MC_chain(wf)
