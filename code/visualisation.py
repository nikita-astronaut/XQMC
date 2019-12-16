import matplotlib.pyplot as plt
import models
import numpy as np
from copy import deepcopy

R_hexagonal = np.array([[np.sqrt(3) / 2, 1 / 2.], [np.sqrt(3) / 2., -1 / 2.]])
G_hexagonal = 2 * np.pi * np.array([[1 / np.sqrt(3), 1.], [1. / np.sqrt(3), -1.]])

def K_FT(k, K, config, R):
    L = config.Ls
    n_internal = config.n_sublattices * config.n_orbitals
    result = np.zeros((n_internal, n_internal)) * 1.0j
    x1, y1 = 0, 0  # rely on translational invariance
    for sublattice_orbit1 in range(n_internal):
        sublattice1 = sublattice_orbit1 % config.n_sublattices
        orbit1 = (sublattice_orbit1 // config.n_sublattices) % config.n_orbitals
        first = models.to_linearized_index(x1, y1, sublattice1, orbit1, L, config.n_orbitals, config.n_sublattices)
        for second in range(K.shape[0]):
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), L, config.n_orbitals, config.n_sublattices)
            element = K[first, second]
            r2_real = np.einsum('j,jk->k', np.array([x2 - x1, y2 - y1]), R)

            ft_factor = np.exp(1.0j * np.einsum('i,i', r2_real, k))
            result[orbit1 + config.n_orbitals * sublattice1, \
                   orbit2 + config.n_orbitals * sublattice2] += ft_factor * element
    return result

def plot_fermi_surface(config):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'

    if geometry == 'hexagonal':
        R = R_hexagonal
        G = G_hexagonal
    else:
        R = R_square
        G = G_square

    K = config.model(config, 0.0)
    k_vectors = []
    energies = []
    for kx in range(config.Ls):
        for ky in range(config.Ls):
            k_real = np.einsum('ji,j->i', G, np.array([kx / config.Ls, ky / config.Ls]))
            K_fourier = K_FT(k_real, K, config, R)
            e_k = np.linalg.eig(K_fourier)[0]
            k_vectors.append((k_real[0], k_real[1]))
            energies.append(np.sort(e_k))
    E_max = np.sort(np.array(energies).reshape(-1))[config.N_electrons // 2]  # 2-spin degeneracy

    kx_array = []
    ky_array = []
    s_array = []
    for i, k in enumerate(k_vectors):
        if np.sum(energies[i] <= E_max) > 0:
            s_array.append(np.sum(energies[i] <= E_max) * 10)
            kx_array.append(k[0])
            ky_array.append(k[1])

    plt.rc('text', usetex = True)
    plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
    plt.rc("pdf", fonttype=42)
    plt.scatter(kx_array, ky_array, s = s_array)

    plt.grid(True)

    if geometry == 'hexagonal':
        K_point = 2 * np.pi * np.array([2 / np.sqrt(3), 2.0 / 3.0]) / 2.
        Gamma_point = 2 * np.pi * np.array([0., 0.]) / 2.
        M_point = 2 * np.pi * np.array([2 / np.sqrt(3), 0]) / 2.
        Kprime_point = 2 * np.pi * np.array([2 / np.sqrt(3), -2.0 / 3.0]) / 2.

        plt.scatter(K_point[0], K_point[1], s = 20, marker = '*', color = 'red')
        plt.scatter(Kprime_point[0], Kprime_point[1], s = 20, marker = '*', color = 'red')
        plt.scatter(Gamma_point[0], Gamma_point[1], s = 20, marker = '*', color = 'red')
        plt.scatter(M_point[0], M_point[1], s = 20, marker = '*', color = 'red')
    plt.show()
