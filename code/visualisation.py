import matplotlib.pyplot as plt
import models
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
    energies = np.array(energies).real
    k_vectors = np.array(k_vectors)
    E_max = np.sort(np.array(energies).reshape(-1))[config.N_electrons // 2]  # 2-spin degeneracy

    fig = plt.figure()
    ax = Axes3D(fig)

    for band in range(energies.shape[1]):
        ax.scatter(k_vectors[:, 0], k_vectors[:, 1], energies[:, band])
    plt.show()

    k_array = []
    s_array = []
    for i, k in enumerate(k_vectors):
        if np.sum(energies[i] <= E_max) > 0:
            s_array.append(np.sum(energies[i] <= E_max) * 10)
            k_array.append(k)

    k_array = np.array(k_array)
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
    plt.rc("pdf", fonttype=42)
    
    textshift = np.array([0.1, 0])
    if geometry == 'hexagonal':
        rotate_K = np.array([[np.cos(np.pi / 3), np.sin(np.pi / 3.)], [-np.sin(np.pi / 3.), np.cos(np.pi / 3)]])
        K_point = 2 * np.pi * np.array([2 / np.sqrt(3), 2.0 / 3.0]) / 2.
        Gamma_point = 2 * np.pi * np.array([0., 0.]) / 2.
        M_point = 2 * np.pi * np.array([2 / np.sqrt(3), 0]) / 2.
        Kprime_point = 2 * np.pi * np.array([2 / np.sqrt(3), -2.0 / 3.0]) / 2.
        for i in range(6):
            rotation_matrix = np.linalg.matrix_power(rotate_K, i)

            plt.scatter(k_array.dot(rotation_matrix)[:, 0], k_array.dot(rotation_matrix)[:, 1], s = s_array)
            plt.scatter(*K_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*K_point.dot(rotation_matrix) + textshift, '$K$', fontsize = 14)
            plt.scatter(*Kprime_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*Kprime_point.dot(rotation_matrix) + textshift, '$K\'$', fontsize = 14)
            plt.scatter(*Gamma_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*Gamma_point.dot(rotation_matrix) + textshift, '$\\Gamma$', fontsize = 14)
            plt.scatter(*M_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*M_point.dot(rotation_matrix) + textshift, '$M$', fontsize = 14)
    plt.grid(True)
    plt.show()
