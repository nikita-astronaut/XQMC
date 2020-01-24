import matplotlib.pyplot as plt
import models
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pairings

from copy import deepcopy

def set_style():
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
    plt.rc("pdf", fonttype=42)
    plt.grid(True, linestyle='--', alpha=0.5)
    return

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
        R = models.R_hexagonal
        G = models.G_hexagonal
    else:
        R = models.R_square
        G = models.G_square

    K = config.model(config, 0.0, spin = +1.0)[0]
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

    set_style()
    fig = plt.figure()
    ax = Axes3D(fig)

    for band in range(energies.shape[1]):
        ax.scatter(k_vectors[:, 0] / np.pi, k_vectors[:, 1] / np.pi, energies[:, band])
    plt.xlabel('$k_x,\\, \\pi$')
    plt.ylabel('$k_y,\\, \\pi$')
    ax.set_zlabel('$E(\\vec k) / t$')
    plt.savefig('../plots/bands.pdf')
    plt.clf()


    set_style()
    k_array = []
    s_array = []
    for i, k in enumerate(k_vectors):
        if np.sum(energies[i] <= E_max) > 0:
            s_array.append(np.sum(energies[i] <= E_max) * 10)
            k_array.append(k)

    k_array = np.array(k_array)
    
    
    textshift = np.array([0.1, 0])
    if geometry == 'hexagonal':
        rotate_K = np.array([[np.cos(np.pi / 3), np.sin(np.pi / 3.)], [-np.sin(np.pi / 3.), np.cos(np.pi / 3)]])
        K_point = 2 * np.pi * np.array([2 / np.sqrt(3), 2.0 / 3.0]) / 2.
        Gamma_point = 2 * np.pi * np.array([0., 0.]) / 2.
        M_point = 2 * np.pi * np.array([2 / np.sqrt(3), 0]) / 2.
        Kprime_point = 2 * np.pi * np.array([2 / np.sqrt(3), -2.0 / 3.0]) / 2.
        for i in range(6):
            rotation_matrix = np.linalg.matrix_power(rotate_K, i)

            plt.scatter(k_array.dot(rotation_matrix)[:, 0], k_array.dot(rotation_matrix)[:, 1], s = s_array, color = 'blue')
            plt.scatter(*K_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*K_point.dot(rotation_matrix) + textshift, '$K$', fontsize = 14)
            plt.scatter(*Kprime_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*Kprime_point.dot(rotation_matrix) + textshift, '$K\'$', fontsize = 14)
            plt.scatter(*Gamma_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*Gamma_point.dot(rotation_matrix) + textshift, '$\\Gamma$', fontsize = 14)
            plt.scatter(*M_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*M_point.dot(rotation_matrix) + textshift, '$M$', fontsize = 14)
    else:
        rotate_K = np.array([[np.cos(np.pi / 2), np.sin(np.pi / 2.)], [-np.sin(np.pi / 2.), np.cos(np.pi / 2)]])
        K_point = 2 * np.pi * np.array([1, 1]) / 2.
        Gamma_point = 2 * np.pi * np.array([0., 0.]) / 2.

        for i in range(4):
            rotation_matrix = np.linalg.matrix_power(rotate_K, i)

            plt.scatter(k_array.dot(rotation_matrix)[:, 0], k_array.dot(rotation_matrix)[:, 1], s = s_array, color = 'blue')
            plt.scatter(*Gamma_point.dot(rotation_matrix), s = 20, marker = '*', color = 'red')
            plt.text(*Gamma_point.dot(rotation_matrix) + textshift, '$\\Gamma$', fontsize = 14)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('../plots/fermi_surface.pdf')
    plt.clf()
    return

def plot_all_pairings(config):
    for gap, name in zip(config.pairings_list, config.pairings_list_names):
        gap_expanded = pairings.combine_product_terms(config, gap)

        plot_pairing(config, gap_expanded, name)

def plot_pairing(config, gap_expanded, name):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'

    if geometry == 'hexagonal':
        R = models.R_hexagonal
    else:
        R = models.R_square
    set_style()

    textshift = np.array([0.1, 0.1])

    x1, y1 = config.Ls // 2, config.Ls // 2
    for sublattice1 in range(config.n_sublattices):
        for orbit1 in range(config.n_orbitals):
            first = models.to_linearized_index(x1, y1, sublattice1, orbit1, config.Ls, config.n_orbitals, config.n_sublattices)
            for second in range(config.total_dof // 2):
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), config.Ls, \
                                                                           config.n_orbitals, config.n_sublattices)

                if gap_expanded[first, second] == 0:
                    continue
                value = gap_expanded[first, second]

                labelstring = str(value)
                if np.abs(value.imag) < 1e-10:
                    labelstring = str(value.real)
                elif geometry == 'hexagonal':
                    if np.abs(value - np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$\\omega$'
                    if np.abs(value + np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$-\\omega$'
                    if np.abs(value - np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$\\omega^*$'
                    if np.abs(value + np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$-\\omega^*$'

                    if np.abs(value - 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$i \\omega$'
                    if np.abs(value + 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$-i \\omega$'
                    if np.abs(value - 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$i \\omega^*$'
                    if np.abs(value + 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring = '$-i \\omega^*$'

                labelstring = '(' + str(orbit1) + '-' + str(orbit2) + '), ' + labelstring


                r1 = np.array([x1, y1]).dot(R) + sublattice1 * np.array([1, 0]) / np.sqrt(3)  # always 0 in the square case
                r2 = np.array([x2, y2]).dot(R) + sublattice2 * np.array([1, 0]) / np.sqrt(3)

                r1_origin = np.array([x1, y1]).dot(R)
                r1 = r1 - r1_origin
                r2 = r2 - r1_origin
                if sublattice2 == 0:
                    plt.scatter(*r2, s=20, color='red')
                else:
                    plt.scatter(*r2, s=20, color='blue')
                plt.annotate(s='', xy=r2, xytext=r1, arrowprops=dict(arrowstyle='->'))
                
                textshift = np.array([r2[1] - r1[1], r1[0] - r2[0]])
                textshift = textshift / np.sqrt(np.sum(textshift ** 2) + 1e-5)
                shiftval = 0.2 - (orbit1 * config.n_orbitals + orbit2) * 0.1
                plt.text(*(r2 + shiftval * textshift), labelstring, zorder=10, fontsize=8)
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim([-0.5, 1.25])
    plt.ylim([-0.75, 0.75])
    # plt.title(name + ' pairing')
    plt.savefig('../plots/' + name + '.pdf')
    plt.clf()
    return

def plot_all_Jastrow(config):
    for index, jastrow in enumerate(models.get_adjacency_list(config)[0]):
        plot_Jastrow(config, jastrow, index)
    return

def plot_Jastrow(config, Jastrow, index):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'

    pairing, orbit1, orbit2, dist = Jastrow

    if geometry == 'hexagonal':
        R = models.R_hexagonal
    else:
        R = models.R_square
    set_style()

    textshift = np.array([0.1, 0.1])

    x1, y1 = config.Ls // 2, config.Ls // 2
    for sublattice1 in range(config.n_sublattices):
        for orbit1 in range(config.n_orbitals):
            first = models.to_linearized_index(x1, y1, sublattice1, orbit1, config.Ls, config.n_orbitals, config.n_sublattices)
            for second in range(config.total_dof // 2):
                orbit2, sublattice2, x2, y2 = models.from_linearized_index(deepcopy(second), config.Ls, \
                                                                           config.n_orbitals, config.n_sublattices)

                if pairing[first, second] == 0:
                    continue
                value = config.initial_jastrow_parameters[index]

                labelstring = str(value)
                labelstring = '(' + str(orbit1) + '-' + str(orbit2) + '), ' + labelstring + ' ' + str(index)


                r1 = np.array([x1, y1]).dot(R) + sublattice1 * np.array([1, 0]) / np.sqrt(3)  # always 0 in the square case
                r2 = np.array([x2, y2]).dot(R) + sublattice2 * np.array([1, 0]) / np.sqrt(3)

                r1_origin = np.array([x1, y1]).dot(R)
                r1 = r1 - r1_origin
                r2 = r2 - r1_origin
                if sublattice2 == 0:
                    plt.scatter(*r2, s=20, color='red')
                else:
                    plt.scatter(*r2, s=20, color='blue')

                if np.sum(np.abs(r1 - r2)) < 1e-5:
                    plt.plot([r1[0]], [r1[1]], marker = '*', ms = 10)
                else:
                    plt.annotate(s='', xy=r2, xytext=r1, arrowprops=dict(arrowstyle='->'))
                
                textshift = np.array([r2[1] - r1[1], r1[0] - r2[0]])
                textshift = textshift / np.sqrt(np.sum(textshift ** 2) + 1e-5)
                shiftval = 0.1 - (orbit1 * config.n_orbitals + orbit2) * 0.1 / 2
                plt.text(*(r2 + shiftval * textshift + np.random.uniform(-0.05, 0.05, 2)), labelstring, zorder=10, fontsize=8)
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('jastrow-' + str(orbit1) + '-' + str(orbit2) + ' , $r = ' + str(dist) + '$')
    plt.savefig('../plots/jastrow_' + str(orbit1) + '-' + str(orbit2) + '_' + str(dist) + '.pdf')
    plt.clf()
    return
