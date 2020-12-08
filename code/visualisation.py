import matplotlib.pyplot as plt
import models
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from opt_parameters import pairings
from numba import jit
from copy import deepcopy
import scipy

def set_style():
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'TeX Gyre Adventor', size = 14)
    plt.rc("pdf", fonttype=42)
    plt.grid(True, linestyle='--', alpha=0.5)
    return


def K_FT(k, K, config, R, spin_dof = 1):
    L = config.Ls
    n_internal = config.n_sublattices * config.n_orbitals * spin_dof
    result = np.zeros((n_internal, n_internal)) * 1.0j
    #result = np.zeros((n_internal // 2, n_internal // 2)) * 1.0j
    x1, y1 = 0, 0  # rely on translational invariance
    for sublattice_orbit_spin1 in range(n_internal):
        ### parse mega-index into components ###
        spin1 = sublattice_orbit_spin1 % spin_dof
        sublattice1 = (sublattice_orbit_spin1 // spin_dof) % config.n_sublattices
        orbit1 = ((sublattice_orbit_spin1 // spin_dof) // config.n_sublattices) % config.n_orbitals

        first = models.to_linearized_index(x1, y1, sublattice1, orbit1, L, \
                                           config.n_orbitals, config.n_sublattices) + spin1 * (K.shape[0] // spin_dof)
        

        for second in range(K.shape[0]):
            spin2 = second // (K.shape[0] // spin_dof)
            orbit2, sublattice2, x2, y2 = models.from_linearized_index(second % (K.shape[0] // spin_dof), L, \
                                          config.n_orbitals, config.n_sublattices)
            element = K[first, second]
            r2_real = np.einsum('j,jk->k', np.array([x2 - x1, y2 - y1]), R)

            ft_factor = np.exp(1.0j * np.einsum('i,i', r2_real, k))
            result[(orbit1 + config.n_orbitals * sublattice1) + spin1 * result.shape[0] // 2, \
                   (orbit2 + config.n_orbitals * sublattice2) + spin2 * result.shape[0] // 2] += ft_factor * element
            #if spin1 == 0 and spin2 == 0:
            #    result[orbit1 + config.n_orbitals * sublattice1, \
            #       orbit2 + config.n_orbitals * sublattice2] += ft_factor * element
    return result

def plot_DOS(config):
    model = config.model(config, 0.0, spin = +1.0)[0]
    model = models.xy_to_chiral(model, 'K_matrix', config, config.chiral_basis)  # this option is only valid for Koshino model

    K_0_plus = model[:, np.arange(0, config.total_dof // 2, 2)]; K_0_plus = K_0_plus[np.arange(0, config.total_dof // 2, 2), :]
    K_0_minus = model[:, np.arange(1, config.total_dof // 2, 2)]; K_0_minus = K_0_minus[np.arange(1, config.total_dof // 2, 2), :]
    Ep, _ = np.linalg.eigh(K_0_plus)
    Em, _ = np.linalg.eigh(K_0_minus)

    set_style()
    fig = plt.figure()

    plt.hist(Ep, bins = np.linspace(Ep.min() - 0.1, Ep.max() + 0.1, 300), color = 'red')
    plt.hist(Em + 0.01, bins = np.linspace(Ep.min() - 0.1, Ep.max() + 0.1, 300), color = 'blue')
    for ep, em in zip(Ep, Em):
        print(ep - em)
    plt.xlabel('$dn(E) / dE$')
    plt.ylabel('$E$')
    plt.grid(True)
    plt.savefig('DoS.pdf')
    plt.show()
    plt.clf()

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
    E_max = np.sort(np.array(energies).reshape(-1))[config.total_dof // 2 // 2 - 14]  # 2-spin degeneracy

    set_style()
    fig = plt.figure()
    ax = Axes3D(fig)

    for band in range(energies.shape[1]):
        ax.scatter(k_vectors[:, 0] / np.pi, k_vectors[:, 1] / np.pi, energies[:, band])
    plt.xlabel('$k_x,\\, \\pi$')
    plt.ylabel('$k_y,\\, \\pi$')
    ax.set_zlabel('$E(\\vec k) / t$')
    plt.savefig('../plots/bands.pdf')
    plt.show()
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
        gap_expanded = models.xy_to_chiral(pairings.combine_product_terms(config, gap), 'pairing', config, config.chiral_basis)
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

                    if np.abs(value - 1.0j) < 1e-7:
                        labelstring = '$i$'
                    if np.abs(value + 1.0j) < 1e-7:
                        labelstring = '$-i$'

                    if np.abs(value - 1.0) < 1e-7:
                        labelstring = '$1$'
                    if np.abs(value + 1.0) < 1e-7:
                        labelstring = '$-1$'

                    if np.abs(value - 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring = '$i\\sqrt{3}$'
                    if np.abs(value + 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring = '$-i\\sqrt{3}$'

                    if np.abs(value - np.sqrt(3)) < 1e-7:
                        labelstring = '$\\sqrt{3}$'
                    if np.abs(value + np.sqrt(3)) < 1e-7:
                        labelstring = '$-\\sqrt{3}$'

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


def plot_all_waves(config):
    for wave, name in zip(config.waves_list, config.waves_list_names):
        wave_chiral = models.xy_to_chiral(wave[0], 'wave', config, config.chiral_basis)
        plot_wave(config, wave_chiral, name)
        

def plot_wave(config, wave, name):
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
                if wave[first, second] == 0:
                    continue
                value_uu = wave[first, second]
                value_dd = wave[first + config.total_dof // 2, second + config.total_dof // 2]

                labelstring_up = str(value_uu)
                if np.abs(value_uu.imag) < 1e-10:
                    labelstring_up = str(value_uu.real)
                elif geometry == 'hexagonal':
                    if np.abs(value_uu - np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$\\omega$'
                    if np.abs(value_uu + np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$-\\omega$'
                    if np.abs(value_uu - np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$\\omega^*$'
                    if np.abs(value_uu + np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$-\\omega^*$'

                    if np.abs(value_uu - 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$i \\omega$'
                    if np.abs(value_uu + 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$-i \\omega$'
                    if np.abs(value_uu - 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$i \\omega^*$'
                    if np.abs(value_uu + 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_up = '$-i \\omega^*$'

                    if np.abs(value_uu - 1.0j) < 1e-7:
                        labelstring_up = '$i$'
                    if np.abs(value_uu + 1.0j) < 1e-7:
                        labelstring_up = '$-i$'

                    if np.abs(value_uu - 1.0) < 1e-7:
                        labelstring_up = '$1$'
                    if np.abs(value_uu + 1.0) < 1e-7:
                        labelstring_up = '$-1$'

                    if np.abs(value_uu - 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring_up = '$i\\sqrt{3}$'
                    if np.abs(value_uu + 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring_up = '$-i\\sqrt{3}$'

                    if np.abs(value_uu - np.sqrt(3)) < 1e-7:
                        labelstring_up = '$\\sqrt{3}$'
                    if np.abs(value_uu + np.sqrt(3)) < 1e-7:
                        labelstring_up = '$-\\sqrt{3}$'

                labelstring_up = '(' + str(orbit1) + '-' + str(orbit2) + '), ' + labelstring_up + ', $\\uparrow$'

                labelstring_down = str(value_dd)
                if np.abs(value_dd.imag) < 1e-10:
                    labelstring_down = str(value_dd.real)
                elif geometry == 'hexagonal':
                    if np.abs(value_dd - np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$\\omega$'
                    if np.abs(value_dd + np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$-\\omega$'
                    if np.abs(value_dd - np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$\\omega^*$'
                    if np.abs(value_dd + np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$-\\omega^*$'

                    if np.abs(value_dd - 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$i \\omega$'
                    if np.abs(value_dd + 1.0j * np.exp(2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$-i \\omega$'
                    if np.abs(value_dd - 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$i \\omega^*$'
                    if np.abs(value_dd + 1.0j * np.exp(-2.0 * np.pi / 3.0 * 1.0j)) < 1e-11:
                        labelstring_down = '$-i \\omega^*$'

                    if np.abs(value_dd - 1.0j) < 1e-7:
                        labelstring_down = '$i$'
                    if np.abs(value_dd + 1.0j) < 1e-7:
                        labelstring_down = '$-i$'

                    if np.abs(value_dd - 1.0) < 1e-7:
                        labelstring_down = '$1$'
                    if np.abs(value_dd + 1.0) < 1e-7:
                        labelstring_down = '$-1$'

                    if np.abs(value_dd - 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring_down = '$i\\sqrt{3}$'
                    if np.abs(value_dd + 1.0j * np.sqrt(3)) < 1e-7:
                        labelstring_down = '$-i\\sqrt{3}$'

                    if np.abs(value_dd - np.sqrt(3)) < 1e-7:
                        labelstring_down = '$\\sqrt{3}$'
                    if np.abs(value_dd + np.sqrt(3)) < 1e-7:
                        labelstring_down = '$-\\sqrt{3}$'

                labelstring_down = '(' + str(orbit1) + '-' + str(orbit2) + '), ' + labelstring_down + ', $\\downarrow$'

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
                shiftval = 0.4 - ((orbit1 * config.n_orbitals + orbit2) + 4) * 0.1
                plt.text(*(r2 + np.random.uniform(-0.003, 0.003, size=2) + shiftval * textshift), labelstring_up, zorder=10, fontsize=8, color='orange')
                shiftval = 0.4 - ((orbit1 * config.n_orbitals + orbit2)) * 0.1
                plt.text(*(r2 + np.random.uniform(-0.003, 0.003, size=2) + shiftval * textshift), labelstring_down, zorder=10, fontsize=8, color='violet')
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #plt.xlim([-0.5, 1.25])
    #plt.ylim([-0.75, 0.75])
    # plt.title(name + ' pairing')
    plt.savefig('../plots/' + name + '.pdf')
    plt.clf()
    return



def plot_all_Jastrow(config):
    for index, jastrow in enumerate(config.jastrows_list):
        plot_Jastrow(config, jastrow, index)
    return

def plot_Jastrow(config, Jastrow, index):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'

    pairing, name = Jastrow

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
                value = 1.

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
    plt.title(name)
    plt.savefig('../plots/' + name + '.pdf')
    plt.clf()
    return


### given kx, ky, returns the fourier transform of H_mean_field (2xn_orbitalsxn_sublattices)-dim matrix (spin included)
def get_spectrum_K(HMF, config, k_real):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'
    
    if geometry == 'hexagonal':
        R = models.R_hexagonal
        G = models.G_hexagonal
    else:
        R = models.R_square
        G = models.G_square

    Ls, n_orbitals, n_sublattices = config.Ls, config.n_orbitals, config.n_sublattices

    HMF_fourier = K_FT(k_real, HMF, config, R, spin_dof = 2)

    return np.linalg.eigh(HMF_fourier)

def is_commensurate(L, k):
    m_x = L * (np.sqrt(3) * k[0] + k[1]) / 2 / (2 * np.pi)
    m_y = L * (np.sqrt(3) * k[0] - k[1]) / 2 / (2 * np.pi)

    if np.abs(m_x - np.rint([m_x])[0]) < 0.03 and np.abs(m_y - np.rint([m_y])[0]) < 0.03:
        print(m_x, m_y)
        return True
    return False

def get_MFH(config, mu_given = None, gap_given = None, only_free = False):
    K_up = config.model(config, config.mu, spin = +1.0)[0]
    K_up = models.xy_to_chiral(K_up, 'K_matrix', config, config.chiral_basis)
    K_up = models.apply_TBC(config, config.twist, deepcopy(K_up), inverse = False)

    K_down = config.model(config, config.mu, spin = -1.0)[0].T
    K_down = models.xy_to_chiral(K_down, 'K_matrix', config, config.chiral_basis)
    K_down = models.apply_TBC(config, config.twist, deepcopy(K_down), inverse = True)

    mu, fugacity, waves, gap, jastrow = config.unpack_parameters(config.initial_parameters)

    C2y = pairings.C2y_symmetry_map_chiral
    C2y = np.kron(np.eye(2), C2y)

    C3z = pairings.C3z_symmetry_map_chiral
    C3z = np.kron(np.eye(2), C3z)

    if mu_given is not None:
        Deltas_twisted = [models.apply_TBC(config, config.twist, deepcopy(gap), inverse = False) for gap in config.pairings_list_unwrapped]
        Delta = pairings.get_total_pairing_upwrapped(config, Deltas_twisted, gap)
        reg = models.apply_TBC(config, config.twist, deepcopy(config.reg_gap_term), inverse = False) * config.reg_gap_val
        Ts = []

        for dmu in mu_given:
            T = scipy.linalg.block_diag(K_up - np.eye(K_up.shape[0]) * (mu + dmu), -(K_down - np.eye(K_down.shape[0]) * (mu + dmu))) + 0.0j
            T[:config.total_dof // 2, config.total_dof // 2:] = Delta + reg
            T[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T + reg.conj().T


            Ts.append(T.copy() * 1.)

            Deltaonly = T * 0.0
            Deltaonly[:config.total_dof // 2, config.total_dof // 2:] = Delta
            Deltaonly[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T

            eigenvalues, eigenvectors = np.linalg.eigh(Ts[-1])
            idxs = np.argsort(np.abs(eigenvalues))[:2]
            # print(eigenvalues[idxs])
            vplus = eigenvectors[:, idxs[0]]
            vminus = eigenvectors[:, idxs[1]]

            print(np.dot(vplus.conj(), C3z.dot(vplus)))
            print(np.dot(vminus.conj(), C3z.dot(vminus)))
            #print(C3z.shape)
            #print(Deltaonly.shape)
            print(np.trace(Delta.T.conj().dot(pairings.C3z_symmetry_map_chiral.dot(Delta.dot(pairings.C3z_symmetry_map_chiral.T)))) \
                / np.trace(np.dot(Delta, Delta.conj().T))) 

            print(np.abs(np.dot(vplus.conj(), np.dot(Deltaonly, vminus))), np.abs(np.dot(vplus.conj(), vminus)), eigenvalues[idxs], idxs, dmu, \
                  np.trace(np.dot(Deltaonly, Deltaonly.T.conj())))
        #exit(-1)
        return Ts


    K_up -= np.eye(K_up.shape[0]) * mu
    K_down -= np.eye(K_down.shape[0]) * mu

    T = scipy.linalg.block_diag(K_up, -K_down) + 0.0j
    if only_free:
        return T

    

    if gap_given is None:
        Delta = pairings.get_total_pairing_upwrapped(config, config.pairings_list_unwrapped, gap)

    
        T[:config.total_dof // 2, config.total_dof // 2:] = Delta
        T[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T
        return T

    else:
        Ts = []
        Delta = pairings.get_total_pairing_upwrapped(config, config.pairings_list_unwrapped, [1])
        for gap in gap_given:
            T_gap = T.copy() * 1.0
    
            T_gap[:config.total_dof // 2, config.total_dof // 2:] = Delta * gap
            T_gap[config.total_dof // 2:, :config.total_dof // 2] = Delta.conj().T * gap
            Ts.append(T_gap.copy() * 1.0)
        return Ts


def plot_MF_spectrum_profile(config):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'
    if geometry == 'square':
        print('Not supported!')
        exit(-1)

    K_point = 2 * np.pi * np.array([2 / np.sqrt(3), 2.0 / 3.0]) / 2.
    Gamma_point = 2 * np.pi * np.array([0., 0.]) / 2.
    M_point = 2 * np.pi * np.array([2 / np.sqrt(3), 0]) / 2.
    Kprime_point = 2 * np.pi * np.array([2 / np.sqrt(3), -2.0 / 3.0]) / 2.

    k_real_list = [(1 - alpha) * K_point + alpha * Gamma_point for alpha in np.linspace(0, 1, 100)] + \
                  [(1 - alpha) * Gamma_point + alpha * M_point for alpha in np.linspace(0, 1, 100)] + \
                  [(1 - alpha) * M_point + alpha * Kprime_point for alpha in np.linspace(0, 1, 100)]

    MFH = get_MFH(config, only_free = True)

    energies = []
    for k_real in k_real_list:
        if not is_commensurate(config.Ls, k_real):
            energies.append(np.zeros(8))
            continue
        E, v = get_spectrum_K(MFH, config, k_real)
        adding = []
        for i in range(v.shape[1]):
            if np.sum(np.abs(v[:, i] ** 2)) > 0.5:
                adding.append(E[i].real)
        for i in range(v.shape[1]):
            if np.sum(np.abs(v[:, i] ** 2)) < 0.5:
                adding.append(E[i].real)
        energies.append(np.array(adding))
    energies = np.array(energies)

    #energies = np.array([ if is_commensurate(config.Ls, k_real) else  ])
    [plt.plot(np.where(np.abs(energies[:, i]) > 0.0)[0], energies[:, i][np.where(np.abs(energies[:, i]) > 0.0)[0]], ls = '--')\
              for i in range(energies.shape[1])]
    #plt.plot([0, 97], [0, 0], ls = '--', color = 'black')
    #plt.xlim([0, 97])
    plt.grid(True)
    plt.show()



def plot_levels_evolution_gap(config):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'
    if geometry == 'square':
        print('Not supported!')
        exit(-1)
    gaps = np.linspace(0, 1e-1, 100)


    eigvls = []
    MFHs = get_MFH(config, gap_given = gaps)
    eigvls = [np.linalg.eigh(MFH)[0] for MFH in MFHs]
    eigvls = np.array(eigvls).T
    for level in eigvls:
        #print(level)
        plt.plot(gaps, level)

    plt.grid(True, ls='--', color='black', alpha=0.15)
    plt.ylim([-0.1, 0.1])
    plt.show()


def plot_levels_evolution_mu(config):
    geometry = 'hexagonal' if config.n_sublattices == 2 else 'square'
    if geometry == 'square':
        print('Not supported!')
        exit(-1)
    dmus = np.linspace(-1, 1, 400)


    eigvls = []
    gaps = []
    MFHs = get_MFH(config, mu_given = dmus)
    eigvls = [np.linalg.eigh(MFH)[0] for MFH in MFHs]
    for point in eigvls:
        gap = (np.sort(point)[config.total_dof // 2] - np.sort(point)[config.total_dof // 2 - 1]) / 2
        gaps.append(gap)
    eigvls = np.array(eigvls).T
    for level in eigvls:
        #print(level)
        plt.plot(dmus, level)

    plt.grid(True, ls='--', color='black', alpha=0.15)
    #plt.ylim([-0.03, 0.03])
    plt.show()

    plt.plot(dmus, gaps)

    plt.grid(True, ls='--', color='black', alpha=0.15)
    #plt.ylim([-0.03, 0.03])
    plt.show()