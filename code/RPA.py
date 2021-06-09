import models
import numpy as np
import os
import sys
import config_vmc as cv_module
from numba import jit
from scipy.linalg import schur
from copy import deepcopy
from time import time

@jit(nopython=True)
def from_linearized_index(index, L, n_orbitals, n_sublattices = 2):
    orbit = index % n_orbitals
    coord = index // n_orbitals
    sublattice = coord % n_sublattices
    coord = coord // n_sublattices

    x = coord // L
    y = coord % L
    return orbit, sublattice, x, y


@jit(nopython=True)
def to_linearized_index(x, y, sublattice, orbit, L, n_orbitals, n_sublattices = 2):
    return orbit + n_orbitals * (sublattice + n_sublattices * (y + x * L))

@jit(nopython=True)
def nearest_neighbor_hexagonal(r1, r2, L):
    if r1[1] == r2[1] and r1[0] == r2[0]:
        return True, 1
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        return True, 2
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True, 3
    return False, 0

@jit(nopython=True)
def next_nearest_neighbor_hexagonal(r1, r2, L):
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True, 1

    if diff_modulo(r1[0], r2[0], L, 1) and r1[1] == r2[1]:
        return True, 2

    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, -1):
        return True, 4

    if diff_modulo(r1[0], r2[0], L, -1) and r1[1] == r2[1]:
        return True, 5

    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -1):
        return True, 3

    if diff_modulo(r1[0], r2[0], L, -1) and diff_modulo(r1[1], r2[1], L, 1):
        return True, 6

    return False, -1


@jit(nopython=True)
def third_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -1):
        return True
    if diff_modulo(r1[0], r2[0], L, 0) and diff_modulo(r1[1], r2[1], L, -1):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, 0):
        return True
    return False


@jit(nopython=True)
def fifth_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -2):
        return True
    if diff_modulo(r1[0], r2[0], L, -2) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    return False

@jit(nopython=True)
def diff_modulo(x, y, L, d):
    if d >= 0:
        return (x - y + L) % L == d  # or (x - y + L) % L == L - d
    return (x - y + L) % L == L + d

@jit(nopython=True)
def _model_hex_2orb_Koshino(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    t1, t2, t5, t4 = 0.331, (-0.010 * 0. + 1.0j * 0.097), 0 * 0.119, 0 * 0.036


    K = np.zeros((total_dof // 2, total_dof // 2)) * 1.0j
    for first in range(total_dof // 2):
        for second in range(total_dof // 2):
            orbit1, sublattice1, x1, y1 = from_linearized_index(first, Ls, n_orbitals, n_sublattices)
            orbit2, sublattice2, x2, y2 = from_linearized_index(second, Ls, n_orbitals, n_sublattices)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor_hexagonal(r1, r2, Ls)[0] and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] += t1

            if orbit2 == orbit1 and fifth_nearest_neighbor(r1, r2, Ls) and sublattice2 == sublattice1:
                K[first, second] += np.real(t2)
            if orbit2 != orbit1 and fifth_nearest_neighbor(r1, r2, Ls) and sublattice2 == sublattice1:
                if orbit1 == 0 and orbit2 == 1:
                    K[first, second] += np.imag(t2)
                else:
                    K[first, second] += -np.imag(t2)

    K = K + K.conj().T
    return K


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

@jit(nopython=True)
def get_fft(N, n_bands):
    W = np.zeros((N ** 2, N ** 2), dtype=np.complex128)
    for kx in range(N):
        for ky in range(N):
            for x in range(N):
                for y in range(N):
                    W[x * N + y, kx * N + ky] = np.exp(2.0j * np.pi / N * kx * x + 2.0j * np.pi / N * ky * y)
    return np.kron(W, np.eye(n_bands)) / N / np.sqrt(n_bands)

@jit(nopython=True)
def fermi(energy, beta, mu):
    energy = energy.real
    #if energy < mu:
    #    return 1.
    #elif energy == mu:
    #    return 0.5
    #return 0.

    return 1 / (1. + np.exp((energy - mu) * beta))

@jit(nopython=True)
def lattice_to_physical(lattice):
    x = (lattice[0] + lattice[1]) * np.sqrt(3) / 2
    y = (lattice[0] - lattice[1]) / 2

    if lattice[2] == 1:
        x += 1. / np.sqrt(3)
    return np.array([x, y])

@jit(nopython=True)
def physical_to_lattice(physical):
    x, y = physical
    lattice = [0, 0, 0]
    if np.abs(int(np.rint(2 * x.real / np.sqrt(3))) - 2 * x.real / np.sqrt(3)) >= 1e-5:
        lattice[2] = 1
        x = x - 1. / np.sqrt(3)


    lattice[1] = int(np.rint((x.real - np.sqrt(3) * y.real) / np.sqrt(3)))
    lattice[0] = int(np.rint((x.real + np.sqrt(3) * y.real) / np.sqrt(3)))

    #print(physical, lattice, (x.real + np.sqrt(3) * y.real) / np.sqrt(3), (x.real - np.sqrt(3) * y.real) / np.sqrt(3))
    return np.array(lattice)

@jit(nopython=True)
def get_C3z_symmetry_map(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2

    mapping = np.zeros((total_dof // 2, total_dof // 2), dtype=np.complex128)  # trivial mapping
    rotation_matrix = np.array([[np.cos(2 * np.pi / 3.), np.sin(2 * np.pi / 3.)], \
                                [-np.sin(2 * np.pi / 3.), np.cos(2 * np.pi / 3.)]], dtype=np.complex128)
    rotation_matrix_orbital = np.array([[np.exp(-2.0j * np.pi / 3), 0], [0, np.exp(2.0j * np.pi / 3)]], dtype=np.complex128)

    for preindex in range(total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            from_linearized_index(preindex, Ls, n_orbitals, n_sublattices)

        orbit_preimage_vector = np.zeros(n_orbitals, dtype=np.complex128); orbit_preimage_vector[orbit_preimage] = 1.
        r_preimage = lattice_to_physical([x_preimage, y_preimage, sublattice_preimage]).astype(np.complex128)

        orbit_image_vector = rotation_matrix_orbital.dot(orbit_preimage_vector) #np.einsum('ij,j->i', rotation_matrix_orbital, orbit_preimage_vector)

        r_image = rotation_matrix.dot(r_preimage)#np.einsum('ij,j->i', rotation_matrix, r_preimage)
        x_image, y_image, sublattice_image = physical_to_lattice(r_image)
        #print(r_preimage, '-->', r_image)
        # print(sublattice_preimage, x_preimage, y_preimage, '-->', sublattice_image, x_image, y_image)
        x_image = int(np.rint(x_image)) % Ls; y_image = int(np.rint(y_image)) % Ls
        
        for orbit_image in range(n_orbitals):
            coefficient = orbit_image_vector[orbit_image]
            index = to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               Ls, n_orbitals, n_sublattices)
            #if coefficient != 0.:
            #    print(preindex, '-->', index)
            mapping[preindex, index] += coefficient
                
    assert np.sum(np.abs(mapping.dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_3z^3 = I
    return mapping

@jit(nopython=True)
def get_C2y_symmetry_map(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    mapping = np.zeros((total_dof // 2, total_dof // 2), dtype=np.complex128)  # trivial mapping

    for preindex in range(total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            from_linearized_index(preindex, Ls, n_orbitals, n_sublattices)     

        orbit_image = 1 - orbit_preimage
        coefficient = -1.0

        r_preimage = lattice_to_physical([x_preimage, y_preimage, sublattice_preimage]).astype(np.complex128)
        r_preimage -= np.array([1. / np.sqrt(3) / 2, 0.0])
        r_image = np.array([-r_preimage[0], r_preimage[1]]) + np.array([1. / np.sqrt(3) / 2, 0.0])

        x_image, y_image, sublattice_image = physical_to_lattice(r_image)
        x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
        x_image = (x_image % Ls); y_image = (y_image % Ls)
        
        index = to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                           Ls, n_orbitals, n_sublattices)

        mapping[preindex, index] += coefficient

    assert np.sum(np.abs(mapping.dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_2y^2 = I
    return mapping + 0.0j

@jit(nopython=True)
def get_TRS_symmetry_map(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2
    mapping = np.zeros((total_dof // 2, total_dof // 2), dtype=np.complex128)  # trivial mapping

    for preindex in range(total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            from_linearized_index(preindex, Ls, n_orbitals, n_sublattices)     

        orbit_image = 1 - orbit_preimage

        r_preimage = lattice_to_physical([x_preimage, y_preimage, sublattice_preimage]).astype(np.complex128)
        r_image = r_preimage

        x_image, y_image, sublattice_image = physical_to_lattice(r_image)
        x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
        x_image = (x_image % Ls); y_image = (y_image % Ls)
        
        index = to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                           Ls, n_orbitals, n_sublattices)

        mapping[preindex, index] += 1

    assert np.sum(np.abs(mapping.dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_2y^2 = I
    return mapping + 0.0j


@jit(nopython=True)
def get_Tx_symmetry_map(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2

    mapping = np.zeros((total_dof // 2, total_dof // 2), dtype=np.complex128)  # trivial mapping
    for preindex in range(total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            from_linearized_index(preindex, Ls, n_orbitals, n_sublattices)

        orbit_preimage_vector = np.zeros(n_orbitals, dtype=np.complex128); orbit_preimage_vector[orbit_preimage] = 1.
        r_preimage = lattice_to_physical([x_preimage, y_preimage, sublattice_preimage]).astype(np.complex128)

        orbit_image_vector = orbit_preimage_vector

        r_image = r_preimage + np.array([np.sqrt(3), 1]) / 2. #rotation_matrix.dot(r_preimage)#np.einsum('ij,j->i', rotation_matrix, r_preimage)
        x_image, y_image, sublattice_image = physical_to_lattice(r_image)
        #print(r_preimage, '-->', r_image)
        # print(sublattice_preimage, x_preimage, y_preimage, '-->', sublattice_image, x_image, y_image)
        x_image = int(np.rint(x_image)) % Ls; y_image = int(np.rint(y_image)) % Ls
        
        for orbit_image in range(n_orbitals):
            coefficient = orbit_image_vector[orbit_image]
            index = to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               Ls, n_orbitals, n_sublattices)
            mapping[preindex, index] += coefficient
                
    #assert np.sum(np.abs(mapping.dot(mapping).dot(mapping).dot(mapping).dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_3z^3 = I
    return mapping

@jit(nopython=True)
def get_Ty_symmetry_map(Ls):
    n_orbitals = 2
    n_sublattices = 2
    total_dof = Ls ** 2 * n_orbitals * n_sublattices * 2

    mapping = np.zeros((total_dof // 2, total_dof // 2), dtype=np.complex128)  # trivial mapping
    for preindex in range(total_dof // 2):
        orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
            from_linearized_index(preindex, Ls, n_orbitals, n_sublattices)

        orbit_preimage_vector = np.zeros(n_orbitals, dtype=np.complex128); orbit_preimage_vector[orbit_preimage] = 1.
        r_preimage = lattice_to_physical([x_preimage, y_preimage, sublattice_preimage]).astype(np.complex128)

        orbit_image_vector = orbit_preimage_vector

        r_image = r_preimage + np.array([np.sqrt(3), -1]) / 2. #rotation_matrix.dot(r_preimage)#np.einsum('ij,j->i', rotation_matrix, r_preimage)
        x_image, y_image, sublattice_image = physical_to_lattice(r_image)
        #print(r_preimage, '-->', r_image)
        # print(sublattice_preimage, x_preimage, y_preimage, '-->', sublattice_image, x_image, y_image)
        x_image = int(np.rint(x_image)) % Ls; y_image = int(np.rint(y_image)) % Ls
        
        for orbit_image in range(n_orbitals):
            coefficient = orbit_image_vector[orbit_image]
            index = to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               Ls, n_orbitals, n_sublattices)
            mapping[preindex, index] += coefficient
                
    #assert np.sum(np.abs(mapping.dot(mapping).dot(mapping).dot(mapping).dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-5  # C_3z^3 = I
    return mapping

import sys
L = int(sys.argv[1])
U = float(sys.argv[2])
J = float(sys.argv[3])

Ls = L
n_bands = 4
beta = L
fft = get_fft(L, n_bands)
mapping = get_C3z_symmetry_map(L)
C2 = get_C2y_symmetry_map(L)
Tx = get_Tx_symmetry_map(L)
Ty = get_Ty_symmetry_map(L)

K0 = _model_hex_2orb_Koshino(L)
U_xy_to_chiral = np.kron(np.eye(K0.shape[0] // 2), np.array([[1, 1], [-1.0j, +1.0j]]) / np.sqrt(2))                                                   

K0 = U_xy_to_chiral.conj().T.dot(K0).dot(U_xy_to_chiral)
assert np.isclose(np.dot(K0.conj().flatten(), mapping.dot(K0).dot(mapping.T.conj()).flatten()) / np.dot(K0.conj().flatten(), K0.flatten()), 1)
assert np.isclose(np.dot(K0.conj().flatten(), Tx.dot(K0).dot(Tx.T.conj()).flatten()) / np.dot(K0.conj().flatten(), K0.flatten()), 1)
assert np.isclose(np.dot(K0.conj().flatten(), Ty.dot(K0).dot(Ty.T.conj()).flatten()) / np.dot(K0.conj().flatten(), K0.flatten()), 1)
TRS = np.concatenate([np.array([2 * i + 1, 2 * i]) for i in range(K0.shape[0] // 2)])
assert np.allclose(K0, K0.conj().T)

K0_TRS = K0[TRS, :]
K0_TRS = K0_TRS[:, TRS].conj()

assert np.isclose(np.vdot(K0.flatten(), K0_TRS.flatten()) / np.dot(K0.conj().flatten(), K0.flatten()), 1.0)
TRS = get_TRS_symmetry_map(L)

K0_plus = K0[::2, :]
K0_plus = K0_plus[:, ::2]

K0_minus = K0[1::2, :]
K0_minus = K0_minus[:, 1::2]

assert np.allclose(K0_plus, K0_minus.conj())
assert np.allclose(np.linalg.eigh(K0_plus)[0], np.linalg.eigh(K0_minus)[0])

energies_solid = np.linalg.eigh(K0)[0]
print(energies_solid)
### doing the FFT of K0 ###
K0_fft = fft.conj().T.dot(K0).dot(fft)
K0_fft_plus = K0_fft[::2, :]
K0_fft_plus = K0_fft_plus[:, ::2]

K0_fft_minus = K0_fft[1::2, :]
K0_fft_minus = K0_fft_minus[:, 1::2]

assert np.allclose(K0_fft_plus, K0_fft_plus.conj().T)
assert np.allclose(K0_fft_minus, K0_fft_minus.conj().T)


K0_check = K0_fft.copy()
for i in range(K0_check.shape[0] // n_bands):
    K0_check[i * n_bands:i * n_bands + n_bands,i * n_bands:i * n_bands + n_bands] = 0.0
assert np.isclose(np.sum(np.abs(K0_check)), 0.0)

A = np.zeros((L, L, n_bands, n_bands), dtype=np.complex128)
energies = np.zeros((L, L, n_bands), dtype=np.complex128)

energies_plus = np.zeros((L, L, n_bands // 2), dtype=np.float64)
energies_minus = np.zeros((L, L, n_bands // 2), dtype=np.float64)
for i in range(K0_check.shape[0] // n_bands):
    kx, ky = i % L, i // L
    mat = K0_fft[i * n_bands:i * n_bands + n_bands,i * n_bands:i * n_bands + n_bands] * 4.
    assert np.allclose(mat.conj().T, mat)
    A[kx, ky, ...] = np.linalg.eigh(mat)[1]
    
    nonch = 0
    for a in range(n_bands):
        for b in range(n_bands):
            if (a + b) % 2 == 1:
                nonch += np.abs(mat[a, b])
    assert np.isclose(nonch, 0.0)
    
    
    energies[kx, ky, ...] = np.linalg.eigh(mat)[0]
    energies_plus[kx, ky, ...] = \
        np.linalg.eigh(K0_fft_plus[i * n_bands // 2:i * n_bands // 2 + n_bands // 2, \
                                   i * n_bands // 2:i * n_bands // 2 + n_bands // 2])[0].real
    energies_minus[kx, ky, ...] = \
        np.linalg.eigh(K0_fft_minus[i * n_bands // 2:i * n_bands // 2 + n_bands // 2, \
                                    i * n_bands // 2:i * n_bands // 2 + n_bands // 2])[0].real

assert np.allclose(np.sort(energies_plus.flatten()), np.sort(energies_minus.flatten()))
@jit(nopython=True)
def get_susc_zero(Ls, n_bands, A, A_plus_q, energies, energies_plus_q, omega, temp, mu):
    chi = np.zeros((n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    for p in range(n_bands):
        for q in range(n_bands):
            for s in range(n_bands):
                for t in range(n_bands):
                    if ((p + t) % 2) != ((q + s) % 2):
                        continue
                    for kx in range(Ls):
                        for ky in range(Ls):
                            for alpha in range(n_bands):
                                for beta in range(n_bands):
                                    chi[p, q, s, t] -= A[kx, ky, s, alpha] * \
                                               np.conj(A[kx, ky, p, alpha]) * \
                                                       A_plus_q[kx, ky, q, beta] * \
                                               np.conj(A_plus_q[kx, ky, t, beta]) / \
                                                (omega + energies_plus_q[kx, ky, beta] - energies[kx, ky, alpha] + 1.0j * temp) * \
                                                (fermi(energies_plus_q[kx, ky, beta], temp, mu) - fermi(energies[kx, ky, alpha], temp, mu))
                                    if np.abs(energies_plus_q[kx, ky, beta] - energies[kx, ky, alpha]) < 1e-10:
                                        chi[p, q, s, t] -= A[kx, ky, s, alpha] * \
                                               np.conj(A[kx, ky, p, alpha]) * \
                                                       A_plus_q[kx, ky, q, beta] * \
                                               np.conj(A_plus_q[kx, ky, t, beta]) * (-1.0j * np.pi * 0. - temp * np.exp(temp * energies[kx, ky, alpha]) / (1 + np.exp(temp * energies[kx, ky, alpha])) ** 2)

    return chi / n_bands / Ls ** 2

@jit(nopython=True)
def get_Gsinglet(Ls, n_bands, U_s, U_c, chi_s, chi_c):
    Gsinglet = np.zeros((Ls, Ls, Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    sh = (n_bands ** 2, n_bands ** 2)
    sh2 = (n_bands, n_bands, n_bands, n_bands)
    chi_s = chi_s.reshape((Ls, Ls, n_bands ** 2, n_bands ** 2))
    chi_c = chi_c.reshape((Ls, Ls, n_bands ** 2, n_bands ** 2))

    for qx in range(Ls):
        for qy in range(Ls):
            for kx in range(Ls):
                for ky in range(Ls):
                    Gsinglet[kx, ky, qx, qy, ...] = \
                                             U_s[(qx + kx) % Ls, (qy + ky) % Ls, ...] / 4. + \
                                             U_c[(qx + kx) % Ls, (qy + ky) % Ls, ...] / 4. + \
                                             U_s[qx - kx, qy - ky, ...].transpose((0, 2, 1, 3)) / 4. + \
                                             U_c[qx - kx, qy - ky, ...].transpose((0, 2, 1, 3)) / 4. + \
                    3. / 4. * (U_s[(qx + kx) % Ls, (qy + ky) % Ls, ...].reshape(sh) @ chi_s[(qx + kx) % Ls, (qy + ky) % Ls, ...] @ U_s[(qx + kx) % Ls, (qy + ky) % Ls, ...].reshape(sh)).reshape(sh2) - \
                    1. / 4. * (U_c[(qx + kx) % Ls, (qy + ky) % Ls, ...].reshape(sh) @ chi_c[(qx + kx) % Ls, (qy + ky) % Ls, ...] @ U_c[(qx + kx) % Ls, (qy + ky) % Ls, ...].reshape(sh)).reshape(sh2) + \
                    3. / 4. * (U_s[qx - kx, qy - ky, ...].reshape(sh) @ chi_s[(qx - kx) % Ls, (qy - ky) % Ls, ...] @ U_s[qx - kx, qy - ky, ...].reshape(sh)).reshape(sh2).transpose((0, 2, 1, 3)) - \
                    1. / 4. * (U_c[qx - kx, qy - ky, ...].reshape(sh) @ chi_c[(qx - kx) % Ls, (qy - ky) % Ls, ...] @ U_c[qx - kx, qy - ky, ...].reshape(sh)).reshape(sh2).transpose((0, 2, 1, 3))
    return Gsinglet


Alat = np.array([[np.sqrt(3) / 2., 1. / 2.], [np.sqrt(3) / 2., -1. / 2.]])
G = 2 * np.pi * np.array([[1. / np.sqrt(3), +1], [1. / np.sqrt(3), -1.]])  # is this momentum compatible?

print(Alat.dot(G) /2 / np.pi)

@jit(nopython=True)
def get_gf_sum(Ls, n_bands, A, energy, temp, mu):
    gf_sum = np.zeros((Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    n_active_sites = 0.
    
    for qx in range(Ls):
        for qy in range(Ls):
            for a in range(n_bands):
                for b in range(n_bands):
                    for c in range(n_bands):
                        for d in range(n_bands):
                            for alpha in range(n_bands):
                                for beta in range(n_bands):
                                    delta_factor = (fermi(energy[qx, qy, alpha], temp, mu) - fermi(-energy[-qx, -qy, beta], temp, -mu)) / \
                                                   (energy[qx, qy, alpha] + energy[-qx, -qy, beta] - 2 * mu + 1e-10)

                                    if np.abs(energy[qx, qy, alpha] + energy[-qx, -qy, beta] - 2 * mu) < 1e-7:
                                        delta_factor = -np.exp(temp * (energy[qx, qy, alpha] - mu)) / ( 1 + np.exp(temp * (energy[qx, qy, alpha] - mu)) ) ** 2
                                    if np.abs(delta_factor) > temp / 10:
                                        n_active_sites += 1
                                    gf_sum[qx, qy, a, b, c, d] += A[qx, qy, a, alpha] * \
                                                          np.conj(A[qx, qy, b, alpha]) * \
                                                                  A[-qx, -qy, c, beta] * \
                                                          np.conj(A[-qx, -qy, d, beta]) * delta_factor
    return gf_sum, n_active_sites / 3072.0 * 2.

@jit(nopython=True)
def construct_op(Ls, n_bands, Gsinglet, gf_sum):
    op = np.zeros((Ls, Ls, Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    for kx in range(Ls):
        for ky in range(Ls):
            for qx in range(Ls):
                for qy in range(Ls):
                    for abar in range(n_bands):
                        for bbar in range(n_bands):
                            for cbar in range(n_bands):
                                for dbar in range(n_bands):
                                    if ((abar + bbar) % 2 != ((cbar + dbar)) % 2):
                                        continue

                                    for a in range(n_bands):
                                        for b in range(n_bands):
                                            op[kx, ky, qx, qy, abar, bbar, cbar, dbar] -= 1. / Ls ** 2 * \
                                                   Gsinglet[kx, ky, -qx, -qy, a, abar, bbar, b] * \
                                                   gf_sum[qx, qy, a, cbar, b, dbar]
                                                   #green[qx, qy, a, cbar] * \
                                                   #green[-qx, -qy, b, dbar] # FIXME --> need conj?#np.conj(green[(-qx) % Ls, (-qy) % Ls, b, dbar])
                                            
    return op


def get_interaction(n_bands, qx, qy, U, J):
    # k -- z1, q -- z2
    U_total = np.zeros((2 * n_bands, 2 * n_bands, 2 * n_bands, 2 * n_bands), dtype=np.complex128)
    
    q_phys = (G[0] * (qx) + G[1] * (qy)) / L
    exp_q1_pp = np.exp(1.0j * np.dot(q_phys, Alat[0]))
    exp_q2_pp = np.exp(1.0j * np.dot(q_phys, Alat[1]))
    
    q_phys = (G[0] * (-qx) + G[1] * (-qy)) / L
    exp_q1_mm = np.exp(1.0j * np.dot(q_phys, Alat[0]))
    exp_q2_mm = np.exp(1.0j * np.dot(q_phys, Alat[1]))
    
    AB_factor_pp = (1. + exp_q1_pp + exp_q2_pp) / 2
    #AB_factor_pm = (1. + exp_q1_pm + exp_q2_pm) / 2
    #AB_factor_mp = (1. + exp_q1_mp + exp_q2_mp) / 2
    AB_factor_mm = (1. + exp_q1_mm + exp_q2_mm) / 2
    
    BA_factor_pp = np.conj(AB_factor_pp)
    #BA_factor_pm = np.conj(AB_factor_pm)
    #BA_factor_mp = np.conj(AB_factor_mp)
    BA_factor_mm = np.conj(AB_factor_mm)
    
    #print(exp_q1, exp_q2)


    for band in range(4):
        U_total[band, band, band + 4, band + 4] += -U
        #U_total[band + 4, band, band + 4, band] += +U
        #U_total[band, band + 4, band, band + 4] += +U
        U_total[band + 4, band + 4, band, band] += -U
        
    for band in range(4):
        U_total[band + 4, band + 4, band, band] += -U
        #U_total[band, band + 4, band, band + 4] += +U
        #U_total[band + 4, band, band + 4, band] += +U
        U_total[band, band, band + 4, band + 4] += -U

    for subl in range(2):
        for nu in range(2):
            nubar = 1 - nu
            band = subl * 2 + nu
            bandbar = subl * 2 + nubar
            
            for s in range(2):
                for sbar in range(2):
                    #print('')
                    U_total[band + 4 * s, band + 4 * s, bandbar + 4 * sbar, bandbar + 4 * sbar] += -U
                    #U_total[bandbar + 4 * sbar, band + 4 * s, bandbar + 4 * sbar, band + 4 * s] += +U
                    #U_total[band + 4 * s, bandbar + 4 * sbar, band + 4 * s, bandbar + 4 * sbar] += +U
                    U_total[bandbar + 4 * sbar, bandbar + 4 * sbar, band + 4 * s, band + 4 * s] += -U

    
    for nuA in range(2):
        for nuB in range(2):
            for sA in range(2):
                for sB in range(2):
                    U_total[0 * 2 + nuA + 4 * sA, 1 * 2 + nuA + 4 * sA, 0 * 2 + nuB + 4 * sB, 1 * 2 + nuB + 4 * sB] = -J * AB_factor_pp
                    #U_total[1 * 2 + nuB + 4 * sB, 1 * 2 + nuA + 4 * sA, 0 * 2 + nuB + 4 * sB, 0 * 2 + nuA + 4 * sA] = +J * AB_factor_pm
                    #U_total[0 * 2 + nuA + 4 * sA, 0 * 2 + nuB + 4 * sB, 1 * 2 + nuA + 4 * sA, 1 * 2 + nuB + 4 * sB] = +J * AB_factor_mp
                    U_total[1 * 2 + nuB + 4 * sB, 0 * 2 + nuB + 4 * sB, 1 * 2 + nuA + 4 * sA, 0 * 2 + nuA + 4 * sA] = -J * AB_factor_mm
                    
                    U_total[1 * 2 + nuA + 4 * sA, 0 * 2 + nuA + 4 * sA, 1 * 2 + nuB + 4 * sB, 0 * 2 + nuB + 4 * sB] = -J * BA_factor_pp
                    #U_total[0 * 2 + nuB + 4 * sB, 0 * 2 + nuA + 4 * sA, 1 * 2 + nuB + 4 * sB, 1 * 2 + nuA + 4 * sA] = +J * BA_factor_pm
                    #U_total[1 * 2 + nuA + 4 * sA, 1 * 2 + nuB + 4 * sB, 0 * 2 + nuA + 4 * sA, 0 * 2 + nuB + 4 * sB] = +J * BA_factor_mp
                    U_total[0 * 2 + nuB + 4 * sB, 1 * 2 + nuB + 4 * sB, 0 * 2 + nuA + 4 * sA, 1 * 2 + nuA + 4 * sA] = -J * BA_factor_mm   
                    
    #assert np.allclose(U_total, U_total.transpose((3, 2, 1, 0)).conj())    
    return U_total


for mu in np.linspace(-0.20, -0.05, 21):

    print('density:', np.sum(energies < mu) * 2 / np.sum(energies > -1e+6))
    t = time()
    susc_0 = np.zeros((L, L, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    for qx in range(Ls):
        for qy in range(Ls):
            A_plus_q = np.roll(A, shift = -qx, axis=0) 
            A_plus_q = np.roll(A_plus_q, shift = -qy, axis=1)

            energies_plus_q = np.roll(energies, shift = -qx, axis=0)
            energies_plus_q = np.roll(energies_plus_q, shift = -qy, axis=1)
            susc_0[qx, qy, ...] = get_susc_zero(Ls, n_bands, A, A_plus_q, energies, energies_plus_q, 0., beta, mu)
    print('t_susc0: ', time() - t)
    
    
    
    U_s = np.zeros((Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    U_c = np.zeros((Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    U_t = np.zeros((Ls, Ls, 2 * n_bands, 2 * n_bands, 2 * n_bands, 2 * n_bands), dtype=np.complex128)

    chi_s = np.zeros((Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    chi_c = np.zeros((Ls, Ls, n_bands, n_bands, n_bands, n_bands), dtype=np.complex128)
    
    t = time()
    eigs = []
    for qx in range(Ls):
        for qy in range(Ls):
            inter = get_interaction(n_bands, qx, qy, U, J).transpose((0, 2, 1, 3)).conj()
            U_s[qx, qy, ...] = inter[:4, :4, :4, :4] - inter[:4, :4, 4:, 4:]
            U_c[qx, qy, ...] = -inter[:4, :4, :4, :4] - inter[:4, :4, 4:, 4:]
            U_t[qx, qy, ...] = inter
            chi_s[qx, qy, ...] = (np.linalg.inv(np.eye(n_bands ** 2) - susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)) @ U_s[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2))) @ susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2))).reshape((n_bands, n_bands, n_bands, n_bands))
            chi_c[qx, qy, ...] = (np.linalg.inv(np.eye(n_bands ** 2) + susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)) @ U_c[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2))) @ susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2))).reshape((n_bands, n_bands, n_bands, n_bands))

            eig_s = np.linalg.eigh(np.eye(n_bands ** 2) - susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)) @ U_s[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)))[0]
            eig_c = np.linalg.eigh(np.eye(n_bands ** 2) + susc_0[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)) @ U_c[qx, qy, ...].reshape((n_bands ** 2, n_bands ** 2)))[0]
            #print(np.sum(np.abs(chi_s[qx, qy, ...] - chi_c[qx, qy, ...]) ** 2), qx, qy)
            #print(qx, qy)
            #print(eig_s.min(), eig_s.max())
            #print(eig_c.min(), eig_c.max())
            assert np.isclose(eig_s.max(), eig_c.max()) # property guaranteed by the P-symmetry
            assert np.isclose(eig_s.min(), eig_c.min())
            assert np.sum(eig_s < 0) == 0.
            eigs.append(eig_s.min())
            
    print('t_Ucs: ', time() - t)
    print('min \chi eigenvalue', np.min(eigs))
    green = np.zeros((Ls, Ls, n_bands, n_bands), dtype=np.complex128)

    t = time()
    for kx in range(Ls):
        for ky in range(Ls):
            for a in range(n_bands):
                for b in range(n_bands):
                    for band in range(n_bands):
                        green[kx, ky, a, b] += A[kx, ky, a, band] * \
                                       np.conj(A[kx, ky, b, band]) / \
                                      (-energies[kx, ky, band].real + mu)
                    if (a + b) % 2 == 1:
                        assert np.isclose(green[kx, ky, a, b], 0.)
    print('t_green: ', time() - t)          
   
    gf_sum, n_sites = get_gf_sum(Ls, n_bands, A, energies.real, beta, mu)
    print('n_active_sites =', n_sites)
    
    
    susc_0 = susc_0.reshape((Ls, Ls, n_bands, n_bands, n_bands, n_bands))
    t = time()
    Gsinglet = get_Gsinglet(Ls, n_bands, U_s, U_c, chi_s, chi_c)
    print('t_gsinglet: ', time() - t)
    t = time()
    op = construct_op(Ls, n_bands, Gsinglet, gf_sum)
    print('t_constructop: ', time() - t)
    op = op.transpose((0, 1, 4, 5, 2, 3, 6, 7)).reshape((Ls ** 2 * n_bands ** 2, Ls ** 2 * n_bands ** 2))
    
    from scipy.sparse.linalg import eigsh, eigs
    t = time()
    E, eig = eigs(op, k=4, which='LR', maxiter=10000000)
    print('t_val: ', time() - t)
    def gram_schmidt_columns(X):
        Q, R = np.linalg.qr(X)
        return Q

    eig = eig[:, np.argsort(-E.real)]
    E = E[np.argsort(-E.real)].real

    eig = gram_schmidt_columns(eig)
    
    print(E[0].real, mu, E[0].real * np.min(np.abs(energies - mu)) ** 2, np.min(np.abs(energies - mu)))

    vals = []
    c2vals = []
    trsvals = []

    gaps_realspace = []
    t = time()
    for j in range(len(E)):
        if np.abs(E[j]) < 1e-6:
            vals.append(0.)
            c2vals.append(0.)
            trsvals.append(0.)
            continue
        vec = eig[:, j].reshape((Ls ** 2, n_bands, n_bands))

        gap_matrix = np.zeros((Ls ** 2 * n_bands, Ls ** 2 * n_bands), dtype=np.complex128)
        gap_matrix_TRS = np.zeros((Ls ** 2 * n_bands, Ls ** 2 * n_bands), dtype=np.complex128)
        for i in range(Ls ** 2):
            gap_matrix[i * n_bands:i * n_bands + n_bands, \
                       i * n_bands:i * n_bands + n_bands] = vec[i, ...]
            kx = i // Ls
            ky = i % Ls

            kx = (-kx) % Ls
            ky = (-ky) % Ls

            ip = kx * Ls + ky
            gap_matrix_TRS[i * n_bands:i * n_bands + n_bands, \
                           i * n_bands:i * n_bands + n_bands] = vec[ip, ...].conj()

        gap_relspace = fft.dot(gap_matrix).dot(fft.conj().T)
        assert np.isclose(np.vdot(gap_relspace.flatten(), gap_relspace.T.flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()), 1.0)

        gap_relspace_TRS = fft.conj().dot(gap_matrix.conj()).dot(fft.T)
        gap_relspace_TRS_true = fft.dot(gap_matrix_TRS).dot(fft.conj().T)
        assert np.allclose(gap_relspace_TRS, gap_relspace_TRS_true)

        gaps_realspace.append(gap_relspace * 1.)
        assert np.isclose(np.vdot(gap_relspace_TRS.flatten(), gap_relspace_TRS.flatten()) * 16., 1.0)

        vals.append(np.vdot(gap_relspace.flatten(), mapping.dot(gap_relspace).dot(mapping.T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()))
        c2vals.append(np.vdot(gap_relspace.flatten(), C2.dot(gap_relspace).dot(C2.T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()))
        trsvals.append(np.vdot(gap_relspace.flatten(), TRS.dot(gap_relspace_TRS).dot(TRS.T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()))
        print(E[j], \
                '\nC3:', np.vdot(gap_relspace.flatten(), mapping.dot(mapping).dot(gap_relspace).dot(mapping.dot(mapping).T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()), \
                '\nTRS:', np.abs(np.vdot(gap_relspace.flatten(), TRS.dot(gap_relspace_TRS).dot(TRS.T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten())),
                '\nC2:', np.vdot(gap_relspace.flatten(), C2.dot(gap_relspace).dot(C2.T).flatten()) / np.vdot(gap_relspace.flatten(), gap_relspace.flatten()))
        print('-----')
    print('t_gapsymmetries: ', time() - t)
    print('\n\n\n\n')
    exit(-1)
