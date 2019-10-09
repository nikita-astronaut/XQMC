import numpy as np
from copy import deepcopy

def diff_modulo(x, y, L, d):
    if d >= 0:
        return (x - y + L) % L == d  # or (x - y + L) % L == L - d
    else:
        return (x - y + L) % L == L + d

def nearest_neighbor(r1, r2, L):
    if r1[1] == r2[1] and r1[0] == r2[0]:
        return True
    if r1[1] == r2[1] and diff_modulo(r1[0], r2[0], L, 1):
        return True
    if r1[0] == r2[0] and diff_modulo(r1[1], r2[1], L, 1):
        return True
    return False

def fifth_nearest_neighbor(r1, r2, L):
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, -2):
        return True
    if diff_modulo(r1[0], r2[0], L, -2) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    if diff_modulo(r1[0], r2[0], L, 1) and diff_modulo(r1[1], r2[1], L, 1):
        return True
    return False

def from_linearized_index(index, L, n_orbitals):
    orbit = index % n_orbitals
    coord = index // n_orbitals
    sublattice = coord % 2
    coord = coord // 2

    x = coord // L
    y = coord % L
    return orbit, sublattice, x, y

def to_linearized_index(x, y, sublattice, orbit, L, n_orbitals):
    return orbit + n_orbitals * (sublattice + 2 * (y + x * L))

def H_TB_simple(L, mu):
    n_orbitals = 2
    t1, t2 = 0.331, (-0.010 + 1.0j * 0.097)

    K = np.zeros((2 * n_orbitals * L * L, 2 * n_orbitals * L * L))
    for first in range(2 * n_orbitals * L * L):
        for second in range(2 * n_orbitals * L * L):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), L, n_orbitals)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), L, n_orbitals)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor(r1, r2, L) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

            if orbit2 == orbit1 and fifth_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
                K[first, second] = np.real(t2)
            if orbit2 != orbit1 and fifth_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
                if orbit1 == 0 and orbit2 == 1:
                    K[first, second] = np.imag(t2)
                else:
                    K[first, second] = -np.imag(t2)

    K = K + K.conj().T
    K = K + np.diag(mu * np.ones(2 * n_orbitals * L * L))
    return K

def H_TB_Sorella(L, mu):
    t1 = 1.
    n_orbitals = 1
    K = np.zeros((2 * n_orbitals * L * L, 2 * n_orbitals * L * L))
    for first in range(2 * n_orbitals * L * L):
        for second in range(2 * n_orbitals * L * L):
            orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), L, n_orbitals)
            orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), L, n_orbitals)

            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            if orbit1 == orbit2 and nearest_neighbor(r1, r2, L) and sublattice1 == 0 and sublattice2 == 1:
                K[first, second] = t1

    K = K + K.conj().T
    K = K + np.diag(mu * np.ones(2 * n_orbitals * L * L))
    return K
