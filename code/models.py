import numpy as np

def from_linearized_index(index, L):
	orbit = index % 2
	coord = index // 2
	sublattice = coord % 2
	coord = coord // 2

	x = coord // L
	y = coord % L
	return orbit, sublattice, x, y

def to_linearized_index(x, y, sublattice, orbit, L):
	return orbit + 2 * (sublattice + 2 * (y + x * L))

def H_TB_simple(L):
	t1, t2 = 0.331, (-0.010 + 1.0j * 0.097)

	K = np.zeros((4 * L * L, 4 * L * L))
	for first in range(4 * L * L):
		for second in range(4 * L * L):
			orbit1, sublattice1, x1, y1 = from_linearized_index(deepcopy(first), L)
			orbit2, sublattice2, x2, y2 = from_linearized_index(deepcopy(second), L)
			coord1, coord2 = y1 + x1 * L, y2 + x2 * L

			r1 = np.array([x1, y1])
			r2 = np.array([x2, y2])

			if orbit1 == orbit2 and nearest_neighbor_below(r1, r2, L) and sublattice1 == 0 and sublattice2 == 1:
				K[first, second] = t1

			if orbit2 == orbit1 and next_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
				K[first, second] = np.real(t2)
			if orbit2 != orbit1 and next_nearest_neighbor(r1, r2, L) and sublattice2 == sublattice1:
				if orbit1 == 0 and orbit2 == 1:
					K[first, second] = np.imag(t2)
				else:
					K[first, second] = -np.imag(t2)

	K = K + K.conj().T
	K = K + np.diag(mu * np.ones(4 * L * L))
	return K
