import numpy as np
import models

class on_site_and_nn_pairing:
    def __init__(self, config):
        self.config = config
        self.parameters = np.random.random(64)  # (3 nns + 1 on-site) x 4 spin x 4 orbit

    def get_pairing_f(self, index1, index2):
        spin1 = index1 % 2
        index1 = index1 // 2

        spin2 = index2 % 2
        index2 = index2 // 2

        orbit1, sublattice1, x1, y1 = models.from_linearized_index(index1, self.config.Ls, self.config.n_orbitals)
        orbit2, sublattice2, x2, y2 = models.from_linearized_index(index2, self.config.Ls, self.config.n_orbitals)
        space1 = x1 * self.config.Ls + y1
        space2 = x2 * self.config.Ls + y2


        
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])

        if models.nearest_neighbor_hexagonal(r1, r2, self.config.Ls) and sublattice1 == 0 and sublattice2 == 1:
            d = models.nearest_neighbor_hexagonal_dir(r1, r2, self.config.Ls)
            return self.parameters[d * 16 + mat_index1 * 4 + mat_index2]

        if models.nearest_neighbor_hexagonal(r2, r1, self.config.Ls) and sublattice1 == 1 and sublattice2 == 0:
            d = models.nearest_neighbor_hexagonal_dir(r2, r1, self.config.Ls)
            return -self.parameters[d * 16 + mat_index2 * 4 + mat_index1]

        ##if x1 == x2 and y1 == y2 and sublattice1 == sublattice2:  # on-site pairing
        #    return (self.parameters[0 * 16 + mat_index1 * 4 + mat_index2] - self.parameters[0 * 16 + mat_index2 * 4 + mat_index1]) / 2.  # in fact there are only 6 independent parameters, perhaps later use that for convergence
        return 0.0
