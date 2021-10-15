import models
import numpy as np
from copy import deepcopy
from numba import jit

def construct_total_jastrow(jastow_list, jastrow_coeff):
    return np.sum(np.array([factor * A[0] for factor, A in zip(jastrow_coeff, jastow_list)]), axis = 0)

def get_jastrow(config, orb_degenerate = False, max_distance = np.inf):
    adjacency_list = config.adjacency_list
    jastrow_list = []
    n_per_dist = (config.n_orbitals * (config.n_orbitals + 1)) // 2

    current_real_matrix = np.zeros(adjacency_list[0][0].shape)

    if orb_degenerate:
        for idx, adj in enumerate(adjacency_list):
            #print(idx, adj, n_per_dist)
            if adj[-1] > max_distance:
                break
            current_real_matrix += adj[0]
            if ((idx + 1) % n_per_dist) == 0:
                jastrow_list.append([deepcopy(current_real_matrix), 'jastrow-deg-deg-{:.2f}'.format(adj[-1])])
                current_real_matrix = np.zeros(adjacency_list[0][0].shape)
        return jastrow_list
    
    for idx, adj in enumerate(adjacency_list):
        if adj[-1] > max_distance:
            break
        if adj[1] == adj[2]:
            continue
        jastrow_list.append([adj[0], 'jastrow-{:d}-{:d}-{:.2f}'.format(*adj[1:])])

    for idx, adj in enumerate(adjacency_list):
        if adj[-1] > max_distance:
            break
        if adj[1] != adj[2]:
            continue
        current_real_matrix += adj[0]
        if ((idx + 1) % n_per_dist) == 0:
            jastrow_list.append([deepcopy(current_real_matrix), 'jastrow-same-same-{:.2f}'.format(adj[-1])])
            current_real_matrix = np.zeros(adjacency_list[0][0].shape)

    return jastrow_list

@jit(nopython=True)
def get_spatial_mask(Ls, x_shift, y_shift):
    mask = np.zeros((Ls ** 2, Ls ** 2))
    for i in range(Ls ** 2):
        for j in range(Ls ** 2):
            xi, yi = i // Ls, i % Ls
            xj, yj = j // Ls, j % Ls
            if (xi - xj) % Ls == x_shift and (yi - yj) % Ls == y_shift:
                mask[i, j] = 1.
    return mask


def get_jastrow_Koshino(config):
    def get_C2y_symmetry_map_chiral_jastrow(config):
        mapping = np.zeros((config.total_dof // 2, config.total_dof // 2))

        for preindex in range(config.total_dof // 2):
            orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
                models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)
            orbit_image = 1 - orbit_preimage

            r_preimage = np.array(models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], 'hexagonal'))
            r_preimage -= np.array([1. / np.sqrt(3) / 2, 0.0])
            r_image = np.array([-r_preimage[0], r_preimage[1]]) + np.array([1. / np.sqrt(3) / 2, 0.0])

            x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, 'hexagonal')
            x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
            x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)

            index = models.to_linearized_index(x_image, y_image, sublattice_image, orbit_image, \
                                               config.Ls, config.n_orbitals, config.n_sublattices)

            mapping[preindex, index] += 1.

        assert np.sum(np.abs(mapping.dot(mapping) - np.eye(mapping.shape[0]))) < 1e-12  # C_2y^2 = I
        return mapping

    def get_C3z_symmetry_map_chiral_jastrow(config):
        assert config.n_sublattices == 2
        geometry = 'hexagonal'

        mapping = np.zeros((config.total_dof // 2, config.total_dof // 2)) # trivial mapping
        rotation_matrix = np.array([[np.cos(2 * np.pi / 3.), np.sin(2 * np.pi / 3.)], \
                                    [-np.sin(2 * np.pi / 3.), np.cos(2 * np.pi / 3.)]])

        for preindex in range(config.total_dof // 2):
            orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
                models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)

            r_preimage = models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], geometry)
            r_image = np.einsum('ij,j->i', rotation_matrix, r_preimage)

            x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, geometry)

            x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
            x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)
            
            index = models.to_linearized_index(x_image, y_image, sublattice_image, orbit_preimage, \
                                               config.Ls, config.n_orbitals, config.n_sublattices)
            mapping[preindex, index] = 1.
        assert np.sum(np.abs(mapping.dot(mapping).dot(mapping) - np.eye(mapping.shape[0]))) < 1e-12  # C_3z^3 = I
        return mapping

    def get_TRS_symmetry_map_chiral_jastrow(config):
        assert config.n_sublattices == 2
        geometry = 'hexagonal'

        mapping = np.zeros((config.total_dof // 2, config.total_dof // 2)) # trivial mapping

        for preindex in range(config.total_dof // 2):
            orbit_preimage, sublattice_preimage, x_preimage, y_preimage = \
                models.from_linearized_index(preindex, config.Ls, config.n_orbitals, config.n_sublattices)

            r_preimage = models.lattice_to_physical([x_preimage, y_preimage, sublattice_preimage], geometry)
            r_image = r_preimage

            x_image, y_image, sublattice_image = models.physical_to_lattice(r_image, geometry)

            x_image = int(np.rint(x_image)); y_image = int(np.rint(y_image))
            x_image = (x_image % config.Ls); y_image = (y_image % config.Ls)

            index = models.to_linearized_index(x_image, y_image, sublattice_image, 1 - orbit_preimage, \
                                               config.Ls, config.n_orbitals, config.n_sublattices)
            mapping[preindex, index] = 1.
        assert np.sum(np.abs(mapping.dot(mapping) - np.eye(mapping.shape[0]))) < 1e-12  # T^2 = I
        return mapping


    R = get_C3z_symmetry_map_chiral_jastrow(config)
    M = get_C2y_symmetry_map_chiral_jastrow(config)
    TRS = get_TRS_symmetry_map_chiral_jastrow(config)

    jastrow_list = []
    for x_shift in range(config.Ls):
        for y_shift in range(config.Ls):
            mask = get_spatial_mask(config.Ls, x_shift, y_shift)
            for subl_valley_i in range(config.n_orbitals * config.n_sublattices):
                for subl_valley_j in range(config.n_orbitals * config.n_sublattices):
                    jastrow_inner = np.zeros((config.n_orbitals * config.n_sublattices, config.n_orbitals * config.n_sublattices))
                    jastrow_inner[subl_valley_i, subl_valley_j] = 1.
                    # jastrow_inner[subl_valley_j, subl_valley_i] = 1.
                    
                    J = np.kron(mask, jastrow_inner)
                    J_symm = J + R.T.dot(J).dot(R) + R.T.dot(R.T).dot(J).dot(R).dot(R)
                    J_symm = J_symm + M.T.dot(J_symm).dot(M)
                    J_symm = J_symm + J_symm.T
                    # J_symm = J_symm + TRS.T.dot(J_symm).dot(TRS)
                    J_symm = (J_symm > 0).astype(np.float64)
                    jastrow_list.append([J_symm, 'jastrow-{:d}-{:d}-{:d}-{:d}'.format(x_shift, y_shift, subl_valley_i, subl_valley_j)])

    # select unique
    jastrow_list_unique = []
    for J in jastrow_list:
        unique = True
        for J_found in jastrow_list_unique:
            if np.allclose(J_found[0], J[0]):
                unique = False
                # print(J_found[1], J[1])
                break
        if unique:
            jastrow_list_unique.append(J)
    jastrow = np.sum(np.array([j[0] for j in jastrow_list_unique]) , axis = 0)
    return jastrow_list_unique[:-3]  # cut redundant Jastrows

def get_jastrow_Koshino_simple_TRS(config):
    jastrow_list = []

    for site in range(len(config.adjacency_list) // 3):
        r = np.sqrt(config.adjacency_list[3 * site][-1])
        jastrow_list.append([np.array([adj[0] for adj in config.adjacency_list[3 * site:3 * site + 1]]).sum(axis = 0) + \
                             np.array([adj[0] for adj in config.adjacency_list[3 * site + 2:3 * site + 3]]).sum(axis = 0), 'J_intra_({:.2f})'.format(r)])
        jastrow_list.append([np.array([adj[0] for adj in config.adjacency_list[3 * site + 1:3 * site + 2]]).sum(axis = 0), 'J_inter_({:.2f})'.format(r)])
        print(site, 'intra', np.array([adj[0] for adj in config.adjacency_list[3 * site:3 * site + 1]]).sum() + \
                             np.array([adj[0] for adj in config.adjacency_list[3 * site + 2:3 * site + 3]]).sum())
        print(site, 'inter', np.array([adj[0] for adj in config.adjacency_list[3 * site + 1:3 * site + 2]]).sum())

    return jastrow_list[:-1]  # only NN and nearest-neighbors  # cut redundant Jastrows


jastrow_on_site_1orb = None
jastrow_long_range_1orb = None
jastrow_long_range_2orb_degenerate = None
jastrow_long_range_2orb_nondegenerate = None
jastrow_Koshino = None
jastrow_Koshino_Gutzwiller = None
jastrow_Koshino_simple = None

def obtain_all_jastrows(config):
    global jastrow_on_site_1orb, jastrow_long_range_1orb, \
           jastrow_long_range_2orb_degenerate, jastrow_long_range_2orb_nondegenerate, jastrow_Koshino, jastrow_Koshino_Gutzwiller, jastrow_Koshino_simple

    if config.n_orbitals == 1:
        jastrow_on_site_1orb = get_jastrow(config, orb_degenerate = False, max_distance = 0.1)
        jastrow_long_range_1orb = get_jastrow(config, orb_degenerate = False)
        return
    if config.n_orbitals == 2:
        # jastrow_long_range_2orb_degenerate = get_jastrow(config, orb_degenerate = True)
        # jastrow_long_range_2orb_nondegenerate = get_jastrow(config, orb_degenerate = False)
        # jastrow_Koshino = get_jastrow_Koshino(config)
        # jastrow_Koshino_Gutzwiller = [jastrow_Koshino[0], jastrow_Koshino[1], jastrow_Koshino[4]]  # only on-site orders
        jastrow_Koshino_simple = get_jastrow_Koshino_simple_TRS(config)
        return
    raise NotImplementedError()
