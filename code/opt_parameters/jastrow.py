import models
import numpy as np
from copy import deepcopy

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
        jastrow_list.append([adj[0], 'jastrow-{:d}-{:d}-{:.2f}'.format(*adj[1:])])

    return jastrow_list

jastrow_on_site_1orb = None
jastrow_long_range_1orb = None
jastrow_long_range_2orb_degenerate = None
jastrow_long_range_2orb_nondegenerate = None

def obtain_all_jastrows(config):
    global jastrow_on_site_1orb, jastrow_long_range_1orb, \
           jastrow_long_range_2orb_degenerate, jastrow_long_range_2orb_nondegenerate

    if config.n_orbitals == 1:
        jastrow_on_site_1orb = get_jastrow(config, orb_degenerate = False, max_distance = 0.1)
        jastrow_long_range_1orb = get_jastrow(config, orb_degenerate = False)
        return
    if config.n_orbitals == 2:
        jastrow_long_range_2orb_degenerate = get_jastrow(config, orb_degenerate = True)
        jastrow_long_range_2orb_nondegenerate = get_jastrow(config, orb_degenerate = False)
        return
    raise NotImplementedError()