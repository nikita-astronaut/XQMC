import wavefunction_vmc
import pairings
import numpy as np
from numba import jit
import models

@jit(nopython=True)
def gap_gap_correlator(state, pairing):
    '''
        <\\Delta^{\\dag} \\Delta> = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
                                  = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} d_{j + L} d^{\\dag}_{i} d_k d^{\\dag}_{l + L} = 
                                  = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} F(i, j, k, l)
    '''

    correlator = 0.0
    for i in range(pairing.shape[0]):
        for j in np.where(pairing[i, :] != 0.0)[0]:
            for k in range(pairing.shape[0]):
                for l in np.where(pairing[k, :] != 0.0)[0]:
                    correlator += np.real(wavefunction_vmc.get_wf_ratio_double_exchange(*state, i, j, k, l) * \
                                          np.conj(pairing[i, j]) * pairing[k, l])

    # normalize pairing for the number of bonds coming from every site
    correlator /= np.sum(np.abs(pairing[0, :]))
    return correlator


@jit(nopython=True)
def n_up_n_down_correlator(state, adj):
    '''
        <n_up(i) n_down(j)> = \\sum\\limits_{ij} A[i, j] c^{\\dag}_{i, up} c_{i, up} c^{\\dag}_{j, down} c_{j, down} = 
                            = \\sum\\limits_{ij} A[i, j] d^{\\dag}_{i} d_{i} d_{j + L} d^{\\dag}_{j + L} = 
                            = \\sum\\limits_{ij} A[i, j] d_{j + L} d^{\\dag}_{i} d_{i} d^{\\dag}_{j + L} = 
                            = \\sum\\limits_{ij} A[i, j] F(i, j, i, j)
    '''

    L = len(state[3]) // 2
    correlator = 0.0
    for i in range(L):
        for j in np.where(adj[i, :] > 0)[0]:
            correlator += np.real(wavefunction_vmc.get_wf_ratio_double_exchange(*state, i, j, i, j))

    # normalize to the total number of accounted links
    correlator /= np.sum(np.abs(adj))
    return correlator

@jit(nopython=True)
def Sz_Sz_correlator(state, adj):
    '''
        <n_up(i) n_down(j)> = \\sum\\limits_{ij} A[i, j] c^{\\dag}_{i, up} c_{i, up} c^{\\dag}_{j, down} c_{j, down} = 
                            = \\sum\\limits_{ij} A[i, j] d^{\\dag}_{i} d_{i} d_{j + L} d_{j + L} =
                            = \\sum\\limits_{ij} A[i, j] n_i (1 - n_{j + L}) [after particle-hole transform] 
    '''

    L = len(state[3]) // 2
    correlator = 0.0
    for i in range(L):
        for j in np.where(adj[i, :] > 0)[0]:
            correlator += np.real(wavefunction_vmc.get_wf_ratio(*state, i, i) * (1. - wavefunction_vmc.get_wf_ratio(*state, j + L, j + L)))

    # normalize to the total number of accounted links
    correlator /= np.sum(np.abs(adj))
    return correlator

def compute_observables(wf):
    state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
    observables = []
    names = []

    adj_list = models.get_adjacency_list(wf.config, len(wf.config.initial_jastrow_parameters))  # by default we look only at the correlations within the Jastrow factor

    for i, adj in enumerate(adj_list):
        observables.append(n_up_n_down_correlator(state, adj))
        names.append('⟨n_↑(i) n_↓(j)⟩_' + str(i))

    for name, pairing_unwrapped in zip(wf.config.pairings_list_names, wf.pairings_list_unwrapped):
        observables.append(gap_gap_correlator(state, pairing_unwrapped) / (wf.config.total_dof // 2))
        names.append('Δ^† Δ_' + name)
    return observables, names
