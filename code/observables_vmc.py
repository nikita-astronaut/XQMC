import wavefunction_vmc
import pairings
import numpy as np
from numba import jit


@jit(nopython=True)
def gap_gap_correlator(state, pairing):
    '''
        <\\Delta^{\\dag} \\Delta> = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
                                  = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} d_{j + L} d^{\\dag}_{i} d_k d^{\\dag}_{l + L}.
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


def compute_observables(wf):
    state = (wf.Jastrow, wf.W_GF, wf.place_in_string, wf.state, wf.occupancy)
    observables = []
    names = []
    for name, pairing_unwrapped in zip(wf.config.pairings_list_names, wf.pairings_list_unwrapped):
        observables.append(gap_gap_correlator(state, pairing_unwrapped) / (wf.config.total_dof // 2))
        names.append('Δ^† Δ_' + name)
    return observables, names
