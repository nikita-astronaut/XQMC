import wavefunction_vmc
import pairings
import numpy as np


def gap_gap_correlator(wf, pairing):
	'''
		<\\Delta^{\\dag} \\Delta> = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
		                          = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} d_{j + L} d^{\\dag}_{i} d_k d^{\\dag}_{l + L}.
	'''

	correlator = 0.0 + 0.0j
	for i in range(pairing.shape[0]):
		for j in np.where(pairing[i, :] != 0.0)[0]:
			for k in range(pairing.shape[0]):
				for l in np.where(pairing[k, :] != 0.0)[0]:
					correlator += wf.get_wf_ratio_double_exchange(i, j, k, l) * np.conj(pairing[i, j]) * pairing[k, l]

	correlator /= np.sum(np.abs(pairing[0, :])) * (wf.config.total_dof // 2)
	return correlator


def compute_observables(wf):
	for name, pairing_unwrapped in zip(wf.config.pairings_list_names, wf.pairings_list_unwrapped):
		correlator = gap_gap_correlator(wf, pairing_unwrapped)
		print('<\\Delta \\Delta>_' + name, correlator)
	return
