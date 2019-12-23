import wavefunction
import pairings

def gap_gap_correlator(wf, pairing):
	'''
		<\\Delta^{\\dag} \\Delta> = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} c^{\\dag}_{j, down} c^{\\dag}_{i, up} c_{k, up} c_{l, down} = 
		                          = \\sum\\limits_{ijkl} \\Delta_{ij}^* \\Delta_{kl} d_{j + L} d^{\\dag}_{i} d_k d^{\\dag}_{l + L}.
	'''

	correlator = 0.0 + 0.0j
	for i in range(pairing.shape[0]):
		for j in np.where(pairing[i, :] != 0.0):
			for k in range(pairing.shape[0]):
				for l in np.where(pairing[k, :] != 0.0):
					correlator += wf.get_wf_ratio_double_exchange(i, j, k, l) * np.conj(pairing[i, j]) * pairing[k, l]

	correlator /= np.sum(np.abs(pairing[0, :]))
	return correlator