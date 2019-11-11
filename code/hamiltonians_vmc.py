import numpy as np
import models

class HubbardHamiltonian(object):
	def __init__(self, config):
		self.config = config
		self.edges_quadratic, self.edges_quadric = self._get_edges()

	def _get_edges(self):
		raise NotImplementedError()

	def _get_matrix_elements(self, base_state):  # base_state is a sequence of 0 and 1 on the electron positions
		'''	
			This function only returns the states connected with the base_state via hoppings
			The other energy contributions are diagonal and accounted separately
		'''
		states, matrix_elements = [], []
		for edge in self.edges_quadratic:  # TODO: this can be paralellized with joblib
			i, j, Hij = edge
			if base_state[j] == 1 and base_state[i] == 0:
				new_state = deepcopy(base_state)
				new_state[j] = 0
				new_state[i] = 1
				states.append((new_state, Hij))
				matrix_elements.append(Hij)
		return states, matrix_elements

	def get_energy_basestate(self, machine, base_state):
		states, matrix_elements = self._get_matrix_elements(base_state)
		base_wf = machine.get_wf([base_state])
		adj_wfs = machine.get_wf(states)

		energy = 0.0
		energy += self.config.mu * np.sum(base_state) * np.abs(base_wf) ** 2

		for edge in self.edges_quadric:
			energy += edge[2] * np.abs(base_wf) ** 2 * base_state[edge[0]] * base_state[edge[1]]
		for adj_wf, Hij in zip(adj_wfs, matrix_elements):
			energy += Hij * base_wf * np.conj(adj_wf)
		return energy

	def get_energy_derivative(self, machine, base_state):
		


class hamiltonian_4bands(HubbardHamiltonian)
	def __init__(self, config):
		self.super().__init__(config)

	def _get_edges(self):
		edges_quadratic = []  # for t_{ij} c^{\dag}_i c_j interactions
		K_matrix = models.H_TB_simple(config.Ls, config.mu)
		for i in range(K_matrix.shape[0]):
			for j in range(K_matrix.shape[1]):
				if K_matrix[i, j] != 0.0 and i != j:  # only for hoppings, \mu is accounted separately
					edges_quadratic.append((i, j, K_matrix[i, j]))

		edges_quadric = []  # for V_{ij} n_i n_j density--density interactions
		for i in range(K_matrix.shape[0]):  # only on--site for now
			edges_quadric.append((i, i, self.config.U))
		return edges_quadratic, edges_quadric
