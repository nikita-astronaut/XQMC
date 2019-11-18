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
		base_state_up, base_state_down = base_state
		states, matrix_elements = [], []

		for edge in self.edges_quadratic:  # TODO: this can be paralellized with joblib

			i, j, Hij = edge
			if base_state_up[j] == 1 and base_state_up[i] == 0:
				new_state = deepcopy(base_state_up)
				new_state[j] = 0
				new_state[i] = 1
				states.append((new_state, base_state_down))
				matrix_elements.append(Hij)

			if base_state_down[j] == 1 and base_state_down[i] == 0:
				new_state = deepcopy(base_state_down)
				new_state[j] = 0
				new_state[i] = 1
				states.append((base_state_up, new_state))
				matrix_elements.append(Hij)

		return states, matrix_elements

	def __call__(self, machine, base_state):
		'''
			performs the summation 
			E_loc(i) = \\sum_{j ~ i} H_{ij} \\psi_j / \\psi_i,
			where j ~ i are the all indixes having non-zero matrix element with H_{ij}
		'''
		states, matrix_elements = self._get_matrix_elements(base_state)
		base_wf = machine.get_wf([base_state])
		adj_wfs = machine.get_wf(states)

		energy = 0.0
		energy += self.config.mu * np.sum(base_state[0] + base_state[1])

		for edge in self.edges_quadric:  # TODO: this can be parallized
			energy += edge[2] * \
			          (base_state[0][edge[0]] + base_state[1][edge[0]] - 1) * \
			          (base_state[0][edge[1]] + base_state[1][edge[1]] - 1)
		for adj_wf, Hij in zip(adj_wfs, matrix_elements):
			energy += Hij * adj_wf / base_wf

		return energy




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
