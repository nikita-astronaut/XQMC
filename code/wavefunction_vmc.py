import numpy as np
import time
from config_generator import simulation_parameters as config
import models_vmc

xp = np
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
except ImportError:
    pass

# TODO: make this parallel (multiprocessing / GPU-threads)
class wavefunction(Object):
    def __init__(self, config):
        self.config = config
        self.var_params = self._init_var_params()
        self.U_matrix = self._construct_U_matrix()

    def get_O(self, base_state):
        '''
            O_i = \\partial{\\psi_i} / \\psi_i
        '''
    def _construct_U_matrix(self):
        K = self.config.model(self.config.L, self.config.mu)
        Delta = self.config.pairing(self.config.L)  # the magnitude of the pairing is not yet included here