import numpy as np

xp = np  # by default the code is executed on the CPU
gpu = False
try:
    import cupy as cp
    xp = cp  # if the cp is imported, the code MAY run on GPU if the one is available
    gpu = True
except ImportError:
    pass

import auxiliary_field

def density_spin(h_configuration, K, spin, config):
	G_function = auxiliary_field.get_green_function(h_configuration, K, spin, config)
	return xp.trace(G_function) / (4 * config.Ls ** 2)

def total_density(h_configuration, K, config):
	return (density_spin(h_configuration, K, 1.0, config) + density_spin(h_configuration, K, -1.0, config)) / 2.
    