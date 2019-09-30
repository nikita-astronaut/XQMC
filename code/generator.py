import numpy as np
import cupy as cp

import scipy.linalg
import scipy.sparse as scs
import time
from . import auxiliary_field

from config_generator import simulation_parameters as config

current_field = auxiliary_field.get_initial_field_configuration(config)

