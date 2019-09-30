from models import H_TB_simple
import numpy as np
dt_in_inv_t1 = 1. / 6.
U_in_t1 = 4.
nu = np.arccosh(np.exp(U_in_t1 * dt_in_inv_t1 / 2.))
main_hopping = 0.331

class simulation_parameters:
    def __init__(self):
        self.Ls = 8  # spatial size, the lattice will be of size Ls x Ls
        self.Nt = 10  # the number of time slices for the Suzuki-Trotter procedure
        self.main_hopping = main_hopping  # (meV) main hopping is the same for all models, we need it to put down U and dt in the units of t1 (common)
        self.U = U_in_t1 * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.dt = dt_in_inv_t1 / main_hopping  # the imaginary time step size in the Suzuki-Trotter procedure, dt x Nt = \beta (inverse T),
        self.nu = nu
        self.mu = 0.5  # (meV), chemical potential of the lattice
        self.model = H_TB_simple
        self.start_type = 'hot'  # 'hot' -- initialize spins randomly | 'cold' -- initialize spins all unity | 'path' -- from saved file
        self.n_generator = 100000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)
        self.n_save_frequency = 1000  # every n-th configuration will be stored during generation
        self.save_path = './configurations/'  # where the configurations will be stored | they will have the name save_path/conf_genN.npy, where N is the generated number
        self.n_print_frequency = 1  # write to log every n_print_frequency spin flips
        self.n_smoothing = 10000 # the number of configurations used for smoothing during the generation log output