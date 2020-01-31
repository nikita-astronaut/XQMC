import models
import numpy as np
import auxiliary_field
import pairings

dt_in_inv_t1 = 0.1
U_in_t1 = np.array([2, 2, 2, 2, 2, 2])
V_in_t1 = np.array([2, 2, 2, 2, 2, 2])
main_hopping = 1.0

class simulation_parameters:
    def __init__(self):
        self.Ls = 6  # spatial size, the lattice will be of size Ls x Ls
        self.Nt = 40  # the number of time slices for the Suzuki-Trotter procedure
        self.main_hopping = main_hopping  # (meV) main hopping is the same for all models, we need it to put down U and dt in the units of t1 (common)
        self.U = U_in_t1 * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.V = V_in_t1 * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.dt = dt_in_inv_t1 / main_hopping  # the imaginary time step size in the Suzuki-Trotter procedure, dt x Nt = \beta (inverse T),
        self.nu_V = None
        self.nu_U = None
        self.BC_twist = False; self.twist = (1.0, 1.0)
        self.mu = np.array([0.05, 0.1, 0.15, 0.20, 0.25]) # (meV), chemical potential of the lattice
        self.model = models.model_hex_2orb_Koshino
        self.n_orbitals = 2
        self.field = auxiliary_field.auxiliary_field_interorbital
        self.n_sublattices = 2
        self.start_type = 'hot'  # 'hot' -- initialize spins randomly | 'cold' -- initialize spins all unity | 'path' -- from saved file
        self.n_sweeps = 400  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)
        self.n_save_frequency = 10  # every n-th configuration will be stored during generation
        self.save_path = './configurations/'  # where the configurations will be stored | they will have the name save_path/conf_genN.npy, where N is the generated number
        self.n_print_frequency = 3  # write to log every n_print_frequency spin flips
        self.n_smoothing = 60000 # the number of configurations used for smoothing during the generation log output
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.s_refresh = 5
        self.workdir = '/home/astronaut/Documents/DQMC_TBG/logs_dqmc/1/'
        self.thermalization = 0  # after how many sweeps start computing observables

        pairings.obtain_all_pairings(self)
        self.pairings_list = pairings.on_site_2orb_hex_real + pairings.NN_2orb_hex_real

        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        

        self.n_adj_density = self.n_orbitals * (self.n_orbitals + 1) // 2 * 2
        self.n_adj_pairings = self.n_orbitals * (self.n_orbitals + 1) // 2
