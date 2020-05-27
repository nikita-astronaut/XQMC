import models
import numpy as np
import auxiliary_field
from opt_parameters import pairings, waves
import pickle

dt_in_inv_t1 = 1. / 20
U_in_t1 = np.array([2.20])
V_in_t1 = np.array([2.20])
main_hopping = 1.0

class simulation_parameters:
    def __init__(self):
        self.gpu = False
        self.Ls = 8  # spatial size, the lattice will be of size Ls x Ls
        self.Nt = np.array([40])
        self.BC_twist = False; self.twist = (1.0, 1.0)
        self.model = models.model_hex_2orb_Koshino
        self.n_orbitals = 2; self.n_sublattices = 2
        self.field = auxiliary_field.AuxiliaryFieldInterorbitalAccurate
        self.n_copy = 0

        self.main_hopping = main_hopping  # (meV) main hopping is the same for all models, we need it to put down U and dt in the units of t1 (common)
        self.U = U_in_t1 * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.V = V_in_t1 * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.dt = dt_in_inv_t1 / main_hopping  # the imaginary time step size in the Suzuki-Trotter procedure, dt x Nt = \beta (inverse T),
        self.nu_V = None
        self.nu_U = None
        self.mu = np.array([-0.08])
        self.offset = 0


        self.field = auxiliary_field.AuxiliaryFieldInterorbitalAccurate
        self.start_type = 'presaved'  # 'hot' -- initialize spins randomly | 'cold' -- initialize spins all unity | 'path' -- from saved file
        self.n_sweeps = 50000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)
        self.n_save_frequency = 200  # every n-th configuration will be stored during generation
        self.save_path = './configurations/'  # where the configurations will be stored | they will have the name save_path/conf_genN.npy, where N is the generated number
        self.n_print_frequency = 600  # write to log every n_print_frequency spin flips
        self.n_smoothing = 60000 # the number of configurations used for smoothing during the generation log output
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        
        self.s_refresh = 5
        self.workdir = '/galileo/home/userexternal/nastrakh/XQMC/logs_dqmc/-0-08-3-1-20-8x8/'
        self.workdir_heavy = '/gpfs/scratch/userexternal/nastrakh/logs_dqmc/-0-08-3-1-20-8x8/'
        self.thermalization = 1000  # after how many sweeps start computing observables
        
        self.tests = False
        self.adj_list = models.get_adjacency_list(self)[0]
        self.n_copy = 1

        ### pairings parameters setting (only for measurements) ###
        pairings.obtain_all_pairings(self)
        self.pairings_list = pairings.twoorb_hex_all
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.max_square_pairing_distance = 1.  # on-site + NN case on hex lattice
        self.name_group_dict = {'(S_0)x(S_0&S_x)x(δ)': 3, '(S_z)x(iS_y&S_y)x(δ)': 2, '(S_x)x(S_0&S_x)x(v_1)': 3, '(iS_y)x(iS_y&S_y)x(v_1)': 3, '[(S_x)x(S_1)x(v_2)+(S_x)x(S_2)x(v_3)]': 3, '(S_0)x(S_0&S_x)x(u_1)': 3, '[(S_0)x(S_1)x(u_2)+(S_0)x(S_2)x(u_3)]': 3, '[(S_z)x(S_1)x(u_2)-(S_z)x(S_2)x(u_3)]': 3, '[(iS_y)x(S_1)x(v_2)-(iS_y)x(S_2)x(v_3)]': 2, '(S_z)x(iS_y&S_y)x(u_1)': 2, '(S_z)x(S_0&S_x)x(δ)': 1, '(S_0)x(iS_y&S_y)x(δ)': 0, '[(S_x)x(S_1)x(v_2)-(S_x)x(S_2)x(v_3)]': 1, '[(S_0)x(S_1)x(u_2)-(S_0)x(S_2)x(u_3)]': 1, '(S_z)x(S_0&S_x)x(u_1)': 1, '[(S_z)x(S_1)x(u_2)+(S_z)x(S_2)x(u_3)]': 1, '(S_x)x(iS_y&S_y)x(v_1)': 0, '(iS_y)x(S_0&S_x)x(v_1)': 0, '[(iS_y)x(S_1)x(v_2)+(iS_y)x(S_2)x(v_3)]': 0, '(S_0)x(iS_y&S_y)x(u_1)': 0, '(S_0)x(S_1)x(δ)': 11, '(S_0)x(S_2)x(δ)': 7, '(S_z)x(S_1)x(δ)': 9, '(S_z)x(S_2)x(δ)': 5, '(S_x)x(S_1)x(v_1)': 11, '(S_x)x(S_2)x(v_1)': 7, '(S_x)x(S_0&S_x)x(v_2)': 7, '(S_x)x(S_0&S_x)x(v_3)': 11, '(iS_y)x(iS_y&S_y)x(v_2)': 7, '(iS_y)x(iS_y&S_y)x(v_3)': 11, '(S_x)x(S_1)x(v_3)': 7, '(S_x)x(S_2)x(v_2)': 11, '(S_0)x(S_1)x(u_1)': 11, '(S_0)x(S_2)x(u_1)': 7, '(S_0)x(S_0&S_x)x(u_2)': 7, '(S_0)x(S_0&S_x)x(u_3)': 11, '(iS_y)x(S_1)x(v_1)': 8, '(iS_y)x(S_2)x(v_1)': 4, '(iS_y)x(S_0&S_x)x(v_2)': 4, '(iS_y)x(S_0&S_x)x(v_3)': 8, '(S_0)x(S_1)x(u_3)': 7, '(S_0)x(S_2)x(u_2)': 11, '(S_z)x(iS_y&S_y)x(u_2)': 6, '(S_z)x(iS_y&S_y)x(u_3)': 10, '(S_x)x(iS_y)x(v_2)': 4, '(S_x)x(iS_y)x(v_3)': 8, '(iS_y)x(S_1)x(v_3)': 4, '(iS_y)x(S_2)x(v_2)': 8, '(S_z)x(S_1)x(u_1)': 9, '(S_z)x(S_2)x(u_1)': 5, '(S_z)x(S_0&S_x)x(u_2)': 5, '(S_z)x(S_0&S_x)x(u_3)': 9, '(S_0)x(iS_y)x(u_2)': 4, '(S_0)x(iS_y)x(u_3)': 8, '(S_z)x(S_1)x(u_3)': 5, '(S_z)x(S_2)x(u_2)': 9}

        ### SDW/CDW parameters setting ###
        waves.obtain_all_waves(self)
        self.waves_list = waves.hex_2orb
        self.waves_list_names = [w[-1] for w in self.waves_list]
        self.max_square_order_distance = 0.  # on-site only


        self.n_adj_density = self.n_orbitals * (self.n_orbitals + 1) // 2 * 2
        self.n_adj_pairings = self.n_orbitals * (self.n_orbitals + 1) // 2
