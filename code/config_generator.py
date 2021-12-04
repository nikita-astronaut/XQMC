import models
import numpy as np
import auxiliary_field
from opt_parameters import pairings, waves
import pickle
import sys

dt_in_inv_t1 = 1. / 10
main_hopping = 1.0

class simulation_parameters:
    def __init__(self, Nt, U, mu, rank):
        self.gpu = False
        
        self.Ls = 6
        # spatial size, the lattice will be of size Ls x Ls
        self.Nt = np.array([Nt])
        self.BC_twist = False; self.twist = (1.0, 1.0)
        self.model = models.model_hex_2orb_Koshino
        self.n_orbitals = 2; self.n_sublattices = 2
        self.field = auxiliary_field.AuxiliaryFieldInterorbitalAccurateCluster
        self.chiral_basis = True
        self.n_spins = 2
        self.n_copy = 0
        self.BC = 'PBC'

        self.main_hopping = main_hopping  # (meV) main hopping is the same for all models, we need it to put down U and dt in the units of t1 (common)
        self.alpha = float(sys.argv[5])
        self.tlong = float(sys.argv[6])
        self.U = np.array([U]) * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.V = np.array([0.]) * main_hopping  # the force of on-site Coulomb repulsion in the units of t1
        self.dt = dt_in_inv_t1 / main_hopping  # the imaginary time step size in the Suzuki-Trotter procedure, dt x Nt = \beta (inverse T),
        self.nu_V = None
        self.nu_U = None
        self.BC_twist = False; self.twist = (1.0, 1.0)
        self.mu = np.array([mu])
        self.offset = 0
        self.total_dof = self.Ls ** 2 * self.n_sublattices * self.n_orbitals
        self.far_indices = models._jit_get_far_indices(self.Ls, self.total_dof, self.n_sublattices, self.n_orbitals)

        self.start_type = 'presaved'  # 'hot' -- initialize spins randomly | 'cold' -- initialize spins all unity | 'path' -- from saved file
        self.obs_start_type = 'presaved'
        self.n_sweeps = 500000  # the number of spin flips starting from the initial configuration (can be used both for thermalization and generation)
        self.n_save_frequency = 2  # every n-th configuration will be stored during generation
        self.save_path = './configurations/'  # where the configurations will be stored | they will have the name save_path/conf_genN.npy, where N is the generated number
        self.n_print_frequency = 100  # write to log every n_print_frequency spin flips
        self.n_smoothing = 60000 # the number of configurations used for smoothing during the generation log output
        self.total_dof = self.Ls ** 2 * 2 * self.n_sublattices * self.n_orbitals
        self.s_refresh = 20

        self.workdir = '/scratch/e1000/nastrakh/6x6_{:.3f}_{:.3f}_{:.3f}_{:.3f}/logs_dqmc_real_{:d}_{:d}/'.format(self.U[0], self.alpha, self.tlong, mu, self.Nt[0], rank)
        self.workdir_heavy = '/scratch/e1000/nastrakh/6x6_{:.3f}_{:.3f}_{:.3f}_{:.3f}/logs_dqmc_real_{:d}_{:d}/'.format(self.U[0], self.alpha, self.tlong, mu, self.Nt[0], rank)
        self.thermalization = 2000  # after how many sweeps start computing observables

        self.tests = False; self.test_gaps = False;
        self.adj_list = models.get_adjacency_list(self)[0]
        self.n_copy = 1

        ### pairings parameters setting (only for measurements) ###
        #pairings.obtain_all_pairings(self)
        self.pairings_list = []#pairings.twoorb_hex_all_dqmc
        self.pairings_list_names = [p[-1] for p in self.pairings_list]
        self.pairings_list_unwrapped = [pairings.combine_product_terms(self, gap) for gap in self.pairings_list]
        self.max_square_pairing_distance = 1. / 3.  # on-site + NN case on hex lattice
        self.name_group_dict = {'(S_0)x(S_0&S_x)x(δ)': 3, '(S_z)x(iS_y&S_y)x(δ)': 2, '(S_x)x(S_0&S_x)x(v_1)': 3, '(iS_y)x(iS_y&S_y)x(v_1)': 3, '(S_z)x(S_0&S_x)x(δ)': 1, '(S_0)x(iS_y&S_y)x(δ)': 0, '(S_x)x(iS_y&S_y)x(v_1)': 0, '(iS_y)x(S_0&S_x)x(v_1)': 0, '(S_0)x(S_1)x(δ)': 7, '(S_0)x(S_2)x(δ)': 11, '(S_z)x(S_1)x(δ)': 5, '(S_z)x(S_2)x(δ)': 9, '(S_x)x(S_1)x(v_2)': 11, '(S_x)x(S_2)x(v_3)': 7, '(S_x)x(S_1)x(v_1)': 7, '(S_x)x(S_2)x(v_1)': 11, '(S_x)x(S_0&S_x)x(v_2)': 7, '(S_x)x(S_0&S_x)x(v_3)': 11, '(iS_y)x(iS_y&S_y)x(v_2)': 7, '(iS_y)x(iS_y&S_y)x(v_3)': 11, '(S_x)x(S_1)x(v_3)': 3, '(S_x)x(S_2)x(v_2)': 3, '(S_x)x(iS_y)x(v_2)': 4, '(S_x)x(iS_y)x(v_3)': 8, '(iS_y)x(S_1)x(v_3)': 0, '(iS_y)x(S_2)x(v_2)': 0, '(iS_y)x(S_0&S_x)x(v_2)': 4, '(iS_y)x(S_0&S_x)x(v_3)': 8, '(iS_y)x(S_1)x(v_1)': 4, '(iS_y)x(S_2)x(v_1)': 8, '(iS_y)x(S_1)x(v_2)': 8, '(iS_y)x(S_2)x(v_3)': 4}

        ### SDW/CDW parameters setting ###
        #waves.obtain_all_waves(self)
        self.waves_list = []#waves.hex_Koshino
        self.waves_list_names = [w[-1] for w in self.waves_list]
        self.max_square_order_distance = 1. / 3.  # on-site only


        self.n_adj_density = self.n_orbitals * (self.n_orbitals + 1) // 2 * 2
        self.n_adj_pairings = self.n_orbitals * (self.n_orbitals + 1) // 2
