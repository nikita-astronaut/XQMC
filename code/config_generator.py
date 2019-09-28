from models import H_TB_simple

dt_in_inv_t1 = 1. / 6.
U_in_t1 = 4.
nu = np.arccosh(np.exp(U_in_t1 * dt_in_inv_t1 / 2.))

simulation_parameters = dict(
    Ls = 16,  # spatial size, the lattice will be of size Ls x Ls
    Nt = 26,  # the number of time slices for the Suzuki-Trotter procedure
    main_hopping = 0.331,  # (meV) main hopping is the same for all models, we need it to put down U and dt in the units of t1 (common)
    U_in_t1 = U_in_t1,  # the force of on-site Coulomb repulsion in the units of t1
    dt_in_inv_t1 = dt_in_inv_t1,  # the imaginary time step size in the Suzuki-Trotter procedure, dt x Nt = \beta (inverse T),
    nu = nu,
    mu = 0.5,  # (meV), chemical potential of the lattice
    model = H_TB_simple,
    start_parameters = 'hot'  # 'hot' -- initialize spins randomly | 'cold' -- initialize spins all unity | 'path' -- from saved file
)