import models
import numpy as np
import os
from opt_parameters import pairings
import monte_carlo_vmc
import sys
import config_vmc as cv_module
from numba import jit
from scipy.linalg import schur
@jit(nopython=True)
def get_orb(L, mod):
    orb_ij = []; orb_k = []; n_orb = 0
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64)
    seen_shifts = []
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64)
    for i in range(4 * L ** 2):
        for j in range(4 * L ** 2):
            li, xi, yi = i % 4, (i // 4) // (L), (i // 4) % (L)
            lj, xj, yj = j % 4, (j // 4) // (L), (j // 4) % (L)
            shift_ij = (li, lj, (xi - xj) % L, (yi - yj) % L, xi % mod, yi % mod)

            if shift_ij in seen_shifts:
                idx = seen_shifts.index(shift_ij)
                orb_k.append(orb_k[idx])
                seen_shifts.append(shift_ij)
                orb_ij.append((i, j))
                matrix[i, j] = orb_k[-1]
            else:
                orb_k.append(n_orb); n_orb += 1
                seen_shifts.append(shift_ij)
                orb_ij.append((i, j))
                matrix[i, j] = orb_k[-1]

    return orb_ij, orb_k, n_orb, matrix, seen_shifts

@jit(nopython=True)
def get_jastrow(L, mod):
    orb_ij = []; orb_k = []; n_orb = 0
    seen_shifts = []
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64)
    for i in range(4 * L ** 2):
        for j in range(4 * L ** 2):
            if i == j:
                continue
            li, xi, yi = i % 4, (i // 4) // (L), (i // 4) % (L)
            lj, xj, yj = j % 4, (j // 4) // (L), (j // 4) % (L)
            shift_ij = (li, lj, (xi - xj) % L, (yi - yj) % L, xi % mod, yi % mod)
            shift_ji = (lj, li, (xj - xi) % L, (yj - yi) % L, xj % mod, yj % mod)

            if shift_ij in seen_shifts:
                idx = seen_shifts.index(shift_ij)
                orb_k.append(orb_k[idx])
                seen_shifts.append(shift_ij)
                matrix[i, j] = orb_k[idx]
            elif shift_ji in seen_shifts:
                idx = seen_shifts.index(shift_ji)
                orb_k.append(orb_k[idx])
                seen_shifts.append(shift_ji)
                matrix[i, j] = orb_k[idx]
            else:
                matrix[i, j] = n_orb
                orb_k.append(n_orb); n_orb += 1
                seen_shifts.append(shift_ij)

            orb_ij.append((i, j))

    
    return orb_ij, orb_k, n_orb, seen_shifts, matrix



@jit(nopython=True)
def get_jastrow_fromshift(L, mod, all_distances, dist_threshold = 1):
    orb_ij = []; orb_k = []; n_orb = 0
    seen_shifts = [(0, 0)]
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64) - 1
    cur_jastrow = 0

    for modx in range(mod):
        for mody in range(mod):
            for li in range(4):
                for lj in range(4):
                    for dx in range(L):
                        for dy in range(L):
                            net_unknown = False
                            for x in range(L // mod):
                                for y in range(L // mod):
                                    i = li + (mod * x + modx) * 4 + (mod * y + mody) * L * 4
                                    j = lj + ((mod * x + modx + dx) % L) * 4 + ((mod * y + mody + dy) % L) * 4 * L

                                    if i == j:
                                        continue

                                    if all_distances[i, j] > dist_threshold + 1e-5:
                                        continue

                                    if matrix[i, j] == -1:
                                        matrix[i, j] = cur_jastrow; matrix[j, i] = cur_jastrow
                                        orb_ij.append((i, j)); orb_ij.append((j, i));
                                        orb_k.append(cur_jastrow); orb_k.append(cur_jastrow)
                                        net_unknown = True

                            if net_unknown:
                                cur_jastrow += 1

    return orb_ij, orb_k, cur_jastrow, seen_shifts, matrix



    for i in range(4 * L ** 2):
        for j in range(4 * L ** 2):
            if i == j:
                continue
            li, xi, yi = i % 4, (i // 4) // (L), (i // 4) % (L)
            lj, xj, yj = j % 4, (j // 4) // (L), (j // 4) % (L)
            shift_ij = (li, lj, (xi - xj) % L, (yi - yj) % L, xi % mod, yi % mod)
            shift_ji = (lj, li, (xj - xi) % L, (yj - yi) % L, xj % mod, yj % mod)

            if shift_ij in seen_shifts:
                idx = seen_shifts.index(shift_ij)
                orb_k.append(orb_k[idx])
                seen_shifts.append(shift_ij)
                matrix[i, j] = orb_k[idx]
            elif shift_ji in seen_shifts:
                idx = seen_shifts.index(shift_ji)
                orb_k.append(orb_k[idx])
                seen_shifts.append(shift_ji)
                matrix[i, j] = orb_k[idx]
            else:
                matrix[i, j] = n_orb
                orb_k.append(n_orb); n_orb += 1
                seen_shifts.append(shift_ij)

            orb_ij.append((i, j))

    
    return orb_ij, orb_k, n_orb, seen_shifts, matrix

def W_ij(U, xi, rhat):  # https://arxiv.org/pdf/1905.01887.pdf
    if rhat == 0:
        return U

    d = xi / rhat
    ns = np.arange(-100000, 100001)
    W = 11.077 * U / rhat * np.sum((-1.) ** ns / (1 + (ns * d) ** 2) ** 0.5)
    res = U_0 / (1. + (U_0 / W) ** 5) ** 0.2
    # print('W', W)
    return res if res > 0.05 else 0.0  # Ohno relations


def generate_Imada_format_Koshino(config, U, mod, doping=0, periodic=True):
    H = config.hamiltonian(config_vmc)
    K0 = config.K_0 # already chiral
    Ls = config.Ls

    Tx, Ty = pairings.Tx_symmetry_map, pairings.Ty_symmetry_map
    C3z = np.argmax(np.abs(pairings.C3z_symmetry_map_chiral), axis = 0)
    C2y = np.argmax(np.abs(pairings.C2y_symmetry_map_chiral), axis = 0)
    tx, ty = [], []

    for i in range(Tx.shape[0]):
        assert len(np.where(Tx[i, :] == 1)[0]) == 1
        assert len(np.where(Ty[i, :] == 1)[0]) == 1

        tx.append(np.where(Tx[i, :] == 1)[0][0])
        ty.append(np.where(Ty[i, :] == 1)[0][0])

    tx, ty = np.array(tx), np.array(ty)
    assert np.allclose(tx[ty], ty[tx])

    tx_valley = tx[::2] // 2; ty_valley = ty[::2] // 2;
    assert np.allclose(tx_valley[ty_valley], ty_valley[tx_valley])

    valley = np.concatenate([np.array([2 * i + 1, 2 * i]) for i in range(config.Ls ** 2 * 2)])


    np.save('./files_mvmc/tx_{:d}.npy'.format(config.Ls), tx)
    np.save('./files_mvmc/ty_{:d}.npy'.format(config.Ls), ty)
    np.save('./files_mvmc/C3z_{:d}.npy'.format(config.Ls), C3z)
    np.save('./files_mvmc/C2y_{:d}.npy'.format(config.Ls), C2y)
    np.save('./files_mvmc/valley_{:d}.npy'.format(config.Ls), valley)

    pref = 'real' if np.allclose(K0, K0.real) else 'imag'

    np.save('./files_mvmc/K0_{:s}_{:d}.npy'.format(pref, config.Ls), K0)
    np.save('./files_mvmc/distances_{:d}.npy'.format(config.Ls), config.all_distances)
    
    #np.save('H_edges_{:d}.npy'.format(config.Ls), H.edges_quadric)
    #exit(-1)


    path = os.path.join(config.workdir, 'imada_format_L_{:d}_Ne_{:d}_U_{:.3f}_mod_{:d}_periodic_{:b}'.format(config.Ls, config.Ne, U, mod, periodic))
    os.makedirs(path, exist_ok=True)



    ########### writing the K--matrix ##########
    f = open(os.path.join(path, 'trans.def'), 'w')

    f.write('========================\n')
    f.write('NTransfer      {:d}\n'.format(2 * np.sum(np.abs(config.K_0) > 1e-7)))
    f.write('========================\n')
    f.write('========i_j_s_tijs======\n')
    f.write('========================\n')
    assert np.allclose(K0, K0.T.conj())

    for i in range(K0.shape[0]):
        for j in range(K0.shape[1]):
            if np.abs(K0[i, j]) > 1e-7:
                f.write('    {:d}     0     {:d}     0   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0[i, j]), np.imag(-K0[i, j])))
                f.write('    {:d}     1     {:d}     1   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0[i, j]), np.imag(-K0[i, j])))
                #K0[i, j] = K0[i, j].real + 100j if i > j else -100j
    f.close()

    for i in range(K0.shape[0]):
        for j in range(K0.shape[1]):
            if (i + j) % 2 == 1 and np.abs(K0[i, j]) > 1e-12:
                print(i, j)
                exit(-1)

    K_0_plus = K0[:, np.arange(0, 4 * Ls ** 2, 2)]; K_0_plus = K_0_plus[np.arange(0, 4 * Ls ** 2, 2), :]
    K_0_minus = K0[:, np.arange(1, 4 * Ls ** 2, 2)]; K_0_minus = K_0_minus[np.arange(1, 4 * Ls ** 2, 2), :]
    
    K_0_plus_x = K_0_plus[tx_valley, :]; K_0_plus_x = K_0_plus_x[:, tx_valley]
    assert np.allclose(K_0_plus_x, K_0_plus)
    K_0_plus_y = K_0_plus[ty_valley, :]; K_0_plus_y = K_0_plus_y[:, ty_valley]
    assert np.allclose(K_0_plus_y, K_0_plus)

    energies_plus, orbitals_plus = np.linalg.eigh(K_0_plus)
    energies_minus, orbitals_minus = np.linalg.eigh(K_0_minus)
    # energies_plus, _ = np.linalg.eigh(K_0_plus.real)
    # energies_minus, _ = np.linalg.eigh(K_0_minus.real)  # FIXME

    ##################### Tx #########################
    

    e_round = np.around(energies_plus, decimals=7)
    orbitals_plus_tx = orbitals_plus * 0.0 + 0.0j

    tx_plus_momenta = np.zeros(orbitals_plus.shape[0], dtype=np.complex128)  # this is later needed to select joint (e, t_x) sectors
    for e_sector in np.unique(e_round):
        idxs = np.where(e_round == e_sector)[0]
        
        tx_matrix = np.zeros((len(idxs), len(idxs)), dtype=np.complex128)
        for i, ii in enumerate(idxs):
            for j, jj in enumerate(idxs):
                tx_matrix[i, j] = np.dot(orbitals_plus[:, ii].conj().T, orbitals_plus[:, jj][tx_valley])

        momenta, Um = schur(tx_matrix)
        momenta = np.diag(momenta)
        orbitals_plus_tx[:, idxs] = orbitals_plus[:, idxs].dot(Um)
        tx_plus_momenta[idxs] = momenta
    orbitals_plus = orbitals_plus_tx
    tx_plus_momenta = np.around(tx_plus_momenta, decimals=5)
    print(tx_plus_momenta)

    e_round = np.around(energies_minus, decimals=7)
    orbitals_minus_tx = orbitals_minus * 0.0 + 0.0j
    tx_minus_momenta = np.zeros(orbitals_minus.shape[0], dtype=np.complex128)
    for e_sector in np.unique(e_round):
        idxs = np.where(e_round == e_sector)[0]

        tx_matrix = np.zeros((len(idxs), len(idxs)), dtype=np.complex128)
        for i, ii in enumerate(idxs):
            for j, jj in enumerate(idxs):
                tx_matrix[i, j] = np.dot(orbitals_minus[:, ii].conj().T, orbitals_minus[:, jj][tx_valley])

        momenta, Um = schur(tx_matrix)
        momenta = np.diag(momenta)
        orbitals_minus_tx[:, idxs] = orbitals_minus[:, idxs].dot(Um)
        tx_minus_momenta[idxs] = momenta
    orbitals_minus = orbitals_minus_tx
    tx_minus_momenta = np.around(tx_minus_momenta, decimals=5)
    print(tx_minus_momenta)

    #################### Ty ######################
    e_round = np.around(energies_plus, decimals=7)
    orbitals_plus_ty = orbitals_plus * 0.0 + 0.0j

    for e_sector in np.unique(e_round):
        for kx_sector in np.unique(tx_plus_momenta):
            idxs = np.where((e_round == e_sector) & (tx_plus_momenta == kx_sector))[0]
            if len(idxs) == 0:
                continue

            ty_matrix = np.zeros((len(idxs), len(idxs)), dtype=np.complex128)
            for i, ii in enumerate(idxs):
                for j, jj in enumerate(idxs):
                    ty_matrix[i, j] = np.dot(orbitals_plus[:, ii].conj().T, orbitals_plus[:, jj][ty_valley])

            momenta, Um = schur(ty_matrix)
            momenta = np.diag(momenta)
            orbitals_plus_ty[:, idxs] = orbitals_plus[:, idxs].dot(Um)
    orbitals_plus = orbitals_plus_ty

    
    e_round = np.around(energies_minus, decimals=7)
    orbitals_minus_ty = orbitals_minus * 0.0 + 0.0j
    for e_sector in np.unique(e_round):
        for kx_sector in np.unique(tx_minus_momenta):
            idxs = np.where((e_round == e_sector) & (tx_minus_momenta == kx_sector))[0]
            if len(idxs) == 0:
                continue

            ty_matrix = np.zeros((len(idxs), len(idxs)), dtype=np.complex128)
            for i, ii in enumerate(idxs):
                for j, jj in enumerate(idxs):
                    ty_matrix[i, j] = np.dot(orbitals_minus[:, ii].conj().T, orbitals_minus[:, jj][ty_valley])

            momenta, Um = schur(ty_matrix)
            momenta = np.diag(momenta)
            orbitals_minus_ty[:, idxs] = orbitals_minus[:, idxs].dot(Um)
    orbitals_minus = orbitals_minus_ty
    



    selected_plus = np.argsort(energies_plus)[:len(energies_plus) // 2 - doping]
    orbitals_plus = orbitals_plus[:, selected_plus]
    
    selected_minus = np.argsort(energies_minus)[:len(energies_minus) // 2 - doping]
    orbitals_minus = orbitals_minus[:, selected_minus]

    assert np.allclose(energies_plus, energies_minus)
    print(energies_plus)
    energies_plus = energies_plus[selected_plus]
    print(energies_plus)
    #exit(-1)
    energies_minus = energies_minus[selected_minus]
    

    orbitals = np.kron(orbitals_minus, np.array([[0, 0], [0, 1]])) + np.kron(orbitals_plus, np.array([[1, 0], [0, 0]]))

    energies = np.concatenate([np.array([energies_plus[i], energies_minus[i]]) for i in range(len(energies_minus))], axis = -1)

    for i in range(len(energies)):
        o = orbitals[:, i]
        assert np.allclose(K0.dot(o), o * energies[i])
        print(i, np.dot(o.conj().T, o[tx]), np.dot(o.conj().T, o[ty]), np.dot(o.conj().T, o[C3z]), np.dot(o.conj().T, o[C2y]), energies[i])

        K0realo = K0.real.dot(o)
        angle = np.dot(o, K0realo) / np.sqrt(np.dot(o, o) * np.dot(K0realo, K0realo))

    overlaps = orbitals.T.conj().dot(orbitals)
    assert np.allclose(overlaps, np.eye(overlaps.shape[0]))
    # print(orbitals)  # PLUS HELPED
    #orbitals_all = np.concatenate([orbitals_minus, orbitals_plus], axis = -1)
    print((np.sum(energies_minus) + np.sum(energies_plus)) * 2., 'expected free energy')
    print(2 * np.sum(np.linalg.eigh(K0)[0][:K0.shape[0] // 2 - 2 * doping]), 2 * np.sum(np.linalg.eigh(K0.real)[0][:K0.shape[0] // 2 - 2 * doping]), 'full energy full/real')
    print(2 * np.sum(energies), \
          2 * np.trace(orbitals.conj().T.dot(K0).dot(orbitals)), \
          2 * np.trace(orbitals.conj().T.dot(K0.real).dot(orbitals)), \
          2 * np.trace(orbitals.real.conj().T.dot(K0.real).dot(orbitals.real)), \
          2 * np.trace(orbitals.real.conj().T.dot(K0).dot(orbitals.real)))

    print(np.linalg.eigh(K0)[0])
    print(np.linalg.eigh(K0.real)[0])

    if periodic:
        f_ij = (orbitals.dot(orbitals.T.conj()))
    else:
        f_ij = (orbitals.dot(orbitals.T))
    #for i in range(config.Ls ** 2 * 4):
    #    for j in range(config.Ls ** 2 * 4):
    #        if i != j:
    #            continue
    #        print('f[{:d}, {:d}] = {:10f} + I {:10f}'.format(i, j, f_ij[i, j].real, f_ij[i, j].imag))
    #print(f_ij)
    if periodic:
        f_ij_tx = f_ij[tx, :]; f_ij_tx = f_ij_tx[:, tx]
        assert np.allclose(f_ij, f_ij_tx)

        f_ij_ty = f_ij[ty, :]; f_ij_ty = f_ij_ty[:, ty]
        assert np.allclose(f_ij, f_ij_ty)
        print('f_ij initial orbitals are translationally invariant')
    else:
        print('f_ij initial orbitals are exact (non-translationally invariant)')


    ## test: can det f_ij be complex?  #

    sites_down = np.array([1, 8, 17, 32])
    sites_up = np.array([6, 9, 12, 29])

    f_selected = f_ij[sites_up, :]; f_selected = f_selected[:, sites_down]
    det_0 = np.linalg.det(f_selected)
    
    def parity(array):
        parity = 0
        for i in range(len(array)):
            for j in range(i + 1, len(array)):
                if array[i] > array[j]:
                    parity += 1
        return (-1) ** parity
                
    total_energy = 0.0 + 0.0j
    for i in range(config.Ls ** 2 * 4):
        for j in range(config.Ls ** 2 * 4):
            if i in sites_down or j not in sites_down:
                continue
            if np.abs(K0[i, j]) < 1e-7:
                continue
            new_sites_down = sites_down.copy()
            j_pos = np.where(new_sites_down == j)[0][0]
            new_sites_down = np.concatenate([new_sites_down[:j_pos], new_sites_down[j_pos + 1:], np.array([i])], axis = -1)

            f_selected = f_ij[sites_up, :]; f_selected = f_selected[:, new_sites_down]
            det_new = np.linalg.det(f_selected)
            print(i, j, det_new / det_0 * (-1) ** j_pos * (-1))
            total_energy += det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j]
            print(det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j], K0[i, j], total_energy)
    print('energy assessed within Slater determinants down', total_energy)
    
    total_energy = 0.0 + 0.0j
    for i in range(config.Ls ** 2 * 4):
        for j in range(config.Ls ** 2 * 4):
            if i in sites_up or j not in sites_up:
                continue
            if np.abs(K0[i, j]) < 1e-7:
                continue
            new_sites_up = sites_up.copy()
            j_pos = np.where(new_sites_up == j)[0][0]
            new_sites_up = np.concatenate([new_sites_up[:j_pos], new_sites_up[j_pos + 1:], np.array([i])], axis = -1)

            f_selected = f_ij[new_sites_up, :]; f_selected = f_selected[:, sites_down]
            det_new = np.linalg.det(f_selected)
            print(i, j, det_new / det_0 * (-1) ** j_pos * (-1))
            total_energy += det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j]
            print(det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j], K0[i, j], total_energy)
    print('energy assessed within Slater determinants up', total_energy)


    #rotation = np.array([0, 1, 14, 15, 8, 9, 6, 7, 12, 13, 2, 3, 4, 5, 10, 11])
    #assert np.allclose(valley[valley], np.arange(16))
    assert np.allclose(C3z[C3z[C3z]], np.arange(config.Ls ** 2 * 4))
    assert np.allclose(C2y[C2y], np.arange(config.Ls ** 2 * 4))


    symmetries = [np.arange(len(tx))]
    if mod == 2:
        symmetries = [np.arange(len(tx)), tx, ty, ty[tx]]

    #symmetries = symmetries + [symm[valley] for symm in symmetries]
    #symmetries = symmetries + [symm[C3z] for symm in symmetries] + [symm[C3z[C3z]] for symm in symmetries]
    #symmetries = symmetries + [symm[C2y] for symm in symmetries]

    ########### writing the translational symmetries ##########
    f = open(os.path.join(path, 'qptransidx.def'), 'w')
    f.write('=============================================\n')
    f.write('NQPTrans          {:d}\n'.format(len(symmetries)))
    f.write('=============================================\n')
    f.write('======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n')
    f.write('=============================================\n')

    print(tx, ty)
    test = np.arange(len(tx))
    for i in range(config.Ls):
        if i > 0:
            assert not np.allclose(test, np.arange(len(tx)))

        test = test[tx]
    assert np.allclose(test, np.arange(len(tx)))

    test = np.arange(len(ty))
    for i in range(config.Ls):
        if i > 0:
            assert not np.allclose(test, np.arange(len(ty)))
        test = test[ty]

    assert np.allclose(test, np.arange(len(ty)))


    for i in range(len(symmetries)):
        f.write('{:d}    1.00000\n'.format(i))

    for i, symm in enumerate(symmetries):
        for i_from in range(symm.shape[0]):
            f.write('    {:d}      {:d}      {:d}\n'.format(i, i_from, symm[i_from]))
    f.close()


    ########### writing the spin locations (none) ##########
    f = open(os.path.join(path, 'locspn.def'), 'w')
    f.write('================================\n')
    f.write('NlocalSpin     0\n')
    f.write('================================\n')
    f.write('========i_0LocSpn_1IteElc ======\n')
    f.write('================================\n')
    for i in range(K0.shape[0]):
        f.write('    {:d}      0\n'.format(i))
    f.close()






    ########### writing the jastrows ##########
    ## we use the mod/mod structure
    f = open(os.path.join(path, 'jastrowidx.def'), 'w')
    jastrow_ij, jastrow_k, n_jastrows, seen_shifts, matrix_jastrows = get_jastrow_fromshift(config.Ls, mod, config.all_distances, dist_threshold=1.)
    print(len(np.unique(jastrow_k)))
    assert np.allclose(matrix_jastrows, matrix_jastrows.T)


    if mod == 2:
        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, tx[tx]]    
        matrix_jastrows_trans = matrix_jastrows_trans[tx[tx], :]
        assert np.allclose(matrix_jastrows_trans, matrix_jastrows)

        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, ty[ty]]    
        matrix_jastrows_trans = matrix_jastrows_trans[ty[ty], :]
        assert np.allclose(matrix_jastrows_trans, matrix_jastrows)

        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, tx]    
        matrix_jastrows_trans = matrix_jastrows_trans[tx, :]
        assert not np.allclose(matrix_jastrows_trans, matrix_jastrows)

        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, ty]    
        matrix_jastrows_trans = matrix_jastrows_trans[ty, :]
        assert not np.allclose(matrix_jastrows_trans, matrix_jastrows)
    elif mod == 1:
        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, tx]    
        matrix_jastrows_trans = matrix_jastrows_trans[tx, :]
        assert np.allclose(matrix_jastrows_trans, matrix_jastrows)

        matrix_jastrows_trans = matrix_jastrows.copy()
        matrix_jastrows_trans = matrix_jastrows_trans[:, ty]    
        matrix_jastrows_trans = matrix_jastrows_trans[ty, :]
        assert np.allclose(matrix_jastrows_trans, matrix_jastrows)


    real_jastrow = False

    f.write('=============================================\n')
    f.write('NJastrowIdx         {:d}\n'.format(n_jastrows + 1))
    f.write('ComplexType          {:d}\n'.format(0 if real_jastrow else 1))
    f.write('=============================================\n')
    f.write('=============================================\n')
    for i in range(config.Ls ** 2 * 4):
        for j in range(config.Ls ** 2 * 4):
            if i == j:
                continue
            if (i, j) not in jastrow_ij:
                f.write('    {:d}      {:d}      {:d}\n'.format(i, j, n_jastrows))
            else:
                f.write('    {:d}      {:d}      {:d}\n'.format(i, j, jastrow_k[jastrow_ij.index((i, j))]))
    for i in range(n_jastrows):
        f.write('    {:d}      1\n'.format(i))
    f.write('    {:d}      0\n'.format(n_jastrows))
    f.close()





    f = open(os.path.join(path, 'InJastrow.def'), 'w')
    f.write('======================\n')
    f.write('NJastrowIdx  {:d}\n'.format(n_jastrows + 1))
    f.write('======================\n')
    f.write('== i_j_JastrowIdx  ===\n')
    f.write('======================\n')
    for i in range(n_jastrows):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, \
                np.random.uniform(0.0, 1.0) * 0, np.random.uniform(0.0, 1.0) * 0 if not real_jastrow else 0.0))
    f.write('{:d} {:.10f}  {:.10f}\n'.format(n_jastrows, 0, 0))
    f.close()














    f = open(os.path.join(path, 'gutzwilleridx.def'), 'w')

    f.write('=============================================\n')
    f.write('NGutzwillerIdx          {:d}\n'.format(4 * mod ** 2))
    f.write('ComplexType          {:d}\n'.format(0 if real_jastrow else 1))
    f.write('=============================================\n')
    f.write('=============================================\n')

    for i in range(K0.shape[0]):
        oi, si, xi, yi = models.from_linearized_index(i, config.Ls, config.n_orbitals, config.n_sublattices)

        modx = xi % mod
        mody = yi % mod

        idx = mod ** 2 * (si * 2 + oi) + mod * modx + mody
        f.write('    {:d}      {:d}\n'.format(i, 0))#idx))
    for i in range(4 * mod ** 2):
        f.write('    {:d}      1\n'.format(i))
    f.close()


    f = open(os.path.join(path, 'InGutzwiller.def'), 'w')
    f.write('======================\n')
    f.write('NGutzwillerIdx  {:d}\n'.format(4 * mod ** 2))
    f.write('======================\n')
    f.write('== i_j_GutzwillerIdx  ===\n')
    f.write('======================\n')
    for i in range(4 * mod ** 2):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, np.random.uniform(0.0, 1.0) * 0, np.random.uniform(0.0, 1.0) * 0 if not real_jastrow else 0.0))
    f.close()


    







    orbit_ij, orbit_k, n_orbits, matrix_orb, seen_shifts = get_orb(config.Ls, mod)


    if mod == 2:
        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, tx[tx]]    
        matrix_orb_trans = matrix_orb_trans[tx[tx], :]
        assert np.allclose(matrix_orb_trans, matrix_orb)

        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, ty[ty]]    
        matrix_orb_trans = matrix_orb_trans[ty[ty], :]
        assert np.allclose(matrix_orb_trans, matrix_orb)

        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, tx]    
        matrix_orb_trans = matrix_orb_trans[tx, :]
        assert not np.allclose(matrix_orb_trans, matrix_orb)

        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, ty]    
        matrix_orb_trans = matrix_orb_trans[ty, :]
        assert not np.allclose(matrix_orb_trans, matrix_orb)
    elif mod == 1:
        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, tx]    
        matrix_orb_trans = matrix_orb_trans[tx, :]
        assert np.allclose(matrix_orb_trans, matrix_orb)

        matrix_orb_trans = matrix_orb.copy()
        matrix_orb_trans = matrix_orb_trans[:, ty]    
        matrix_orb_trans = matrix_orb_trans[ty, :]
        assert np.allclose(matrix_orb_trans, matrix_orb)


    ########### writing the orbitals indexes ##########
    f = open(os.path.join(path, 'orbitalidx.def'), 'w')

    f.write('=============================================\n')
    if not periodic:
        f.write('NOrbitalIdx         {:d}\n'.format((config.Ls ** 2 * 4) ** 2))#n_orbits))
    else:
        f.write('NOrbitalIdx         {:d}\n'.format((n_orbits)))

    f.write('ComplexType          {:d}\n'.format(0 if real_jastrow else 1))
    f.write('=============================================\n')
    f.write('=============================================\n')


    if periodic:
        for ij, k in zip(orbit_ij, orbit_k):
            f.write('    {:d}      {:d}      {:d}\n'.format(ij[0], ij[1], k))
        for i in range(n_orbits):
            f.write('    {:d}      1\n'.format(i))  # FIXME
    else:
        orb_num = 0
        for i in range(config.Ls ** 2 * 4):
            for j in range(config.Ls ** 2 * 4):
                f.write('    {:d}      {:d}      {:d}\n'.format(i, j, orb_num))
                orb_num += 1
        for i in range(orb_num):
            f.write('    {:d}      1\n'.format(i))
    f.close()

    f = open(os.path.join(path, 'InOrbital.def'), 'w')
    f.write('======================\n')
    if not periodic:
        f.write('NOrbitalIdx  {:d}\n'.format(orb_num)) #n_orbits))
    else:
        f.write('NOrbitalIdx  {:d}\n'.format(n_orbits))
    f.write('======================\n')
    f.write('== i_j_OrbitalIdx  ===\n')
    f.write('======================\n')
    if not periodic:
        orb_num = 0
        for i in range(config.Ls ** 2 * 4):
            for j in range(config.Ls ** 2 * 4):
                f.write('{:d} {:.10f}  {:.10f}\n'.format(orb_num, f_ij[i, j].real + 0 * np.random.uniform(3e-2, 3e-2), \
                                                                  f_ij[i, j].imag + 0 * np.random.uniform(-3e-2, 3e-2)))
                orb_num += 1
    else:
        for k in range(n_orbits):
            i, j = orbit_ij[orbit_k.index(k)]
            #f.write('{:d} {:.10f}  {:.10f}\n'.format(i, np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0) if not real_jastrow else 0.0))
            f.write('{:d} {:.10f}  {:.10f}\n'.format(k, f_ij[i, j].real, f_ij[i, j].imag if not real_jastrow else 0.0))






    interaction = H.edges_quadric

    f = open(os.path.join(path, 'coulombintra.def'), 'w')

    f.write('=============================================\n')
    f.write('NCoulombIntra          {:d}\n'.format(config.total_dof // 2))
    f.write('=============================================\n')
    f.write('================== CoulombIntra ================\n')
    f.write('=============================================\n')

    for i in range(K0.shape[0]):
        f.write('    {:d}         {:.14f}\n'.format(i, U))
    f.close()



    f = open(os.path.join(path, 'coulombinter.def'), 'w')
    f.write('=============================================\n')
    f.write('NCoulombInter          {:d}\n'.format(int(np.sum((interaction - np.diag(np.diag(interaction))) != 0.0))))
    f.write('=============================================\n')
    f.write('================== CoulombInter ================\n')
    f.write('=============================================\n')

    for i in range(K0.shape[0]):
        for j in range(K0.shape[1]):
            if interaction[i, j] != 0.0 and i != j:
                f.write('   {:d}     {:d}  {:.4f}\n'.format(i, j, U / 2.))
    f.close()








    ########### writing the namedef ##########
    f = open(os.path.join(path, 'namedef.def'), 'w')
    f.write('     ModPara  modpara.def\n')
    f.write('     LocSpin  locspn.def\n')
    f.write('       Trans  trans.def\n')
    f.write('CoulombIntra  coulombintra.def\n')
    f.write('CoulombInter  coulombinter.def\n')
    # f.write('    OneBodyG  greenone.def\n')
    # f.write('    TwoBodyG  greentwo.def\n')
    f.write('  Gutzwiller  gutzwilleridx.def\n')
    f.write('     Jastrow  jastrowidx.def\n')
    f.write('     Orbital  orbitalidx.def\n')
    f.write('    TransSym  qptransidx.def\n')
    f.write('    InJastrow  InJastrow.def\n')
    f.write('    InOrbital InOrbital.def\n')
    f.write('    InGutzwiller  InGutzwiller.def\n')
    f.close()

    ########### writing the modpara ##########
    f = open(os.path.join(path, 'modpara.def'), 'w')

    f.write('--------------------\n')
    f.write('Model_Parameters   0\n')
    f.write('--------------------\n')
    f.write('VMC_Cal_Parameters\n')
    f.write('--------------------\n')
    f.write('CDataFileHead  zvo\n')
    f.write('CParaFileHead  zqp\n')
    f.write('--------------------\n')
    f.write('NVMCCalMode    0\n')
    f.write('--------------------\n')
    f.write('NDataIdxStart  1\n')
    f.write('NDataQtySmp    1\n')
    f.write('--------------------\n')
    f.write('Nsite          {:d}\n'.format(K0.shape[0]))
    f.write('Ncond          {:d}\n'.format(config.Ls ** 2 * 4 - doping * 4))
    f.write('2Sz            0\n')
    f.write('NSPGaussLeg    8\n')
    f.write('NSPStot        0\n')
    f.write('NMPTrans       {:d}\n'.format(len(symmetries)))
    f.write('NSROptItrStep  1000\n')
    f.write('NSROptItrSmp   40\n')
    f.write('DSROptRedCut   0.0010000000\n')
    f.write('DSROptStaDel   0.0200000000\n')
    f.write('DSROptStepDt   0.0200000000\n')
    f.write('NVMCWarmUp     400\n')
    f.write('NVMCInterval   1\n')
    f.write('NVMCSample     4000\n')
    f.write('NExUpdatePath  0\n')
    f.write('RndSeed        1\n')
    f.write('NSplitSize     1\n')
    f.write('NStore         0\n')
    f.write('NSRCG          1\n')
    f.close()

if __name__ == "__main__":
    config_vmc_file = monte_carlo_vmc.import_config(sys.argv[1])
    config_vmc_import = config_vmc_file.MC_parameters(int(sys.argv[2]), 0)

    config_vmc = cv_module.MC_parameters(int(sys.argv[2]), 0)
    config_vmc.__dict__ = config_vmc_import.__dict__.copy()

    monte_carlo_vmc.print_model_summary(config_vmc)

    generate_Imada_format_Koshino(config_vmc, float(sys.argv[3]), int(sys.argv[4]), \
                                  doping=int(sys.argv[5]), periodic = (sys.argv[6] == '1'))
