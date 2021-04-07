import models
import numpy as np
import os
from opt_parameters import pairings
import monte_carlo_vmc
import sys
import config_vmc as cv_module
from numba import jit
from scipy.linalg import schur
from copy import deepcopy

@jit(nopython=True)
def get_fft_APBCy(N):
    W = np.zeros((N ** 2, N ** 2), dtype=np.complex128)
    for kx in range(N):
        for ky in range(N):
            for x in range(N):
                for y in range(N):
                    W[x * N + y, kx * N + ky] = np.exp(2.0j * np.pi / N * kx * x + (2.0j * np.pi / N * ky - 1.0j * np.pi / N) * y)
    return np.kron(W, np.eye(4))

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
def get_jastrow_fromshift(all_translations, L, all_distances_rounded, dist_threshold = 1):
    orb_ij = []; orb_k = []; n_orb = 0
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64) - 1
    cur_jastrow = 0
    dist = list(np.sort(np.unique(all_distances_rounded)))

    for i in range(0, 4):
        for j in range(i + 1, 4 * L ** 2):
            if i == j:
                continue

            oi = i % 2; oj = j % 2

            d_ij = all_distances_rounded[i, j]
            if d_ij > 0 + 1e-5:
                jastrow_idx = dist.index(d_ij) * 4 + oi * 2 + oj - 3
                #if oi == oj:
                #    jastrow_idx = dist.index(d_ij) * 2
                #else:
                #    jastrow_idx = dist.index(d_ij) * 2 - 1
            else:
                jastrow_idx = 0  # then we know the orbitals are different


            for trans in all_translations:
                matrix[trans[i], trans[j]] = jastrow_idx
                matrix[trans[j], trans[i]] = jastrow_idx
                orb_ij.append((trans[i], trans[j]))
                orb_k.append(jastrow_idx)
                orb_ij.append((trans[j], trans[i]))
                orb_k.append(jastrow_idx)
    cur_jastrow = matrix.max()

    return orb_ij, orb_k, cur_jastrow, matrix



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



def get_interaction(config, U_list):
    Ls = config.Ls
    K0 = config.K_0

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

    path = '/home/astronaut/Documents/all_Imada_formats/'


    for U in U_list:
        f = open(os.path.join(path, 'coulombintra_{:d}_{:.3f}.def'.format(Ls, U)), 'w')

        f.write('=============================================\n')
        f.write('NCoulombIntra          {:d}\n'.format(config.total_dof // 2))
        f.write('=============================================\n')
        f.write('================== CoulombIntra ================\n')
        f.write('=============================================\n')

        for i in range(K0.shape[0]):
            f.write('    {:d}         {:.14f}\n'.format(i, U))
        f.close()

        interactions = np.zeros((config.total_dof // 2, config.total_dof // 2))
        dist = np.around(np.sqrt(config.all_distances), decimals=5)
        dist_unique = np.sort(np.unique(dist.flatten()))

        for i in range(config.total_dof // 2):
            for j in range(config.total_dof // 2):
                if i == j:
                    continue
                r = dist[i, j]
                r_number = np.where(dist_unique == r)[0][0]
                if r_number == 0:
                    interactions[i, j] = U
                    continue

                if r_number == 1:
                    interactions[i, j] = 2 * U / 3.
                    continue

                if r_number == 2:
                    interactions[i, j] = U / 3.
                    continue

                if r_number == 3:
                    interactions[i, j] = U / 3.
                    continue

        f = open(os.path.join(path, 'coulombinter_{:d}_{:.3f}.def'.format(Ls, U)), 'w')
        f.write('=============================================\n')
        f.write('NCoulombInter          {:d}\n'.format(int(np.sum(np.abs(interactions) > 1e-5))))
        f.write('=============================================\n')
        f.write('================== CoulombInter ================\n')
        f.write('=============================================\n')

        for i in range(K0.shape[0]):
            for j in range(K0.shape[1]):
                if i == j:
                    continue

                if interactions[i, j] != 0.0:
                    f.write('   {:d}     {:d}  {:.4f}\n'.format(i, j, interactions[i, j] / 2.))
        f.close()  # TODO: check very carefully the factor of 2


        n_terms = (config.total_dof // 2 // 2) * 3 * 16


        f = open(os.path.join(path, 'interall_{:d}_{:.3f}.def'.format(Ls, U)), 'w')
        f.write('======================\n')
        f.write('NInterAll          {:d}\n'.format(n_terms))
        f.write('======================\n')
        f.write('========zInterAll=====\n')
        f.write('======================\n')

        for i in range(config.total_dof // 2 // 2):
            for j in range(config.total_dof // 2 // 2):
                r = dist[i * 2, j * 2]
                r_number = np.where(dist_unique == r)[0][0]
                if r_number != 1:  # only NN terms
                    continue

                for spin_i in range(2):
                    for valley_i in range(2):
                        for spin_j in range(2):
                            for valley_j in range(2):
                                f.write('{:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:.10f} {:.10f}\n'.format(
                                    i * 2 + valley_i, spin_i, \
                                    j * 2 + valley_i, spin_i, \
                                    j * 2 + valley_j, spin_j, \
                                    i * 2 + valley_j, spin_j, \
                                    +0.1 * U, 0.0))
        f.close()
    return


def get_kinetic_orbitals(config, filling):
    Ls = config.Ls
    K0 = config.K_0
    assert config.twist_mesh == 'PBC'

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

    path = '/home/astronaut/Documents/all_Imada_formats/'

    ########### writing the spin locations (none) ##########
    f = open(os.path.join(path, 'locspn_{:d}.def'.format(Ls)), 'w')
    f.write('================================\n')
    f.write('NlocalSpin     0\n')
    f.write('================================\n')
    f.write('========i_0LocSpn_1IteElc ======\n')
    f.write('================================\n')
    for i in range(Ls ** 2 * 4):
        f.write('    {:d}      0\n'.format(i))
    f.close()



    symmetries = [np.arange(Ls ** 2 * 4)]
    ########### writing the translational symmetries ##########
    f = open(os.path.join(path, 'qptransidx_{:d}.def'.format(Ls)), 'w')
    f.write('=============================================\n')
    f.write('NQPTrans          {:d}\n'.format(len(symmetries)))
    f.write('=============================================\n')
    f.write('======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n')
    f.write('=============================================\n')

    for i in range(len(symmetries)):
        f.write('{:d}    1.00000\n'.format(i))

    for i, symm in enumerate(symmetries):
        for i_from in range(symm.shape[0]):
            f.write('    {:d}      {:d}      {:d}\n'.format(i, i_from, symm[i_from]))
    f.close()



    from copy import deepcopy
    ########### writing the jastrows ##########
    all_translations = [np.arange(Ls ** 2 * 4)]
    curr_trans = tx.copy()
    all_new_translations = []
    for kx in range(config.Ls - 1):
        new_translations = [symm[curr_trans] for symm in all_translations]
        all_new_translations.append(deepcopy(new_translations))
        curr_trans = curr_trans[tx]
    for d in all_new_translations:
        all_translations += d

    curr_trans = ty.copy()
    all_new_translations = []
    for kx in range(config.Ls - 1):
        new_translations = [symm[curr_trans] for symm in all_translations]
        all_new_translations.append(deepcopy(new_translations))
        curr_trans = curr_trans[ty]
    for d in all_new_translations:
        all_translations += d



    f = open(os.path.join(path, 'jastrowidx_TRSbroken_{:d}.def'.format(Ls)), 'w')
    jastrow_ij, jastrow_k, n_jastrows, matrix_jastrows = get_jastrow_fromshift(all_translations, config.Ls, np.around(np.array(config.all_distances), decimals = 5), dist_threshold=5.)
    #np.save('check.npy', matrix_jastrows)

    assert np.allclose(matrix_jastrows, matrix_jastrows.T)

    matrix_jastrows_trans = matrix_jastrows.copy()
    matrix_jastrows_trans = matrix_jastrows_trans[:, tx]    
    matrix_jastrows_trans = matrix_jastrows_trans[tx, :]
    assert np.allclose(matrix_jastrows_trans, matrix_jastrows)

    matrix_jastrows_trans = matrix_jastrows.copy()
    matrix_jastrows_trans = matrix_jastrows_trans[:, ty]    
    matrix_jastrows_trans = matrix_jastrows_trans[ty, :]
    assert np.allclose(matrix_jastrows_trans, matrix_jastrows)


    f.write('=============================================\n')
    f.write('NJastrowIdx         {:d}\n'.format(n_jastrows + 1))
    f.write('ComplexType          {:d}\n'.format(0))
    f.write('=============================================\n')
    f.write('=============================================\n')

    uniques = []
    for i in range(config.Ls ** 2 * 4):
        for j in range(config.Ls ** 2 * 4):
            if i == j:
                continue
            f.write('    {:d}      {:d}      {:d}\n'.format(i, j, matrix_jastrows[i, j]))

    for i in range(n_jastrows):
        f.write('    {:d}      1\n'.format(i))
    f.write('    {:d}      0\n'.format(n_jastrows))
    f.close()



    f = open(os.path.join(path, 'InJastrow_TRSbroken_{:d}.def'.format(Ls)), 'w')
    f.write('======================\n')
    f.write('NJastrowIdx  {:d}\n'.format(n_jastrows + 1))
    f.write('======================\n')
    f.write('== i_j_JastrowIdx  ===\n')
    f.write('======================\n')
    for i in range(n_jastrows):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, \
                np.random.uniform(0.0, 1.0) * 0, np.random.uniform(0.0, 1.0) * 0))
    f.write('{:d} {:.10f}  {:.10f}\n'.format(n_jastrows, 0, 0))
    f.close()




    f = open(os.path.join(path, 'gutzwilleridx_{:d}.def'.format(Ls)), 'w')

    f.write('=============================================\n')
    f.write('NGutzwillerIdx          {:d}\n'.format(1))
    f.write('ComplexType          {:d}\n'.format(0))
    f.write('=============================================\n')
    f.write('=============================================\n')

    for i in range(4 * Ls ** 2):
        f.write('    {:d}      {:d}\n'.format(i, 0))#idx))
    for i in range(1):
        f.write('    {:d}      1\n'.format(i))
    f.close()


    f = open(os.path.join(path, 'InGutzwiller.def'), 'w')
    f.write('======================\n')
    f.write('NGutzwillerIdx  {:d}\n'.format(1))
    f.write('======================\n')
    f.write('== i_j_GutzwillerIdx  ===\n')
    f.write('======================\n')
    for i in range(1):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, np.random.uniform(0.0, 1.0) * 0, np.random.uniform(0.0, 1.0) * 0))
    f.close()


    
    ########### writing the modpara ##########
    f = open(os.path.join(path, 'modpara_{:d}_{:d}.def'.format(Ls, filling)), 'w')

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
    f.write('Nsite          {:d}\n'.format(Ls ** 2 * 4))
    f.write('Ncond          {:d}\n'.format(filling))
    f.write('2Sz            0\n')
    f.write('NSPGaussLeg    8\n')
    f.write('NSPStot        0\n')
    f.write('NMPTrans       {:d}\n'.format(1))
    f.write('NSROptItrStep  400\n')
    f.write('NSROptItrSmp   40\n')
    f.write('DSROptRedCut   0.0000001000\n')
    f.write('DSROptStaDel   0.0200000000\n')
    f.write('DSROptStepDt   0.0000020000\n')
    f.write('NVMCWarmUp     400\n')
    f.write('NVMCInterval   1\n')
    f.write('NVMCSample     4000\n')
    f.write('NExUpdatePath  0\n')
    f.write('RndSeed        1\n')
    f.write('NSplitSize     1\n')
    f.write('NStore         0\n')
    f.write('NSRCG          1\n')
    f.close()


    twist = (0, 0.5)
    twist_exp = [np.exp(2 * np.pi * 1.0j * twist[0]), np.exp(2 * np.pi * 1.0j * twist[1])]
    fft = get_fft_APBCy(config.Ls)
    for gap_idx in range(len(config.idx_map)):
        print('gap_idx = ', gap_idx, 'gap_id = ',config.idx_map[gap_idx] )
        if config.idx_map[gap_idx] != 13:
            continue

        for gap_val in [0.00, 0.003, 0.006, 0.010, 0.02, 0.03]:
            if gap_idx == 0:
                g = 0.0 * K0
            else:
                g = gap_val * np.load('the_wave_extended_{:d}.npy'.format(config.Ls))
                #models.xy_to_chiral(pairings.combine_product_terms(config, pairings.twoorb_hex_all[config.idx_map[gap_idx]]), 'pairing', config, True)

            # g = (g + g.conj()) / np.sqrt(2)
            #print(g[0], 'g[0]')
            #print(g[1], 'g[1]')
            swave = 1e-5 * models.xy_to_chiral(pairings.combine_product_terms(config, pairings.twoorb_hex_all[1]), 'pairing', config, True)
            #print(swave[0], 'swave')

            g = g + swave
            gap = models.apply_TBC(config, twist_exp, deepcopy(g), inverse = False)
            #np.save('the_wave_extended_twisted_{:d}.npy'.format(config.Ls), models.apply_TBC(config, twist_exp, deepcopy(np.load('the_wave_extended_{:d}.npy'.format(config.Ls))), inverse = False))
            #exit(-1)
            gapT = models.apply_TBC(config, twist_exp, deepcopy(g).T, inverse = True)

            gap_fft = fft.T.conj().dot(gap).dot(fft)
            gap_check = gap_fft.copy()
            for i in range(gap_check.shape[0] // 4):
                #print(i % 4, i // 4)
                #(np.abs(np.linalg.eig(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4])[0]), i)
                # print(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4])
                #assert np.allclose(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4], gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4].conj().T)
                gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4] = 0.0
            assert np.isclose(np.sum(np.abs(gap_check)), 0.0)

            ############ determine required mu_BCS to start ################
            K0_up = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = False)
            K0_down = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = True)
            K0_downT = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = True).T
            K0_upT = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = False).T
            print('energies {:d}'.format(config.Ls), np.linalg.eigh(K0_up)[0])

            #### check twist is correct ###
            K0_fft_plus = fft.conj().T.dot(K0_up).dot(fft)
            K0_fft_minus = fft.T.dot(K0_up).dot(fft.conj())

            K0_check = K0_fft_plus.copy()
            for i in range(K0_check.shape[0] // 4):
                K0_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4] = 0.0
            assert np.isclose(np.sum(np.abs(K0_check)), 0.0)

            K0_check = K0_fft_minus.copy()
            for i in range(K0_check.shape[0] // 4):
                K0_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4] = 0.0
            assert np.isclose(np.sum(np.abs(K0_check)), 0.0)

            assert np.allclose(K0_up, K0_up.conj().T)
            assert np.allclose(K0_down, K0_down.conj().T)

            L = K0.shape[0]
            totalM = np.zeros((4 * L, 4 * L), dtype=np.complex128)

            totalM[:L, :L] = K0_up; totalM[L:2 * L, L:2 * L] = K0_down; totalM[2 * L:3 * L, 2 * L:3 * L] = -K0_upT; totalM[3 * L:, 3 * L:] = -K0_downT
            totalM[:L, 3 * L:] = gap; totalM[L: 2 * L, 2 * L:3 * L] = -gapT; totalM[2 * L: 3 * L, L:2 * L] = -gapT.conj().T; totalM[3 * L:, :L] = gap.conj().T; 
            energies = np.linalg.eigh(totalM)[0] # energies with BC twist and gap
            
            #print(energies)
            mu_BCS = (energies[filling * 2] + energies[filling * 2 - 1]) / 2.
            print (energies[filling * 2], energies[filling * 2 - 1])
            #assert not np.isclose(energies[filling * 2], energies[filling * 2 - 1])
            print('mu_BCS = ', mu_BCS)


            K0_up = K0_up - np.eye(K0_up.shape[0]) * mu_BCS
            K0_upT = K0_upT - np.eye(K0_upT.shape[0]) * mu_BCS
            K0_down = K0_down - np.eye(K0_down.shape[0]) * mu_BCS
            K0_downT = K0_downT - np.eye(K0_downT.shape[0]) * mu_BCS



            L = K0.shape[0]
            totalM = np.zeros((4 * L, 4 * L), dtype=np.complex128)

            totalM[:L, :L] = K0_up; totalM[L:2 * L, L:2 * L] = K0_down; totalM[2 * L:3 * L, 2 * L:3 * L] = -K0_upT; totalM[3 * L:, 3 * L:] = -K0_downT
            totalM[:L, 3 * L:] = gap; totalM[L: 2 * L, 2 * L:3 * L] = -gapT; totalM[2 * L: 3 * L, L:2 * L] = -gapT.conj().T; totalM[3 * L:, :L] = gap.conj().T; 

            selected_idxs = np.concatenate([np.arange(0, L), np.arange(3 * L, 4 * L)])
            totalM_updown = totalM[:, selected_idxs]; totalM_updown = totalM_updown[selected_idxs, ...]

            #totalM_updown = np.zeros((2 * L, 2 * L), dtype=np.complex128)
            #totalM_updown[:L, :L] = K0; totalM_updown[L:, L:] = -K0.T;
            #totalM_updown[:L, L:] = gap; totalM_updown[L:, :L] = gap.conj().T;

            selected_idxs = np.arange(L, 3 * L)
            totalM_downup = totalM[:, selected_idxs]; totalM_downup = totalM_downup[selected_idxs, ...]

            TRS = np.concatenate([np.array([2 * i + 1, 2 * i], dtype=np.int64) for i in range(L)])


            #totalM_downup = np.zeros((2 * L, 2 * L), dtype=np.complex128)
            #totalM_downup[:L, :L] = K0; totalM_downup[L:, L:] = -K0.T;
            #totalM_downup[:L, L:] = -gap.T; totalM_downup[L:, :L] = -gap.conj();

            en_updown, W_updown = np.linalg.eigh(totalM_updown)

            totalM_updown_TRS = totalM_updown[TRS, ...];totalM_updown_TRS = totalM_updown_TRS[..., TRS]
            totalM_updown_TRS = totalM_updown_TRS.conj()

            print('after TRS, discrepancy', np.sum(np.abs(totalM_updown_TRS - totalM_updown)))

            en_downup, W_downup = np.linalg.eigh(totalM_downup)
            assert np.allclose(en_updown, np.sort(-en_updown))
            assert np.allclose(en_downup, np.sort(-en_downup))
            en_total, W = np.linalg.eigh(totalM)
            #print('energies with gap', gap_val, en_total)
            #print('updown energies', en_updown)
            #print('downup energies', en_downup)


            #for en, state in zip(en_downup, W_downup.T):
            #    if np.abs(en + 0.03085819) < 1e-6:
            #        print(en, state)
            #exit(-1)

            for i in range(W_updown.shape[1] // 2):
                v = W_updown[:, i]
                en = en_updown[i]

                v_conj = v * 0.0;
                v_conj[:len(v) // 2] = v[len(v)//2:].conj()
                v_conj[len(v) // 2:] = v[:len(v)//2].conj()

                en_conj = np.dot(v_conj.conj(), totalM_updown.dot(v_conj)) / np.dot(v_conj.conj(), v_conj)
                #print(en_conj, en, np.dot(v_conj.conj(), v_conj), np.dot(v.conj(), totalM_updown.dot(v)))
                #W_conj.append(v_conj)
                #assert np.isclose(en_conj, -en)
            #exit(-1)


            W_conj = []

            for i in range(W.shape[1] // 2):
                v = W[:, i]
                en = en_total[i]

                v_conj = v * 0.0;
                v_conj[:len(v) // 2] = v[len(v)//2:].conj()
                v_conj[len(v) // 2:] = v[:len(v)//2].conj()

                en_conj = np.dot(v_conj.conj(), totalM.dot(v_conj))
                #print(en_conj, en)
                W_conj.append(v_conj)
                assert np.isclose(en_conj, -en)

            W_conj = np.array(W_conj).T

            W[:, W.shape[1] // 2:] = W_conj  # make the right form -- but this makes no difference: this is only rearrangement of 2nd part of the array, while we only use the 1st part
            # why W does not protect that block form? -- or do we even need this form?


            assert np.allclose(np.diag(W.conj().T.dot(totalM).dot(W)).real, np.diag(W.conj().T.dot(totalM).dot(W)))
            assert np.allclose(np.sort(np.diag(W.conj().T.dot(totalM).dot(W)).real), np.linalg.eigh(totalM)[0])



            # with gap 6, W_pieces does not diagonalize totalM! why?
            W_pieces = np.zeros((4 * L, 4 * L), dtype=np.complex128)
            W_pieces[:L, :L] = W_updown[:L, :L]; W_pieces[3 * L:, 3 * L:] = W_updown[L:, L:];
            W_pieces[3 * L:, :L] = W_updown[L:, :L]; W_pieces[:L, 3 * L:] = W_updown[:L, L:];

            W_pieces[L: 2 * L, L: 2 * L] = W_downup[:L, :L]; W_pieces[2 * L:3 * L, 2 * L:3 * L] = W_downup[L:, L:];
            W_pieces[2 * L:3 * L, L: 2 * L] = W_downup[L:, :L]; W_pieces[L: 2 * L, 2 * L:3 * L] = W_downup[:L, L:];

            

            #assert np.allclose(np.sort(np.diag(W_pieces.conj().T.dot(totalM).dot(W_pieces)).real), np.sort(np.diag(W_pieces.conj().T.dot(totalM).dot(W_pieces))))
            #assert np.isclose(np.sum(np.abs(W_pieces.conj().T.dot(totalM).dot(W_pieces) - np.diag(np.diag(W_pieces.conj().T.dot(totalM).dot(W_pieces))))), 0.0)
            assert np.isclose(np.sum(np.abs(W.conj().T.dot(totalM).dot(W) - np.diag(np.diag(W.conj().T.dot(totalM).dot(W))))), 0.0)

            #print(np.sort(np.diag(W.conj().T.dot(totalM).dot(W)).real) - np.sort(np.diag(W_pieces.conj().T.dot(totalM).dot(W_pieces)).real))
            #assert np.allclose(np.sort(np.diag(W.conj().T.dot(totalM).dot(W)).real), \
            #                   np.sort(np.diag(W_pieces.conj().T.dot(totalM).dot(W_pieces)).real))

            #print(np.linalg.det(W_updown), np.linalg.det(W_downup), np.linalg.det(W_updown) * np.linalg.det(W_downup))
            #print(np.linalg.det(W_pieces))
            #print(np.linalg.det(W))



            for i in range(W_updown.shape[1]):
                v = W_updown[:, i]
                en = en_updown[i]

                v_conj = v * 0.0;
                v_conj[:len(v) // 2] = v[len(v)//2:].conj()
                v_conj[len(v) // 2:] = v[:len(v)//2].conj()

                en_conj = np.dot(v_conj.conj(), totalM_updown.dot(v_conj))
                # print(en_conj, en)
                #assert en_conj == -en



            mask = np.zeros((4 * L, 4 * L), dtype=np.complex128)
            mask[:L, :L] = np.ones((L, L)); mask[L:2 * L, L:2 * L] = np.ones((L, L)); mask[2 * L:3 * L, 2 * L:3 * L] = np.ones((L, L)); mask[3 * L:, 3 * L:] = np.ones((L, L))
            mask[3 * L:, :L] = np.ones((L, L)); mask[2 * L: 3 * L, L:2 * L] = np.ones((L, L)); mask[L: 2 * L, 2 * L:3 * L] = np.ones((L, L)); mask[:L, 3 * L:] = np.ones((L, L)); 


            #totalM[:L, :L] = K0; totalM[L:2 * L, L:2 * L] = K0; totalM[2 * L:3 * L, 2 * L:3 * L] = -K0.T; totalM[3 * L:, 3 * L:] = -K0.T
            #totalM[:L, 2 * L:3 * L] = gap; totalM[L: 2 * L, 3 * L:] = -gap.T; totalM[3 * L:, L: 2 * L] = -gap.conj(); totalM[2 * L:3 * L, :L] = gap.conj().T;

            assert np.allclose(totalM, totalM.conj().T)


            # W = np.linalg.eigh(totalM / 2.)[1]
            #print(np.linalg.eigh(totalM / 2.)[0])
            #assert np.sum(np.abs(W - W * mask)) == 0


            #assert np.allclose(W[:W.shape[0] // 2, :W.shape[0] // 2], W[W.shape[0] // 2:, W.shape[0] // 2:].conj())
            #assert np.allclose(W[W.shape[0] // 2:, :W.shape[0] // 2], W[:W.shape[0] // 2, W.shape[0] // 2:].conj())

            Q, V = W[:W.shape[0] // 2, :W.shape[0] // 2], \
                   W[W.shape[0] // 2:, :W.shape[0] // 2]
            Z = (Q.dot(np.linalg.inv(V)))
            print('max U^{-1} = ', np.max(np.abs(np.linalg.inv(Q))))


            np.save('Z_fast.npy', Z) 
            result = Z[Z.shape[0] // 2:, :Z.shape[0] // 2]

            Z = Z / np.abs(np.max(Z))
            print(np.sum(np.abs(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T)))
            print(np.sum(np.abs(np.real(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T))))
            print(np.sum(np.abs(np.imag(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T))))

            
            assert np.allclose(Z[Z.shape[0] // 2:, :Z.shape[0] // 2], -Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T)
            assert np.allclose(Z[Z.shape[0] // 2:, Z.shape[0] // 2:], Z[Z.shape[0] // 2:, Z.shape[0] // 2:] * 0.0)
            assert np.allclose(Z[:Z.shape[0] // 2, :Z.shape[0] // 2], Z[:Z.shape[0] // 2, :Z.shape[0] // 2] * 0.0)


            ##### preparing orbital idxs and teir initial values ####
            vol = 4 * Ls ** 2
            orbital_idxs = -np.ones((vol, vol), dtype=np.int64)

            f_ij = result
            f_ij = f_ij / np.abs(np.max(f_ij))
            np.save('f_ij_fast.npy', f_ij)

            current_orb_idx = 0
            for xshift in range(Ls):
                for yshift in range(Ls):
                    for iorb in range(4):
                        for jorb in range(4):
                            if yshift > 0:
                                for ipos in range(Ls ** 2):
                                    i = ipos * 4 + iorb
                                    oi, si, xi, yi = models.from_linearized_index(i, config.Ls, config.n_orbitals, config.n_sublattices)
                                    j = models.to_linearized_index((xi + xshift) % Ls, (yi + yshift) % Ls, jorb % 2, jorb // 2, Ls, 2, 2)
                                    if yi + yshift > Ls - 1:
                                        orbital_idxs[i, j] = current_orb_idx
                                current_orb_idx += 1

                            for ipos in range(Ls ** 2):
                                i = ipos * 4 + iorb
                                oi, si, xi, yi = models.from_linearized_index(i, config.Ls, config.n_orbitals, config.n_sublattices)
                                j = models.to_linearized_index((xi + xshift) % Ls, (yi + yshift) % Ls, jorb % 2, jorb // 2, Ls, 2, 2)
                                if yi + yshift <= Ls - 1:
                                    orbital_idxs[i, j] = current_orb_idx
                            current_orb_idx += 1
            print('FAST: orbitals after enforcing APBCy remaining:', current_orb_idx)
            for i in range(current_orb_idx):
                values = f_ij.flatten()[orbital_idxs.flatten() == i]
                assert np.isclose(np.std(values - values.mean()), 0.0)



            if np.allclose(f_ij, f_ij.T):
                print('FAST: symmetric f_ij = f_ji (singlet): restricting su(2) parameters')

                for i in range(vol):
                    for j in range(vol):
                        orb_ij = orbital_idxs[i, j]
                        orb_ji = orbital_idxs[j, i]
                        orbital_idxs[i, j] = np.min([orb_ij, orb_ji])
                        orbital_idxs[j, i] = np.min([orb_ij, orb_ji])
                new_orbitals = np.unique(orbital_idxs.flatten())
                mapping = list(np.sort(new_orbitals))

                for i in range(vol):
                    for j in range(vol):
                        orbital_idxs[i, j] = mapping.index(orbital_idxs[i, j])

                for i in range(len(mapping)):
                    values = f_ij.flatten()[orbital_idxs.flatten() == i]
                    assert np.isclose(np.std(values - values.mean()), 0.0)
                print('FAST: total orbitals su(2) with APBCy', len(mapping))
                current_orb_idx = len(mapping)

            TRS = np.concatenate([[2 * i + 1, 2 * i] for i in range(vol // 2)])
            f_trs = f_ij[:, TRS]
            f_trs = f_trs[TRS, :]
            if np.allclose(f_trs, f_ij):
                print('FAST: f_ij = TRS f_ij: resticting TRS parameters')


                for i in range(vol):
                    for j in range(vol):
                        orb_ij = orbital_idxs[i, j]
                        i_trs = ((i // 2) * 2) + (((i % 2) + 1) % 2)
                        j_trs = ((j // 2) * 2) + (((j % 2) + 1) % 2)
                        orb_ij_trs = orbital_idxs[i_trs, j_trs]
                        #print(f_ij[i, j], f_ij[i_trs, j_trs])
                        assert np.isclose(f_ij[i, j], f_ij[i_trs, j_trs])

                        orbital_idxs[i, j] = np.min([orb_ij, orb_ij_trs])
                        orbital_idxs[i_trs, j_trs] = np.min([orb_ij, orb_ij_trs])

                #for i in range(current_orb_idx):
                #    if np.sum(orbital_idxs.flatten() == i) == 0:
                #        print('orbital', i, 'is missing')
                new_orbitals = np.unique(orbital_idxs.flatten())
                mapping = list(np.sort(new_orbitals))

                for i in range(vol):
                    for j in range(vol):
                        orbital_idxs[i, j] = mapping.index(orbital_idxs[i, j])

                for i in range(len(mapping)):
                    values = f_ij.flatten()[orbital_idxs.flatten() == i]
                    assert np.isclose(np.std(values - values.mean()), 0.0)
                print('FAST: total orbitals su(2) with APBCy and TRS!', len(mapping) + 1)
                current_orb_idx = len(mapping)

            np.save('orbital_idxs_fast.npy', orbital_idxs)



            f = open(os.path.join(path, 'InOrbital_extended_{:d}_{:d}_{:d}_{:.4f}.def'.format(Ls, config.idx_map[gap_idx], filling, gap_val)), 'w')
            f.write('======================\n')
            f.write('NOrbitalIdx  {:d}\n'.format(current_orb_idx))
            f.write('======================\n')
            f.write('== i_j_OrbitalIdx  ===\n')
            f.write('======================\n')
            for k in range(current_orb_idx):
                mask = (orbital_idxs == k)
                val = np.sum(f_ij * mask) / np.sum(mask)
                f.write('{:d} {:.20f}  {:.20f}\n'.format(k, val.real, val.imag))
            f.close()








            ########### writing the orbitals indexes ##########
            f = open(os.path.join(path, 'orbitalidx_extended_{:d}_{:d}_{:d}.def'.format(Ls, config.idx_map[gap_idx], filling)), 'w')

            f.write('=============================================\n')
            f.write('NOrbitalIdx         {:d}\n'.format(current_orb_idx))
            f.write('ComplexType          {:d}\n'.format(1))
            f.write('=============================================\n')
            f.write('=============================================\n')

            for i in range(config.Ls ** 2 * 4):
                for j in range(config.Ls ** 2 * 4):
                    f.write('    {:d}      {:d}      {:d}\n'.format(i, j, orbital_idxs[i, j]))

            for i in range(current_orb_idx):
                f.write('    {:d}      1\n'.format(i))
            f.close()



            twist_exp = [np.exp(2.0j * np.pi * twist[0]), np.exp(2.0j * np.pi * twist[1])]
            K_0_twisted = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = False)


            ########### writing the K--matrix ##########
            K0_up_int = K0_up - np.diag(np.diag(K0_up))
            K0_down_int = K0_down - np.diag(np.diag(K0_down))

            f = open(os.path.join(path, 'trans_{:d}_{:.3f}_{:.3f}.def'.format(Ls, *twist)), 'w')

            f.write('========================\n')
            f.write('NTransfer      {:d}\n'.format(2 * np.sum(np.abs(K0_up_int) > 1e-7)))
            f.write('========================\n')
            f.write('========i_j_s_tijs======\n')
            f.write('========================\n')

            for i in range(K_0_twisted.shape[0]):
                for j in range(K_0_twisted.shape[1]):
                    if np.abs(K0_up_int[i, j]) > 1e-7:
                        f.write('    {:d}     0     {:d}     0   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0_up_int[i, j]), np.imag(-K0_up_int[i, j])))
                        f.write('    {:d}     1     {:d}     1   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0_down_int[i, j]), np.imag(-K0_down_int[i, j])))
            f.close()
    return


    


if __name__ == "__main__":
    config_vmc_file = monte_carlo_vmc.import_config(sys.argv[1])
    config_vmc_import = config_vmc_file.MC_parameters(int(sys.argv[2]), 0)

    config_vmc = cv_module.MC_parameters(int(sys.argv[2]), 0)
    config_vmc.__dict__ = config_vmc_import.__dict__.copy()

    monte_carlo_vmc.print_model_summary(config_vmc)


    #get_interaction(config_vmc, np.linspace(0.0, 0.9, 4))
    get_kinetic_orbitals(config_vmc, int(sys.argv[3]))
