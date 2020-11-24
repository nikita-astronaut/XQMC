import models
import numpy as np
import os
from opt_parameters import pairings
import monte_carlo_vmc
import sys
import config_vmc as cv_module
from numba import jit

@jit(nopython=True)
def get_orb(L, mod):
    orb_ij = []; orb_k = []; n_orb = 0
    matrix = np.zeros((4 * L ** 2,4 * L ** 2), dtype=np.int64)
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
def get_jastrow_fromshift(L, mod):
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


def generate_Imada_format_Koshino(config, U):
    H = config.hamiltonian(config_vmc)
    K0 = config.K_0 # already chiral

    np.save('K0.npy', K0)
    np.save('H_edges.npy', H.edges_quadric)

    mod = 2
    path = os.path.join(config.workdir, 'imada_format_L_{:d}_Ne_{:d}_U_{:.3f}'.format(config.Ls, config.Ne, U))
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
    f.close()




    
    Tx, Ty = pairings.Tx_symmetry_map, pairings.Ty_symmetry_map
    tx, ty = [], []

    for i in range(Tx.shape[0]):
        assert len(np.where(Tx[i, :] == 1)[0]) == 1
        assert len(np.where(Ty[i, :] == 1)[0]) == 1

        tx.append(np.where(Tx[i, :] == 1)[0][0])
        ty.append(np.where(Ty[i, :] == 1)[0][0])

    tx, ty = np.array(tx), np.array(ty)
    assert np.allclose(tx[ty], ty[tx])
    np.save('tx.npy', tx)
    np.save('ty.npy', ty)

    valley = np.concatenate([np.array([2 * i + 1, 2 * i]) for i in range(8)])
    rotation = np.array([0, 1, 14, 15, 8, 9, 6, 7, 12, 13, 2, 3, 4, 5, 10, 11])
    assert np.allclose(valley[valley], np.arange(16))
    assert np.allclose(rotation[rotation[rotation]], np.arange(16))

    symmetries = [np.arange(len(tx)), tx, ty, ty[tx]]

    symmetries = symmetries + [symm[valley] for symm in symmetries]
    #symmetries = symmetries + [symm[rotation] for symm in symmetries] + [symm[rotation[rotation]] for symm in symmetries]

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
    jastrow_ij, jastrow_k, n_jastrows, seen_shifts, matrix_jastrows = get_jastrow_fromshift(config.Ls, mod)
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


    f.write('=============================================\n')
    f.write('NJastrowIdx         {:d}\n'.format(n_jastrows))
    f.write('ComplexType          1\n')
    f.write('=============================================\n')
    f.write('=============================================\n')
    for ij, k in zip(jastrow_ij, jastrow_k):
        f.write('    {:d}      {:d}      {:d}\n'.format(ij[0], ij[1], k))
    for i in range(n_jastrows):
        f.write('    {:d}      1\n'.format(i))
    f.close()





    f = open(os.path.join(path, 'InJastrow.def'), 'w')
    f.write('======================\n')
    f.write('NJastrowIdx  {:d}\n'.format(n_jastrows))
    f.write('======================\n')
    f.write('== i_j_JastrowIdx  ===\n')
    f.write('======================\n')
    for i in range(n_jastrows):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, \
                np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)))
    f.close()














    f = open(os.path.join(path, 'gutzwilleridx.def'), 'w')  # FIXME: impose sublattice structure too

    f.write('=============================================\n')
    f.write('NGutzwillerIdx          {:d}\n'.format(4 * mod ** 2))
    f.write('ComplexType          1\n')
    f.write('=============================================\n')
    f.write('=============================================\n')

    for i in range(K0.shape[0]):
        oi, si, xi, yi = models.from_linearized_index(i, config.Ls, config.n_orbitals, config.n_sublattices)

        modx = xi % mod
        mody = yi % mod

        idx = mod ** 2 * (si * 2 + oi) + mod * modx + mody
        f.write('    {:d}      {:d}\n'.format(i, idx))
    for i in range(16):
        f.write('    {:d}      1\n'.format(i))
    f.close()


    f = open(os.path.join(path, 'InGutzwiller.def'), 'w')
    f.write('======================\n')
    f.write('NGutzwillerIdx  {:d}\n'.format(4 * mod ** 2))
    f.write('======================\n')
    f.write('== i_j_GutzwillerIdx  ===\n')
    f.write('======================\n')
    for i in range(4):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)))
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
    f.write('NOrbitalIdx         {:d}\n'.format(n_orbits))
    f.write('ComplexType          1\n')
    f.write('=============================================\n')
    f.write('=============================================\n')

    for ij, k in zip(orbit_ij, orbit_k):
        f.write('    {:d}      {:d}      {:d}\n'.format(ij[0], ij[1], k))
    for i in range(n_orbits):
        f.write('    {:d}      1\n'.format(i))
    f.close()

    f = open(os.path.join(path, 'InOrbital.def'), 'w')
    f.write('======================\n')
    f.write('NOrbitalIdx  {:d}\n'.format(n_orbits))
    f.write('======================\n')
    f.write('== i_j_OrbitalIdx  ===\n')
    f.write('======================\n')
    for i in range(n_orbits):
        f.write('{:d} {:.10f}  {:.10f}\n'.format(i, np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)))






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
    f.write('Ncond          {:d}\n'.format(config.Ne))
    f.write('2Sz            0\n')
    f.write('NSPGaussLeg    8\n')
    f.write('NSPStot        0\n')
    f.write('NMPTrans       {:d}\n'.format(len(symmetries)))
    f.write('NSROptItrStep  400\n')
    f.write('NSROptItrSmp   40\n')
    f.write('DSROptRedCut   0.0010000000\n')
    f.write('DSROptStaDel   0.0200000000\n')
    f.write('DSROptStepDt   0.0200000000\n')
    f.write('NVMCWarmUp     100\n')
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

    generate_Imada_format_Koshino(config_vmc, float(sys.argv[3]))
