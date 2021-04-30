import models
import numpy as np
import os
from opt_parameters import pairings
import monte_carlo_vmc
import sys
import config_vmc as cv_module
from numba import jit
from scipy.linalg import schur
from pfapack import pfaffian as cpf
from copy import deepcopy

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
def get_jastrow_fromshift(L, all_distances_rounded, dist_threshold = 1):
    orb_ij = []; orb_k = []; n_orb = 0
    matrix = np.zeros((4 * L ** 2, 4 * L ** 2), dtype=np.int64) - 1
    cur_jastrow = 0
    dist = list(np.sort(np.unique(all_distances_rounded)))

    for i in range(4 * L ** 2):
        for j in range(4 * L ** 2):
            if i == j:
                continue

            oi = i % 2; oj = j % 2

            d_ij = all_distances_rounded[i, j]
            if d_ij > 0 + 1e-5:
                if oi == oj:
                    jastrow_idx = dist.index(d_ij) * 2
                else:
                    jastrow_idx = dist.index(d_ij) * 2 - 1
            else:
                jastrow_idx = 0  # then we know the orbitals are different

            matrix[i, j] = jastrow_idx
            orb_ij.append((i, j))
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


@jit(nopython=True)
def get_fft(N, twist_exp):
    #i, j = np.meshgrid(np.arange(N), np.arange(N))
    #A=np.multiply.outer(i.flatten(), i.flatten())
    #B=np.multiply.outer(j.flatten(), j.flatten())
    #omega = np.exp(-2*np.pi*1J/N)
    #W = np.power(omega, A+B)/N

    W = np.zeros((N ** 2, N ** 2), dtype=np.complex128)
    for kx in range(N):
        for ky in range(N):
            for x in range(N):
                for y in range(N):
                    W[x * N + y, kx * N + ky] = np.exp(2.0j * np.pi / N * kx * x + (2.0j * np.pi / N * ky - 1.0j * np.pi / N) * y)
    return np.kron(W, np.eye(4))

def get_fft_small(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    A=np.multiply.outer(i.flatten(), i.flatten())
    B=np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/N)
    W = np.power(omega, A+B)/N

    return np.kron(W, np.ones(4)[np.newaxis, ...])


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


    #np.save('./files_mvmc/tx_{:d}.npy'.format(config.Ls), tx)
    #np.save('./files_mvmc/ty_{:d}.npy'.format(config.Ls), ty)
    #np.save('./files_mvmc/C3z_{:d}.npy'.format(config.Ls), C3z)
    #np.save('./files_mvmc/C2y_{:d}.npy'.format(config.Ls), C2y)
    #np.save('./files_mvmc/valley_{:d}.npy'.format(config.Ls), valley)

    pref = 'real' if np.allclose(K0, K0.real) else 'imag'

    #np.save('./files_mvmc/K0_{:s}_{:d}.npy'.format(pref, config.Ls), K0)
    #np.save('./files_mvmc/distances_{:d}.npy'.format(config.Ls), config.all_distances)
    
    #np.save('H_edges_{:d}.npy'.format(config.Ls), H.edges_quadric)
    #exit(-1)


    path = os.path.join(config.workdir, 'imada_format_L_{:d}_Ne_{:d}_U_{:.3f}_mod_{:d}_periodic_{:b}_forcheck'.format(config.Ls, config.Ne, U, mod, periodic))
    os.makedirs(path, exist_ok=True)

    
    config.mu = -0.15416910392893438
    K0 = K0 - np.eye(K0.shape[0]) * config.mu
    twist = (0, 0.5)
    twist_exp = [np.exp(2 * np.pi * 1.0j * twist[0]), np.exp(2 * np.pi * 1.0j * twist[1])]
    K0_up = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = False)
    K0_down = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = True)


    assert np.allclose(K0_up, K0_up.conj().T)
    assert np.allclose(K0_down, K0_down.conj().T)

 
    ########### writing the K--matrix ##########
    f = open(os.path.join(path, 'trans.def'), 'w')

    f.write('========================\n')
    f.write('NTransfer      {:d}\n'.format(2 * np.sum(np.abs(config.K_0) > 1e-7)))
    f.write('========================\n')
    f.write('========i_j_s_tijs======\n')
    f.write('========================\n')
    assert np.allclose(K0, K0.T.conj())

    K0_trans = K0 * 1.0
    K0_trans = K0_trans[tx, ...]
    K0_trans = K0_trans[..., tx]

    assert np.allclose(K0_trans, K0)

    K0_trans = K0 * 1.0
    K0_trans = K0_trans[ty, ...]
    K0_trans = K0_trans[..., ty]

    assert np.allclose(K0_trans, K0)


    for i in range(K0.shape[0]):
        for j in range(K0.shape[1]):
            if i == j:
                continue
            if np.abs(K0[i, j]) > 1e-7:
                f.write('    {:d}     0     {:d}     0   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0_up[i, j]), np.imag(-K0_up[i, j])))  # why j, i instead of ij? think!
                f.write('    {:d}     1     {:d}     1   {:.6f}  {:.6f}\n'.format(i, j, np.real(-K0_down[i, j]), np.imag(-K0_down[i, j])))
                #K0[i, j] = K0[i, j].real + 100j if i > j else -100j
    f.close()

    assert np.allclose(models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = True).T, models.apply_TBC(config, twist_exp, deepcopy(K0).T, inverse = True))

    

    


    
    g = 0.01 * config.pairings_list_unwrapped[0]


    K0_downT = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = True).T
    K0_upT = models.apply_TBC(config, twist_exp, deepcopy(K0), inverse = False).T
    for i in range(K0_up.shape[0]):
        for j in range(K0_up.shape[1]):
            if (i + j) % 2 == 1 and np.abs(K0_up[i, j]) > 1e-12:
                print(i, j)
                exit(-1)

    swave = 0. * models.xy_to_chiral(pairings.combine_product_terms(config, pairings.twoorb_hex_all[1]), 'pairing', config, True)

    g = g + swave
    gap = models.apply_TBC(config, twist_exp, deepcopy(g), inverse = False)
    gapT = models.apply_TBC(config, twist_exp, deepcopy(g).T, inverse = True)
    #print(gap); exit(-1)

    
    print('energies = ', np.linalg.eigh(K0_upT)[0])
    print(np.diag(K0_upT))

    #exit(-1)
    print(np.min(np.abs(np.linalg.eig(gap)[0])) / 0.03, 'minimum gap mode')
    # assert np.allclose(gap, gap.T)
    #print(config.pairings_list_unwrapped[0])
    #exit(-1)


    #twist = (0.123, -1.23)
    #twist_exp = [np.exp(2.0j * np.pi * twist[0]), np.exp(2.0j * np.pi * twist[1])]



    fft = get_fft(config.Ls, twist_exp)
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
    
    gap_fft = fft.T.conj().dot(gap).dot(fft)
    gap_check = gap_fft.copy()
    for i in range(gap_check.shape[0] // 4):
        #print(i % 4, i // 4)
        print(np.abs(np.linalg.eig(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4])[0]), i)
        # print(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4])
        #assert np.allclose(gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4], gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4].conj().T)
        gap_check[i * 4:i * 4 + 4,i * 4:i * 4 + 4] = 0.0
    assert np.isclose(np.sum(np.abs(gap_check)), 0.0)

    #exit(-1)
    #exit(-1)
    # first basis
    u_up = []
    u_down = []

    v_up = []
    v_down = []


    '''
    for i in range(gap_check.shape[0] // 4):
        hk = np.zeros((8, 8), dtype=np.complex128)
        hk[:4, :4] = K0_fft_plus[i * 4:i * 4 + 4, i * 4:i * 4 + 4]
        hk[4:, 4:] = -K0_fft_plus[i * 4:i * 4 + 4, i * 4:i * 4 + 4]

        print(hk[:4, :4])
        print(-hk[4:, 4:])

        print()
        print()

        hk[:4, 4:] = gap_fft[i * 4:i * 4 + 4,i * 4:i * 4 + 4]
        hk[4:, :4] = hk[:4, 4:].conj().T

        assert np.allclose(hk, hk.conj().T)

        gamma = np.kron(np.array([[0, -1], [1, 0]]), np.eye(4))


        #print(hk)
        #print(gamma.dot(hk.conj()).dot(gamma))
        #assert np.allclose(hk, gamma.dot(hk.conj()).dot(gamma))

        en, eigvals = np.linalg.eigh(hk)
        assert np.allclose(en, np.sort(-en))

        u_up.append(eigvals[:4, 4:])
        v_up.append(eigvals[4:, 4:])

        print(en, i)
        print(fft[np.arange(0, 64, 4), i * 4])
        print()
        print()
        for e, level in zip(en, eigvals.T):
            level_anti = level * 0.0
            level_anti[:4] = -level[4:].conj()
            level_anti[4:] = level[:4].conj()
            e_anti = np.dot(level_anti.conj(), hk.dot(level_anti))
            #assert np.isclose(e, -e_anti)


        #print(eigvals)
        #print(en)
        #print(eigvals[:4, 4:])
        #print(-eigvals[4:, :4].conj())
        #assert np.allclose(eigvals[:4, 4:], -eigvals[4:, :4].conj())
        #assert np.allclose(eigvals[4:, :4], eigvals[:4, 4:].conj())

        #print(en)

    # second basis
    for i in range(gap_check.shape[0] // 4):
        hk = np.zeros((8, 8), dtype=np.complex128)
        hk[:4, :4] = K0_fft_minus[i * 4:i * 4 + 4,i * 4:i * 4 + 4]
        hk[4:, 4:] = -K0_fft_plus[i * 4:i * 4 + 4,i * 4:i * 4 + 4]
        hk[:4, 4:] = -gap_fft[i * 4:i * 4 + 4,i * 4:i * 4 + 4]
        hk[4:, :4] = hk[:4, 4:].conj().T

        assert np.allclose(hk, hk.conj().T)

        en, eigvals = np.linalg.eigh(hk)
        assert np.allclose(en, np.sort(-en))
        u_down.append(eigvals[:4, :4])
        v_down.append(eigvals[4:, :4])
        #print(en)


    u_up = np.array(u_up)  # (k, beta, alpha)
    v_up = np.array(v_up)
    u_down = np.array(u_down)
    v_down = np.array(v_down)

    print(u_up.shape, v_up.shape, u_down.shape, v_down.shape)

    denom = (v_up * u_down).sum(axis = 1)  # (k, alpha)

    i = 0
    for u, v in zip(u_up, v_up):
        print(np.abs(np.linalg.eig(u)[0]))#, np.linalg.eig(v)[0])
        print(u)
        print(np.linalg.eigh(K0_fft_plus[i * 4:i * 4 + 4, i * 4:i * 4 + 4])[0])
        print()
        i += 1
        #print(np.linalg.inv(v).conj())
    #fraction = np.einsum('kab,kga->kbg', np.linalg.inv(v_up.conj()), u_up.conj())
    fraction = np.einsum('kba,kag->kgb', v_up, np.linalg.inv(u_up))
    print(np.linalg.inv(u_up))
    #fraction = np.einsum('kba,kga->kabg', u_up, u_down)
    #fraction = np.einsum('kabg,ka->kbg', fraction, denom)
    print(fraction.shape)

    fraction = np.tile(fraction, (1, config.Ls ** 2, 1))
    fraction = np.tile(fraction, (1, 1, config.Ls ** 2))

    print(fraction.shape, fft.shape)

    fft_k = get_fft_small(config.Ls)  # FIXME: is this correct?
    '''




    # oldresult = np.einsum('kij,ki,kj->ij', fraction, fft_k.conj(), fft_k)


    # assert np.allclose(gap, gap.T)  # consider only singlets
    L = K0.shape[0]
    totalM = np.zeros((4 * L, 4 * L), dtype=np.complex128)
    totalM[:L, :L] = K0_up; totalM[L:2 * L, L:2 * L] = K0_down; totalM[2 * L:3 * L, 2 * L:3 * L] = -K0_upT; totalM[3 * L:, 3 * L:] = -K0_downT
    totalM[:L, 3 * L:] = gap; totalM[L: 2 * L, 2 * L:3 * L] = -gapT; totalM[2 * L: 3 * L, L:2 * L] = -gapT.conj().T; totalM[3 * L:, :L] = gap.conj().T;

    selected_idxs = np.concatenate([np.arange(0, L), np.arange(2 * L, 3 * L)])
    totalM_updown = totalM[:, selected_idxs]; totalM_updown = totalM_updown[selected_idxs, ...]

    #totalM_updown = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    #totalM_updown[:L, :L] = K0; totalM_updown[L:, L:] = -K0.T;
    #totalM_updown[:L, L:] = gap; totalM_updown[L:, :L] = gap.conj().T;

    selected_idxs = np.arange(L, 3 * L)
    totalM_downup = totalM[:, selected_idxs]; totalM_downup = totalM_downup[selected_idxs, ...]




    #totalM_downup = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    #totalM_downup[:L, :L] = K0; totalM_downup[L:, L:] = -K0.T;
    #totalM_downup[:L, L:] = -gap.T; totalM_downup[L:, :L] = -gap.conj();

    en_updown, W_updown = np.linalg.eigh(totalM_updown)


    en_downup, W_downup = np.linalg.eigh(totalM_downup)
    assert np.allclose(en_updown, np.sort(-en_updown))
    assert np.allclose(en_downup, np.sort(-en_downup))
    en_total, W = np.linalg.eigh(totalM)
    print(en_total)

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

    #state = np.array([2, 4, 6, 7, 10, 13, 14, 24, 29, 32, 33, 34, 35, 37, 38, 39, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 59, 60, 61, 63, \
    #                  65, 69, 70, 71, 72, 75, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 96, 98, 105, 106, 109, 110, 112, 114, 115, 118, 119, 120, 121, 125])
    state =  np.array([0, 1, 4, 5, 6, 9, 12, 13, 15, 17, 18, 21, 24, 26, 27, 30, 32, 36, 37, 38, 41, 42, 44, 46, 47, 49, 53, 55, 57, 58, 59, 62, 64, 65, 67, 68, 71, 76, 77, 78, 79, 82, 85, 86, 87, 90, 91, 92, 96, 98, 100, 101, 105, 106, 107, 108, 111, 114, 115, 117, 119, 124, 126, 127])
    state_ph = []
    for s in state:
        if s < 64:
            state_ph.append(s)
    for j in range(64, 128):
        if j not in state:
            state_ph.append(j)
    state = np.array(state_ph)
    #print(repr(state))
    #exit(-1)


    #print(Q)
    #print(V)
    #Z = Q.dot(np.linalg.inv(V))
    #Z = Z#.conj()

    #print(np.abs(V) > 1e-10)
    ##print( )
    #print(np.abs(np.linalg.inv(Q)) > 1e-10)


    #Z = (Q.dot(np.linalg.inv(V + 0. * np.eye(V.shape[0]))))
    #Z = (V.dot(np.linalg.inv(Q))).conj()
    #Z = np.linalg.inv(Z).conj()



    Z = (Q.dot(np.linalg.inv(V)))
    print('max U^{-1} = ', np.max(np.abs(np.linalg.inv(Q))))
    #exit(-1)
    #Z = Z.conj()

    np.save('Z.npy', Z)
    #print(np.abs(Z) > 1e-6)
    result = Z[Z.shape[0] // 2:, :Z.shape[0] // 2]

    #pfa1 = cpf.pfaffian(Q.conj().T.dot(V.conj()))

    ##print(result.shape, state.shape)
    #det = result[state[:32], ...]
    ##det = det[..., state[32:] - 64]
    #det = np.linalg.det(det)
    

    sites_up = state[:32]
    sites_down = state[32:] - 64


    res = result




    f_selected = res[sites_up, :]; f_selected = f_selected[:, sites_down]
    det_0 = np.linalg.det(f_selected)

    total_energy = 0.0 + 0.0j
    for j in range(config.Ls ** 2 * 4):
        for i in range(config.Ls ** 2 * 4):
            if i in sites_up or j not in sites_up:
                continue
            if np.abs(K0[i, j]) < 1e-7:
                continue
            new_sites_up = sites_up.copy()
            j_pos = np.where(new_sites_up == j)[0][0]
            new_sites_up = np.concatenate([new_sites_up[:j_pos], new_sites_up[j_pos + 1:], np.array([i])], axis = -1)

            f_selected = res[new_sites_up, :]; f_selected = f_selected[:, sites_down]
            det_new = np.linalg.det(f_selected)
            print(i, j, det_new / det_0 * (-1) ** j_pos * (-1))
            total_energy += det_new / det_0 * (-1) ** j_pos * (-1) * K0.T[i, j]
            #print(det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j], K0[i, j], total_energy)
    print('energy assessed within Slater determinants up', -total_energy)


    total_energy = 0.0 + 0.0j
    for j in range(config.Ls ** 2 * 4):
        for i in range(config.Ls ** 2 * 4):
            if i in sites_down or j not in sites_down:
                continue
            if np.abs(K0[i, j]) < 1e-7:
                continue
            new_sites_down = sites_down.copy()
            j_pos = np.where(new_sites_down == j)[0][0]
            new_sites_down = np.concatenate([new_sites_down[:j_pos], new_sites_down[j_pos + 1:], np.array([i])], axis = -1)

            f_selected = res[sites_up, :]; f_selected = f_selected[:, new_sites_down]
            det_new = np.linalg.det(f_selected)
            print(i, j, det_new / det_0 * (-1) ** j_pos * (-1))
            total_energy += det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j]
            # print(det_new / det_0 * (-1) ** j_pos * (-1) * K0[i, j], K0[i, j])
    print('energy assessed within Slater determinants down', -total_energy)


    
    # assert np.allclose(gap, gap.conj().T)


    # print(Z[Z.shape[0] // 2:, :Z.shape[0] // 2])
    # print(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T)

    print(config.pairings_list_names)
    Z = Z / np.abs(np.max(Z))
    print(np.sum(np.abs(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T)))
    print(np.sum(np.abs(np.real(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T))))
    print(np.sum(np.abs(np.imag(Z[Z.shape[0] // 2:, :Z.shape[0] // 2] + Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T))))
    assert np.allclose(Z[Z.shape[0] // 2:, :Z.shape[0] // 2], -Z[:Z.shape[0] // 2, Z.shape[0] // 2:].T)  # how does this condition look for triplet?
    assert np.allclose(Z[Z.shape[0] // 2:, Z.shape[0] // 2:], Z[Z.shape[0] // 2:, Z.shape[0] // 2:] * 0.0)  # indeed, there are no such terms even for triplet [cool]
    assert np.allclose(Z[:Z.shape[0] // 2, :Z.shape[0] // 2], Z[:Z.shape[0] // 2, :Z.shape[0] // 2] * 0.0)
    


    u = Q[Q.shape[0] // 2:, Q.shape[0] // 2:]



    result_trans = result.copy()
    result_trans = result_trans[tx, :]
    result_trans = result_trans[:, tx]

    #print(result[0, :])
    #assert np.isclose(np.sum(np.abs(result - result_trans)), 0.0)

    result_trans = result.copy()
    result_trans = result_trans[ty, :]
    result_trans = result_trans[:, ty]
    print(tx, np.sum(np.abs(result - result_trans)))
    print(ty, np.sum(np.abs(result - result_trans)))
    #print(result[0, :])
    print()
    #assert np.isclose(np.sum(np.abs(result - result_trans)), 0.0)

    vol = 4 * Ls ** 2
    orbital_idxs = -np.ones((vol, vol), dtype=np.int64)

    f_ij = result
    f_ij = f_ij / np.abs(np.max(f_ij))
    np.save('f_ij.npy', f_ij)

    '''
    energies, orbitals = np.linalg.eigh(K0_up)
    orbitals = orbitals.T
    orbitals = orbitals[energies < 0.]
    energies = energies[energies < 0.]
    orbitals = orbitals.T
    overlaps = orbitals.dot(orbitals.T)
    # f_ij = overlaps
    np.save('orbitals.npy', orbitals)
    np.save('f_ij.npy', f_ij)
    '''

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
    print('orbitals after enforcing APBCy remaining:', current_orb_idx)
    for i in range(current_orb_idx):
        values = f_ij.flatten()[orbital_idxs.flatten() == i]
        assert np.isclose(np.std(values - values.mean()), 0.0)



    if np.allclose(f_ij, f_ij.T):
        print('symmetric f_ij = f_ji (singlet): restricting su(2) parameters')

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
        print('total orbitals su(2) with APBCy', len(mapping))
        current_orb_idx = len(mapping)

    TRS = np.concatenate([[2 * i + 1, 2 * i] for i in range(vol // 2)])
    f_trs = f_ij[:, TRS]
    f_trs = f_trs[TRS, :]
    if np.allclose(f_trs, f_ij):
        print('f_ij = TRS f_ij: resticting TRS parameters')


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
        print('total orbitals su(2) with APBCy and TRS!', len(mapping) + 1)
        current_orb_idx = len(mapping)

    np.save('orbital_idxs.npy', orbital_idxs)
    '''
    zero_idxs = []
    nonzero_idxs = []
    for i in range(len(mapping)):
        values = f_ij.flatten()[orbital_idxs.flatten() == i]
        assert np.isclose(np.std(values - values.mean()), 0.0)
        if np.isclose(np.abs(values.mean()), 0.):
            zero_idxs.append(i)
        else:
            nonzero_idxs.append(i)

    nonzero_idxs = np.array(nonzero_idxs, dtype=np.int64)
    mapping = list(np.sort(nonzero_idxs))
    n_nonzero = len(nonzero_idxs)

    for i in range(64):
        for j in range(64):
            if orbital_idxs[i, j] in zero_idxs:
                orbital_idxs[i, j] = n_nonzero
            else:
                orbital_idxs[i, j] = mapping.index(orbital_idxs[i, j])

    print('total orbitals su(2) with APBCy and TRS, nonzero!', len(mapping) + 1)
    '''



    
    eig, _ = np.linalg.eigh(K0)
    print(2 * (np.sum(eig[eig < 0]) + config.mu * np.sum(eig < 0)))

    print(2 * np.trace(u.conj().T.dot(K0 + np.eye(K0.shape[0]) * config.mu).dot(u)))
    print(eig + config.mu)


    

    #print(Z)
    #exit(-1)

    '''
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
    '''


    ### trying now the normal state -- will it blend? ###
    '''
    energies, orbitals = np.linalg.eigh(K0_up)
    orbitals = orbitals.T
    orbitals = orbitals[energies < 0.]
    energies = energies[energies < 0.]
    orbitals = orbitals.T
    overlaps = orbitals.T.dot(orbitals)
    f_ij = overlaps
    
    for i in range(len(energies)):
        o = orbitals[:, i]
        assert np.allclose(K0.dot(o), o * energies[i])
        print(i, np.dot(o.conj().T, o[tx]), np.dot(o.conj().T, o[ty]), np.dot(o.conj().T, o[C3z]), np.dot(o.conj().T, o[C2y]), energies[i])

        K0realo = K0.real.dot(o)
        angle = np.dot(o, K0realo) / np.sqrt(np.dot(o, o) * np.dot(K0realo, K0realo))
    

    overlaps = orbitals.T.dot(orbitals)
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
    '''
    #if periodic:
    ##    f_ij = (orbitals.dot(orbitals.T.conj()))
    #else:
    #    f_ij = (orbitals.dot(orbitals.T))
    # f_ij = overlaps
    #for i in range(config.Ls ** 2 * 4):
    #    for j in range(config.Ls ** 2 * 4):
    #        if i != j:
    #            continue
    #        print('f[{:d}, {:d}] = {:10f} + I {:10f}'.format(i, j, f_ij[i, j].real, f_ij[i, j].imag))
    #print(f_ij)
    '''
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
    '''
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
    jastrow_ij, jastrow_k, n_jastrows, matrix_jastrows = get_jastrow_fromshift(config.Ls, np.around(np.array(config.all_distances), decimals = 5), dist_threshold=5.)

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



    f = open(os.path.join(path, 'InJastrow_{:d}.def'.format(Ls)), 'w')
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


    f = open(os.path.join(path, 'InOrbital.def'), 'w')
    f.write('======================\n')
    f.write('NOrbitalIdx  {:d}\n'.format(current_orb_idx))
    f.write('======================\n')
    f.write('== i_j_OrbitalIdx  ===\n')
    f.write('======================\n')
    for k in range(current_orb_idx):
        mask = (orbital_idxs == k)
        val = np.sum(f_ij * mask) / np.sum(mask)
        f.write('{:d} {:.20f}  {:.20f}\n'.format(k, val.real, val.imag))





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
    f.write('NVMCCalMode    1\n')
    f.write('--------------------\n')
    f.write('NDataIdxStart  1\n')
    f.write('NDataQtySmp    1\n')
    f.write('--------------------\n')
    f.write('Nsite          {:d}\n'.format(K0.shape[0]))
    f.write('Ncond          {:d}\n'.format(144 - 10 * 2))
    f.write('2Sz            0\n')
    f.write('NSPGaussLeg    1\n')
    f.write('NSPStot        0\n')
    f.write('NMPTrans       {:d}\n'.format(len(symmetries)))
    f.write('NSROptItrStep  1\n')
    f.write('NSROptItrSmp   40\n')
    f.write('DSROptRedCut   0.0010000000\n')
    f.write('DSROptStaDel   0.0200000000\n')
    f.write('DSROptStepDt   0.0000000000\n')
    f.write('NVMCWarmUp     0\n')
    f.write('NVMCInterval   1\n')
    f.write('NVMCSample     1\n')
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
