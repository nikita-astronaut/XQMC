import numpy as np
import models

def obtain_all_hoppings_Koshino_real(config, pairings, threshold=3):
    K_all = np.eye(config.total_dof // 2) * 0.0

    C3z = pairings.C3z_symmetry_map_chiral
    C2y = pairings.C2y_symmetry_map_chiral
    Tx = pairings.Tx_symmetry_map
    Ty = pairings.Ty_symmetry_map

    TRS = np.kron(np.eye(config.total_dof // 2 // 2), np.array([[0, 1], [1, 0]]))

    symmetries = [np.eye(config.total_dof // 2)]
    symmetries = symmetries + [symm.dot(C2y) for symm in symmetries]
    symmetries = symmetries + [symm.dot(C3z) for symm in symmetries] + [symm.dot(C3z).dot(C3z) for symm in symmetries]
    K_hopp = []

    for li in range(4):
        for lj in range(4):
            if (li + lj) % 2 == 1:
                continue

            for dx in range(config.Ls):
                for dy in range(config.Ls):
                    K_bare = K_all * 0.
                    if config.all_distances[li, lj + 4 * ((dx) % config.Ls) + 4 * config.Ls * ((dy) % config.Ls)] > threshold + 1e-4:
                        continue

                    for x in range(config.Ls):
                        for y in range(config.Ls):
                            idx_i = li + 4 * x + 4 * config.Ls * y
                            idx_j = lj + 4 * ((x + dx) % config.Ls) + 4 * config.Ls * ((y + dy) % config.Ls)

                            K_bare[idx_i, idx_j] = 1

                    K_bare = np.sum(np.array([symm.dot(K_bare.dot(symm.T.conj())) for symm in symmetries]), axis = 0)  # point-group symmetries
                    K_bare = K_bare + TRS.dot(K_bare.dot(TRS.T.conj())).conj()  # TRS


                    K_bare = K_bare + K_bare.conj().T
                    K_bare = np.abs(K_bare > 1e-7) * 1.0

                    #print(np.abs(np.trace(K_bare.conj().T.dot(K_all))), np.abs(np.trace(K_bare.conj().T.dot(K_bare))))
                    if np.abs(np.trace(K_bare.conj().T.dot(K_bare))) == 0:
                        print('WTF')
                    assert np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), 0) or \
                           np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), np.abs(np.trace(K_bare.conj().T.dot(K_bare))))

                    if np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), 0):
                        K_hopp.append(['hopp_re_{:d}_{:d}_{:d}_{:d}'.format(dx, dy, li, lj), K_bare])
                        K_all += K_bare
                        print(dx, dy, li, lj)
                        print(np.where(np.abs(K_bare[0, :]) > 1e-7)[0])
                        K_all = np.abs(K_all > 1e-7) * 1.0
                    #else:
                    #    print('REJECTED')
    return K_hopp



def obtain_all_hoppings_Koshino_complex(config, pairings, threshold=3):
    K_all = np.eye(config.total_dof // 2) * 0.0 + 0.0j

    C3z = pairings.C3z_symmetry_map_chiral
    C2y = pairings.C2y_symmetry_map_chiral
    Tx = pairings.Tx_symmetry_map
    Ty = pairings.Ty_symmetry_map

    TRS = np.kron(np.eye(config.total_dof // 2 // 2), np.array([[0, 1], [1, 0]]))

    symmetries = [np.eye(config.total_dof // 2)]
    symmetries = symmetries + [symm.dot(C2y) for symm in symmetries]
    symmetries = symmetries + [symm.dot(C3z) for symm in symmetries] + [symm.dot(C3z).dot(C3z) for symm in symmetries]
    K_hopp = []

    for li in range(4):
        for lj in range(4):
            if (li + lj) % 2 == 1:
                continue

            for dx in range(config.Ls):
                for dy in range(config.Ls):
                    K_bare = K_all * 0. + 0.0j
                    if config.all_distances[li, lj + 4 * ((dx) % config.Ls) + 4 * config.Ls * ((dy) % config.Ls)] > threshold + 1e-4:
                        continue

                    for x in range(config.Ls):
                        for y in range(config.Ls):
                            idx_i = li + 4 * x + 4 * config.Ls * y
                            idx_j = lj + 4 * ((x + dx) % config.Ls) + 4 * config.Ls * ((y + dy) % config.Ls)
                            K_bare[idx_i, idx_j] = 1.0j

                    K_bare = np.sum(np.array([symm.dot(K_bare.dot(symm.T.conj())) for symm in symmetries]), axis = 0)  # point-group symmetries
                    K_bare = K_bare + TRS.dot(K_bare.dot(TRS.T.conj())).conj()  # TRS


                    K_bare = K_bare + K_bare.conj().T
                    #K_bare = np.abs(K_bare > 1e-7) * 1.0
                    #print(np.unique(K_bare.imag)[np.abs(np.unique(K_bare.imag)) > 1e-8])
                    #continue

                    #print(np.abs(np.trace(K_bare.conj().T.dot(K_all))), np.abs(np.trace(K_bare.conj().T.dot(K_bare))))
                    if np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_bare))), 0):
                        continue

                    K_bare = K_bare / np.max(np.abs(np.imag(K_bare)))

                    #print(np.abs(np.trace(K_bare.conj().T.dot(K_all))), np.abs(np.trace(K_bare.conj().T.dot(K_bare))))
                    assert np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), 0) or \
                           np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), np.abs(np.trace(K_bare.conj().T.dot(K_bare))))

                    if np.isclose(np.abs(np.trace(K_bare.conj().T.dot(K_all))), 0):
                        K_hopp.append(['hopp_im_{:d}_{:d}_{:d}_{:d}'.format(dx, dy, li, lj), K_bare])
                        K_all += K_bare
                        print(dx, dy, li, lj)
                        #K_all = np.abs(K_all > 1e-7) * 1.0
                    #else:
                    #    print('REJECTED')
    return K_hopp
