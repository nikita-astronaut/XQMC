import numpy as np
import pandas as pd
import os
import sys

def partner(name):
    if name[1] == 'j':
        return name[0] + name[2:]
    else:
        return name[0] + 'j' + name[1:]
def save_reader(path):
    file = open(path, 'r')
    lines = []
    for i, line in enumerate(file):
        if i > 0 and 'sweep' in line:
            continue
        lines.append(line)
    if len(lines) == 0:
        return None
    path = path + 'c'
    file.close()
    file = open(path, 'w')
    [file.write(line) for line in lines]
    file.close()
    

    return pd.read_csv(path,sep = "\s+", skipinitialspace=True,  index_col = False)

E_dict = {
    '(S_0)x(S_1)x(δ)': '(S_0)x(S_2)x(δ)',
    '(S_z)x(S_1)x(δ)': '(S_z)x(S_2)x(δ)',
    '(S_x)x(S_1)x(v_1)': '(S_x)x(S_2)x(v_1)',
    '(S_x)x(S_0&S_x)x(v_2)': '(S_x)x(S_0&S_x)x(v_3)',
    '(iS_y)x(iS_y&S_y)x(v_2)': '(iS_y)x(iS_y&S_y)x(v_3)',
    '(S_x)x(S_1)x(v_3)': '(S_x)x(S_2)x(v_2)',
    '(S_0)x(S_1)x(u_1)': '(S_0)x(S_2)x(u_1)',
    '(S_0)x(S_0&S_x)x(u_2)' : '(S_0)x(S_0&S_x)x(u_3)',
    '(iS_y)x(S_1)x(v_1)' : '(iS_y)x(S_2)x(v_1)',
    '(iS_y)x(S_0&S_x)x(v_2)' : '(iS_y)x(S_0&S_x)x(v_3)',
    '(S_0)x(S_1)x(u_3)' : '(S_0)x(S_2)x(u_2)', # ??????
    '(S_z)x(iS_y&S_y)x(u_2)' : '(S_z)x(iS_y&S_y)x(u_3)',
    '(S_x)x(iS_y)x(v_2)' : '(S_x)x(iS_y)x(v_3)',
    '(iS_y)x(S_1)x(v_3)' : '(iS_y)x(S_1)x(v_2)',
    '(S_z)x(S_1)x(u_1)' : '(S_z)x(S_2)x(u_1)',
    '(S_z)x(S_0&S_x)x(u_2)': '(S_z)x(S_0&S_x)x(u_3)',
    '(S_0)x(iS_y)x(u_2)': '(S_0)x(iS_y)x(u_3)',
    '(S_z)x(S_1)x(u_3)' : '(S_z)x(S_2)x(u_2)'
}
E_list_select = E_dict.keys()

def parse_chi(steps, chis):
    return np.array(chis)
    L = steps[1] - steps[0]
    zetas = []

    for i in range(len(steps) - 1):
        zetas.append(chis[i + 1] * (i + 1) - chis[i] * i)
    return np.array(zetas)

# lenghts = np.array([2, 2, 2 * 2, 4, 4, 8 * 2])
lenghts = np.array([1, 1, 1, 0, 0, 2])


boundaries = [int(np.sum(lenghts[:i])) for i in range(len(lenghts) + 1)]
print(boundaries)
avg_ratio = 0.0 + 0.0j
n = 0
def form_chi_matrix(df):
    names = []
    for name in df.columns.values:
        if '_chi_vertex' in name:
            names = names + [name[:-16].split('*')[0]]
    names = np.array(names)
    _, idx = np.unique(names, return_index=True)
    
    names = names[np.sort(idx)]
    chi_matrix = np.zeros((len(df), len(names), len(names))) + 0.0j
    dchi_matrix = np.zeros((len(names), len(names)))


    for idx_a, a_name in enumerate(names):
        for idx_b, b_name in enumerate(names):
            vals_real = df[a_name + '*' + b_name + '_chi_vertex_real'].values
            vals_imag = df[a_name + '*' + b_name + '_chi_vertex_imag'].values
            chi_matrix[:, idx_a, idx_b] = vals_real + 1.0j * vals_imag
            dchi_matrix[idx_a, idx_b] = np.real(np.mean(vals_real) / np.sqrt(len(vals_real) - 1))
    print(len(vals_real), 'measurements!!')
    return chi_matrix, dchi_matrix, names

def get_jackknife_max_eig(allmatrix, irrep):
    global avg_ratio, n
    matrix = 1.0 * allmatrix[:, irrep, :]
    matrix = matrix[:, :, irrep]
    
    means_slide = []
    for i in range(matrix.shape[0]):
        means_slide.append(np.sum(np.abs(matrix[:i].mean(axis = 0).conj() - matrix[:i].mean(axis = 0))))
    plt.plot(np.arange(matrix.shape[0]), means_slide)
    plt.show()
    energies = []
    for i in range(matrix.shape[0]):
        jmatrix = np.concatenate([matrix[:i], matrix[i + 1:]], axis=0).mean(axis = 0)
        jmatrix = 0.5 * (jmatrix + jmatrix.conj().T)
        
        E, U = np.linalg.eigh(jmatrix)
        energies.append(E.max())
    n = len(energies)
    energies = np.array(energies)
    E_total = np.linalg.eigh(matrix.mean(axis = 0))[0].max()
    
    E, U = np.linalg.eigh((matrix.mean(axis = 0).T.conj() + matrix.mean(axis = 0)) / 2.)
    if U.shape[0] == 2:
        avg_ratio += np.abs(U[:, np.argmax(E)][0] / U[:, np.argmax(E)][1])
        print(matrix.mean(axis = 0))
        n += 1
        print('attention')
    print(avg_ratio / n)
    print(U[:, np.argmax(E)], np.max(E), ((matrix.mean(axis = 0).T.conj() + matrix.mean(axis = 0)) / 2.)[0, 0])
    return energies.mean(), np.sqrt((n - 1.) / n * np.sum((energies - E_total) ** 2))

def get_jackknife_1d(data):
    means = []
    for i in range(data.shape[0]):
        jmatrix = np.concatenate([data[:i], data[i + 1:]], axis=0).mean(axis = 0)
        means.append(jmatrix)

    n = len(means)
    means = np.array(means)
    return means.mean(), np.sqrt((n - 1.) / n * np.sum(np.abs(means - np.mean(means)) ** 2))

seen_prefixes = []
max_chi_dict = {}


def plot_smth_whole_dirs(master_dir):
    df_gaps = pd.DataFrame(columns = ['chi_name', 'chi_val', 'chi_err', 'Nt', 'mu'])
    df_gaps_full = pd.DataFrame(columns = ['gap_name_bra', 'gap_name_ket', 'chi_val', 'chi_err', 'Nt', 'mu'])
    df_Sq0 = pd.DataFrame(columns = ['name', 'Sq0_val', 'Sq0_err', 'Nt', 'mu'])
    df_cl = pd.DataFrame(columns = ['name', 'cl_val', 'cl_err', 'Nt', 'mu'])
    df_irreps = pd.DataFrame(columns = ['irreps_name', 'irreps_val', 'irreps_err', 'Nt', 'mu'])
    df_orders = pd.DataFrame(columns = ['order_name', 'order_val', 'order_err', 'Nt', 'mu'])
    df_general = pd.DataFrame(columns = ['Nt', 'mu', 'rho', 'sign'])
    
    

    for root, subdirs, files in os.walk(master_dir):
        if '_c_' not in root:
            continue
        if root[-2] == '_' and root[:-4] in seen_prefixes:
            continue
        if root[-3] == '_' and root[:-5] in seen_prefixes:
            continue
        if root[-3] == '_':
            seen_prefixes.append(root[:-5])
        else:
            seen_prefixes.append(root[:-4])
        chi_vertex_accumulated = []#np.empty((0, 32, 32))
        chi_total_accumulated = []#np.empty((0, 32, 32))
        chi_vertex_accumulated_max = []#np.empty((0, 32, 32))
        chi_total_accumulated_max = []#np.empty((0, 32, 32))
        
        Sq0_accumulated = []#np.empty((0, 32))
        Sq0_accumulated_max = []#np.empty((0, 32))
        
        corr_len_accumulated = []#np.empty((0, 32))
        corr_len_accumulated_max = []#np.empty((0, 32))
        
        order_accumulated = []#np.empty((0, 10, 6, 6))
        order_accumulated_max = []#np.empty((0, 10, 6, 6))
        gap_names = None
        orders_names = None
        print(root, flush=True)
        info = root.split('_')
        mu = float(info[-5])
        Nt = int(info[-3])
        
        for c in range(100):
            if c == 0:
                df_gen = save_reader(os.path.join(root, 'general_log.dat'))
                if df_gen is not None:
                    mean_sign = df_gen['⟨sign_obs_l⟩'].values.mean()
                    density = df_gen['⟨density⟩'].values.mean() / mean_sign
                    df_general.loc[len(df_general)] = [Nt, mu, density, mean_sign]  # general information abour the launch
            chi_vertex_accumulated_fixedc = np.empty((0, 56, 56))
            ids = []
            rootc = root[:-1] + '{:d}'.format(c) if root[-2] == '_' else root[:-2] + '{:d}'.format(c)
            if not os.path.isdir(rootc):
                break
            
            ### find last entry id for this c -- it contains the mean ###
            for _, _, files in os.walk(rootc):
                for file in files:
                    if 'chi_vertex_' in file and 'final' not in file:
                        ids.append(int(file.split('_')[-1][:-4]))

            if len(ids) == 0:
                continue

            idx_max = np.max(ids)
            #if idx_max < 6000:
            #    continue
            print(idx_max)
            ids = np.sort(ids)            
            
            def accumulate(ids, name, accumulated, accumulated_max):
                for n in range(len(ids) - 1):
                    file_next = os.path.join(rootc, '{:s}_{:d}.npy'.format(name, ids[n + 1]))
                    file_prev = os.path.join(rootc, '{:s}_{:d}.npy'.format(name, ids[n]))
                    try:
                        data_next = np.load(file_next)
                        data_prev = np.load(file_prev)
                        if np.all(~np.isnan(data_next)) and np.all(~np.isnan(data_prev)):
                            accumulated.append(data_next * (n + 1) - data_prev * n)  # FIXME: if I relaunch the orders -- this fails
                            if n == len(ids) - 2:
                                accumulated_max.append(data_next)
                    except:
                        pass
                return accumulated, accumulated_max
            order_accumulated, order_accumulated_max = accumulate(ids, 'order', order_accumulated, order_accumulated_max)
            Sq0_accumulated, Sq0_accumulated_max = accumulate(ids, 'Sq0', Sq0_accumulated, Sq0_accumulated_max)
            corr_len_accumulated, corr_len_accumulated_max = accumulate(ids, 'corr_lengths', corr_len_accumulated, corr_len_accumulated_max)
            chi_vertex_accumulated, chi_vertex_accumulated_max = accumulate(ids, 'chi_vertex', chi_vertex_accumulated, chi_vertex_accumulated_max)
            chi_total_accumulated, chi_total_accumulated_max = accumulate(ids, 'chi_total', chi_total_accumulated, chi_total_accumulated_max)
            name = os.path.join(rootc, 'gap_names.npy')
            gap_names = np.load(name)
            name = os.path.join(rootc, 'orders_names.npy')
            orders_names = np.load(name)

        chi_total_accumulated = np.array(chi_total_accumulated)
        chi_total_accumulated_max = np.array(chi_total_accumulated_max)
        chi_vertex_accumulated = np.array(chi_vertex_accumulated)
        chi_vertex_accumulated_max = np.array(chi_vertex_accumulated_max)
        corr_len_accumulated = np.array(corr_len_accumulated)
        corr_len_accumulated_max = np.array(corr_len_accumulated_max)
        Sq0_accumulated = np.array(Sq0_accumulated)
        Sq0_accumulated_max = np.array(Sq0_accumulated_max)
        order_accumulated = np.array(order_accumulated)
        order_accumulated_max = np.array(order_accumulated_max)

        if gap_names is None:
            continue
        print(len(gap_names))
        
        if len(chi_vertex_accumulated) < 1:
            continue

        for n, gap_name_n in enumerate(gap_names):
            for m, gap_name_m in enumerate(gap_names):
                corr_full_max = chi_vertex_accumulated_max[:, n, m]
                corr_full = chi_vertex_accumulated[:, n, m]
                if len(corr_full) > 0:
                    c, _ = get_jackknife_1d(corr_full_max)
                    _, e = get_jackknife_1d(corr_full)
                    df_gaps_full.loc[len(df_gaps_full)] = [gap_name_n, gap_name_m, c, e.real, Nt, mu]

        for n, gap_name in enumerate(gap_names):
            corr_diag_max = chi_vertex_accumulated_max[:, n, n]
            corr_diag = chi_vertex_accumulated[:, n, n]
            max_chi_dict[gap_name + str(Nt)] = chi_vertex_accumulated_max[:, n, n]
            if len(corr_diag) > 0:
                c, _ = get_jackknife_1d(corr_diag_max)
                _, e = get_jackknife_1d(corr_diag)
                df_gaps.loc[len(df_gaps)] = [gap_name, c.real, e.real, Nt, mu]
            Sq0 = Sq0_accumulated_max[:, n] # TODO
            if len(Sq0) > 0:
                
                c, e = get_jackknife_1d(Sq0)
                df_Sq0.loc[len(df_gaps)] = [gap_name, c.real, e.real, Nt, mu]
            
            cl = corr_len_accumulated_max[:, n] # TODO
            c, e = get_jackknife_1d(cl)
            df_cl.loc[len(df_gaps)] = [gap_name, c.real, e.real, Nt, mu]
    return df_gaps_full, df_gaps, df_Sq0, df_cl, df_irreps, df_orders, df_general


df_gaps_full, df_gaps, df_Sq0, df_cl, df_irreps, df_orders, df_general = plot_smth_whole_dirs(sys.argv[1])

for data_dir in sys.argv[2:-1]:
    df_gaps_fulld, df_gapsd, df_Sq0d, df_cld, df_irrepsd, df_ordersd, df_generald = plot_smth_whole_dirs(data_dir)
    df_gaps_full = pd.concat([df_gaps_full, df_gaps_fulld])
    df_gaps = pd.concat([df_gaps, df_gapsd])
    df_Sq0 = pd.concat([df_Sq0, df_Sq0d])
    df_cl = pd.concat([df_cl, df_cld])
    df_irreps = pd.concat([df_irreps, df_irrepsd])
    df_orders = pd.concat([df_orders, df_ordersd])
    df_general = pd.concat([df_general, df_generald])

df_gaps.to_csv(os.path.join(sys.argv[-1], 'df_gaps.csv'))
df_gaps.to_csv(os.path.join(sys.argv[-1], 'df_gaps_full.csv'))
df_Sq0.to_csv(os.path.join(sys.argv[-1], 'df_Sq0.csv'))
df_cl.to_csv(os.path.join(sys.argv[-1], 'df_cl.csv'))
df_irreps.to_csv(os.path.join(sys.argv[-1], 'df_irreps.csv'))
df_orders.to_csv(os.path.join(sys.argv[-1], 'df_orders.csv'))
df_general.to_csv(os.path.join(sys.argv[-1], 'df_general.csv'))
