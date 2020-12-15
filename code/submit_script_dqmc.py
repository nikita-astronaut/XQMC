import subprocess
import os
import sys

U, V, name, mu, Nt, Nc, offset, Ls = [x for x in sys.argv[5:]]
U = float(U)
V = float(V)
mu = float(mu)
offset = int(offset)

Nt = int(Nt)
Nc = int(Nc)
offset = int(offset)
Ls = int(Ls)

path_to_configs = list(sys.argv[2])
for i, s in enumerate(path_to_configs):
    if s == '.' and ''.join(path_to_configs[i:]) != '.py':
        path_to_configs[i] = '-'
path_to_configs = ''.join(path_to_configs)

os.makedirs(path_to_configs, exist_ok=True)

path_to_sbatch = sys.argv[3]
path_to_logs = list(sys.argv[4])

for i, s in enumerate(path_to_logs):
    if s == '.' and ''.join(path_to_logs[i:]) != '.py':
        path_to_logs[i] = '-'
path_to_logs = ''.join(path_to_logs)

os.makedirs(path_to_logs, exist_ok=True)



bare_config = open(sys.argv[1], 'r')
lines = [line for line in bare_config]


for i, line in enumerate(lines):
    if 'self.Ls = ' in line:
        lines[i] = '        self.Ls = {:d}\n'.format(Ls)
    if 'U_in_t1 =' in line:
        lines[i] = 'U_in_t1 = np.array([{:.2f}])\n'.format(U)
    if 'V_in_t1 =' in line:
        lines[i] = 'V_in_t1 = np.array([{:.2f}])\n'.format(V)
    if 'self.mu =' in line:
        lines[i] = '        self.mu = np.array([{:.2f}])\n'.format(mu)
    if 'self.Nt = ' in line:
        lines[i] = '        self.Nt = np.array([{:d}])\n'.format(Nt)
    if 'self.offset = ' in line:
        lines[i] = '        self.offset = {:d}\n'.format(offset)
    if 'self.workdir = ' in line:
        lines[i] = '        self.workdir = \'/s/ls4/users/astrakhantsev/DQMC_TBG/logs_dqmc/{:s}/\'\n'.format(name)
    if 'self.workdir_heavy = ' in line:
        lines[i] = '        self.workdir_heavy = \'/s/ls4/users/astrakhantsev/DQMC_TBG/logs_dqmc/{:s}/\'\n'.format(name)

config_name = list(os.path.join(path_to_configs, 'config_{:d}_{:d}_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_o_{:d}.py'.format(Ls, Ls, U, V, mu, Nt, int(offset))))

for i, s in enumerate(config_name):
    if s == '.' and ''.join(config_name[i:]) != '.py':
        config_name[i] = '-'
config_name = ''.join(config_name)

f = open(config_name, 'w')
[f.write(line) for line in lines]
f.close()

sbatch_file = open(path_to_sbatch, 'r')
lines = [line for line in sbatch_file]

for i, line in enumerate(lines):
    if '#SBATCH -o' in line:
        lines[i] = '#SBATCH -o {:s}/output_{:d}x{:d}_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}.out\n'.format(path_to_logs, Ls, Ls, U, V, mu, Nt)
    if '#SBATCH -e' in line:
        lines[i] = '#SBATCH -e {:s}/error_{:d}x{:d}_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}.err\n'.format(path_to_logs, Ls, Ls, U, V, mu, Nt)
    if '#SBATCH --job-name' in line:
        lines[i] = '#SBATCH --job-name {:d}x{:d}_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}\n'.format(Ls, Ls, U, V, mu, Nt)
    if '#SBATCH --ntasks' in line:
        lines[i] = '#SBATCH --ntasks={:d}\n'.format(Nc)

    if 'mpirun' in line:
        pieces = line.split()
        pieces[2] = str(Nc)
        pieces[-1] = config_name + '\n'
        lines[i] = ' '.join(pieces)
sbatch_file.close()
f = open(path_to_sbatch, 'w')
[f.write(line) for line in lines]
f.close()

print(subprocess.run(['sbatch', path_to_sbatch]))
