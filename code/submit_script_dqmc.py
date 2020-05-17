import subprocess
import os
import sys

U, V, mu, Nt, Nc, offset = [x for x in sys.argv[5:]]
U = float(U)
V = float(V)
mu = float(mu)

Nt = int(Nt)
Nc = int(Nc)

path_to_configs = sys.argv[2]
path_to_sbatch = sys.argv[3]
path_to_logs = sys.argv[4]

bare_config = open(sys.argv[1], 'r')
lines = [line for line in bare_config]
for i, line in enumerate(lines):
    if 'U_in_t1 =' in line:
        lines[i] = 'U_in_t1 = np.array([{:.2f}])\n'.format(U)
    if 'V_in_t1 =' in line:
        lines[i] = 'V_in_t1 = np.array([{:.2f}])\n'.format(V)
    if 'self.mu =' in line:
        lines[i] = '        self.mu = np.array([{:.2f}])\n'.format(mu)
    if 'self.Nt = ' in line:
        lines[i] = '        self.Nt = np.array([{:d}])\n'.format(Nt)

config_name = list(os.path.join(path_to_configs, 'config_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}_o_{:d}.py'.format(U, V, mu, Nt, int(offset))))
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
        lines[i] = '#SBATCH -o {:s}/output_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}.out\n'.format(path_to_logs, U, V, mu, Nt)
    if '#SBATCH -e' in line:
        lines[i] = '#SBATCH -e {:s}/error_U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}.err\n'.format(path_to_logs, U, V, mu, Nt)
    if '#SBATCH --job-name' in line:
        lines[i] = '#SBATCH --job-name U_{:.2f}_V_{:.2f}_mu_{:.2f}_Nt_{:d}\n'.format(U, V, mu, Nt)
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
