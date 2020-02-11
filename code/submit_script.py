import subprocess
import os
import sys

U, V, J, mu, Ne, gaps = [x for x in sys.argv[6:]]
U = float(U)
V = float(V)
J = float(J)
mu = float(mu)

Ne = int(Ne)

bare_config = open(sys.argv[1], 'r')
path_to_configs = sys.argv[2]
path_to_sbatch = sys.argv[3]
path_to_logs = sys.argv[4]
name = sys.argv[5]

lines = [line for line in bare_config]
for i, line in enumerate(lines):
    if 'self.U =' in line:
        lines[i] = '        self.U = {:.2f}\n'.format(U)
    if 'self.V =' in line:
        lines[i] = '        self.V = {:.2f}\n'.format(V)
    if 'self.J =' in line:
        lines[i] = '        self.J = {:.2f}\n'.format(J)
    if 'self.mu =' in line:
        lines[i] = '        self.mu = {:.2f}\n'.format(mu)
    if 'self.Ne = ' in line:
        lines[i] = '        self.Ne = {:d}\n'.format(Ne)
    if 'self.pairings_list = ' in line:
        lines[i] = '        ' + gaps + '\n'

config_name = list(os.path.join(path_to_configs, 'config_{s}_U_{:.2f}_V_{:.2f}_J_{:.2f}_mu_{:.2f}_Ne_{:d}.py'.format(name, U, V, J, mu, Ne)))
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
        lines[i] = '#SBATCH -o {:s}/output_{s}_U_{:.2f}_V_{:.2f}_J_{:.2f}_mu_{:.2f}_Ne_{:d}.out\n'.format(path_to_logs, name, U, V, J, mu, Ne)
    if '#SBATCH -e' in line:
        lines[i] = '#SBATCH -e {:s}/error_{s}_U_{:.2f}_V_{:.2f}_J_{:.2f}_mu_{:.2f}_Ne_{:d}.err\n'.format(path_to_logs, name, U, V, J, mu, Ne)
    if '#SBATCH --job-name' in line:
        lines[i] = '#SBATCH --job-name U_{s}_{:.2f}_V_{:.2f}_J_{:.2f}_mu_{:.2f}_Ne_{:d}\n'.format(name, U, V, J, mu, Ne)

    if 'python' in line:
        pieces = line.split()
        pieces[-1] = config_name + '\n'
        lines[i] = ' '.join(pieces)
sbatch_file.close()
f = open(path_to_sbatch, 'w')
[f.write(line) for line in lines]
f.close()

print(subprocess.run(['sbatch', path_to_sbatch]))
