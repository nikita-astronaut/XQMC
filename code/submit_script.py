import subprocess
import os
import sys

epsilon, xi, name, Ne = [x for x in sys.argv[5:]]
xi = float(xi)
epsilon = float(epsilon)

Ne = int(Ne)

def remove_dots(string):
    config_name = list(string)
    for i, s in enumerate(config_name):
        if s == '.' and ''.join(config_name[i:]) != '.py':
            config_name[i] = '-'
    return ''.join(config_name)


bare_config = open(sys.argv[1], 'r')
path_to_configs = sys.argv[2] + name + '/e_{:.2f}_xi_{:.2f}_Ne_{:d}/'.format(epsilon, xi, Ne)
path_to_sbatch = sys.argv[3]
path_to_logs = sys.argv[4] + '/' + name + '/e_{:.2f}_xi_{:.2f}_Ne_{:d}/'.format(epsilon, xi, Ne)
path_to_configs = remove_dots(path_to_configs)
path_to_logs = remove_dots(path_to_logs)
os.makedirs(path_to_configs, exist_ok=True)
os.makedirs(path_to_logs, exist_ok=True)



lines = [line for line in bare_config]
for i, line in enumerate(lines):
    if 'self.epsilon =' in line:
        lines[i] = '        self.epsilon = {:.2f}\n'.format(epsilon)
    if 'self.xi =' in line:
        lines[i] = '        self.xi = {:.2f}\n'.format(xi)
    if 'self.Ne = ' in line:
        lines[i] = '        self.Ne = {:d}\n'.format(Ne)
    if 'self.workdir =' in line:
        lines[i] = '        self.workdir = \'{:s}\'\n'.format(path_to_logs)

config_name = os.path.join(path_to_configs, 'config_{:s}_e_{:.2f}_xi_{:.2f}_Ne_{:d}.py'.format(name, epsilon, xi, Ne))
config_name = remove_dots(config_name)

f = open(config_name, 'w')
[f.write(line) for line in lines]
f.close()

sbatch_file = open(path_to_sbatch, 'r')
lines = [line for line in sbatch_file]

for i, line in enumerate(lines):
    if '#SBATCH -o' in line:
        lines[i] = '#SBATCH -o {:s}/output_{:s}_e_{:.2f}_xi_{:.2f}_Ne_{:d}.out\n'.format(path_to_logs, name, epsilon, xi, Ne)
    if '#SBATCH -e' in line:
        lines[i] = '#SBATCH -e {:s}/error_{:s}_e_{:.2f}_xi_{:.2f}_Ne_{:d}.err\n'.format(path_to_logs, name, epsilon, xi, Ne)
    if '#SBATCH --job-name' in line:
        lines[i] = '#SBATCH --job-name {:s}_e_{:.2f}_xi_{:.2f}_Ne_{:d}\n'.format(name, epsilon, xi, Ne)

    if 'python' in line:
        pieces = line.split()
        pieces[-1] = config_name + '\n'
        lines[i] = ' '.join(pieces)
sbatch_file.close()
f = open(path_to_sbatch, 'w')
[f.write(line) for line in lines]
f.close()

print(subprocess.run(['sbatch', path_to_sbatch]))
