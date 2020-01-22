import subprocess
import os
import sys

U, V, J, fugacity = [float(x) for x in sys.argv[5:]]
bare_config = open(sys.argv[1], 'r')
path_to_configs = sys.argv[2]
path_to_sbatch = sys.argv[3]
path_to_logs = sys.argv[4]

lines = [line for line in bare_config]
for i, line in enumerate(lines):
    if 'self.U =' in line:
        lines[i] = '        self.U = np.array([{:.2f}])\n'.format(U)
    if 'self.V =' in line:
        lines[i] = '        self.V = np.array([{:.2f}])\n'.format(V)
    if 'self.J =' in line:
        lines[i] = '        self.J = np.array([{:.2f}])\n'.format(J)
    if 'self.fugacity =' in line:
        lines[i] = '        self.fugacity = np.array([{:.2f}])\n'.format(fugacity)

config_name = os.path.join(path_to_configs, 'config_U_{:.2f}_V_{:.2f}_J_{:.2f}_f_{:.2f}.py'.format(U, V, J, fugacity))
for i, s in enumerate(config_name):
    if s == '.' and config_name[i:] != '.py':
        config_name[i] = '-'

f = open(config_name, 'w')
[f.write(line) for line in lines]
f.close()

sbatch_file = open(path_to_sbatch, 'r')
lines = [line for line in sbatch_file]

for i, line in enumerate(lines):
    if '#SBATCH -o' in line:
        lines[i] = '#SBATCH -o {:s}/output_U_{:.2f}_V_{:.2f}_J_{:.2f}_f_{:.2f}.out\n'.format(U, V, J, fugacity)
    if '#SBATCH -e' in line:
        lines[i] = '#SBATCH -e {:s}/error_U_{:.2f}_V_{:.2f}_J_{:.2f}_f_{:.2f}.err\n'.format(U, V, J, fugacity)
    if '#SBATCH --job-name' in line:
        lines[i] = '#SBATCH --job-name U_{:.2f}_V_{:.2f}_J_{:.2f}_f_{:.2f}\n'.format(U, V, J, fugacity)

    if 'python' in line:
        pieces = line.split()
        pieces[-1] = config_name + '\n'
        lines[i] = ' '.join(pieces)
sbatch_file.close()
f = open(path_to_sbatch, 'w')
[f.write(line) for line in lines]
f.close()

print(subprocess.run(['sbatch', path_to_sbatch]))