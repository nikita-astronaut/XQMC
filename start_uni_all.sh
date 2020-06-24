#for n in 72 84 94 114 124 134 144 154 164 174 184 194 204 216
#for n in 110 112 114 116 118
path_logs="/gpfs/scratch/userexternal/nastrakh/"
path="/galileo/home/userexternal/nastrakh/XQMC/"
for eps in 3.00; do
    for xi in 2.0 1.5 1.0 0.7 0.3 0.1; do
        for n in 124 128 132 136 152 156 160 164; do
            python3 ./code/submit_script.py $path/code/config_vmc.py $path/config_files/ $path/launch_scripts/launch_generated.sh $path_logs/logs/ $eps $xi $1 $n
        done
    done
done
