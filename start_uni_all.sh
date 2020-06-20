#for n in 72 84 94 114 124 134 144 154 164 174 184 194 204 216
#for n in 110 112 114 116 118
for eps in 0.30; do
    for xi in 2.0 1.5 1.0 0.7 0.3 0.1; do
        for n in 136; do
            python3 ./code/submit_script.py /s/ls4/users/astrakhantsev/DQMC_TBG/code/config_vmc.py /s/ls4/users/astrakhantsev/DQMC_TBG/config_files/ /s/ls4/users/astrakhantsev/DQMC_TBG/launch_scripts/launch_generated.sh /s/ls4/users/astrakhantsev/DQMC_TBG/logs/ $eps $xi $1 $n
        done
    done
done
