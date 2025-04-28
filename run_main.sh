#!/bin/bash
#SBATCH -J mypyjb          # Specify job name
#SBATCH -p compute         # Use partition shared
#SBATCH -N 1               # Specify number of nodes (1 for serial applications!)
#SBATCH -n 1               # Specify max. number of tasks to be invoked
#SBATCH -t 02:00:00        # Set a limit on the total run time
#SBATCH -A bb1005          # Charge resources on this project account
#SBATCH -o myjob.o%j       # File name for standard and error output
#SBATCH --mem-per-cpu=1920M

set -e

# Environment:
module load python3
ulimit -s # 204800
ulimit -c 0
#export OMP_STACKSIZE=900M

echo "Start python script execution at $(date)"

python -u main.py
#python -u main_trend_per_ice.py
#python -u plot_main_dict.py
