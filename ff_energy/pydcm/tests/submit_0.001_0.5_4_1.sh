#!/bin/bash
#SBATCH --job-name=my_job_0.001_0.5_4_1
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=4G
#SBATCH --output=logs/dcm_output_0.001_0.5_4_1.log
#SBATCH --error=errors/dcm_error_0.001_0.5_4_1.log

# Your actual SLURM job command here
# For example, let's echo the job ID
echo "Job ID: $SLURM_JOB_ID"

conda activate p
cd /cluster/home/boittier/ff_energy/ff_energy/pydcm/tests/
python test_dcm.py --alpha 0.001 --n_factor 4 --l2 0.5 --json dcm.json --fname dcm test_N_repeats 




