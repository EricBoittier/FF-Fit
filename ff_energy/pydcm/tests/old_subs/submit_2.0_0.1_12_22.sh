#!/bin/bash
#SBATCH --job-name=my_job_2.0_0.1_12_22
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=4G
#SBATCH --output=logs/dcm_output_2.0_0.1_12_22.log
#SBATCH --error=errors/dcm_error_2.0_0.1_12_22.log

# Your actual SLURM job command here
# For example, let's echo the job ID
echo "Job ID: $SLURM_JOB_ID"

conda activate p
cd /cluster/home/boittier/ff_energy/ff_energy/pydcm/tests/
python test_dcm.py --alpha 2.0 --n_factor 12 --l2 0.1 --json dcm.json --fname dcm test_N_repeats 




