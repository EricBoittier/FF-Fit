# !/bin/bash
# Define the SLURM job options
job_name="my_job"
partition="long"
time_limit="01:00:00"
num_tasks="1"
cpus_per_task="48"
memory="4G"

# Define the arrays
#alphas=(0.0 0.1 0.01 0.001 0.0001 0.00001 0.5 1.0)
#alphas=(0.0001 0.00001 0.00001 0.5 )
alphas=(2.0 10.0 100.0)
l2s=(0.0 0.1 0.5 1.0 2.0 4.0)
n_factors=(6 8 10 12)
repeats=30
json="water.json"
fname="water"
# Loop over the arrays
for alpha in "${alphas[@]}"
do
    for l2 in "${l2s[@]}"
    do
        for n in "${n_factors[@]}"
        do
            for (( repeat=1; repeat<=repeats; repeat++ ))
            do
                # Define the output and error log filenames
                output_log="${fname}_output_${alpha}_${l2}_${n}_${repeat}.log"
                error_log="${fname}_error_${alpha}_${l2}_${n}_${repeat}.log"

                # Create the SLURM submission script for each job copy
                submission_script="#!/bin/bash
#SBATCH --job-name=${job_name}_${alpha}_${l2}_${n}_${repeat}
#SBATCH --partition=${partition}
#SBATCH --time=${time_limit}
#SBATCH --ntasks=${num_tasks}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${memory}
#SBATCH --output=${output_log}
#SBATCH --error=${error_log}

# Your actual SLURM job command here
# For example, let's echo the job ID
echo \"Job ID: \$SLURM_JOB_ID\"

conda activate p
cd /cluster/home/boittier/ff_energy/ff_energy/pydcm/tests/
python test_dcm.py --alpha ${alpha} --n_factor ${n} --l2 ${l2} --json ${json} --fname ${fname} test_N_repeats 



"

                # Write the submission script to a file
                script_file="submit_${alpha}_${l2}_${n}_${repeat}.sh"
                echo "$submission_script" > "$script_file"

                # Submit the SLURM job
                sbatch "$script_file"

                # Optional: Add a delay between job submissions (in seconds)
                sleep 1
            done
        done
    done
done

