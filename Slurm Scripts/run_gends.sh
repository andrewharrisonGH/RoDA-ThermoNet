#!/bin/bash

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_gends"

# The project ID which this job should run under:
#SBATCH --account="*insert_job_no*"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# The amount of memory in megabytes per node:
#SBATCH --mem=65536

# Use this email address:
#SBATCH --mail-user="*insert_email*"

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-12:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# The modules to load:
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate thermonet

# Run the job from the directory where it was launched (default)

# The job command(s):
python gends.py --input Q1744_direct.csv --output Q1744_tensorsi180 --pdb_dir ./PDB_relaxed  --rotations rotations.csv --boxsize 16 --voxelsize 1 --ncores 32

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s