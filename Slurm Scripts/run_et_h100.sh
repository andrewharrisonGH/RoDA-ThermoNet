#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpu-h100

# Number of nodes
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_ensemble_train"

# The project ID which this job should run under:
#SBATCH --account="""

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per node:
#SBATCH --mem=65536

# Use this email address:
#SBATCH --mail-user=

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=2-00:00:00

#SBATCH --array=1-10
#SBATCH --output=logs/ens_%A_%a.out
#SBATCH --error=logs/ens_%A_%a.err

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
# Run ensemble training (each task trains one member)
MEMBER_IDX=$SLURM_ARRAY_TASK_ID

srun python train_ensemble.py \
    --direct_features Q1744_pptensors_fwd.npy \
    --inverse_features Q1744_pptensors_rev.npy \
    --direct_targets Q1744_tensors_fwd_ddg.txt \
    --inverse_targets Q1744_tensors_rev_ddg.txt \
    --epochs 200 \
    --prefix 0TN \
    --member $MEMBER_IDX \
    --k 10
    

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s