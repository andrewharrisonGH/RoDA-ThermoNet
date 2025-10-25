#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Apr 15 2025 10:11:16 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_da_workflow"

# The project ID which this job should run under:
#SBATCH --account=""

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-20:00:00

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
PDB_ID="$1"
INPUT_PDB="./PDBs/$PDB_ID.pdb"
VARIANT_LIST="./Variants/${PDB_ID}_variants.csv"
OUT_DIR="./PDB_relaxed/$PDB_ID"

# Create output directory
mkdir -p "$OUT_DIR"

# Step 1 - Relax stating structure
path/to/relax.static.linuxgccrelease -in:file:s "$INPUT_PDB" -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false -out:path:all "$OUT_DIR"

# Step 2 - Simulate variants
python ./rosetta_relax.py --rosetta-bin path/to/relax.static.linuxgccrelease -l "$VARIANT_LIST" --base-dir ./PDB_relaxed

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s