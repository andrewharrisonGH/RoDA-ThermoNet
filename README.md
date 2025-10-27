# RoDA-ThermoNet
An end-to-end workflow as part of a MC-SCIBIF Research Project to apply rotational data augmentation (RoDA) to [ThermoNet](https://github.com/gersteinlab/ThermoNet) (Li et al., 2020).
ThermoNet is a computational tool for predicting protein thermostability change (ΔΔG) due to missense mutations. Its ensemble 3D-CNN architecture learns from voxelised PDB environments, and has shown promise in accurate prediction from solely structure-derived features. RoDA for ThermoNet seeks to expand the number of trainable tensors by rotating PDB structures, and preserving the scarce experimental stability data labels. 

Here we detail instructions on how to execute the developed workflow to expand training datasets for ThermoNet. Key optimisations have been made from legacy code to improve space and time costs in generating tensors for training and prediction.

## Contents
1. [Installation](#installation)
2. [Use RoDA-ThermoNet](#use-roda-thermonet)
3. [Example](#example)
    - [Training](#training)
    - [Testing](#testing)

## Project Structure

Below is an overview of the repository structure and purpose of each main component.

```plaintext
RoDA-ThermoNet
|
├── Datasets/                       # Formatted datasets for RoDA-ThermoNet
|   ├── Q1744_direct.csv            # Original training dataset for ThermoNet
|   ├── s669_ref.csv                # Common benchmarking dataset across literature
|   └── ssym_ref.csv                # Original testing dataset for ThermoNet
|
├── Models/                         # Previously Trained Models
|   ├── models_0/                   # 10 "No Augmentation" ThermoNet Ensembles
|   ├── models_60/                  # 10 incremental 60° (x,y) RoDa-ThermoNet Ensembles
|   ├── models_72/                  # 10 incremental 72° (x,y) RoDa-ThermoNet Ensembles
|   ├── models_90/                  # 10 incremental 90° (x,y) RoDa-ThermoNet Ensembles
|   ├── models_120/                 # 10 incremental 120° (x,y) RoDa-ThermoNet Ensembles
|   ├── models_180/                 # 10 incremental 180° (x,y) RoDa-ThermoNet Ensembles
|   └── ThermoNet_Model/            # Original ThermoNet Ensemble
|
├── Slurm Scripts/                  # Example Slurm Script for job Submission
|
├── utils/                          # Utility scripts for ThermoNet
|
├── amino_acids.py                  # Utility script for clean_pdb.py
├── clean_pdb.py                    # PDB pre-processing script for Rosetta
├── environment.yaml                # Conda environment settings
├── gends.py                        # Rotated Tensor generation script
├── get_pdbs.py                     # Fetch and clean dataset PDBs script
├── LICENSE                         # License information
├── predict.py                      # Legacy predict on model script
├── rosetta_relax.py                # Legacy variant PDB creator script
├── rotations.csv                   # Applied rotations list
├── sanity_check.py                 # Fetched PDBs align w/ dataset script
├── split_variants.py               # Separate dataset by unique PDB script
└── train_ensemble.py               # 3D-CNN model trainer script


## Installation

First, setup the environment with required software.

### Clone RoDA-ThermoNet
Clone RoDA-ThermoNet locally.
```bash
git clone https://github.com/andrewharrisonGH/RoDA-ThermoNet.git
```

### Requirements
ThermoNet was built for Linux platforms, thus, RoDA-ThermoNet has only been tested on Linux platforms. To use RoDA-ThermoNet, you will need to install the following third-party software:
  * Rosetta 3.10. Rosetta is a multi-purpose macromolecular modeling suite that is used in ThermoNet for creating and refining protein structures. You can get Rosetta from the Rosetta Commons website: https://www.rosettacommons.org/software
1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.10 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

### Conda environment
```bash
conda env create --name thermonet --file environment.yml
conda activate thermonet
```
The above commands will create a conda environment and install all dependencies for RoDa-ThermoNet tensor creation, model training (CPU-only) and ΔΔG prediction. For GPU-enabled model training and prediction, your HPC system may already have an available environment module (e.g. `load module TensorFlow/2.15.1-CUDA-12.2.0-Python-3.11.3`) ensure that it has:
- tensorflow==2.15.1
- keras==2.15.0
- cudatoolkit==12.2
- cudnn==8.9

## Use RoDA-ThermoNet
Note: this workflow is optimised for use on a HPC cluster using Slurm Workload Manager.
The whole of RoDA-ThermoNet input begins with two CSV files: VARIANT_LIST CSV (e.g. `Q1744_direct.csv`), and a ROTATIONS_LIST CSV (e.g. `rotations.csv`).
Ensure VARIANT_LIST is formatted with `pdb_id,pos,wild_type,mutant,ddg` and ROTATION_LIST with `x_rot,y_rot,z_rot`, example files are found in this repo.

### Get PDBs
Run the following command to download and clean each unique wild-type PDB for your dataset, depositing all downloaded selected PDB chains into `PDBs/`
```bash
python get_pdbs.py --csv_file VARIANT_LIST
```

### Sanity check dataset with PDBs
Before moving onto relaxation ensure wild-type residue and positions for each PDB align with VARIANT_LIST by running.
```bash
python sanity_check.py VARIANT_LIST
```

### Split variants for parallel relaxation
Run the following commands to generate separate CSVs for each unique PDB which will be passed into individual Slurm submission relaxation and mutation jobs.
```bash
mkdir Variants
python split_variants.py --input_csv VARIANT_LIST --output_dir ./Variants
```

### Generate individual job submission commands
Run the following commands to submit separate Slurm jobs for generating relaxed and mutated structures with `da_workflow.sh` (insert preffered partition, project ID, and path/to/relax.static.linuxgccrelease binary), depositing into individualised pdb_id subfolders under `./PDB_relaxed`.
```bash
mkdir log
for i in ./PDBs/*.pdb; do 
filename=$(basename "$i" .pdb) 
echo "sbatch -o log/${filename}.out -e log/${filename}.err da_workflow.sh $filename" 
done > run_rdawf_slurm.sh
cat run_rdawf_slurm.sh | bash
```
Noting `da_workflow` calls the legacy Rosetta relax protocol
```bash
path/to/relax.static.linuxgccrelease -in:file:s "$INPUT_PDB" -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false -out:path:all "$OUT_DIR"
```
and mutation protocol
```bash
python ./rosetta_relax.py --rosetta-bin path/to/relax.static.linuxgccrelease -l "$VARIANT_LIST" --base-dir ./PDB_relaxed
```

### Rotation and Tensor Generation
Run the following commands, with an appropriate number of allocated CPU resources (I have preffered `n=32`), to generate direct and reverse tensors from the relaxed PDB structures that undergo the rotational augmentations specified in ROTATIONS (example separate slurm submission scripts can be found in this Repo as `run_gends` and `run_gends_rev.sh`). If you are generating testing tensors, have only one rotation listed in ROTATIONS_LIST as `0,0,0`.
```bash
python gends.py --input VARIANT_LIST --output output_tensor_name --pdb_dir ./PDB_relaxed  --rotations ROTATION_LIST --boxsize 16 --voxelsize 1 --ncores 32
python gends.py --input VARIANT_LIST --output output_tensor_name --pdb_dir ./PDB_relaxed  --rotations ROTATION_LIST --boxsize 16 --voxelsize 1 --ncores 32 --reverse
```

### Train new ThermoNet ensemble
Run the following command for $MEMBER_IDX 1..10, if you wish to train a new ThermoNet ensemble with the generated tensors, otherwise move to prediction with an already trained ensemble. Note this step is best submitted as a Slurm array as exampled in `run_et_h100.sh`, running on a GPU partition such as `gpu-h100` with the valid environment as specified above.
```bash
python train_ensemble.py \
    --direct_features output_tensor_name_dir.npy \
    --inverse_features output_tensor_name_rev.npy \
    --direct_targets output_tensor_name_dir_ddg.txt \
    --inverse_targets output_tensor_name_rev_ddg.txt \
    --epochs 200 \
    --prefix RoDAThermoNet_ensemble \
    --member $MEMBER_IDX \
    --k 10
```

### Predict
Run the following command, to predict on direct (or reverse, change `dir` to `rev`) generated tensors for each ensemble member.
```bash
for i in `seq 1 10`; do python predict.py -x output_tensor_name_dir.npy -m RoDAThermoNet_ensemble_member_${i}.h5 -o output_tensor_name_dir_predictions_${i}.txt; done
```

Now take the average across members for final ensemble ΔΔG predictions, and compare to target ΔΔG values on your own analysis software of choice (noting some datasets may have differing ΔΔG sign conventions, such as the original Q1744 having stabilising ΔΔG < 0, and S669 and Ssym having stabilising ΔΔG > 0, so adjust final sign to align accordingly)

##  Example
Here we outline execution of RoDA as incremental rotations of 180 degrees around the x and y-axis of PDB structures, to train a new ThermoNet-style ensemble, and predict on Ssym

### Training
1. Move `Q1744_direct.csv` into the main folder and call `get_pdbs.py` to retrieve required PDBs
```bash
python get_pdbs.py --csv_file Q1744_direct.csv
```
2. Conduct sanity check to affirm PDB and listed variant allignment
```bash
python sanity_check.py Q1744_direct.csv
```
3. Create split variant lists for each unique PDB in `Q1744_direct.csv`
```bash
mkdir Variants
python split_variants.py --input_csv Q1744_direct.csv --output_dir ./Variants
```
4. Adjust `da_workflow.sh` and job commands to correct Rosetta binary paths, and move the script to the main folder; to then run relax and mutate Rosetta protocols for each unique PDB and their separated variants with
```bash
mkdir log
for i in ./PDBs/*.pdb; do 
filename=$(basename "$i" .pdb) 
echo "sbatch -o log/${filename}.out -e log/${filename}.err da_workflow.sh $filename" 
done > run_rdawf_slurm.sh
cat run_rdawf_slurm.sh | bash
```
5. List the desired incremental rotations to applied in `rotations.csv`
```
x_rot,y_rot,z_rot
0,0,0
180,0,0
0,180,0
180,180,0
```
6. Move into the main folder and adjust `run_gends.sh` to have the job command
```
python gends.py --input Q1744_direct.csv --output Q1744_tensorsi180 --pdb_dir ./PDB_relaxed  --rotations rotations.csv --boxsize 16 --voxelsize 1 --ncores 32
```
7. Submit `sbatch run_gends.sh` to generate augmented direct tensors and target values; add the `--reverse` flag to the `run_gends.sh` job command and resubmit to generate augmented reverse tensors and target values.
8. Move to the main folder and adjust the job commands of `run_et_h100.sh` to
```
srun python train_ensemble.py \
    --direct_features Q1744_tensorsi180_dir.npy \
    --inverse_features Q1744_tensorsi180_rev.npy \
    --direct_targets Q1744_tensorsi180_dir_ddg.txt \
    --inverse_targets Q1744_tensorsi180_rev_ddg.txt \
    --epochs 200 \
    --prefix i180_RoDAThermoNet \
    --member $MEMBER_IDX \
    --k 10
```
9. Submit `sbatch run_et_h100.sh` on appropriate (CPU/GPU) partition and environment to train new ensemble members

### Testing
1. Remove all PDBs currently populating `./PDBs` directory
```bash
rm -r ./PDBs
```
2. Move `ssym_ref.csv` into the main folder and call `get_pdbs.py` to retrieve required PDBs
```bash
python get_pdbs.py --csv_file ssym_ref.csv
```
3. Conduct sanity check to affirm PDB and listed variant allignment
```bash
python sanity_check.py ssym_ref.csv
```
4. Create split variant lists for each unique PDB in `ssym_ref.csv`
```bash
python split_variants.py --input_csv ssym_ref.csv --output_dir ./Variants
```
4. Run relax and mutate Rosetta protocols for each unique PDB and their separated variants with
```bash
for i in ./PDBs/*.pdb; do 
filename=$(basename "$i" .pdb) 
echo "sbatch -o log/${filename}.out -e log/${filename}.err da_workflow.sh $filename" 
done > run_rdawf_slurm.sh
cat run_rdawf_slurm.sh | bash
```
5. List the desired rotations to be applied in `rotations.csv`
```
x_rot,y_rot,z_rot
0,0,0
```
6. Adjust `run_gends.sh` to have the job command
```
python gends.py --input ssym_ref.csv --output ssym_tensors --pdb_dir ./PDB_relaxed  --rotations rotations.csv --boxsize 16 --voxelsize 1 --ncores 32
```
7. Submit `sbatch run_gends.sh` to generate augmented direct tensors and target values; add the `--reverse` flag to the `run_gends.sh` job command and resubmit to generate augmented reverse tensors and target values.
8. Run predictions on direct and reverse tensors for each RoDA-ThermoNet ensemble member
```bash
for i in `seq 1 10`; do python predict.py -x ssym_tensors_dir.npy -m i180_RoDAThermoNet_member_${i}.h5 -o ssym_dir_i180_pred_${i}.txt; done
for i in `seq 1 10`; do python predict.py -x ssym_tensors_rev.npy -m i180_RoDAThermoNet_member_${i}.h5 -o ssym_rev_i180_pred_${i}.txt; done
```
9. Average the predictions across the ensemble members for both direct and reverse final predictions ΔΔG (adjusted sign to convention if necessary) and compare with the generated targets in `ssym_tensors_dir_ddg.txt` and `ssym_tensors_rev_ddg.txt`. Example linux command to average each line across each file for dir and rev:
```bash
paste ssym_dir_i180_pred_{1..10}.txt | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}' > final_dir_pred.txt
paste ssym_rev_i180_pred_{1..10}.txt | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}' > final_rev_pred.txt
```