# RoDA-ThermoNet
An end-to-end workflow as part of a MC-SCIBIF Research Project to apply rotational data augmentation (RoDA) to [ThermoNet](https://github.com/gersteinlab/ThermoNet) (Li et al., 2020).
ThermoNet is a computational tool for predicting protein thermostability change (ΔΔG) due to missense mutations. Its ensemble 3D-CNN architecture learns from voxelised PDB environments, and has shown promise in accurate prediction from solely structure-derived features. RoDA for ThermoNet seeks to expand trainable instances by rotating PDB structures, and preserving the scarce experimental stability data. 
Here we detail instructions on how to execute the developed workflow to expand training datasets for ThermoNet. Key optimisations have been made from legacy code to improve space and time costs in generating tensors for training and prediction.

## Installation

First, setup the environment with required software.

# Clone RoDA-ThermoNet
Clone RoDA-ThermoNet locally.
```bash
git clone https://github.com/andrewharrisonGH/RoDA-ThermoNet.git
```

# Requirements
ThermoNet was built for Linux platforms, thus, RoDA-ThermoNet has only been tested on Linux platforms. To use ThermoNet, you would need to install the following third-party software:
  * Rosetta 3.10. Rosetta is a multi-purpose macromolecular modeling suite that is used in ThermoNet for creating and refining protein structures. You can get Rosetta from the Rosetta Commons website: https://www.rosettacommons.org/software
1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.10 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

# Conda environment
```bash
conda env create --name thermonet --file environment.yaml
conda activate thermonet
```
The above commands will create a conda environment and install all dependencies for RoDa-ThermoNet tensor creation, model training and ΔΔG prediction. Change `prefix: /your/path/.conda/envs/thermonet` in environment.yaml to your conda environment path (e.g. `/home/user/.conda/envs/thermonet`)

## Use RoDA-ThermoNet
Note: this workflow is optimised for use on a HPC cluster using Slurm Workload Manager.
The whole of RoDA-ThermoNet input begins with two CSV files: VARIANT_LIST CSV (e.g. `Q1744_direct.csv`), and a ROTATIONS_LSIT CSV (e.g. `rotations.csv`).
Ensure variant CSV is formatted with `pdb_id,pos,wild_type,mutant,ddg` and rotations CSV with `x_rot,y_rot,z_rot`, example files are found in this repo.

# Get PDBs
Run the following command to download and clean each unique wild-type PDB for your dataset, depositing all downloaded selected PDB chains into `PDBs/`
```bash
python get_pdbs.py --csv_file VARIANT_LIST
```

# Sanity check dataset with PDBs
Before moving onto relaxation ensure wild-type residue and positions for each PDB align with VARIANT_LIST by running.
```bash
python sanity_check.py VARIANT_LIST
```

# Split variants for parallel relaxation
Run the following commands to generate separate CSVs for each unique PDB which will be passed into individual Slurm submission relaxation and mutation jobs.
```bash
mkdir Variants
python split_variants --input_csv VARIANT_LIST --output_dir ./Variants
```

# Generate individual job submission commands
Run the following commands to submit separate Slurm jobs for relaxation and mutation with `da_workflow.sh` (insert preffered partition, project ID, and path/to/relax.static.linuxgccrelease binary)
```bash
mkdir log
for i in ./PDBs/*.pdb; do 
filename=$(basename "$i" .pdb) 
echo "sbatch -o log/${filename}.out -e log/${filename}.err da_workflow.sh $filename" 
done > run_rdawf_slurm.sh
cat run_rdawf_slurm.sh | bash
```
Noting `da_workflow` calls the ThermoNet ROSETTA relax protocol
```bash
path/to/relax.static.linuxgccrelease -in:file:s "$INPUT_PDB" -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false -out:path:all "$OUT_DIR"
```
and mutation protocol
```bash
python ./rosetta_relax.py --rosetta-bin path/to/relax.static.linuxgccrelease -l "$VARIANT_LIST" --base-dir ./PDB_relaxed
```

# Rotation and Tensor Generation
Run the following commands, with an appropriate number of allocated CPU resources (I have preffered `n=32`), to generate direct and reverse tensors from the relaxed PDB structure that undergo the rotational augmentations specified in ROTATIONS (example separate slurm submission scripts can be found in this Repo). If you are generating testing tensors, have only one rotation listed in ROTATIONS_LIST as `0,0,0`.
```bash
python gends.py --input VARIANT_LIST --output output_tensor_name --pdb_dir ./PDB_relaxed  --rotations ROTATION_LIST --boxsize 16 --voxelsize 1 --ncores 32
python gends.py --input VARIANT_LIST --output output_tensor_name --pdb_dir ./PDB_relaxed  --rotations ROTATION_LIST --boxsize 16 --voxelsize 1 --ncores 32 --reverse
```

# Train new ThermoNet ensemble
Run the following commands, if you wish to train a new ThermoNet ensemble with the generated tensors, otherwise move to prediction with an already trained ensemble. Note this step is best submitted as Slurm array as exampled in `run_et_h100.sh`, running on a GPU partition such as `gpu-h100`.
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

# Predict
Run the following command, to predict on direct (or reverse, change `dir` to `rev`) generated tensors for each ensemble member.
```bash
for i in `seq 1 10`; do python predict.py -x output_tensor_name_dir.npy -m RoDAThermoNet_ensemble_member_${i}.h5 -o output_tensor_name_dir_predictions_${i}.txt; done
```

Now take the average across members for final ensemble ΔΔG predictions, and compare to target ΔΔG values on your own analysis software of choice.
