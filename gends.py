#!/usr/bin/env python3
# Author: Andrew Harrison | 1580584
import os
import time
import gc
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from numpy.lib import format as npformat
from utils import pdb_utils


def parse_cmd() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='File containing a list of protein mutations.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help='Base filename to write the dataset.')
    parser.add_argument('-p', '--pdb_dir', dest='pdb_dir', type=str, required=True,
                        help='Directory where PDB files are stored.')
    parser.add_argument('--rotations', dest='rotations', type=str, required=True,
                        help='CSV file containing x, y, z rotations.')
    parser.add_argument('--boxsize', dest='boxsize', type=int,
                        help='Bounding box size around mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int,
                        help='Voxel size.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Print verbose messages from HTMD.')
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='If set, only the reverse dataset is created and ddGs are sign-flipped.')
    parser.add_argument('--ncores', dest='ncores', type=int, default=1,
                        help='Number of cores for multiprocessing.')
    return parser.parse_args()


def compute_features(task: tuple) -> dict:
    """Calculates voxel features for a single mutation task."""
    pdb_chain, pos, wt, mt, ddg, rot, pdb_dir, boxsize, voxelsize, verbose, reverse_flag = task
    x_deg, y_deg, z_deg = map(float, rot)
    res_num = int(pos)

    # Load WT and MT structures
    wt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed.pdb")
    mt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed_{wt}{pos}{mt}_relaxed.pdb")
    if not os.path.exists(wt_path) or not os.path.exists(mt_path):
        raise FileNotFoundError(f"Missing PDB files: {wt_path} or {mt_path}")

    # Convert rotation from degrees to radians for compute_voxel_features
    rot_rad = np.radians([x_deg, y_deg, z_deg])

    features_wt = pdb_utils.compute_voxel_features(res_num, wt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)
    features_mt = pdb_utils.compute_voxel_features(res_num, mt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)

    # Remove property channel 6
    features_wt = np.delete(features_wt, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)

    # Combine channels and assign ddG
    if reverse_flag:
        tensor = np.concatenate((features_mt, features_wt), axis=0).astype('float32')
        ddg_val = -float(ddg)
    else:
        tensor = np.concatenate((features_wt, features_mt), axis=0).astype('float32')
        ddg_val = float(ddg)

    tensor = np.transpose(tensor, (1, 2, 3, 0))

    del features_wt, features_mt
    gc.collect()
    return {'tensor': tensor, 'ddg': ddg_val}


def generate_tasks(input_file: str, rotations_list: list, pdb_dir: str, boxsize: int, voxelsize: int, verbose: bool, reverse_flag: bool) -> iter:
    """
    Yields tasks one by one from the input file.
    """
    with open(input_file, 'r') as f:
        for line in f:
            # Skip header line
            if 'pdb_id' in line:
                continue
            
            pdb_chain, pos, wt, mt, ddg = line.strip().split(',')
            pdb_chain = pdb_chain.upper()
            
            for rot in rotations_list:
                yield (pdb_chain, pos, wt, mt, ddg, rot,
                       pdb_dir, boxsize, voxelsize, verbose, reverse_flag)


def main() -> None:
    """Main script execution to process mutations and save features."""
    args = parse_cmd()
    pdb_dir = os.path.abspath(args.pdb_dir)

    # Load rotations
    rotations = []
    with open(args.rotations, 'r') as f:
        for line in f:
            rots = line.strip().split(',')
            if rots[0] != 'x_rot':
                rotations.append(tuple(rots))

    # Probe shape using a temporary generator
    # C, X, Y, Z are dimensions BEFORE transposition
    C = X = Y = Z = None 
    probe_generator = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                     args.voxelsize, args.verbose, args.reverse)
    for task in probe_generator:
        try:
            result = compute_features(task)
            if result is not None:
                # Shape returned by compute_features is (D, H, W, C)
                D, H, W, C = result['tensor'].shape 
                # Dimensions needed for output shape
                break
        except FileNotFoundError:
            continue
    
    if D is None:
        raise RuntimeError("Could not generate any valid features to determine the output shape.")

    # Estimate total number of samples for memory-mapped file
    num_mutations = sum(1 for line in open(args.input)) - 1 # Subtract 1 for header
    n_samples = num_mutations * len(rotations)

    # Create main generator for the multiprocessing pool
    tasks_for_pool = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                    args.voxelsize, args.verbose, args.reverse)
    
    # Decide dataset name
    dset_name = "rev" if args.reverse else "dir"
    npy_path = args.output + f"_{dset_name}.npy"
    txt_path = args.output + f"_{dset_name}_ddg.txt"

    # Preallocate .npy file as memory-mapped array with (N, D, H, W, C) shape
    # NOTE: D, H, W, C are dimensions from the probe
    mmap = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype="float32", shape=(n_samples, D, H, W, C)
    )
    print(f"Allocated memory-mapped array with shape: ({n_samples}, {D}, {H}, {W}, {C})")

    with open(txt_path, "w") as ddg_file:
        idx = 0
        # maxtasksperchild=1 ensures memory from each task is released
        with Pool(min(args.ncores, cpu_count()), maxtasksperchild=1) as pool:
            # Pass generator directly to the pool
            for res in pool.imap_unordered(compute_features, tasks_for_pool, chunksize=2):
                if res is None:
                    continue
                
                if idx < n_samples:
                    mmap[idx] = res['tensor']
                    ddg_file.write(f"{res['ddg']}\n")
                    idx += 1
                else:
                    print(f"Warning: generated more results than allocated space ({n_samples}).")
                
                del res
                gc.collect()

    # Flush memory map to disk
    mmap.flush() 
    # .txt file lines give true count of successful tasks
    print(f"Saved {idx} tensors to {npy_path} with allocated shape {mmap.shape}")
    print(f"Saved ddGs to {txt_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds")