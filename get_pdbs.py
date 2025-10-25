#!/usr/bin/env python3
# Author: Andrew Harrison | 1580584
import os
import argparse
import pandas as pd
import subprocess

def download_and_clean_pdb(pdb_id_full: str, output_dir: str, clean_pdb_path: str) -> None:
    """
    Downloads a PDB file using pdb_fetch and cleans it using Rosetta's clean_pdb.py.
    """
    pdb_code = pdb_id_full[:4].lower()
    chain_id = pdb_id_full[4].upper()
    output_filename = f"{pdb_id_full.upper()}.pdb"  # Ensure uppercase filename

    raw_pdb_path = os.path.join(output_dir, f'{pdb_code}.pdb')
    output_pdb_path = os.path.join(output_dir, output_filename)

    # Fetch PDB
    try:
        with open(raw_pdb_path, 'w') as out_f:
            subprocess.run(
                ['pdb_fetch', pdb_code],
                check=True,
                stdout=out_f
            )
        print(f'Downloaded {pdb_code}.pdb')
    except subprocess.CalledProcessError:
        print(f'Error: Failed to fetch {pdb_code} using pdb_fetch')
        return

    # Run Rosetta's clean_pdb.py
    try:
        subprocess.run(
            ['python', clean_pdb_path, os.path.abspath(raw_pdb_path), chain_id],
            cwd=output_dir,
            check=True,
            stdout=subprocess.DEVNULL,  # suppress output
            stderr=subprocess.DEVNULL
        )
        cleaned_file = os.path.join(
            output_dir, f"{pdb_code}_{chain_id}.pdb"
        )
        fasta_file = os.path.join(
            output_dir, f"{pdb_code}_{chain_id}.fasta"
        )

        if os.path.exists(cleaned_file):
            os.rename(cleaned_file, output_pdb_path)
            print(f'Successfully wrote: {output_filename} (chain {chain_id})')
        else:
            print(f'Error: Expected output {cleaned_file} not found')

        # Remove fasta
        if os.path.exists(fasta_file):
            os.remove(fasta_file)

    except subprocess.CalledProcessError:
        print(f'Error: clean_pdb.py failed on {pdb_code} chain {chain_id}')

    # Cleanup raw PDB
    if os.path.exists(raw_pdb_path):
        os.remove(raw_pdb_path)


def main() -> None:
    """
    Main function to parse arguments, load the list of PDBs, and process them.
    """
    parser = argparse.ArgumentParser(description='Download PDBs (via pdb_fetch) and clean them with Rosetta clean_pdb.py.')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to the input CSV file with a "pdb_id" column (e.g., 1abcA).')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_pdb_path = os.path.join(script_dir, 'clean_pdb.py')

    df = pd.read_csv(args.csv_file)

    if 'pdb_id' not in df.columns:
        print('Error: CSV file must contain a column named "pdb_id" (e.g. 1abcA)')
        return
    
    unique_pdb_ids = df['pdb_id'].dropna().unique()

    # Set up output directory
    csv_dir = os.path.dirname(os.path.abspath(args.csv_file))
    pdb_dir = os.path.join(csv_dir, 'PDBs')
    os.makedirs(pdb_dir, exist_ok=True)

    for pdb_id_full in unique_pdb_ids:
        if len(pdb_id_full) != 5:
            print(f'Skipping malformed pdb_id: {pdb_id_full}')
            continue
        download_and_clean_pdb(pdb_id_full, pdb_dir, clean_pdb_path)
    
    print('All PDB files processed.')


if __name__ == '__main__':
    main()