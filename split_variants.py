#!/usr/bin/env python3
# Author: Andrew Harrison | 1580584
import pandas as pd
from pathlib import Path
import os
import argparse

def split_mutation_csvs(input_csv: str, output_dir: str) -> None:
    """
    Splits a single master CSV file containing multiple protein mutations
    into separate CSV files, one for each unique protein (identified by 'pdb_id').
    """
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Processing {input_path.name}")

    if "pdb_id" not in df.columns:
        print(f"Skipping {input_path.name} (no 'pdb_id' column)")
        return None

    for pdb_id_chain, group_df in df.groupby("pdb_id"):
        # Use the full PDB ID including the chain letter for the filename
        pdb_full_id = pdb_id_chain.upper()
        out_path = output_path / f"{pdb_full_id}_variants.csv"
        group_df.to_csv(out_path, index=False)
        print(f"Wrote {len(group_df)} rows to {out_path.name}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Split mutation CSVs into per-protein files.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file containing mutations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for per-protein CSVs ('Variants')")
    args = parser.parse_args()

    split_mutation_csvs(args.input_csv, args.output_dir)

if __name__ == "__main__":
    main()