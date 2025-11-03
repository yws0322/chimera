import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict


def load_clinical_data_to_dataframe(data_dir: str = "../clinlical_data/") -> pd.DataFrame:
    """Loads all clinical data JSON files and converts to DataFrame."""
    clinical_data = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    print(f"Loading clinical data from {data_dir}...")
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['patient_id'] = filename.replace('.json', '')
                    clinical_data.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    df = pd.DataFrame(clinical_data)
    print(f"Loaded {len(df)} patients")
    return df


def assign_folds_uniform(
    df: pd.DataFrame,
    stratify_by_column: str,
    fold_column_name: str = 'fold',
    n_folds: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Assigns fold numbers to each sample using uniform stratification.
    
    Args:
        df: DataFrame with clinical data
        stratify_by_column: Column name to stratify by
        fold_column_name: Name for the new fold column
        n_folds: Number of folds (default: 10)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with added fold column
    """
    rng = np.random.default_rng(random_state)
    
    # Work with non-null values only
    df_valid = df.dropna(subset=[stratify_by_column]).copy()
    
    # Sort by the stratification column
    df_valid = df_valid.sort_values(by=stratify_by_column).reset_index(drop=True)

    # Create bins (value-based, not quantiles)
    min_val, max_val = df_valid[stratify_by_column].min(), df_valid[stratify_by_column].max()
    bins = np.linspace(min_val, max_val, n_folds + 1)
    df_valid['bin'] = np.digitize(df_valid[stratify_by_column], bins) - 1
    df_valid['bin'] = df_valid['bin'].clip(0, n_folds - 1)

    # Assign fold numbers uniformly within each bin
    df_valid[fold_column_name] = -1
    
    for b in range(n_folds):
        bin_mask = df_valid['bin'] == b
        bin_indices = df_valid[bin_mask].index.tolist()
        rng.shuffle(bin_indices)
        
        # Assign fold numbers in round-robin
        for idx, data_idx in enumerate(bin_indices):
            fold_num = idx % n_folds
            df_valid.loc[data_idx, fold_column_name] = fold_num
    
    # Drop temporary bin column
    df_valid = df_valid.drop('bin', axis=1)
    
    return df_valid


def print_fold_statistics(
    df: pd.DataFrame,
    fold_column: str,
    stratify_by_column: str
):
    """Prints statistics for each fold."""
    print(f"\n--- Fold Statistics (Stratified by '{stratify_by_column}') ---")
    print("=" * 60)
    
    for fold_num in sorted(df[fold_column].unique()):
        fold_data = df[df[fold_column] == fold_num]
        mean_val = fold_data[stratify_by_column].mean()
        
        print(f"Fold {fold_num:02d}: {len(fold_data):3d} samples, "
              f"Mean {stratify_by_column}: {mean_val:6.1f}")
    
    print("-" * 60)
    print(f"Total samples: {len(df)}")


def create_and_save_csv_with_folds(
    df: pd.DataFrame,
    stratify_by: str,
    fold_column_name: str,
    output_file: str,
    n_folds: int = 10
):
    """
    Create fold assignments and save to CSV.
    
    Args:
        df: DataFrame with clinical data
        stratify_by: Column to stratify by
        fold_column_name: Name for fold column
        output_file: Output CSV file path
        n_folds: Number of folds
    """
    print(f"\n{'='*70}")
    print(f"Creating {n_folds}-fold splits stratified by: {stratify_by}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}")
    
    # Assign folds
    df_with_folds = assign_folds_uniform(
        df,
        stratify_by_column=stratify_by,
        fold_column_name=fold_column_name,
        n_folds=n_folds
    )
    
    # Print statistics
    print_fold_statistics(df_with_folds, fold_column_name, stratify_by)
    
    # Save to CSV
    df_with_folds.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return df_with_folds


def main():
    """Main function to create CSV files with fold assignments."""
    print("Fold Creation Script Started (CSV-based)")
    print("=" * 70)
    
    try:
        df = load_clinical_data_to_dataframe()
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return
    
    # Create CSV with folds stratified by age
    df_age = create_and_save_csv_with_folds(
        df=df.copy(),
        stratify_by='age_at_prostatectomy',
        fold_column_name='fold_age',
        output_file='clinical_data_with_folds_age.csv',
        n_folds=10
    )
    
    # Create CSV with folds stratified by follow-up time
    df_followup = create_and_save_csv_with_folds(
        df=df.copy(),
        stratify_by='time_to_follow-up/BCR',
        fold_column_name='fold_followup',
        output_file='clinical_data_with_folds_followup.csv',
        n_folds=10
    )
    
    print("\n" + "=" * 70)
    print("Fold creation completed successfully")
    print("=" * 70)
    print("\nCreated CSV files:")
    print("  clinical_data_with_folds_age.csv      - Stratified by age")
    print("  clinical_data_with_folds_followup.csv - Stratified by follow-up time")
    print("\nEach CSV contains all data with a fold column (0-9).")


if __name__ == "__main__":
    main()

