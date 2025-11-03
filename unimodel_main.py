import numpy as np
import pandas as pd
import json
import os
import random
from datetime import datetime

import torch
from sklearn.model_selection import StratifiedKFold
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

from models.mlp_model import train_mlp_model, MLPSurvival
from models.real_mlp_model import train_real_mlp_model, RealMLPSurvival
from models.tabm_model import train_tabm_model, TabMSurvival
from models.traditional_model import (
    train_cox_model,
    train_rsf_model,
    train_xgbse_model
)
import pickle
import joblib


def load_clinical_csv(csv_path='clinical_data_with_folds_followup.csv'):
    """Load clinical data from CSV with fold assignments"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}\nPlease run create_fold.py first!")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def preprocess_categorical_features(df):
    """Preprocess categorical features in DataFrame"""
    df = df.copy()
    
    # Earlier therapy
    if 'earlier_therapy' in df.columns:
        df['earlier_therapy_none'] = (df['earlier_therapy'] == 'none').astype(int)
        df['earlier_therapy_cryo'] = (df['earlier_therapy'] == 'radiotherapy + cryotherapy').astype(int)
        df['earlier_therapy_hormones'] = (df['earlier_therapy'] == 'radiotherapy + hormones').astype(int)
    
    # Positive lymph nodes
    if 'positive_lymph_nodes' in df.columns:
        df['positive_lymph_nodes'] = df['positive_lymph_nodes'].astype(str)
        df['positive_lymph_nodes_x'] = (df['positive_lymph_nodes'] == 'x').astype(int)
        df['positive_lymph_nodes_0'] = (df['positive_lymph_nodes'].isin(['0', '0.0'])).astype(int)
        df['positive_lymph_nodes_1'] = (df['positive_lymph_nodes'].isin(['1', '1.0'])).astype(int)
    
    # pT stage
    if 'pT_stage' in df.columns:
        df['pT_stage'] = df['pT_stage'].astype(str).str.extract('(\d+)', expand=False).fillna(0).astype(int)
    
    return df

def prepare_survival_data(df):
    """Prepare survival data from clinical features"""
    df = df.copy()

    if 'BCR' not in df.columns or 'time_to_follow-up/BCR' not in df.columns:
        print("Warning: 'BCR' or 'time_to_follow-up/BCR' column not found.")
        df['event'] = 0
        df['time'] = np.nan
        return df

    df['event'] = pd.to_numeric(df['BCR'], errors='coerce').fillna(0).astype(int)
    df['time'] = pd.to_numeric(df['time_to_follow-up/BCR'], errors='coerce').fillna(1.0)

    max_observed_time = df.loc[df['event'] == 1, 'time'].max()
    if pd.notna(max_observed_time):
        df.loc[df['event'] == 0, 'time'] = df.loc[df['event'] == 0, 'time'].apply(lambda x: max(x, max_observed_time + 1))

    df = df.drop(columns=['BCR', 'time_to_follow-up/BCR'], errors='ignore')

    print(f"BCR events: {df['event'].sum()}/{len(df)} ({df['event'].sum()/len(df)*100:.1f}%)")
    print(f"Time range: {df['time'].min():.1f} - {df['time'].max():.1f}")

    return df


def convert_to_numerical(df):
    """Convert object columns to numerical values"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    return df


def load_data_for_fold_from_csv(df, fold_num, fold_column='fold_followup'):
    """
    Load and preprocess data for a specific fold from CSV DataFrame.
    
    Args:
        df: DataFrame with all clinical data and fold assignments
        fold_num: Fold number to use as test set (0-9)
        fold_column: Column name containing fold assignments
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Split by fold
    test_mask = df[fold_column] == fold_num
    train_mask = ~test_mask
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print(f"Error: Empty train or test set for fold {fold_num}")
        return None, None, None, None
    
    # Preprocess categorical features
    train_df = preprocess_categorical_features(train_df)
    test_df = preprocess_categorical_features(test_df)
    
    # Prepare survival data
    train_df = prepare_survival_data(train_df)
    test_df = prepare_survival_data(test_df)
    
    # Extract survival outcomes
    y_train = train_df[['event', 'time']]
    y_test = test_df[['event', 'time']]
    
    # Drop columns
    cols_to_drop = ['event', 'time', 'earlier_therapy', 'positive_lymph_nodes', 
                    'patient_id', fold_column, 'fold_age', 'fold_followup']
    
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Convert to numerical
    X_train = convert_to_numerical(X_train)
    X_test = convert_to_numerical(X_test)
    
    # Align columns
    train_cols = X_train.columns
    test_cols = X_test.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    
    X_test = X_test[train_cols]
    
    return X_train, y_train, X_test, y_test

def perform_fold_evaluation(fold_num, df, fold_column='fold_followup', device='cpu', save_models=False, models_dir='saved_models'):
    """Perform evaluation on a specific fold, capturing both train and test C-index."""
    results = defaultdict(lambda: {'test_cindex': 0.0, 'train_cindex': 0.0, 'model': None})
    
    print(f"\n--- Fold {fold_num} ---")
    
    X_train, y_train_df, X_test, y_test_df = load_data_for_fold_from_csv(df, fold_num, fold_column)
    
    if X_train is None:
        print(f"Failed to load data for fold {fold_num}, skipping.")
        return dict(results)
    
    print(f"Train: {len(X_train)} samples, Features: {X_train.shape[1]}, Events: {y_train_df['event'].sum()}/{len(y_train_df)} ({y_train_df['event'].sum()/len(y_train_df)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples, Features: {X_test.shape[1]}, Events: {y_test_df['event'].sum()}/{len(y_test_df)} ({y_test_df['event'].sum()/len(y_test_df)*100:.1f}%)")
    
    models_to_train = [
        ('MLP', train_mlp_model), 
        ('RealMLP', train_real_mlp_model), 
        ('TabM', train_tabm_model),
        ('CoxPH', train_cox_model), 
        ('RandomSurvivalForest', train_rsf_model), 
        ('XGBoost', train_xgbse_model),
    ]
    
    for model_name, train_func in models_to_train:
        try:
            print(f"Training {model_name}...")
            # Modify train functions to return model as well
            result = train_func(X_train, y_train_df, X_test, y_test_df, device, return_model=True)
            
            if len(result) == 3:
                test_score, train_score, trained_model = result
            else:
                test_score, train_score = result
                trained_model = None
            
            results[model_name] = {
                'test_cindex': test_score,
                'train_cindex': train_score,
                'model': trained_model
            }
            print(f"{model_name} - Test C-index: {test_score:.4f}, Train C-index: {train_score:.4f}")
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'test_cindex': 0.0, 'train_cindex': 0.0, 'model': None}
            continue
    
    return dict(results)

def save_top3_models_per_fold(fold_name, fold_num, fold_results, split_type='followup', models_dir='saved_models'):
    """Save top 3 models based on test C-index for each fold"""
    # Create directory structure
    fold_dir = os.path.join(models_dir, f'{split_type}_fold_{fold_num:02d}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Get models sorted by test C-index
    model_scores = []
    for model_name, result in fold_results.items():
        if result.get('model') is not None and result.get('test_cindex', 0) > 0:
            model_scores.append({
                'name': model_name,
                'test_cindex': result['test_cindex'],
                'train_cindex': result['train_cindex'],
                'model': result['model']
            })
    
    # Sort by test C-index descending
    model_scores.sort(key=lambda x: x['test_cindex'], reverse=True)
    
    # Save top 3 models
    top3 = model_scores[:3]
    saved_models = []
    
    for rank, model_info in enumerate(top3, 1):
        model_name = model_info['name']
        model = model_info['model']
        test_cindex = model_info['test_cindex']
        train_cindex = model_info['train_cindex']
        
        try:
            # Save model based on type
            if model_name in ['MLP', 'RealMLP', 'TabM']:
                # PyTorch models - save entire object with pickle
                model_path = os.path.join(fold_dir, f'{model_name.lower()}_rank{rank}.pkl')
                joblib.dump(model, model_path)
                print(f"  Saved {model_name} (rank {rank}) - Test C-index: {test_cindex:.4f}")
            else:
                # Scikit-learn models (CoxPH, RSF, XGBoost)
                model_path = os.path.join(fold_dir, f'{model_name.lower()}_rank{rank}.pkl')
                joblib.dump(model, model_path)
                print(f"  Saved {model_name} (rank {rank}) - Test C-index: {test_cindex:.4f}")
            
            saved_models.append({
                'rank': rank,
                'model_name': model_name,
                'model_path': model_path,
                'test_cindex': test_cindex,
                'train_cindex': train_cindex
            })
            
        except Exception as e:
            print(f"  Failed to save {model_name}: {e}")
            continue
    
    # Save fold metadata
    metadata = {
        'fold_name': fold_name,
        'fold_num': fold_num,
        'split_type': split_type,
        'top3_models': saved_models,
        'all_model_results': {
            name: {
                'test_cindex': res['test_cindex'],
                'train_cindex': res['train_cindex']
            } for name, res in fold_results.items()
        }
    }
    
    metadata_path = os.path.join(fold_dir, 'fold_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=float)
    
    print(f"  Saved fold metadata to {metadata_path}")
    
    return saved_models


def aggregate_results_across_folds(all_results):
    """Aggregate results across all folds"""
    aggregated = defaultdict(lambda: {
        'test_all_scores': [], 'test_mean': 0, 'test_std': 0, 'test_min': 0, 'test_max': 0,
        'train_all_scores': [], 'train_mean': 0, 'train_std': 0, 'train_min': 0, 'train_max': 0,
        'count': 0
    })
    
    model_names = all_results[0].keys() if all_results else []

    for model_name in model_names:
        test_scores = []
        train_scores = []
        
        for fold_res in all_results:
            if model_name in fold_res and isinstance(fold_res[model_name], dict):
                test_score = fold_res[model_name].get('test_cindex', 0)
                train_score = fold_res[model_name].get('train_cindex', 0)
                if test_score is not None and not np.isnan(test_score):
                    test_scores.append(test_score)
                if train_score is not None and not np.isnan(train_score):
                    train_scores.append(train_score)
        
        if test_scores or train_scores:
            aggregated[model_name] = {
                'test_all_scores': test_scores,
                'test_mean': np.mean(test_scores) if test_scores else 0,
                'test_std': np.std(test_scores) if test_scores else 0,
                'test_min': np.min(test_scores) if test_scores else 0,
                'test_max': np.max(test_scores) if test_scores else 0,
                'train_all_scores': train_scores,
                'train_mean': np.mean(train_scores) if train_scores else 0,
                'train_std': np.std(train_scores) if train_scores else 0,
                'train_min': np.min(train_scores) if train_scores else 0,
                'train_max': np.max(train_scores) if train_scores else 0,
                'count': len(test_scores)
            }
    
    return dict(aggregated)

def print_aggregated_results(aggregated_results, num_folds):
    """Print aggregated results across all folds"""
    print(f"\n{'='*100}")
    print(f"AGGREGATED RESULTS ACROSS {num_folds} FOLDS")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Test C-index':<25} {'Train C-index':<25} {'Runs':<10}")
    print(f"{'':<20} {'Mean±Std (Min-Max)':<25} {'Mean±Std (Min-Max)':<25}")
    print("-" * 100)
    
    if not aggregated_results:
        print("No valid results to display.")
        return

    sorted_models = sorted(aggregated_results.items(), key=lambda item: item[1]['test_mean'], reverse=True)

    for model_name, stats in sorted_models:
        test_info = f"{stats['test_mean']:.3f}±{stats['test_std']:.3f} ({stats['test_min']:.3f}-{stats['test_max']:.3f})"
        train_info = f"{stats['train_mean']:.3f}±{stats['train_std']:.3f} ({stats['train_min']:.3f}-{stats['train_max']:.3f})"
        
        print(f"{model_name:<20} {test_info:<25} {train_info:<25} {stats['count']:<10}")
    
    print("-" * 100)
    
    if sorted_models:
        best_model, best_stats = sorted_models[0]
        print(f"\nBest performing model (Test): {best_model} (Mean C-index: {best_stats['test_mean']:.4f})")
        
        print(f"\nOverfitting Analysis:")
        for model_name, stats in sorted_models:
             if stats['count'] > 0:
                diff = stats['train_mean'] - stats['test_mean']
                if diff > 0.1:
                    print(f"  {model_name}: Potential overfitting (Train-Test diff: {diff:.3f})")
                else:
                    print(f"  {model_name}: Good generalization (Train-Test diff: {diff:.3f})")

def main():
    # Record program start time
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"PROGRAM STARTED AT: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models_dir = 'saved_models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Define both splits to process
    csv_configs = [
        {
            'csv_path': 'clinical_data_with_folds_followup.csv',
            'fold_column': 'fold_followup',
            'split_type': 'followup'
        },
        {
            'csv_path': 'clinical_data_with_folds_age.csv',
            'fold_column': 'fold_age',
            'split_type': 'age'
        }
    ]
    
    all_results_combined = {}
    all_saved_models_combined = []
    
    # Process both CSV files
    for config in csv_configs:
        csv_path = config['csv_path']
        fold_column = config['fold_column']
        split_type = config['split_type']
        
        print(f"\n{'='*80}")
        print(f"PROCESSING SPLIT: {split_type.upper()}")
        print(f"CSV: {csv_path}")
        print(f"{'='*80}")
        
        # Load clinical data from CSV
        try:
            df = load_clinical_csv(csv_path)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}")
            print("Please run create_fold.py first!")
            continue
        
        # Get number of unique folds
        num_folds = int(df[fold_column].max() + 1)
        print(f"Found {num_folds} folds in column '{fold_column}'")
        
        all_results = []
        all_saved_models = []
        
        for fold_num in range(num_folds):
            print(f"\n{'='*60}\nSTARTING FOLD {fold_num}/{num_folds} ({split_type})\n{'='*60}")
            
            np.random.seed(fold_num)
            torch.manual_seed(fold_num)
            random.seed(fold_num)
            
            results = perform_fold_evaluation(fold_num, df, fold_column=fold_column,
                                             device=device, save_models=True, models_dir=models_dir)
            all_results.append(results)
            
            # Save top 3 models for this fold
            print(f"\nSaving top 3 models for fold {fold_num}...")
            saved_models = save_top3_models_per_fold(f'fold_{fold_num}', fold_num, results, 
                                                      split_type=split_type, models_dir=models_dir)
            all_saved_models.extend(saved_models)
            
            print(f"Completed fold {fold_num}/{num_folds}")
        
        if not all_results:
            print(f"No results were generated for {split_type}. Skipping.")
            continue

        aggregated_results = aggregate_results_across_folds(all_results)
        
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {split_type.upper()} SPLIT")
        print(f"{'='*80}")
        print_aggregated_results(aggregated_results, num_folds)
        
        # Store results for this split
        all_results_combined[split_type] = {
            'num_folds': num_folds,
            'csv_path': csv_path,
            'fold_column': fold_column,
            'aggregated_results': aggregated_results,
            'detailed_results_by_fold': {f'fold_{i}': {
                name: {'test_cindex': res[name]['test_cindex'], 'train_cindex': res[name]['train_cindex']}
                for name in res.keys()
            } for i, res in enumerate(all_results)},
            'saved_models': all_saved_models
        }
        all_saved_models_combined.extend(all_saved_models)
    
    # Generate combined plot if we have results from both splits
    if all_results_combined:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total models saved: {len(all_saved_models_combined)}")
        print(f"Splits processed: {len(all_results_combined)}")
    
    # Save combined results
    results_summary = {
        'device': device,
        'total_models_saved': len(all_saved_models_combined),
        'models_per_fold': 3,
        'total_folds': sum([config['num_folds'] for config in all_results_combined.values()]),
        'splits_processed': list(all_results_combined.keys()),
        'results_by_split': all_results_combined,
        'all_saved_models': all_saved_models_combined
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/unimodel_evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4, default=float)
    
    print(f"\nDetailed results saved to results/unimodel_evaluation_results.json")
    print("Fold-based evaluation completed")
    
    # Record program end time and calculate total duration
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"PROGRAM COMPLETED AT: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL DURATION: {total_duration}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Add timing information to results summary
    results_summary['timing'] = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_seconds': total_duration.total_seconds(),
        'total_duration_str': str(total_duration)
    }
    
    # Save updated results
    with open('results/unimodel_evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4, default=float)

if __name__ == "__main__":
    main()