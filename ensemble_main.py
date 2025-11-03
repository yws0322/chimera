import numpy as np
import pandas as pd
import json
import os
import glob
import torch
import torch.nn as nn
import joblib
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import model classes
from models.mlp_model import MLPSurvival
from models.real_mlp_model import RealMLPSurvival
from models.tabm_model import TabMSurvival
from tabm import TabM

# Import data loading utilities
from unimodel_main import (
    load_clinical_csv,
    load_data_for_fold_from_csv,
    preprocess_categorical_features,
    prepare_survival_data,
    convert_to_numerical
)

# Import survival metrics
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler


class EnsemblePredictor:
    """Ensemble predictor using 60 saved models"""
    
    def __init__(self, models_dir='saved_models', device='cpu'):
        self.models_dir = models_dir
        self.device = device
        self.loaded_models = []
        self.model_metadata = {}
        
    def load_all_models(self):
        """Load all saved models"""
        print(f"Loading all models from {self.models_dir}...")
        
        if not os.path.exists(self.models_dir):
            print(f"Models directory {self.models_dir} not found!")
            return 0
        
        models_loaded = 0
        
        # Find all fold directories
        fold_dirs = glob.glob(os.path.join(self.models_dir, '*_fold_*'))
        
        for fold_dir in sorted(fold_dirs):
            # Load fold metadata
            metadata_path = os.path.join(fold_dir, 'fold_metadata.json')
            if not os.path.exists(metadata_path):
                print(f"  No metadata found in {fold_dir}, skipping...")
                continue
            
            with open(metadata_path, 'r') as f:
                fold_metadata = json.load(f)
            
            fold_name = os.path.basename(fold_dir)
            
            # Load each model in this fold
            for model_info in fold_metadata.get('top3_models', []):
                model_name = model_info['model_name']
                model_path = model_info['model_path']
                
                try:
                    model = self._load_single_model(model_name, model_path)
                    if model is not None:
                        self.loaded_models.append({
                            'model': model,
                            'model_name': model_name,
                            'fold': fold_name,
                            'rank': model_info['rank'],
                            'test_cindex': model_info['test_cindex'],
                            'train_cindex': model_info['train_cindex'],
                            'model_path': model_path
                        })
                        models_loaded += 1
                        print(f"  Loaded {model_name} from {fold_name} (rank {model_info['rank']})")
                except Exception as e:
                    print(f"  Failed to load {model_name} from {fold_name}: {e}")
                    continue
        
        print(f"\nTotal models loaded: {models_loaded}")
        return models_loaded
    
    def _load_single_model(self, model_name, model_path):
        """Load a single model"""
        if not os.path.exists(model_path):
            print(f"  Model file not found: {model_path}")
            return None
        
        try:
            # All models are saved as pickle files (.pkl)
            model = joblib.load(model_path)
            return model
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            return None
    
    def _get_base_predictions(self, X):
        """Get predictions from all base models"""
        base_predictions = []
        successful_models = []
        
        for i, model_info in enumerate(self.loaded_models):
            model = model_info['model']
            model_name = model_info['model_name']
            
            try:
                # Get risk scores from models
                if model_name in ['MLP', 'RealMLP', 'TabM']:
                    # PyTorch models
                    pred = -model.predict_risk_score(X)
                elif model_name == 'CoxPH':
                    # CoxPH model
                    if hasattr(model, 'scaler'):
                        X_scaled = model['scaler'].transform(X)
                        pred = -model['model'].predict(X_scaled)
                    else:
                        pred = -model.predict(X)
                elif model_name == 'RandomSurvivalForest':
                    # RSF model
                    X_np = np.asarray(X, dtype=np.float64)
                    X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
                    pred = -model.predict(X_np)
                elif model_name == 'XGBoost':
                    # XGBoost model
                    try:
                        survival_probs = model.predict(X)
                        pred = np.sum(survival_probs, axis=1)
                    except:
                        continue
                else:
                    continue
                
                base_predictions.append(pred)
                successful_models.append(i)
                
            except Exception as e:
                continue
        
        if not base_predictions:
            raise ValueError("No valid base predictions generated")
        
        return np.column_stack(base_predictions), successful_models
    
    def predict_simple_average(self, X):
        """Simple average ensemble"""
        base_preds, successful_models = self._get_base_predictions(X)
        final_prediction = np.mean(base_preds, axis=1)
        return final_prediction, len(successful_models)
    
    def evaluate_ensemble(self, X, y):
        """Evaluate ensemble performance"""
        predictions, models_used = self.predict_simple_average(X)
        
        # Calculate C-index
        c_index_tuple = concordance_index_censored(
            y['event'].astype(bool),
            y['time'],
            -predictions  # Negate for C-index calculation
        )
        c_index = c_index_tuple[0]
        
        return c_index, models_used


def evaluate_split(ensemble_predictor, df, fold_column, split_type, num_folds):
    """Evaluate ensemble on a single split"""
    test_c_indices = []
    train_c_indices = []
    fold_results = []
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {split_type.upper()} SPLIT")
    print(f"{'='*60}")
    
    for fold_num in range(num_folds):
        print(f"\nFold {fold_num}/{num_folds}")
        
        try:
            # Load fold data from CSV
            X_train, y_train, X_test, y_test = load_data_for_fold_from_csv(df, fold_num, fold_column)
            
            if X_train is None or X_test is None:
                print(f"  Failed to load data for fold {fold_num}, skipping...")
                continue
            
            # Evaluate on test set
            test_c_index, test_models_used = ensemble_predictor.evaluate_ensemble(X_test, y_test)
            
            # Evaluate on train set
            train_c_index, train_models_used = ensemble_predictor.evaluate_ensemble(X_train, y_train)
            
            test_c_indices.append(test_c_index)
            train_c_indices.append(train_c_index)
            
            fold_results.append({
                'fold_num': fold_num,
                'test_c_index': test_c_index,
                'train_c_index': train_c_index,
                'n_test_samples': len(X_test),
                'n_train_samples': len(X_train),
                'n_test_events': y_test['event'].sum(),
                'n_train_events': y_train['event'].sum(),
                'models_used': test_models_used
            })
            
            print(f"  Test C-index: {test_c_index:.4f}")
            print(f"  Train C-index: {train_c_index:.4f}")
            print(f"  Models used: {test_models_used}")
            
        except Exception as e:
            print(f"  Error processing fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not test_c_indices:
        return None
    
    return {
        'split_type': split_type,
        'overall_test_c_index': np.mean(test_c_indices),
        'overall_train_c_index': np.mean(train_c_indices),
        'test_std_c_index': np.std(test_c_indices),
        'train_std_c_index': np.std(train_c_indices),
        'test_min_c_index': np.min(test_c_indices),
        'test_max_c_index': np.max(test_c_indices),
        'train_min_c_index': np.min(train_c_indices),
        'train_max_c_index': np.max(train_c_indices),
        'overfitting_gap': np.mean(train_c_indices) - np.mean(test_c_indices),
        'total_folds': len(test_c_indices),
        'fold_results': fold_results
    }


def evaluate_ensemble_on_all_folds():
    """Evaluate ensemble performance on all folds from both splits"""
    print("Starting Ensemble Evaluation on All Folds")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize ensemble predictor and load models
    ensemble_predictor = EnsemblePredictor(models_dir='saved_models', device=device)
    total_models = ensemble_predictor.load_all_models()
    
    if total_models == 0:
        print("No models loaded. Please run unimodel_main.py first.")
        return False
    
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
    
    all_splits_results = {}
    
    for config in csv_configs:
        csv_path = config['csv_path']
        fold_column = config['fold_column']
        split_type = config['split_type']
        
        # Load clinical data from CSV
        try:
            df = load_clinical_csv(csv_path)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_path}. Skipping {split_type} split.")
            continue
        
        # Get number of folds
        num_folds = int(df[fold_column].max() + 1)
        
        # Evaluate this split
        split_results = evaluate_split(ensemble_predictor, df, fold_column, split_type, num_folds)
        
        if split_results:
            all_splits_results[split_type] = split_results
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 80)
    
    if all_splits_results:
        print(f"\n{'Split':<15} {'Test C-index':<20} {'Train C-index':<20} {'Overfitting':<12}")
        print("-" * 67)
        
        for split_type, results in all_splits_results.items():
            test_str = f"{results['overall_test_c_index']:.4f} +/- {results['test_std_c_index']:.4f}"
            train_str = f"{results['overall_train_c_index']:.4f} +/- {results['train_std_c_index']:.4f}"
            overfitting = results['overfitting_gap']
            
            print(f"{split_type:<15} {test_str:<20} {train_str:<20} {overfitting:<12.4f}")
        
        # Save results
        final_results = {
            'total_models_in_ensemble': total_models,
            'ensemble_method': 'Simple Average',
            'results_by_split': all_splits_results
        }
        
        os.makedirs('results', exist_ok=True)
        output_file = 'results/ensemble_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=4, default=float)
        
        print(f"\nResults saved to {output_file}")
        print("Ensemble evaluation completed")
        
        return True
    else:
        print("No valid results computed")
        return False


def main():
    """Main execution function"""
    print("Starting 60-Model Ensemble Evaluation")
    print("=" * 80)
    
    try:
        success = evaluate_ensemble_on_all_folds()
        if success:
            print("\nEnsemble evaluation completed successfully")
        else:
            print("\nEnsemble evaluation failed")
        return success
    except Exception as e:
        print(f"\nError during ensemble evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

