# traditional_models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

# 라이브러리 임포트 (이전과 동일)
try:
    from xgbse import XGBSEKaplanNeighbors
    XGBSE_AVAILABLE = True
except ImportError:
    XGBSE_AVAILABLE = False

try:
    from auton_survival.models.cph import DeepCoxPH
    from auton_survival.models.dsm import DeepSurvivalMachines as AutoDSM
    AUTON_SURVIVAL_AVAILABLE = True
except ImportError:
    AUTON_SURVIVAL_AVAILABLE = False


def train_cox_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train CoxPH model. Assumes y_train/y_test are DataFrames."""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_train_sksurv = np.array(
            list(zip(y_train['event'], y_train['time'])), 
            dtype=[('event', bool), ('time', float)]
        )
        
        model = CoxPHSurvivalAnalysis(alpha=0.1, ties='efron', n_iter=100)
        model.fit(X_train_scaled, y_train_sksurv)
        
        risk_scores = model.predict(X_test_scaled)
        
        c_index_tuple = concordance_index_censored(
            y_test['event'].astype(bool), y_test['time'], risk_scores
        )

        train_risk_scores = model.predict(X_train_scaled)
        train_c_index_tuple = concordance_index_censored(
            y_train['event'].astype(bool), 
            y_train['time'], 
            train_risk_scores
        )
        
        if return_model:
            # Return model with scaler for later use
            model_with_scaler = {'model': model, 'scaler': scaler}
            return c_index_tuple[0], train_c_index_tuple[0], model_with_scaler
        else:
            return c_index_tuple[0], train_c_index_tuple[0]
    except Exception as e:
        print(f"Cox model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0

# --- MODIFIED FUNCTION ---
def train_rsf_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train RSF model with added stability measures to prevent segfaults."""
    try:
        # Convert X data to float64 numpy arrays explicitly
        X_train_np = np.asarray(X_train, dtype=np.float64)
        X_test_np = np.asarray(X_test, dtype=np.float64)

        # Replace NaN or infinite values with safe values
        X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Extract y data (time, event)
        time_train = y_train['time'].values
        event_train = y_train['event'].values

        # Replace time values <= 0 with small positive value for numerical stability
        time_train[time_train <= 0] = 1e-8
        
        # Create structured array for scikit-survival
        y_train_sksurv = np.array(
            list(zip(event_train, time_train)), 
            dtype=[('event', bool), ('time', float)]
        )

        # Hyperparameter tuning for stability
        rsf_model = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,  # Increase minimum samples to split for stability
            min_samples_leaf=5,    # Increase minimum samples in leaf for stability
            n_jobs=1,              # Disable parallel processing to prevent memory conflicts
            random_state=42
        )
        
        rsf_model.fit(X_train_np, y_train_sksurv)
        
        risk_scores = rsf_model.predict(X_test_np)
        
        c_index_tuple = concordance_index_censored(
            y_test['event'].astype(bool), y_test['time'], risk_scores
        )

        train_risk_scores = rsf_model.predict(X_train_np)
        train_c_index_tuple = concordance_index_censored(
            y_train['event'].astype(bool), 
            y_train['time'], 
            train_risk_scores
        )
        
        if return_model:
            return c_index_tuple[0], train_c_index_tuple[0], rsf_model
        else:
            return c_index_tuple[0], train_c_index_tuple[0]
    except Exception as e:
        print(f"RSF model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0

def train_xgbse_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train XGBSE model. Assumes y_train/y_test are DataFrames."""
    if not XGBSE_AVAILABLE: return 0.0
    try:
        y_train_xgb = np.array(
            list(zip(1 - y_train['event'], y_train['time'])), 
            dtype=[('censor', bool), ('time', float)]
        )

        num_bins = 11
        _, bin_edges = pd.cut(
            y_train_xgb['time'], bins=num_bins, retbins=True, labels=False, include_lowest=True
        )
        time_bins = bin_edges.tolist()
        # time_bins = list(range(1, 110))
        
        model = XGBSEKaplanNeighbors(n_neighbors=30)
        model.fit(X_train, y_train_xgb, time_bins=time_bins)
        
        survival_probs = model.predict(X_test)
        predicted_times = np.sum(survival_probs, axis=1)
            
        c_index_tuple = concordance_index_censored(
            y_test['event'].astype(bool), y_test['time'], -predicted_times
        )

        train_survival_probs = model.predict(X_train)
        train_predicted_times = np.sum(train_survival_probs, axis=1)
        train_c_index_tuple = concordance_index_censored(
            y_train['event'].astype(bool), 
            y_train['time'], 
            -train_predicted_times
        )
        
        if return_model:
            model_with_bins = {'model': model, 'time_bins': time_bins}
            return c_index_tuple[0], train_c_index_tuple[0], model_with_bins
        else:
            return c_index_tuple[0], train_c_index_tuple[0]
    except Exception as e:
        print(f"XGBSE model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0

