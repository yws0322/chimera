# real_mlp_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
from sksurv.metrics import concordance_index_censored

def discretize_time(time_values, num_bins):
    discretized = pd.cut(time_values, bins=num_bins, labels=False, include_lowest=True)
    return discretized.astype(int)

class NLLSurvLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, y_time, y_event):
        y_time = y_time.type(torch.int64).unsqueeze(1)
        y_event = y_event.type(torch.int64).unsqueeze(1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones_like(y_event, dtype=torch.float), S], 1)
        s_prev = torch.gather(S_padded, 1, y_time).clamp(min=1e-7)
        h_this = torch.gather(hazards, 1, y_time).clamp(min=1e-7)
        log_lik_uncensored = torch.log(s_prev) + torch.log(h_this)
        log_lik_censored = torch.log(torch.gather(S_padded, 1, y_time + 1).clamp(min=1e-7))
        neg_log_lik = - (y_event * log_lik_uncensored + (1 - y_event) * log_lik_censored)
        if self.reduction == 'mean':
            return neg_log_lik.mean()
        return neg_log_lik.sum()

class ScalingLayer(nn.Module):
    def __init__(self, n_features: int): 
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x * self.scale[None, :]

class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x.mul(torch.tanh(F.softplus(x)))

class SimpleMLP(BaseEstimator):
    def __init__(self, num_bins: int = 11, device: str = 'cpu'):
        self.num_bins, self.device = num_bins, device

    def fit(self, X, y, X_val=None, y_val=None):
        input_dim = X.shape[1]
        
        y_time, y_event = y['time'].values, y['event'].values
        
        y_time_discrete = discretize_time(y_time, self.num_bins)
        
        model = nn.Sequential(
            ScalingLayer(input_dim), NTPLinear(input_dim, 256), Mish(),
            NTPLinear(256, 256), Mish(), NTPLinear(256, 256), Mish(),
            NTPLinear(256, self.num_bins, zero_init=True),
        ).to(self.device)
        train_ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                                 torch.as_tensor(y_time_discrete, dtype=torch.long),
                                 torch.as_tensor(y_event, dtype=torch.long))
        train_dl = DataLoader(train_ds, batch_size=min(256, len(X)), shuffle=True, drop_last=True)
        opt, criterion, n_epochs = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95)), NLLSurvLoss(), 40
        
        for epoch in range(n_epochs):
            model.train()
            for x_batch, t_batch, e_batch in train_dl:
                x_batch, t_batch, e_batch = x_batch.to(self.device), t_batch.to(self.device), e_batch.to(self.device)
                logits = model(x_batch)
                loss = criterion(logits, t_batch, e_batch)
                opt.zero_grad(); loss.backward(); opt.step()
        self.model_ = model
        return self

    def predict_time(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(X, dtype=torch.float32).to(self.device))
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            predicted_times = torch.sum(survival, dim=1)
        return predicted_times.cpu().numpy()

    def predict_risk_score(self, X):
        return -self.predict_time(X)
    
    def predict(self, X):
        return self.predict_risk_score(X)

class RealMLPSurvival(BaseEstimator):
    def __init__(self, num_bins=11, device='cpu'):
        self.num_bins, self.device = num_bins, device

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = SimpleMLP(num_bins=self.num_bins, device=self.device)
        self.model_.fit(X_scaled, y)
        return self

    def predict_time(self, X):
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_time(X_scaled)

    def predict_risk_score(self, X):
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_risk_score(X_scaled)
    
    def predict(self, X):
        return self.predict_risk_score(X)

def train_real_mlp_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train Real MLP model. Assumes y_train/y_test are pandas DataFrames."""
    try:
        # RealMLPSurvival class handles data processing internally, pass DataFrame as-is
        model = RealMLPSurvival(num_bins=11, device=device)
        model.fit(X_train, y_train)
        
        test_risk_scores = model.predict_risk_score(X_test)
        
        c_index_tuple = concordance_index_censored(
            y_test['event'].astype(bool), 
            y_test['time'], 
            test_risk_scores
        )

        train_risk_scores = model.predict_risk_score(X_train)
        train_c_index_tuple = concordance_index_censored(
            y_train['event'].astype(bool), 
            y_train['time'], 
            train_risk_scores
        )

        if return_model:
            return c_index_tuple[0], train_c_index_tuple[0], model
        else:
            return c_index_tuple[0], train_c_index_tuple[0]
        
    except Exception as e:
        print(f"Real MLP model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0