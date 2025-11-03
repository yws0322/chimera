# mlp_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sksurv.metrics import concordance_index_censored

def discretize_time(time_values, num_bins):
    discretized = pd.cut(time_values, bins=num_bins, labels=False, include_lowest=True)
    return discretized.astype(int)

class NLLSurvLoss(nn.Module):
    """Negative Log-Likelihood loss for discrete-time survival models."""
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

class SurvivalNet(nn.Module):
    def __init__(self, input_dim, num_time_bins=11):
        super(SurvivalNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_time_bins)
    
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)
        return logits

class MLPSurvival(BaseEstimator):
    def __init__(self, num_bins: int = 11, device: str = 'cpu', epochs: int = 20, lr: float = 0.001):
        self.num_bins = num_bins
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # y는 항상 DataFrame으로 가정
        time_values = y['time'].values
        event_values = y['event'].values
        
        T_train_discrete = discretize_time(time_values, self.num_bins)
        
        X_train_tensor = torch.FloatTensor(X_scaled).to(self.device)
        T_train_tensor = torch.LongTensor(T_train_discrete).to(self.device)
        Y_train_tensor = torch.LongTensor(event_values).to(self.device)
        
        self.model_ = SurvivalNet(input_dim=X.shape[1], num_time_bins=self.num_bins).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = NLLSurvLoss(reduction='mean')
        
        for epoch in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(X_train_tensor)
            loss = criterion(logits, T_train_tensor, Y_train_tensor)
            loss.backward()
            optimizer.step()
        
        return self

    def predict_time(self, X):
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            test_logits = self.model_(X_tensor)
            test_hazards = torch.sigmoid(test_logits)
            test_survival = torch.cumprod(1 - test_hazards, dim=1)
            predicted_times = torch.sum(test_survival, dim=1)
            
        return predicted_times.cpu().numpy()

    def predict_risk_score(self, X):
        return -self.predict_time(X)
    
    def predict(self, X):
        return self.predict_risk_score(X)

def train_mlp_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train standard MLP model. Assumes y_train/y_test are pandas DataFrames."""
    try:
        # MLPSurvival class handles data processing internally, pass DataFrame as-is
        model = MLPSurvival(num_bins=11, device=device, epochs=50)
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
        print(f"MLP model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0