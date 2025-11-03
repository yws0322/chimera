# tabm_model.py

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sklearn.base import BaseEstimator
from torch.utils.data import TensorDataset, DataLoader
import warnings

# --- TabM Model Imports ---
import abc
import collections.abc
import typing
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict, Union
import rtdl_num_embeddings
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Unpack
from tabm import TabM

warnings.filterwarnings('ignore')

def discretize_time(time_values, num_bins):
    discretized = pd.cut(time_values, bins=num_bins, labels=False, include_lowest=True)
    return discretized.astype(int)

# --- Survival Analysis Wrapper for TabM ---
class NLLSurvLoss(nn.Module):
    """Negative Log-Likelihood loss for discrete-time survival models."""
    def forward(self, logits: Tensor, y_time: Tensor, y_event: Tensor) -> Tensor:
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
        return neg_log_lik.mean()

class TabMSurvival(BaseEstimator):
    def __init__(self, num_bins: int = 110, k: int = 16, device: str = 'cpu', epochs: int = 50, lr: float = 1e-3):
        self.num_bins = num_bins
        self.k = k
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        self.model_ = TabM.make(
            n_num_features=X.shape[1],
            cat_cardinalities=None,
            d_out=self.num_bins,
            k=self.k,
        ).to(self.device)
        
        y_time, y_event = y['time'].values, y['event'].values
        y_time_discrete = discretize_time(y_time, self.num_bins)

        train_ds = TensorDataset(
            torch.as_tensor(X_scaled, dtype=torch.float32),
            torch.as_tensor(y_time_discrete, dtype=torch.long),
            torch.as_tensor(y_event, dtype=torch.long)
        )
        train_dl = DataLoader(train_ds, batch_size=min(128, len(X)), shuffle=True)
        
        opt = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = NLLSurvLoss()

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0
            for x_batch, t_batch, e_batch in train_dl:
                x_batch, t_batch, e_batch = x_batch.to(self.device), t_batch.to(self.device), e_batch.to(self.device)
                
                ensemble_logits = self.model_(x_num=x_batch)
                
                loss = 0
                for i in range(self.k):
                    loss += criterion(ensemble_logits[:, i, :], t_batch, e_batch)
                loss /= self.k
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        return self

    def predict_time(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            ensemble_logits = self.model_(x_num=X_tensor)
            avg_logits = torch.mean(ensemble_logits, dim=1)
            hazards = torch.sigmoid(avg_logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            expected_survival_time = torch.sum(survival, dim=1)
        return expected_survival_time.cpu().numpy()

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.predict_time(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_risk_score(X)

def train_tabm_model(X_train, y_train, X_test, y_test, device='cpu', return_model=False):
    """Train TabM model. Assumes y_train and y_test are pandas DataFrames."""
    try:
        # y_train and y_test are already DataFrames, no conversion needed
        model = TabMSurvival(
            num_bins=11, 
            k=16,
            device=device, 
            epochs=50,
            lr=1e-3
        )
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
        print(f"TabM model training failed: {e}")
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0