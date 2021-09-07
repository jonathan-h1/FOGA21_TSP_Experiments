#!/usr/bin/env python3.8
# coding: utf-8

# Import all required libraries
#import setGPU
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Specify Python Parser
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--normed', type=bool, default=False)
args = parser.parse_args()

# Specify Training Hyperparameter
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-04

VAL_FOLD = [args.fold]
NORMED = args.normed
RUN = args.run

# Specify required data paths
FEATURE_SET = '../data/feature_set.csv'
PERFORMANCE_DATA = '../data/combined_performance_data.csv'
MODEL_NAME = 'autoencoder_combined_mlp_{}_{}_{}'.format(('normed' if NORMED else 'unnormed'), RUN, VAL_FOLD[0])


# Considered features hardcoded as list
feature_sup_set = [ 'mst_depth_mean',
                    'mst_depth_median',
                    'mst_depth_max',
                    'mst_depth_span',
                    'nng_3_n_weak',
                    'nng_3_n_strong',
                    'nng_3_weak_components_mean',
                    'nng_3_weak_components_median',
                    'nng_3_weak_components_min',
                    'nng_3_weak_components_max',
                    'nng_3_weak_components_span',
                    'nng_3_strong_components_mean',
                    'nng_3_strong_components_median',
                    'nng_3_strong_components_max',
                    'nng_3_strong_components_span',
                    'nng_5_n_weak',
                    'nng_5_n_strong',
                    'nng_5_weak_components_mean',
                    'nng_5_weak_components_median',
                    'nng_5_weak_components_min',
                    'nng_5_weak_components_max',
                    'nng_5_weak_components_span',
                    'nng_5_strong_components_mean',
                    'nng_5_strong_components_median',
                    'nng_5_strong_components_min',
                    'nng_5_strong_components_max',
                    'nng_5_strong_components_span',
                    'nng_7_n_weak',
                    'nng_7_n_strong',
                    'nng_7_weak_components_mean',
                    'nng_7_weak_components_median',
                    'nng_7_weak_components_min',
                    'nng_7_weak_components_max',
                    'nng_7_weak_components_span',
                    'nng_7_strong_components_mean',
                    'nng_7_strong_components_median',
                    'nng_7_strong_components_min',
                    'nng_7_strong_components_max',
                    'nng_7_strong_components_span']


# Load and build the Training/Validation set
features = pd.read_csv(FEATURE_SET)
performance = pd.read_csv(PERFORMANCE_DATA)

# Merge features and performance datasets
idx = (features['norm'] == NORMED)
features = features[idx].fillna(0.)
performance = performance[['root', 'group', 'prob', 'fold', 'eax.par10', 'lkh.par10']]
combined_set = pd.merge(performance, features, how='inner', on=('root', 'group', 'prob'))
combined_set = combined_set.drop(['norm'], axis=1)
combined_set = combined_set[['root', 'group', 'prob', 'fold', 'eax.par10', 'lkh.par10'] + feature_sup_set]

# Split datasets into training/validation set
valid_idx = combined_set['fold'].isin(VAL_FOLD)
X_train = combined_set[~valid_idx]
X_valid = combined_set[valid_idx]

# Drop all irrelevant columns
X_train = X_train.drop(['root', 'group', 'prob', 'fold', 'eax.par10', 'lkh.par10'], axis=1).to_numpy()
X_valid = X_valid.drop(['root', 'group', 'prob', 'fold', 'eax.par10', 'lkh.par10'], axis=1).to_numpy()
X_total = combined_set.drop(['root', 'group', 'prob', 'fold', 'eax.par10', 'lkh.par10'], axis=1).to_numpy()


# If normalized data are used, no min-max-normalization is used
if not NORMED:
    x_min, x_max = X_train.min(0), X_train.max(0)
    X_train = (X_train - x_min) / (x_max - x_min + 1e-8)
    X_valid = (X_valid - x_min) / (x_max - x_min + 1e-8)
    X_total = (X_total - x_min) / (x_max - x_min + 1e-8)


# Defined PyTorch dataloaders
train_data_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float)), 
                               batch_size=BATCH_SIZE, shuffle=True)

valid_data_loader = DataLoader(TensorDataset(torch.tensor(X_valid, dtype=torch.float)), 
                               batch_size=BATCH_SIZE, shuffle=True)


# Define AE architecture
model = nn.Sequential(
            nn.Sequential( # Encoder
                nn.Linear(X_train.shape[1], 512),
                nn.BatchNorm1d(512),
                nn.GELU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),

                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.GELU(),

                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.GELU(),

                nn.Linear(8, 4),
                nn.BatchNorm1d(4),
                nn.GELU(),

                nn.Linear(4, 2),
                nn.BatchNorm1d(2),
                nn.GELU(),
            ),
            nn.Sequential( # Decoder
                nn.Linear(2, 4),
                nn.BatchNorm1d(4),
                nn.GELU(),

                nn.Linear(4, 8),
                nn.BatchNorm1d(8),
                nn.GELU(),

                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.GELU(),

                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),

                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),

                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),

                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),

                nn.Linear(512, X_train.shape[1]),
            )
        )
_ = model.cuda()

# PyTorch Loss-Function, Optimizer and scheduler
loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=50, verbose=True)


# Training Loop
best_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    train_losses = []
    valid_losses = []
    
    # Training
    model.train()
    for batch in train_data_loader:
        X = batch[0].cuda()
        y = model(X)
        
        loss = loss_func(y, X.detach())
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del X, y, loss
    
    # Evaluation
    model.eval()
    for batch in valid_data_loader:
        with torch.no_grad():
            X = batch[0].cuda()
            y = model(X)

            loss = loss_func(y, X.detach())
            valid_losses.append(loss.item())
        del X, y, loss
    
    t_loss = round(sum(train_losses) / len(train_losses), 4)
    v_loss = round(sum(valid_losses) / len(valid_losses), 4)
    scheduler.step(v_loss)
    print('Epoch {}/{}: Loss {}, Val. Loss {}'.format(epoch, EPOCHS, t_loss, v_loss))
    
    if v_loss <= best_loss:
        best_loss = v_loss
        torch.save(model, 'models/{}.pt'.format(MODEL_NAME))
_ = model.cpu()
del model

# Reload best model
model = torch.load('models/{}.pt'.format(MODEL_NAME)).cuda()

# Use encoder to create embeddings on the whole dataset as we are only interested in the embedding and not in the model..
with torch.no_grad():
    model.eval()
    out = model[0](torch.tensor(X_total, dtype=torch.float).cuda()).cpu()

# Save dataframe
data = pd.DataFrame({'X1' : out.numpy()[:,0], 'X2' : out.numpy()[:,1]})
data = pd.concat([combined_set[['root', 'group', 'prob']], data], axis=1)
data.to_csv('results/{}.csv'.format(MODEL_NAME))