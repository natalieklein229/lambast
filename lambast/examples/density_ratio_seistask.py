"""
Density Ratio Weighting Demo: Seismic Binary Classification

This example demonstrates covariate shift mitigation using
density-ratio weighting on a synthetic seismic dataset.
The dataset is located at: https://zenodo.org/records/17943530

Setup:
    1. Place dataset files under:
       lambast/datasets/seistask/

    2. Expected files:
       - SeisTask_data.h5
       - SeisTask_metadata.csv

Here, we split source and target by source_type (gabor or ricker).

This script:
    - Trains a baseline classifier on source
    - Evaluates on target
    - Learns density-ratio weights using unlabeled source/target
    - Retrains weighted classifier
    - Compares performance

Run:
    python examples/density_ratio_seismic_binary.py
"""
# %%
from importlib import resources

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from lambast.mitigation_methods.density_ratio import DensityRatioWeighter
from lambast.mitigation_methods.density_ratio.models import BinaryCNN
from lambast.mitigation_methods.density_ratio.train_task import (
    eval_binary_accuracy, train_binary_classifier)

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get path to lambast package
with resources.as_file(resources.files("lambast")) as package_root:
    data_dir = package_root / "datasets" / "seistask"

# %% Load data
with h5py.File('%s/SeisTask_data.h5' % data_dir, 'r') as f:
    # List all groups and datasets in the file
    print("Keys: %s" % f.keys())
    print(f['data'].shape)
    data = f['data'][:]

# split source/target and make train/test
meta = pd.read_csv('%s/SeisTask_metadata.csv' % data_dir)
source_ix = np.where(meta['source_type'] == 'ricker')[0]
target_ix = np.where(meta['source_type'] == 'gabor')[0]
source_y = meta.iloc[source_ix]['signal'].to_numpy()
target_y = meta.iloc[target_ix]['signal'].to_numpy()
source_x = data[source_ix]
target_x = data[target_ix]


def train_test_val(x, y, train_size=0.8):
    train_ix, val_ix = train_test_split(
        np.arange(len(x)), test_size=1 - train_size)
    val_ix, test_ix = train_test_split(val_ix, test_size=0.5)
    return {'x': {'train': x[train_ix], 'val': x[val_ix], 'test': x[test_ix]},
            'y': {'train': y[train_ix], 'val': y[val_ix], 'test': y[test_ix]}}


source_split = train_test_val(source_x, source_y)
target_split = train_test_val(target_x, target_y)

x_mean = np.mean(source_split['x']['train'], axis=0, keepdims=True)
x_std = np.std(source_split['x']['train'], axis=0, keepdims=True)
for k in ['train', 'val', 'test']:
    source_split['x'][k] = (source_split['x'][k] - x_mean) / x_std
    target_split['x'][k] = (target_split['x'][k] - x_mean) / x_std

plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(source_split['x']['train'][:10, 0, :].T)
plt.title('Source')
plt.subplot(212)
plt.plot(target_split['x']['train'][:10, 0, :].T)
plt.title('Target')
plt.show()

acc_results = []

# %% Baseline
# Train classifier on source data, test on target data
baseline_bin = BinaryCNN(in_channels=2)
baseline, hist = train_binary_classifier(baseline_bin,
                                         source_split['x']['train'],
                                         source_split['y']['train'],
                                         X_val=source_split['x']['val'],
                                         y_val=source_split['y']['val'],
                                         device=device,
                                         epochs=150, lr=0.01)
src_acc = eval_binary_accuracy(
    baseline,
    source_split['x']['test'],
    source_split['y']['test'],
    device=device)
trg_acc = eval_binary_accuracy(
    baseline,
    target_split['x']['test'],
    target_split['y']['test'],
    device=device)
print(f"Baseline (train on source) test accuracy on source: {
    src_acc['acc']:0.2f}, target: {trg_acc['acc']:0.2f}")

plt.figure()
plt.plot(hist["train_loss"], label='train')
plt.plot(hist["val_loss"], label='val')
plt.legend()
plt.title('Baseline CNN loss')
plt.show()
acc_results.append(
    {'model': 'base_source', 'test_set': 'source', 'acc': src_acc['acc']})
acc_results.append(
    {'model': 'base_source', 'test_set': 'target', 'acc': trg_acc['acc']})

# train classifier on target data, test on target data
baseline_target_bin = BinaryCNN(in_channels=2)
baseline_target, hist = train_binary_classifier(baseline_target_bin,
                                                target_split['x']['train'],
                                                target_split['y']['train'],
                                                X_val=target_split['x']['val'],
                                                y_val=target_split['y']['val'],
                                                device=device,
                                                epochs=150,
                                                lr=0.01)
src_acc = eval_binary_accuracy(
    baseline_target,
    source_split['x']['test'],
    source_split['y']['test'],
    device=device)
trg_acc = eval_binary_accuracy(
    baseline_target,
    target_split['x']['test'],
    target_split['y']['test'],
    device=device)
print(f"Baseline (train on source) test accuracy on source: {
    src_acc['acc']:0.2f}, target: {trg_acc['acc']:0.2f}")
plt.figure()
plt.plot(hist["train_loss"], label='train')
plt.plot(hist["val_loss"], label='val')
plt.legend()
plt.title('Baseline (target) CNN loss')
plt.show()
acc_results.append(
    {'model': 'base_target', 'test_set': 'source', 'acc': src_acc['acc']})
acc_results.append(
    {'model': 'base_target', 'test_set': 'target', 'acc': trg_acc['acc']})

# %% Density ratio weighting
# Train domain classifier to reweight source data
drw = DensityRatioWeighter(epochs=300, batch_size=256, device=device, lr=0.001)
drw = drw.fit(
    source_split['x']['train'],
    target_split['x']['train'],
    source_split['x']['val'],
    target_split['x']['val'])
# print("Weighter diagnostics:", drw.diagnostics_)
print(drw.diagnostics_)

plt.figure()
plt.plot(drw.history_["train_loss"], label='train')
plt.plot(drw.history_["val_loss"], label='val')
plt.legend()
plt.title('Domain classifier loss')
plt.show()

w_source = drw.compute_weights(source_split['x']['train'], alpha=0.3)
print('weight mean: %0.2f, sd: %0.2f, min: %0.2f, max: %0.2f' %
      (w_source.mean(), w_source.std(), w_source.min(), w_source.max()))
ess = w_source.sum()**2 / (w_source**2).sum()
print('ESS/n %0.2f' % (ess / len(w_source)))


# %% Weighted training
# Weighted training: use density ratio weights and apply to target
weighted_bin = BinaryCNN(in_channels=2)
weighted, hist = train_binary_classifier(weighted_bin,
                                         source_split['x']['train'],
                                         source_split['y']['train'],
                                         sample_weight=w_source,
                                         X_val=source_split['x']['val'],
                                         y_val=source_split['y']['val'],
                                         device=device,
                                         epochs=150,
                                         lr=0.01)
src_acc = eval_binary_accuracy(
    weighted,
    source_split['x']['test'],
    source_split['y']['test'],
    device=device)
trg_acc = eval_binary_accuracy(
    weighted,
    target_split['x']['test'],
    target_split['y']['test'],
    device=device)
print(f"Weighted test accuracy on source: {
    src_acc['acc']:0.2f}, target: {trg_acc['acc']:0.2f}")
plt.figure()
plt.plot(hist["train_loss"], label='train')
plt.plot(hist["val_loss"], label='val')
plt.legend()
plt.title('Weighted CNN loss')
plt.show()
acc_results.append(
    {'model': 'weighted', 'test_set': 'source', 'acc': src_acc['acc']})
acc_results.append(
    {'model': 'weighted', 'test_set': 'target', 'acc': trg_acc['acc']})
# %%
acc_results = pd.DataFrame(acc_results)
print(acc_results)
