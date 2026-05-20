"""
Title: MSFA-Net: An Advanced Deep Learning Model for Identifying Blue Horizontal-Branch Stars from LAMOST DR12
Authors: Mingyuan Wang, Xiaoming Kong, Jie Ju, Yude Bu, and Yuchen Liang

Code names: msfa_net.py

Language: Python 3.8

License: MIT License

Code tested under the following compilers/operating systems: Python 3.8 / Ubuntu Linux / NVIDIA A100 GPU

Description of input data: 1D spectral fluxes stored in .npz format (containing 'fluxes', 'labels', and 'obsids' arrays extracted from LAMOST DR12).

Description of output data: Trained model weights (.pth) and console printouts of classification metrics (Loss, Macro F1-score).

System requirements: CUDA-enabled GPU (NVIDIA A100 recommended) and minimum 16GB system RAM for efficient batch processing.

Calls to external routines: PyTorch (torch, torch.nn), NumPy (numpy), and scikit-learn (sklearn.metrics.f1_score).

Additional comments: This script implements and trains the Multi-Scale Frequency Attention Network (MSFA-Net) for stellar spectral classification, incorporating Multi-Scale Convolution, Soft Frequency Attention via FFT, and Enhanced Spatial Pyramid Pooling (SPP).

The AAS gives permission to anyone who wishes to use these subroutines to run their own calculations.
Permission to republish or reuse these routines should be directed to permissions@aas.org.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score
import warnings

# 过滤警告
warnings.filterwarnings("ignore")

# ================== 1. 基础设置 ==================
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 任务配置
NUM_CLASSES = 5
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASS_NAMES = ['BHB', 'A-1', 'B-1', 'htsd', 'wd-1']

# 训练超参数
BATCH_SIZE = 96
LR = 1e-3
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 150
SMOOTHING = 0.1


# ================== 2. MSFANet ==================

class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 4
        self.branch3 = nn.Conv1d(channels, mid_channels, 3, padding=1)
        self.branch5 = nn.Conv1d(channels, mid_channels, 5, padding=2)
        self.branch7 = nn.Conv1d(channels, mid_channels, 7, padding=3)
        self.branch1 = nn.Conv1d(channels, mid_channels, 1)
        self.fusion = nn.Conv1d(mid_channels * 4, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([
            self.branch3(x),
            self.branch5(x),
            self.branch7(x),
            self.branch1(x)
        ], dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        return self.relu(out + x)

class SoftFrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.num_bands = 16
        self.band_pool = nn.AdaptiveAvgPool1d(self.num_bands)
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels, bias=False),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, L = x.shape
        fft = torch.fft.rfft(x, norm='ortho')
        mag = torch.abs(fft)
        pha = torch.angle(fft)

        band_mag = self.band_pool(mag)
        channel_feat = band_mag.mean(dim=-1) + 0.3 * band_mag.min(dim=-1).values

        w = self.attention(channel_feat).unsqueeze(-1)
        mag_mod = mag * w

        fft_mod = torch.polar(mag_mod, pha)
        x_freq = torch.fft.irfft(fft_mod, n=L, norm='ortho')

        return x + self.alpha * x_freq

class EnhancedSPP(nn.Module):
    def __init__(self, in_channels, bins=(1, 2, 4)):
        super().__init__()
        self.bins = bins
        self.out_dim = in_channels * sum(bins) * 3

    def forward(self, x):
        features = []
        for b in self.bins:
            pool_mean = F.adaptive_avg_pool1d(x, b).flatten(1)

            mean = F.adaptive_avg_pool1d(x, b)
            pool_sq_mean = F.adaptive_avg_pool1d(x ** 2, b)
            pool_std = torch.sqrt(
                F.relu(pool_sq_mean - mean ** 2) + 1e-8
            ).flatten(1)

            pool_min = -F.adaptive_max_pool1d(-x, b).flatten(1)

            features.extend([pool_mean, pool_std, pool_min])

        return torch.cat(features, dim=1)

class MSFANet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 31, 1, 15, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            MultiScaleBlock(64),
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            MultiScaleBlock(128),
            nn.MaxPool1d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MultiScaleBlock(256),
            SoftFrequencyAttention(256),
            nn.MaxPool1d(2)
        )

        self.spp = EnhancedSPP(256, bins=(1, 2, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.spp.out_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spp(x)
        x = self.dropout(x)
        return self.fc(x)


# ================== 3. 数据处理 ==================

def load_processed_data(path):
    print(f"Loading {path} ...")
    data = np.load(path)
    return data['fluxes'], data['labels'], data['obsids']


class SpectralDataset(Dataset):
    def __init__(self, x, y, orig_labels, obsids):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.orig_labels = torch.tensor(orig_labels, dtype=torch.long)
        self.obsids = np.array(obsids)

        if torch.isnan(self.x).any() or torch.isinf(self.x).any():
            self.x = torch.nan_to_num(self.x)

        print(f"Dataset ready. Shape: {self.x.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx].unsqueeze(0),
                self.y[idx],
                self.orig_labels[idx],
                self.obsids[idx])


# ================== 4. 损失与训练工具 ==================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        if self.weight is not None:
            w = self.weight.unsqueeze(0).expand_as(pred)
            true_dist = true_dist * w

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            out = model(x)
            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(p)
            targets.extend(y.numpy())
    return np.array(preds), np.array(targets)


# ================== 5. 主训练流程 ==================
if __name__ == '__main__':
    main_data_path = './cache/1_flux_label_6+2.npz'
    fluxes, orig_labels, obsids = load_processed_data(main_data_path)

    valid_labels = list(LABEL_MAP.keys())
    mask = np.isin(orig_labels, valid_labels)

    fluxes = fluxes[mask]
    orig_labels = orig_labels[mask]
    obsids = obsids[mask]

    labels = np.array([LABEL_MAP[l] for l in orig_labels])
    dataset = SpectralDataset(fluxes, labels, orig_labels, obsids)

    indices = np.random.permutation(len(dataset))
    split = int(len(dataset) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    model = MSFANet(num_classes=NUM_CLASSES).to(device)

    class_count = np.bincount(labels[train_indices])
    weights = torch.tensor(
        1.0 / np.log1p(class_count + 1),
        dtype=torch.float32
    ).to(device)

    criterion = LabelSmoothingLoss(
        classes=NUM_CLASSES,
        smoothing=SMOOTHING,
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.5, patience=8,
        min_lr=1e-6
    )

    best_macro_f1 = 0.0

    print("Epoch | Loss | MacroF1")
    print("-" * 30)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        preds, targets = evaluate(model, test_loader)

        macro_f1 = f1_score(targets, preds, average='macro')
        scheduler.step(macro_f1)

        print(f"{epoch:>5} | {loss:.4f} | {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            os.makedirs('./model', exist_ok=True)
            torch.save(model.state_dict(), './model/MSFANet.pth')

    print(f"\nTraining Finished. Best Macro F1: {best_macro_f1:.4f}")