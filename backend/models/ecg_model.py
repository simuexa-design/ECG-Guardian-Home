"""
Deep Learning Model Architecture for Multi-Label ECG Arrhythmia Classification
SE-ResNet-1D: 1D Multi-Scale Residual Network with Squeeze-and-Excitation (SE)
and Multi-Head Attention Pooling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 10 Standard Multi-Label Arrhythmia Classes
CLASSES = [
    {
        'id': 'NORM',
        'name': 'Normal Sinus Rhythm',
        'code': 'NORM',
        'category': 'Normal',
        'description': 'Regular rhythm originating from the SA node (60-100 bpm) with normal P-QRS-T complexes.'
    },
    {
        'id': 'AFIB',
        'name': 'Atrial Fibrillation',
        'code': 'AFIB',
        'category': 'Supraventricular Arrhythmia',
        'description': 'Chaotic atrial depolarization with irregularly irregular ventricular response and absent P waves.'
    },
    {
        'id': 'AFL',
        'name': 'Atrial Flutter',
        'code': 'AFL',
        'category': 'Supraventricular Arrhythmia',
        'description': 'Rapid regular atrial flutter waves in characteristic "sawtooth" morphology (typically 250-350 bpm).'
    },
    {
        'id': 'LBBB',
        'name': 'Left Bundle Branch Block',
        'code': 'LBBB',
        'category': 'Conduction Abnormality',
        'description': 'Delayed left ventricular depolarization causing widened QRS (>120ms) and slurred broad R waves.'
    },
    {
        'id': 'RBBB',
        'name': 'Right Bundle Branch Block',
        'code': 'RBBB',
        'category': 'Conduction Abnormality',
        'description': 'Delayed right ventricular depolarization with characteristic rSR\' "rabbit ears" pattern in precordial leads.'
    },
    {
        'id': 'PAC',
        'name': 'Premature Atrial Contraction',
        'code': 'PAC',
        'category': 'Ectopic Beat',
        'description': 'Premature atrial beat originating outside the SA node, featuring an abnormal early P wave morphology.'
    },
    {
        'id': 'PVC',
        'name': 'Premature Ventricular Contraction',
        'code': 'PVC',
        'category': 'Ectopic Beat',
        'description': 'Ectopic ventricular beat with premature, wide, bizarre QRS complex and discordant T wave.'
    },
    {
        'id': 'STACH',
        'name': 'Sinus Tachycardia',
        'code': 'STACH',
        'category': 'Sinus Rhythm Variant',
        'description': 'Accelerated normal sinus rhythm with heart rate strictly exceeding 100 beats per minute.'
    },
    {
        'id': 'SBRAD',
        'name': 'Sinus Bradycardia',
        'code': 'SBRAD',
        'category': 'Sinus Rhythm Variant',
        'description': 'Slow normal sinus rhythm with resting heart rate below 60 beats per minute.'
    },
    {
        'id': 'MI',
        'name': 'Myocardial Infarction / Ischemia',
        'code': 'MI',
        'category': 'Ischemic Pathology',
        'description': 'Acute or evolving myocardial injury evidenced by significant ST-segment deviation or pathological Q waves.'
    }
]

CLASS_IDS = [c['id'] for c in CLASSES]
CLASS_NAMES = [c['name'] for c in CLASSES]
NUM_CLASSES = len(CLASSES)


class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        # x: (batch, channels, length)
        b, c, _ = x.shape
        w = torch.mean(x, dim=2)  # Global average pooling -> (batch, channels)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))  # (batch, channels)
        w = w.unsqueeze(2)  # (batch, channels, 1)
        return x * w


class ResBlock1D(nn.Module):
    """1D Residual Block with Squeeze-and-Excitation."""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        padding1 = ((7 - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7,
            stride=stride, padding=padding1, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        out = self.dropout(out)
        return out


class AttentionPooling1D(nn.Module):
    """Multi-Head Temporal Attention Pooling for 1D features."""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, channels, length)
        # Permute to (batch, length, channels)
        x_perm = x.permute(0, 2, 1)
        attn = self.attention_weights(x_perm)  # (batch, length, 1)
        attn = F.softmax(attn, dim=1)
        # Weighted sum over length
        pooled = torch.sum(x_perm * attn, dim=1)  # (batch, channels)
        return pooled, attn.squeeze(-1)


class ECGResNet1D(nn.Module):
    """
    Multi-Scale Deep Residual Network for Multi-Label ECG Arrhythmia Classification.
    Accepts 1D ECG signals of shape (batch, 1, 2500).
    Outputs raw logits of shape (batch, NUM_CLASSES).
    """
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem: Initial convolution with wide receptive field
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 32 -> 64 filters
        self.stage1 = nn.Sequential(
            ResBlock1D(base_filters, base_filters * 2, stride=2, dilation=1),
            ResBlock1D(base_filters * 2, base_filters * 2, stride=1, dilation=2)
        )

        # Stage 2: 64 -> 128 filters
        self.stage2 = nn.Sequential(
            ResBlock1D(base_filters * 2, base_filters * 4, stride=2, dilation=1),
            ResBlock1D(base_filters * 4, base_filters * 4, stride=1, dilation=2)
        )

        # Stage 3: Target feature maps for Grad-CAM
        self.stage3 = nn.Sequential(
            ResBlock1D(base_filters * 4, base_filters * 8, stride=2, dilation=1),
            ResBlock1D(base_filters * 8, base_filters * 8, stride=1, dilation=1)
        )

        # Attention Pooling
        self.attn_pool = AttentionPooling1D(base_filters * 8)
        self.global_avg = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        feat_dim = base_filters * 8 * 2  # Combined attention + avg pool
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, L)
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        target_conv = self.stage3(feat)  # (B, 256, L')

        # Combined temporal attention + global avg pooling
        pooled_attn, _ = self.attn_pool(target_conv)
        pooled_avg = self.global_avg(target_conv).squeeze(-1)
        combined = torch.cat([pooled_attn, pooled_avg], dim=1)

        logits = self.classifier(combined)
        return logits

    def get_target_layer(self):
        """Returns the final convolutional block for Grad-CAM activation mapping."""
        return self.stage3[-1].conv2

