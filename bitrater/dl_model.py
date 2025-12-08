"""PyTorch model architecture for deep learning bitrate classification.

Two-stage CNN + BiLSTM architecture:
- Stage 1: CNN feature extractor on dual-band spectrograms (128 bins, 2-second windows)
- Stage 2: BiLSTM + Multi-head Attention + auxiliary features for sequence classification
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """CNN that extracts a fixed-size feature vector from a single spectrogram window.

    Input: (batch, 1, 128, time_frames) — dual-band spectrogram
    Output: (batch, feature_dim)
    """

    def __init__(self, feature_dim=128, dropout=0.2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: (1, 128, T) -> (32, 64, T/2)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            # Block 2: (32, 64, T/2) -> (64, 32, T/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            # Block 3: (64, 32, T/4) -> (128, 16, T/8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            # Block 4: (128, 16, T/8) -> (256, 8, T/16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        return self.project(x)


class WindowClassifier(nn.Module):
    """Stage 1: CNN + linear head for per-window classification.

    Used to warm-start the CNN before sequence training.
    """

    def __init__(self, n_classes=7, feature_dim=128, dropout=0.2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(feature_dim=feature_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.head(features)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling over a sequence.

    Learns which windows in the sequence are most informative.
    """

    def __init__(self, input_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.Linear(input_dim, n_heads)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_logits = self.attention(x)
        attn_weights = torch.softmax(attn_logits, dim=1)

        attended = []
        head_dim = x.size(-1) // self.n_heads
        for h in range(self.n_heads):
            w = attn_weights[:, :, h].unsqueeze(-1)
            x_slice = x[:, :, h * head_dim : (h + 1) * head_dim]
            attended.append((w * x_slice).sum(dim=1))

        return torch.cat(attended, dim=-1)


class SequenceClassifier(nn.Module):
    """Stage 2: BiLSTM + Attention + auxiliary features on CNN feature sequences.

    Combines three representations:
    - Attention-pooled BiLSTM output (learned temporal patterns)
    - Temporal variance of LSTM outputs (VBR/CBR signal)
    - Auxiliary features (SVM spectral features + global modulation)
    """

    def __init__(self, feature_dim=128, hidden_dim=128, n_layers=2,
                 n_classes=7, n_heads=4, dropout=0.3, use_variance_features=True,
                 n_frame_features=0):
        super().__init__()
        self.use_variance_features = use_variance_features
        self.n_frame_features = n_frame_features

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2

        self.attention = MultiHeadAttentionPooling(lstm_out_dim, n_heads=n_heads)

        combined_dim = lstm_out_dim
        if use_variance_features:
            combined_dim += lstm_out_dim * 2
        if n_frame_features > 0:
            combined_dim += n_frame_features

        self.embedding_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, n_classes)

        self.vbr_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, frame_features=None):
        lstm_out, _ = self.lstm(x)
        pooled = self.attention(lstm_out)

        parts = [pooled]
        if self.use_variance_features:
            temporal_std = lstm_out.std(dim=1)
            temporal_range = lstm_out.max(dim=1).values - lstm_out.min(dim=1).values
            parts.extend([temporal_std, temporal_range])
        if frame_features is not None and self.n_frame_features > 0:
            parts.append(frame_features)

        combined = torch.cat(parts, dim=-1)
        embedding = self.embedding_proj(combined)

        logits = self.classifier(embedding)
        vbr_pred = self.vbr_head(embedding)

        return logits, vbr_pred, embedding
