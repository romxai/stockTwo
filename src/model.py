"""
Neural network model architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution for capturing patterns at different time scales"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15]):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for k in kernel_sizes
        ])

        self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        x = x.transpose(1, 2)
        outputs = [conv(x) for conv in self.convs]
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, outputs))
        return fused.transpose(1, 2)


class CrossModalAttention(nn.Module):
    """Bidirectional attention between numerical and text features"""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.num_to_text_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_num_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, num_features, text_features):
        num_enhanced, num_attn_weights = self.num_to_text_attn(
            num_features, text_features, text_features
        )
        num_enhanced = self.norm1(num_features + num_enhanced)
        num_enhanced = num_enhanced + self.ffn(num_enhanced)

        text_enhanced, text_attn_weights = self.text_to_num_attn(
            text_features, num_features, num_features
        )
        text_enhanced = self.norm2(text_features + text_enhanced)
        text_enhanced = text_enhanced + self.ffn(text_enhanced)

        self.num_attn_weights = num_attn_weights
        self.text_attn_weights = text_attn_weights

        return num_enhanced, text_enhanced


class MarketRegimeDetector(nn.Module):
    """Detects market regime (bullish/bearish/volatile/stable)"""

    def __init__(self, input_dim, num_regimes=4):
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_regimes)
        )

    def forward(self, x):
        return self.detector(x)


class AdaptiveFeatureFusion(nn.Module):
    """Learns gating mechanism to weight numerical vs text features"""

    def __init__(self, d_model):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )

        self.interaction = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, num_features, text_features):
        combined = torch.cat([num_features, text_features], dim=-1)
        gates = self.gate(combined)
        interaction = self.interaction(combined)
        fused = (gates[:, 0:1] * num_features +
                gates[:, 1:2] * text_features +
                gates[:, 2:3] * interaction)
        return fused


class UltraAdvancedStockPredictor(nn.Module):
    """
    Novel hybrid architecture combining:
    - Multi-scale TCN for price patterns
    - Transformer for text encoding
    - Cross-modal attention
    - Market regime detection
    - Multi-task learning
    """

    def __init__(self,
                 num_numerical_features,
                 num_text_features,
                 d_model=512,
                 num_heads=8,
                 num_lstm_layers=3,
                 num_transformer_layers=4,
                 num_regimes=4,
                 dropout=0.3):
        super().__init__()

        self.d_model = d_model

        # 1. INPUT PROJECTION
        self.num_projection = nn.Linear(num_numerical_features, d_model)
        self.text_projection = nn.Linear(num_text_features, d_model)

        # 2. NUMERICAL BRANCH
        self.multi_scale_conv = MultiScaleTemporalConv(d_model, d_model)
        self.num_lstm = nn.LSTM(
            d_model, d_model // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # 3. TEXT BRANCH
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.text_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.text_lstm = nn.LSTM(
            d_model, d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 4. CROSS-MODAL ATTENTION
        self.cross_modal_attention = CrossModalAttention(d_model, num_heads, dropout)

        # 5. MARKET REGIME DETECTOR
        self.regime_detector = MarketRegimeDetector(d_model, num_regimes)

        # 6. REGIME-SPECIFIC PROCESSING
        self.regime_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_regimes)
        ])

        # 7. ADAPTIVE FUSION
        self.adaptive_fusion = AdaptiveFeatureFusion(d_model)

        # 8. FINAL LAYERS
        self.final_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 9. PREDICTION HEADS
        self.direction_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, X_num, X_text, return_attention=False):
        batch_size = X_num.size(0)

        # 1. PROJECT
        num_features = self.num_projection(X_num)
        
        # --- MODIFY THIS BLOCK ---
        # text_features = self.text_projection(X_text)
        
        # EXPERIMENT: Mute text features to see if price data works alone
        text_features = torch.zeros_like(num_features).to(num_features.device)
        # --- END MODIFICATION ---

        # 2. PROCESS NUMERICAL
        num_features = self.multi_scale_conv(num_features)
        num_features, _ = self.num_lstm(num_features)

        # 3. PROCESS TEXT
        text_features = self.text_transformer(text_features)
        text_features, _ = self.text_lstm(text_features)

        # 4. CROSS-MODAL ATTENTION
        num_enhanced, text_enhanced = self.cross_modal_attention(
            num_features, text_features
        )

        # 5. TEMPORAL POOLING
        num_final = num_enhanced[:, -1, :]
        text_final = text_enhanced[:, -1, :]

        # 6. DETECT REGIME
        regime_logits = self.regime_detector(num_final)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # 7. REGIME-SPECIFIC PROCESSING
        regime_features = []
        for i, head in enumerate(self.regime_heads):
            regime_feat = head(num_final)
            regime_features.append(regime_feat * regime_probs[:, i:i+1])
        regime_processed = sum(regime_features)

        # 8. ADAPTIVE FUSION
        fused = self.adaptive_fusion(regime_processed, text_final)

        # 9. FINAL ENCODING
        encoded = self.final_encoder(fused)

        # 10. PREDICTIONS
        direction_logits = self.direction_head(encoded)
        magnitude = self.magnitude_head(encoded).squeeze(-1)
        volatility = self.volatility_head(encoded).squeeze(-1)

        if return_attention:
            return (direction_logits, magnitude, volatility,
                   self.cross_modal_attention.num_attn_weights,
                   self.cross_modal_attention.text_attn_weights)

        return direction_logits, magnitude, volatility

    def predict_with_confidence(self, X_num, X_text, mc_samples=10):
        """Predict with uncertainty estimation using MC Dropout"""
        self.train()

        direction_preds = []

        with torch.no_grad():
            for _ in range(mc_samples):
                dir_logits, _, _ = self.forward(X_num, X_text)
                dir_probs = F.softmax(dir_logits, dim=-1)
                direction_preds.append(dir_probs)

        direction_preds = torch.stack(direction_preds)
        mean_probs = direction_preds.mean(dim=0)
        predictions = mean_probs.argmax(dim=-1)
        variance = direction_preds.var(dim=0).mean(dim=-1)
        confidence = 1.0 / (1.0 + variance)

        self.eval()

        return predictions, mean_probs, confidence


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting"""

    def __init__(self):
        super().__init__()
        self.log_var_direction = nn.Parameter(torch.zeros(1))
        self.log_var_magnitude = nn.Parameter(torch.zeros(1))
        self.log_var_volatility = nn.Parameter(torch.zeros(1))
        # --- CHANGE THIS ---
        # self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_loss = nn.CrossEntropyLoss() # Use standard CE Loss
        # --- END CHANGE ---
        self.mse_loss = nn.MSELoss()

    def forward(self, dir_logits, dir_targets, mag_preds, mag_targets, vol_preds, vol_targets):
        # Focus ONLY on direction classification for now
        # Disable magnitude and volatility tasks to simplify the problem
        dir_loss = self.focal_loss(dir_logits, dir_targets)

        # COMMENTED OUT: Multi-task learning - model was ignoring hard direction task
        # mag_loss = self.mse_loss(mag_preds, mag_targets)
        # vol_loss = self.mse_loss(vol_preds, vol_targets)
        #
        # precision_dir = torch.exp(-self.log_var_direction)
        # precision_mag = torch.exp(-self.log_var_magnitude)
        # precision_vol = torch.exp(-self.log_var_volatility)
        #
        # weighted_dir_loss = precision_dir * dir_loss + self.log_var_direction
        # weighted_mag_loss = precision_mag * mag_loss + self.log_var_magnitude
        # weighted_vol_loss = precision_vol * vol_loss + self.log_var_volatility
        #
        # total_loss = weighted_dir_loss + weighted_mag_loss + weighted_vol_loss

        total_loss = dir_loss  # Just use direction loss

        return total_loss, {
            'direction': dir_loss.item(),
            'magnitude': 0,  # mag_loss.item() if 'mag_loss' in locals() else 0,
            'volatility': 0,  # vol_loss.item() if 'vol_loss' in locals() else 0,
            'total': total_loss.item()
        }
