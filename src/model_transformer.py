"""
Transformer-based model architecture for stock prediction
Based on pure attention mechanism with cross-modal fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness"""
    def __init__(self, d_model, max_len=150):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerStockPredictor(nn.Module):
    """
    Pure Transformer architecture for stock prediction
    - Multi-head self-attention for temporal patterns
    - Cross-attention between price and news
    - Multi-task prediction heads
    """
    def __init__(self, num_features, text_features, d_model=256, nhead=8, 
                 num_layers=6, dropout=0.2):
        super().__init__()
        
        # Input projections
        self.num_projection = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for numerical features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.num_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer encoder for text features
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Global pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_num, x_text, return_attention=False):
        # Project inputs
        num_emb = self.num_projection(x_num)  # (B, T, d_model)
        text_emb = self.text_projection(x_text)  # (B, T, d_model)
        
        # Add positional encoding
        num_emb = self.pos_encoder(num_emb)
        text_emb = self.pos_encoder(text_emb)
        
        # Self-attention within each modality
        num_encoded = self.num_transformer(num_emb)  # (B, T, d_model)
        text_encoded = self.text_transformer(text_emb)  # (B, T, d_model)
        
        # Cross-attention: numerical queries text
        cross_attn_out, attn_weights = self.cross_attention(
            num_encoded, text_encoded, text_encoded
        )  # (B, T, d_model)
        
        # Fuse modalities
        fused = torch.cat([num_encoded, cross_attn_out], dim=-1)  # (B, T, 2*d_model)
        fused = self.fusion(fused)  # (B, T, d_model)
        
        # Attention-based pooling (focus on important time steps)
        attn_weights_pool = self.attention_pool(fused)  # (B, T, 1)
        pooled = (fused * attn_weights_pool).sum(dim=1)  # (B, d_model)
        
        # Predictions
        direction = self.direction_head(pooled)  # (B, 2)
        magnitude = self.magnitude_head(pooled).squeeze(-1)  # (B,)
        volatility = self.volatility_head(pooled).squeeze(-1)  # (B,)
        
        if return_attention:
            return direction, magnitude, volatility, attn_weights
        
        return direction, magnitude, volatility
    
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
