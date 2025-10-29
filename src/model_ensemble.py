"""
Ensemble Multi-Model Architecture for Stock Prediction
Combines LSTM + Transformer + TCN for robustness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBranch(nn.Module):
    """LSTM branch for capturing sequential dependencies"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, 2*hidden_dim)
        # Global average pooling
        out = out.transpose(1, 2)  # (B, 2*hidden_dim, T)
        out = self.pool(out).squeeze(-1)  # (B, 2*hidden_dim)
        return out


class TCNBranch(nn.Module):
    """Temporal Convolutional Network branch"""
    def __init__(self, input_dim, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(nn.Conv1d(
                in_ch, out_ch, kernel_size,
                padding=(kernel_size-1) * dilation // 2,
                dilation=dilation
            ))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.tcn(x)  # (B, num_channels[-1], T)
        x = self.pool(x).squeeze(-1)  # (B, num_channels[-1])
        return x


class TransformerBranch(nn.Module):
    """Transformer branch for attention-based modeling"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.projection(x)  # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # Global average pooling: (B, d_model)
        return x


class EnsembleStockPredictor(nn.Module):
    """
    Ensemble of LSTM + TCN + Transformer
    Each learns different aspects, combined via learned weights
    """
    def __init__(self, num_features, text_features, dropout=0.3):
        super().__init__()
        
        # Three branches for numerical features
        self.lstm_num = LSTMBranch(num_features, hidden_dim=64, num_layers=2)
        self.tcn_num = TCNBranch(num_features, num_channels=[64, 128], kernel_size=3)
        self.transformer_num = TransformerBranch(num_features, d_model=128, nhead=4, num_layers=2)
        
        # Three branches for text features
        self.lstm_text = LSTMBranch(text_features, hidden_dim=64, num_layers=2)
        self.tcn_text = TCNBranch(text_features, num_channels=[64, 128], kernel_size=3)
        self.transformer_text = TransformerBranch(text_features, d_model=128, nhead=4, num_layers=2)
        
        # Learned ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Calculate total feature dimension
        lstm_out = 64 * 2  # bidirectional
        tcn_out = 128
        transformer_out = 128
        total_dim = (lstm_out + tcn_out + transformer_out) * 2  # *2 for num+text
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_num, x_text):
        # Process numerical features through all branches
        lstm_num_out = self.lstm_num(x_num)
        tcn_num_out = self.tcn_num(x_num)
        transformer_num_out = self.transformer_num(x_num)
        
        # Process text features through all branches
        lstm_text_out = self.lstm_text(x_text)
        tcn_text_out = self.tcn_text(x_text)
        transformer_text_out = self.transformer_text(x_text)
        
        # Concatenate all branch outputs
        combined = torch.cat([
            lstm_num_out, tcn_num_out, transformer_num_out,
            lstm_text_out, tcn_text_out, transformer_text_out
        ], dim=-1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Predictions
        direction = self.direction_head(fused)
        magnitude = self.magnitude_head(fused).squeeze(-1)
        volatility = self.volatility_head(fused).squeeze(-1)
        
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


print("✅ Ensemble architecture defined!")
