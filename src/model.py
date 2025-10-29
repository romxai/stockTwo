"""
Neural network model architectures

This module implements a hybrid deep learning architecture that combines:
- Multi-scale Temporal Convolutions (capture short/medium/long-term patterns)
- Bidirectional LSTMs (model sequential dependencies)
- Transformers (learn complex attention patterns)
- Cross-modal Attention (fuse price and text information)
- Market Regime Detection (adapt to different market conditions)
- Multi-task Learning (predict direction, magnitude, and volatility simultaneously)

The architecture is designed specifically for financial time series prediction
where different time scales, market regimes, and news sentiment all matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-scale temporal convolution for capturing patterns at different time scales
    
    Uses parallel convolutional layers with different kernel sizes to detect:
    - Short-term patterns (kernel_size=3: ~3 days)
    - Medium-term patterns (kernel_size=7: ~1 week)
    - Long-term patterns (kernel_size=15: ~3 weeks)
    
    Learned fusion weights combine these scales, allowing the model to emphasize
    the most relevant timescale for each prediction.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15]):
        super().__init__()

        # Create parallel Conv1D layers with different receptive fields
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),  # Padding keeps length same
                nn.BatchNorm1d(out_channels),  # Normalize activations
                nn.ReLU(),                      # Non-linearity
                nn.Dropout(0.2)                 # Regularization
            )
            for k in kernel_sizes
        ])

        # Learnable weights for combining multi-scale features (initialized uniformly)
        self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        # Conv1D expects (batch, channels, length), input is (batch, length, channels)
        x = x.transpose(1, 2)
        # Apply all convolutions in parallel
        outputs = [conv(x) for conv in self.convs]
        # Softmax ensures weights sum to 1 (weighted average)
        weights = F.softmax(self.fusion_weights, dim=0)
        # Weighted combination of multi-scale features
        fused = sum(w * out for w, out in zip(weights, outputs))
        # Transpose back to (batch, length, channels)
        return fused.transpose(1, 2)


class CrossModalAttention(nn.Module):
    """
    Bidirectional attention between numerical (price) and text (news) features
    
    This module allows:
    1. Price features to attend to relevant news (which news matters for this price pattern?)
    2. News features to attend to relevant price patterns (which price patterns relate to this news?)
    
    This bidirectional flow helps the model learn which news is relevant for price prediction
    and vice versa, improving the fusion of multi-modal information.
    
    Includes residual connections and feedforward networks (like in Transformer).
    """

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Attention: numerical features query text features
        self.num_to_text_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        # Attention: text features query numerical features
        self.text_to_num_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for stable training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network for additional non-linear transformation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand
            nn.GELU(),                         # Smooth activation
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),  # Project back
            nn.Dropout(dropout)
        )

    def forward(self, num_features, text_features):
        # Cross-attention: numerical features attend to text
        # (query=num, key=text, value=text)
        num_enhanced, num_attn_weights = self.num_to_text_attn(
            num_features, text_features, text_features
        )
        # Residual connection + layer norm (standard Transformer pattern)
        num_enhanced = self.norm1(num_features + num_enhanced)
        # Another residual with feedforward network
        num_enhanced = num_enhanced + self.ffn(num_enhanced)

        # Cross-attention: text features attend to numerical
        text_enhanced, text_attn_weights = self.text_to_num_attn(
            text_features, num_features, num_features
        )
        text_enhanced = self.norm2(text_features + text_enhanced)
        text_enhanced = text_enhanced + self.ffn(text_enhanced)

        # Store attention weights for interpretability (can visualize which news/prices matter)
        self.num_attn_weights = num_attn_weights
        self.text_attn_weights = text_attn_weights

        return num_enhanced, text_enhanced


class MarketRegimeDetector(nn.Module):
    """
    Detects market regime to adapt model behavior
    
    Market regimes have different dynamics:
    - Bullish: prices trending up, momentum matters
    - Bearish: prices trending down, risk aversion matters
    - Volatile: large swings, volatility signals matter
    - Stable: low movement, mean reversion matters
    
    By detecting regime, model can apply regime-specific processing
    via learned regime-specific heads.
    """

    def __init__(self, input_dim, num_regimes=4):
        super().__init__()

        # Multi-layer classifier to categorize market state
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_regimes)  # Output: logits for each regime
        )

    def forward(self, x):
        # Return regime classification logits
        return self.detector(x)


class AdaptiveFeatureFusion(nn.Module):
    """
    Learns gating mechanism to dynamically weight numerical vs text features
    
    Sometimes news matters more (e.g., during earnings), sometimes price patterns
    matter more (e.g., technical breakouts). This module learns three gates:
    1. Gate for numerical features
    2. Gate for text features  
    3. Gate for interaction between them
    
    Gates are computed from the combined features and sum to 1 (via softmax),
    allowing the model to adaptively emphasize what matters most for each sample.
    """

    def __init__(self, d_model):
        super().__init__()

        # Gating network: outputs 3 weights summing to 1
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Process concatenated features
            nn.Tanh(),                         # Smooth non-linearity
            nn.Linear(d_model, 3),             # 3 gates: num, text, interaction
            nn.Softmax(dim=-1)                 # Ensure gates sum to 1
        )

        # Interaction network: learns combined patterns from both modalities
        self.interaction = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, num_features, text_features):
        # Concatenate both modalities
        combined = torch.cat([num_features, text_features], dim=-1)
        # Compute gating weights (sum to 1 via softmax)
        gates = self.gate(combined)
        # Compute interaction features
        interaction = self.interaction(combined)
        # Weighted combination of all three components
        fused = (gates[:, 0:1] * num_features +      # Weight for numerical
                gates[:, 1:2] * text_features +       # Weight for text
                gates[:, 2:3] * interaction)          # Weight for interaction
        return fused


class UltraAdvancedStockPredictor(nn.Module):
    """
    Novel hybrid architecture for stock price prediction combining multiple techniques
    
    Architecture Flow:
    1. Input Projection: Map raw features to common d_model dimension
    2. Numerical Branch: Multi-scale Conv → Bidirectional LSTM
    3. Text Branch: Transformer → Bidirectional LSTM
    4. Cross-Modal Attention: Bidirectional information flow between modalities
    5. Market Regime Detection: Classify current market state
    6. Regime-Specific Processing: Apply regime-specific transformations
    7. Adaptive Fusion: Learn optimal combination of numerical/text features
    8. Prediction Heads: Output direction (classification), magnitude, volatility (regression)
    
    Multi-task learning (direction + magnitude + volatility) helps the model learn
    more robust representations by forcing it to understand different aspects of
    price movements simultaneously.
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

        self.d_model = d_model  # Hidden dimension for all internal representations

        # 1. INPUT PROJECTION
        # Map different feature spaces to common dimension
        self.num_projection = nn.Linear(num_numerical_features, d_model)
        self.text_projection = nn.Linear(num_text_features, d_model)

        # 2. NUMERICAL BRANCH (Process price patterns)
        # Multi-scale convolutions capture patterns at different timescales
        self.multi_scale_conv = MultiScaleTemporalConv(d_model, d_model)
        # Bidirectional LSTM models temporal dependencies (past & future context)
        self.num_lstm = nn.LSTM(
            d_model, d_model // 2,              # Output d_model (d_model//2 * 2 directions)
            num_layers=num_lstm_layers,
            batch_first=True,                    # Input shape: (batch, seq, features)
            bidirectional=True,                  # Process sequence forward & backward
            dropout=dropout if num_lstm_layers > 1 else 0  # Dropout between LSTM layers
        )

        # 3. TEXT BRANCH (Process news sentiment and semantics)
        # Transformer encoder for capturing complex attention patterns in text
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,                     # Multi-head attention
            dim_feedforward=d_model * 4,         # FFN hidden dimension
            dropout=dropout,
            batch_first=True,
            activation='gelu'                    # Smooth activation (better than ReLU)
        )
        self.text_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers    # Stack multiple transformer layers
        )
        # LSTM to capture sequential flow of news over time
        self.text_lstm = nn.LSTM(
            d_model, d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 4. CROSS-MODAL ATTENTION
        # Bidirectional attention allows price and news to inform each other
        self.cross_modal_attention = CrossModalAttention(d_model, num_heads, dropout)

        # 5. MARKET REGIME DETECTOR
        # Classifies current market state (bullish/bearish/volatile/stable)
        self.regime_detector = MarketRegimeDetector(d_model, num_regimes)

        # 6. REGIME-SPECIFIC PROCESSING
        # Separate processing heads for each market regime
        # Model learns different strategies for different market conditions
        self.regime_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_regimes)  # One head per regime
        ])

        # 7. ADAPTIVE FUSION
        # Learns optimal combination of numerical and text features
        self.adaptive_fusion = AdaptiveFeatureFusion(d_model)

        # 8. FINAL LAYERS
        # Progressive dimensionality reduction before prediction
        self.final_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),      # d_model → d_model/2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4), # d_model/2 → d_model/4
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 9. PREDICTION HEADS (Multi-task learning)
        # Direction head: Binary classification (up or down?)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 2 classes: down (0), up (1)
        )
        # Magnitude head: Regression (how much will it move?)
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Continuous output
        )
        # Volatility head: Regression (how uncertain is the prediction?)
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Continuous output
        )

        # Initialize weights using best practices
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using best practices
        
        - Xavier uniform for Linear layers: keeps variance stable across layers
        - Orthogonal for LSTM weights: prevents exploding/vanishing gradients
        - Zero init for biases: standard practice
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Good for layers with similar input/output size
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)  # Orthogonal matrices preserve gradient norms
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, X_num, X_text, return_attention=False):
        """
        Forward pass through the entire architecture
        
        Args:
            X_num: Numerical features (batch, sequence_length, num_features)
            X_text: Text embeddings (batch, sequence_length, text_features)
            return_attention: If True, return attention weights for interpretability
            
        Returns:
            direction_logits: Class logits for up/down prediction (batch, 2)
            magnitude: Predicted move magnitude (batch,)
            volatility: Predicted volatility (batch,)
            [Optional] attention weights if return_attention=True
        """
        batch_size = X_num.size(0)

        # 1. PROJECT to common dimension
        num_features = self.num_projection(X_num)
        
        # text_features = self.text_projection(X_text)
        
        # EXPERIMENT: Mute text features to isolate price signal
        # NOTE: This is a diagnostic modification - uncomment above line to use real text
        text_features = torch.zeros_like(num_features).to(num_features.device)
        # --- END MODIFICATION ---

        # 2. PROCESS NUMERICAL (Multi-scale Conv + LSTM)
        num_features = self.multi_scale_conv(num_features)
        num_features, _ = self.num_lstm(num_features)

        # 3. PROCESS TEXT (Transformer + LSTM)
        text_features = self.text_transformer(text_features)
        text_features, _ = self.text_lstm(text_features)

        # 4. CROSS-MODAL ATTENTION (Bidirectional information flow)
        num_enhanced, text_enhanced = self.cross_modal_attention(
            num_features, text_features
        )

        # 5. TEMPORAL POOLING (Use only last timestep for prediction)
        # Last timestep contains information from entire sequence via LSTM/attention
        num_final = num_enhanced[:, -1, :]
        text_final = text_enhanced[:, -1, :]

        # 6. DETECT MARKET REGIME
        regime_logits = self.regime_detector(num_final)
        regime_probs = F.softmax(regime_logits, dim=-1)  # Probability distribution over regimes

        # 7. REGIME-SPECIFIC PROCESSING
        # Apply each regime head and weight by regime probability
        regime_features = []
        for i, head in enumerate(self.regime_heads):
            regime_feat = head(num_final)
            # Weight by regime probability (soft routing)
            regime_features.append(regime_feat * regime_probs[:, i:i+1])
        regime_processed = sum(regime_features)  # Weighted mixture of regime-specific features

        # 8. ADAPTIVE FUSION (Learn optimal combination of price & news)
        fused = self.adaptive_fusion(regime_processed, text_final)

        # 9. FINAL ENCODING (Dimensionality reduction)
        encoded = self.final_encoder(fused)

        # 10. MULTI-TASK PREDICTIONS
        direction_logits = self.direction_head(encoded)     # Classification: up/down
        magnitude = self.magnitude_head(encoded).squeeze(-1)  # Regression: size of move
        volatility = self.volatility_head(encoded).squeeze(-1)  # Regression: uncertainty

        # Optionally return attention weights for interpretability
        if return_attention:
            return (direction_logits, magnitude, volatility,
                   self.cross_modal_attention.num_attn_weights,
                   self.cross_modal_attention.text_attn_weights)

        return direction_logits, magnitude, volatility

    def predict_with_confidence(self, X_num, X_text, mc_samples=10):
        """
        Predict with uncertainty estimation using MC (Monte Carlo) Dropout
        
        MC Dropout: Make multiple predictions with dropout enabled to get
        a distribution of predictions. Variance in predictions indicates
        model uncertainty (epistemic uncertainty).
        
        High uncertainty → model is not confident (be cautious with this prediction)
        Low uncertainty → model is confident
        
        Args:
            X_num: Numerical input features
            X_text: Text input features
            mc_samples: Number of stochastic forward passes (more = better uncertainty estimate)
            
        Returns:
            predictions: Class predictions (0 or 1)
            mean_probs: Average probabilities across MC samples
            confidence: Inverse of variance (higher = more confident)
        """
        self.train()  # Enable dropout for stochastic predictions

        direction_preds = []

        # Run multiple forward passes with different dropout masks
        with torch.no_grad():
            for _ in range(mc_samples):
                dir_logits, _, _ = self.forward(X_num, X_text)
                dir_probs = F.softmax(dir_logits, dim=-1)
                direction_preds.append(dir_probs)

        # Aggregate predictions
        direction_preds = torch.stack(direction_preds)  # (mc_samples, batch, 2)
        mean_probs = direction_preds.mean(dim=0)        # Average probabilities
        predictions = mean_probs.argmax(dim=-1)         # Take most likely class
        variance = direction_preds.var(dim=0).mean(dim=-1)  # Variance across samples
        confidence = 1.0 / (1.0 + variance)             # Convert variance to confidence score

        self.eval()  # Reset to evaluation mode

        return predictions, mean_probs, confidence


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Standard cross-entropy treats all samples equally. Focal loss down-weights
    easy samples (high confidence correct predictions) and focuses on hard samples.
    
    Formula: FL = -α(1-p_t)^γ * log(p_t)
    - α: class balancing factor
    - γ: focusing parameter (higher γ = more focus on hard samples)
    - p_t: predicted probability of true class
    
    When γ=0, reduces to standard cross-entropy.
    When γ=2 (common), easy samples (p_t→1) contribute very little to loss.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Probability of true class
        p_t = torch.exp(-ce_loss)
        # Apply focal term: down-weight easy samples
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting
    
    Combines three tasks:
    1. Direction prediction (classification)
    2. Magnitude prediction (regression)
    3. Volatility prediction (regression)
    
    Learns task-specific weights based on homoscedastic uncertainty
    (uncertainty inherent to the task, not specific samples).
    Higher uncertainty → lower weight for that task.
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters (one per task)
        # Using log-variance ensures positivity and numerical stability
        self.log_var_direction = nn.Parameter(torch.zeros(1))
        self.log_var_magnitude = nn.Parameter(torch.zeros(1))
        self.log_var_volatility = nn.Parameter(torch.zeros(1))
        # --- CHANGE THIS ---
        # self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)  # For class imbalance
        self.focal_loss = nn.CrossEntropyLoss()  # Use standard CE Loss (simpler)
        # --- END CHANGE ---
        self.mse_loss = nn.MSELoss()  # For regression tasks

    def forward(self, dir_logits, dir_targets, mag_preds, mag_targets, vol_preds, vol_targets):
        """
        Compute multi-task loss
        
        Args:
            dir_logits: Direction prediction logits (batch, 2)
            dir_targets: True direction labels (batch,)
            mag_preds: Magnitude predictions (batch,)
            mag_targets: True magnitudes (batch,)
            vol_preds: Volatility predictions (batch,)
            vol_targets: True volatilities (batch,)
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components for logging
        """
        # Currently focusing ONLY on direction classification
        # Magnitude and volatility tasks disabled to simplify learning
        dir_loss = self.focal_loss(dir_logits, dir_targets)

        # COMMENTED OUT: Multi-task learning with uncertainty weighting
        # Disabled because model was focusing on easier regression tasks
        # and ignoring the harder classification task
        # 
        # mag_loss = self.mse_loss(mag_preds, mag_targets)
        # vol_loss = self.mse_loss(vol_preds, vol_targets)
        #
        # # Convert log-variance to precision (inverse variance)
        # precision_dir = torch.exp(-self.log_var_direction)
        # precision_mag = torch.exp(-self.log_var_magnitude)
        # precision_vol = torch.exp(-self.log_var_volatility)
        #
        # # Weighted losses: precision * loss + log_variance
        # # High uncertainty (high log_var) → low precision → lower weight
        # weighted_dir_loss = precision_dir * dir_loss + self.log_var_direction
        # weighted_mag_loss = precision_mag * mag_loss + self.log_var_magnitude
        # weighted_vol_loss = precision_vol * vol_loss + self.log_var_volatility
        #
        # total_loss = weighted_dir_loss + weighted_mag_loss + weighted_vol_loss

        total_loss = dir_loss  # Only use direction loss for now

        # Return loss components for logging
        return total_loss, {
            'direction': dir_loss.item(),
            'magnitude': 0,  # Disabled
            'volatility': 0,  # Disabled
            'total': total_loss.item()
        }
