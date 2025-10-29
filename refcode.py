# ========================================
# NOISE-ROBUST DENOISING ATTENTION NETWORK
# Designed specifically for noisy financial data
# ========================================

import pywt  # For wavelet denoising

class WaveletDenoising(nn.Module):
    """
    Wavelet-based noise reduction for financial time series
    Removes high-frequency noise while preserving trends
    """
    def __init__(self, wavelet='db4', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: denoised x with same shape
        """
        batch_size, seq_len, n_features = x.shape
        x_denoised = torch.zeros_like(x)
        
        # Apply wavelet denoising to each feature independently
        for b in range(batch_size):
            for f in range(n_features):
                signal = x[b, :, f].cpu().detach().numpy()
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                
                # Soft thresholding (removes noise)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                
                coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
                
                # Reconstruction
                signal_denoised = pywt.waverec(coeffs_thresh, self.wavelet)
                
                # Handle length mismatch
                if len(signal_denoised) > seq_len:
                    signal_denoised = signal_denoised[:seq_len]
                elif len(signal_denoised) < seq_len:
                    signal_denoised = np.pad(signal_denoised, (0, seq_len - len(signal_denoised)))
                
                x_denoised[b, :, f] = torch.tensor(signal_denoised, device=x.device, dtype=x.dtype)
        
        return x_denoised


class RobustFeatureSelection(nn.Module):
    """
    Learns which features are reliable vs noisy
    Uses Gumbel-Softmax for differentiable feature selection
    """
    def __init__(self, n_features, n_selected, temperature=1.0):
        super().__init__()
        self.n_features = n_features
        self.n_selected = n_selected
        self.temperature = temperature
        
        # Feature importance scores (learnable)
        self.feature_scores = nn.Parameter(torch.randn(n_features))
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LayerNorm(n_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: (batch, seq_len, features) with noisy features downweighted
        """
        # Compute feature weights via Gumbel-Softmax
        if self.training:
            # Sample from Gumbel distribution
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.feature_scores) + 1e-10) + 1e-10)
            logits = (self.feature_scores + gumbel_noise) / self.temperature
        else:
            logits = self.feature_scores
        
        # Softmax to get weights
        weights = torch.softmax(logits, dim=0)
        
        # Apply weights
        x_weighted = x * weights.unsqueeze(0).unsqueeze(0)
        
        # Encode
        batch, seq, feat = x.shape
        x_flat = x_weighted.reshape(-1, feat)
        x_encoded = self.feature_encoder(x_flat)
        x_encoded = x_encoded.reshape(batch, seq, feat)
        
        return x_encoded


class TemporalSmoothing(nn.Module):
    """
    EWMA-based temporal smoothing to reduce noise
    """
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: smoothed x
        """
        batch, seq_len, features = x.shape
        smoothed = torch.zeros_like(x)
        
        # EWMA smoothing
        smoothed[:, 0] = x[:, 0]
        for t in range(1, seq_len):
            smoothed[:, t] = self.alpha * x[:, t] + (1 - self.alpha) * smoothed[:, t-1]
        
        return smoothed


class MultiScaleFeaturePyramid(nn.Module):
    """
    Extracts features at multiple timescales
    Helps find stable patterns across different windows
    """
    def __init__(self, in_features, scales=[3, 7, 14, 28]):
        super().__init__()
        self.scales = scales
        
        # Conv for each scale
        self.convs = nn.ModuleList([
            nn.Conv1d(in_features, in_features, kernel_size=s, padding=s//2)
            for s in scales
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(in_features * len(scales), in_features, 1),
            nn.BatchNorm1d(in_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: (batch, seq_len, features)
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # (B, F, T)
        
        # Apply all scales
        multi_scale = [conv(x) for conv in self.convs]
        
        # Concatenate
        concat = torch.cat(multi_scale, dim=1)
        
        # Fuse
        fused = self.fusion(concat)
        
        # Transpose back
        return fused.transpose(1, 2)


class DenoisingAttentionNetwork(nn.Module):
    """
    Complete noise-robust architecture
    Combines all denoising techniques for maximum robustness
    """
    def __init__(self, num_features, text_features, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.3):
        super().__init__()
        
        print("🔧 Building Noise-Robust Denoising Attention Network...")
        
        # === DENOISING LAYERS ===
        # Note: Wavelet denoising commented out for speed; uncomment if needed
        # self.wavelet_denoise_num = WaveletDenoising(wavelet='db4', level=2)
        # self.wavelet_denoise_text = WaveletDenoising(wavelet='db4', level=2)
        
        self.temporal_smooth_num = TemporalSmoothing(alpha=0.3)
        self.temporal_smooth_text = TemporalSmoothing(alpha=0.2)
        
        self.feature_selector_num = RobustFeatureSelection(num_features, num_features, temperature=0.5)
        self.feature_selector_text = RobustFeatureSelection(text_features, text_features, temperature=0.5)
        
        self.multi_scale_num = MultiScaleFeaturePyramid(num_features, scales=[3, 7, 14])
        self.multi_scale_text = MultiScaleFeaturePyramid(text_features, scales=[3, 7, 14])
        
        # === PROJECTION LAYERS ===
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # === ATTENTION LAYERS ===
        # Self-attention for numerical features
        self.num_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Self-attention for text features  
        self.text_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Cross-attention (price ↔ news)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norms_num = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_text = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # === CONFIDENCE ESTIMATION ===
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # === FUSION ===
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === PREDICTION HEADS ===
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
        
        print("   ✅ Denoising layers: Temporal smoothing + Feature selection + Multi-scale")
        print("   ✅ Attention layers: Self-attention + Cross-attention")
        print("   ✅ Confidence estimation: Uncertainty-aware predictions")
    
    def forward(self, x_num, x_text):
        # === DENOISING STAGE ===
        # Wavelet denoising (optional, slow)
        # x_num = self.wavelet_denoise_num(x_num)
        # x_text = self.wavelet_denoise_text(x_text)
        
        # Temporal smoothing
        x_num = self.temporal_smooth_num(x_num)
        x_text = self.temporal_smooth_text(x_text)
        
        # Robust feature selection
        x_num = self.feature_selector_num(x_num)
        x_text = self.feature_selector_text(x_text)
        
        # Multi-scale features
        x_num = self.multi_scale_num(x_num)
        x_text = self.multi_scale_text(x_text)
        
        # === ENCODING STAGE ===
        num_emb = self.num_proj(x_num)  # (B, T, d_model)
        text_emb = self.text_proj(x_text)  # (B, T, d_model)
        
        # Self-attention within modalities
        for i, (num_attn, text_attn, norm_num, norm_text) in enumerate(
            zip(self.num_attention, self.text_attention, self.norms_num, self.norms_text)
        ):
            # Numerical self-attention
            num_attn_out, _ = num_attn(num_emb, num_emb, num_emb)
            num_emb = norm_num(num_emb + num_attn_out)
            
            # Text self-attention
            text_attn_out, _ = text_attn(text_emb, text_emb, text_emb)
            text_emb = norm_text(text_emb + text_attn_out)
        
        # Cross-attention (price queries news)
        cross_out, attn_weights = self.cross_attention(num_emb, text_emb, text_emb)
        
        # === FUSION STAGE ===
        # Pool over time
        num_pooled = num_emb.mean(dim=1)  # (B, d_model)
        cross_pooled = cross_out.mean(dim=1)  # (B, d_model)
        
        # Concatenate
        fused = torch.cat([num_pooled, cross_pooled], dim=-1)  # (B, 2*d_model)
        
        # Estimate confidence
        confidence = self.confidence_estimator(fused)  # (B, 1)
        
        # Final fusion
        fused = self.fusion(fused)  # (B, d_model)
        
        # === PREDICTION STAGE ===
        direction = self.direction_head(fused)  # (B, 2)
        magnitude = self.magnitude_head(fused).squeeze(-1)  # (B,)
        volatility = self.volatility_head(fused).squeeze(-1)  # (B,)
        
        # Weight predictions by confidence
        direction = direction * confidence
        
        return direction, magnitude, volatility


print("✅ Noise-Robust Denoising Attention Network defined!")
print("   Key features:")
print("   - Temporal smoothing (EWMA)")
print("   - Robust feature selection (learns which features to trust)")
print("   - Multi-scale feature pyramid (3/7/14-day patterns)")
print("   - Cross-modal attention (price ↔ news)")
print("   - Confidence-weighted predictions")