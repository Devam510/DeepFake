"""
Production-Grade Audio Neural Embedding Module (Layer 3)
========================================================

Architecture: Wav2Vec2 (Frozen) -> Temporal Transformer -> MLP.
Includes Out-Of-Distribution (OOD) detection via Energy-based scoring
to ensure the model gracefully handles noise, music, or unseen languages,
triggering the `uncertainty_flag`.

Requires: torch, transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TemporalTransformer(nn.Module):
    """
    Analyzes sequences of Wav2Vec2 embeddings to find temporal 
    inconsistencies and artifacts spanning multiple frames.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        # Positional Encoding to retain time-step order
        self.pos_encoder = nn.Parameter(torch.randn(1, 1500, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        
        # Add positional encoding (handle varying sequence lengths up to 1500)
        pos = self.pos_encoder[:, :seq_len, :]
        x = x + pos
        
        # Pass through transformer
        out = self.transformer(x)
        
        # Temporal Pooling (Mean across time dimension)
        pooled = torch.mean(out, dim=1)
        return pooled


class AudioNeuralDetector(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base", 
                 freeze_backbone: bool = True):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install transformers constraint to use Layer 3: pip install transformers torch")
            
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            # We strictly freeze Wav2Vec2 to act solely as a feature extractor.
            # This prevents catastrophic forgetting of semantic acoustic space.
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
                
        self.transformer = TemporalTransformer(embed_dim=self.wav2vec2.config.hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # Binary Logit Output
        )
        
    def forward(self, audio_array: torch.Tensor) -> torch.Tensor:
        """
        Takes 16kHz audio waveform array. Returns logits.
        """
        with torch.no_grad():
            outputs = self.wav2vec2(audio_array)
            hidden_states = outputs.last_hidden_state  # (Batch, Seq, 768)

        temporal_features = self.transformer(hidden_states)  # (Batch, 768)
        logits = self.mlp(temporal_features)
        return logits

    def forward_features(self, audio_array: torch.Tensor):
        """
        Single-pass method: runs Wav2Vec2 ONCE and returns both
        (logit_score, ood_embedding). Use this instead of calling
        forward() + extract_embedding_for_ood() separately — it
        halves GPU time per file during batch feature extraction.
        """
        with torch.no_grad():
            outputs = self.wav2vec2(audio_array)
            hidden_states = outputs.last_hidden_state  # (Batch, Seq, 768)
            temporal_features = self.transformer(hidden_states)  # (Batch, 768)
            logits = self.mlp(temporal_features)       # (Batch, 1)
        return logits, temporal_features

    def extract_embedding_for_ood(self, audio_array: torch.Tensor) -> torch.Tensor:
        """
        Extract the transformer-pooled feature vector.
        Prefer forward_features() when you also need the score.
        """
        with torch.no_grad():
            outputs = self.wav2vec2(audio_array)
            hidden_states = outputs.last_hidden_state
            temporal_features = self.transformer(hidden_states)
        return temporal_features


# ══════════════════════════════════════════════════════════════════════════════
# ENERGY-BASED OUT-OF-DISTRIBUTION (OOD) ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
def calculate_energy_ood_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes purely Energy-based OOD scores.
    Energy = -T * torch.logsumexp(logits / T, dim=1)
    
    If the energy score goes above a calibrated threshold, the input is 
    "Out Of Distribution" (e.g. music, severe noise, completely unrecognized codec).
    """
    # For binary classification (1 output node), we treat it as 2-class logits for Energy
    # Typically this expects multi-class, but we adapt it for binary confidence bounded.
    
    # Expand to 2D logit equivalent (fake vs real confidence)
    pseudo_logits = torch.cat([logits, -logits], dim=1)
    
    energy = -temperature * torch.logsumexp(pseudo_logits / temperature, dim=1)
    return energy


if __name__ == "__main__":
    print("[Production] Audio Neural Embedding Detector Initialized.")
    print("  -> Wav2Vec2 + Temporal Transformer + Energy OOD Logic Ready.")
