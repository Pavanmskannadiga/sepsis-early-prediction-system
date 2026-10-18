"""
Deep Sepsis Ensemble Model.

Combines predictions from BiLSTM+Attention, Transformer Encoder, and
Temporal Convolutional Network (TCN) to maximize predictive robustness,
generalization, and provide epistemic uncertainty quantification.

Key Features:
- Soft-voting probability aggregation with calibrated model weights
- Disagreement & Epistemic Uncertainty Estimation (prediction variance and Shannon entropy)
- Individual sub-model breakdown for clinical transparency

Authors: Pavana M.S., Megharaja S., Srujan S. Sagar, Abubakkar Siddiq
Guide: Prof. Shilpa K.C.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.models.bilstm_attention import BiLSTMAttention
from src.models.transformer_model import TransformerSepsisModel
from src.models.tcn_model import TCNSepsisModel


class DeepSepsisEnsemble(nn.Module):
    """
    Heterogeneous Deep Learning Ensemble for Early Sepsis Prediction.
    
    Combines:
      1. BiLSTMAttention: Bidirectional recurrent sequence dynamics + temporal attention
      2. TransformerSepsisModel: Global multi-head self-attention encoder
      3. TCNSepsisModel: Causal dilated multi-scale temporal convolutions
    """

    def __init__(
        self,
        bilstm: Optional[nn.Module] = None,
        transformer: Optional[nn.Module] = None,
        tcn: Optional[nn.Module] = None,
        weights: Optional[List[float]] = None,
        input_dim: Optional[int] = None
    ):
        super().__init__()
        dim = input_dim or config.NUM_INPUT_FEATURES
        self.bilstm = bilstm or BiLSTMAttention(input_dim=dim)
        self.transformer = transformer or TransformerSepsisModel(input_dim=dim)
        self.tcn = tcn or TCNSepsisModel(input_dim=dim)

        # Default weights: equal weighting normalized to 1.0
        if weights is None:
            self.weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        else:
            w_sum = sum(weights)
            self.weights = [w / w_sum for w in weights]

        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass computing ensemble weighted logit / probability.

        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).
            return_attention: If True, capture attention from attention-enabled submodels.

        Returns:
            Ensemble logit tensor, shape (batch, 1).
        """
        # Forward through each model
        bilstm_logit = self.bilstm(x, return_attention=return_attention)
        trans_logit = self.transformer(x, return_attention=return_attention)
        tcn_logit = self.tcn(x, return_attention=return_attention)

        # Convert logits to probabilities
        p_bilstm = torch.sigmoid(bilstm_logit)
        p_trans = torch.sigmoid(trans_logit)
        p_tcn = torch.sigmoid(tcn_logit)

        # Weighted soft voting
        p_ensemble = (
            self.weights[0] * p_bilstm +
            self.weights[1] * p_trans +
            self.weights[2] * p_tcn
        )
        p_ensemble = torch.clamp(p_ensemble, 1e-7, 1.0 - 1e-7)

        # Store attention weights from Transformer / BiLSTM if available
        if return_attention:
            self._attention_weights = self.bilstm.get_attention_weights()

        # Convert ensemble probability back to logit for loss / consistency
        logit_ensemble = torch.log(p_ensemble / (1.0 - p_ensemble))
        return logit_ensemble

    def predict_detailed(self, x: torch.Tensor) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Detailed prediction returning ensemble score, individual model outputs,
        and uncertainty quantification metrics.

        Returns:
            Dict containing:
              - 'ensemble_risk': float [0, 1]
              - 'model_predictions': dict with 'bilstm', 'transformer', 'tcn'
              - 'uncertainty': float (standard deviation between model probabilities)
              - 'entropy': float (Shannon entropy of the ensemble prediction)
              - 'agreement_level': str ('HIGH', 'MODERATE', 'LOW')
        """
        self.eval()
        with torch.no_grad():
            p_bilstm = float(torch.sigmoid(self.bilstm(x)).squeeze())
            p_trans = float(torch.sigmoid(self.transformer(x)).squeeze())
            p_tcn = float(torch.sigmoid(self.tcn(x)).squeeze())

            probs = [p_bilstm, p_trans, p_tcn]
            p_ensemble = sum(w * p for w, p in zip(self.weights, probs))

            # Epistemic uncertainty = standard deviation of predictions
            variance = np.var(probs)
            uncertainty = float(np.sqrt(variance))

            # Predictive entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
            p_clip = min(max(p_ensemble, 1e-7), 1.0 - 1e-7)
            entropy = float(-(p_clip * math.log2(p_clip) + (1.0 - p_clip) * math.log2(1.0 - p_clip)))

            # Agreement level
            if uncertainty < 0.08:
                agreement = "HIGH (Strong Consensus)"
            elif uncertainty < 0.18:
                agreement = "MODERATE (Minor Divergence)"
            else:
                agreement = "LOW (High Epistemic Uncertainty)"

            return {
                "ensemble_risk": round(p_ensemble, 4),
                "model_predictions": {
                    "BiLSTM+Attention": round(p_bilstm, 4),
                    "Transformer": round(p_trans, 4),
                    "TCN": round(p_tcn, 4),
                },
                "uncertainty_std": round(uncertainty, 4),
                "entropy": round(entropy, 4),
                "agreement_level": agreement,
            }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities (batch, 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Retrieve stored attention weights."""
        return self._attention_weights

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters across all submodels."""
        bilstm_params = self.bilstm.count_parameters()["total"]
        trans_params = self.transformer.count_parameters()["total"]
        tcn_params = self.tcn.count_parameters()["total"]
        total = bilstm_params + trans_params + tcn_params
        return {
            "bilstm": bilstm_params,
            "transformer": trans_params,
            "tcn": tcn_params,
            "total": total,
            "trainable": total
        }

    def summary(self) -> str:
        """Human-readable ensemble summary."""
        params = self.count_parameters()
        lines = [
            "=" * 65,
            "Deep Sepsis Ensemble (BiLSTM + Transformer + TCN)",
            "=" * 65,
            f"  BiLSTM+Attention parameters:  {params['bilstm']:,}",
            f"  Transformer Encoder params:   {params['transformer']:,}",
            f"  TCN parameters:               {params['tcn']:,}",
            "-" * 65,
            f"  Total Ensemble Parameters:    {params['total']:,}",
            f"  Ensemble Weights:             {self.weights}",
            "=" * 65,
        ]
        return "\n".join(lines)
