"""
Core neural network models for Stream-DAD.

This module implements the main Stream-DAD architecture including the encoder-decoder
structure and the complete forward/backward pass logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

from .gating import DynamicGatingNetwork
from .continual import ContinualLearner
from .drift_detection import DriftDetector

logger = logging.getLogger(__name__)


class GRUEncoder(nn.Module):
    """
    GRU-based encoder for processing gated input windows.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension of GRU
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape [batch_size, window_size, input_dim]
            hidden: Previous hidden state

        Returns:
            output: Encoded representations [batch_size, window_size, hidden_dim]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
        """
        output, hidden = self.gru(x, hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output, hidden


class GRUDecoder(nn.Module):
    """
    GRU-based decoder for reconstructing input from encoded representations.

    Args:
        hidden_dim: Hidden dimension of GRU (matches encoder)
        output_dim: Output dimension (should match input_dim)
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            encoded: Encoded representations [batch_size, window_size, hidden_dim]
            hidden: Previous hidden state

        Returns:
            reconstructed: Reconstructed input [batch_size, window_size, output_dim]
        """
        output, _ = self.gru(encoded, hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)
        reconstructed = self.output_projection(output)

        return reconstructed


class StreamDAD(nn.Module):
    """
    Main Stream-DAD model implementing the complete architecture.

    This class integrates all components: encoder-decoder, dynamic gating,
    drift detection, and continual learning mechanisms.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for encoder-decoder
        window_size: Size of input windows
        config: Configuration dictionary with hyperparameters
    """

    def __init__(self, input_dim: int, hidden_dim: int, window_size: int, config: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.config = config

        # Core encoder-decoder architecture
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )

        self.decoder = GRUDecoder(
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )

        # Dynamic gating mechanism
        self.gating_network = DynamicGatingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            config=config.get('gating', {})
        )

        # Drift detection
        self.drift_detector = DriftDetector(
            input_dim=input_dim,
            window_size=config.get('drift_window', 50),
            config=config.get('drift', {})
        )

        # Continual learning components
        self.continual_learner = ContinualLearner(
            model=self,
            config=config.get('continual', {})
        )

        # Normalization statistics (running mean/std)
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.register_buffer('num_samples', torch.tensor(0))

        # Previous hidden states for temporal context
        self.prev_hidden = None
        self.prev_gates = None

        # Training mode flags
        self.adaptation_mode = False

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using running statistics.

        Args:
            x: Input tensor [batch_size, window_size, input_dim]

        Returns:
            normalized: Normalized input tensor
        """
        if self.training:
            # Update running statistics
            batch_mean = x.mean(dim=(0, 1))
            batch_var = x.var(dim=(0, 1), unbiased=False)

            # Exponential moving average
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_samples += x.shape[0] * x.shape[1]

        # Normalize
        normalized = (x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return normalized

    def forward(self, x: torch.Tensor, return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stream-DAD.

        Args:
            x: Input tensor [batch_size, window_size, input_dim]
            return_gates: Whether to return gate values

        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed input
                - gates: Feature gates (if return_gates=True)
                - drift_signal: Drift sensitivity signals
                - anomaly_scores: Per-sample anomaly scores
                - hidden: Final hidden state
        """
        batch_size = x.shape[0]

        # Normalize input
        x_norm = self.normalize_input(x)

        # Compute drift signals
        drift_signal = self.drift_detector(x_norm)

        # Compute spatial correlations
        correlations = self.compute_spatial_correlations(x_norm)

        # Generate dynamic gates
        gates = self.gating_network(
            prev_hidden=self.prev_hidden,
            drift_signal=drift_signal,
            correlations=correlations
        )

        # Apply gating
        x_gated = x_norm * gates.unsqueeze(1)  # Broadcasting over window dimension

        # Encode gated input
        encoded, hidden = self.encoder(x_gated, self.prev_hidden)

        # Decode to reconstruct
        reconstructed = self.decoder(encoded)

        # Compute anomaly scores (reconstruction error)
        anomaly_scores = torch.norm(x_norm - reconstructed, dim=-1, p=2)  # [batch_size, window_size]

        # Update previous states
        if batch_size == 1:  # Only for streaming inference
            self.prev_hidden = hidden.detach()
            self.prev_gates = gates.detach()

        # Prepare output
        output = {
            'reconstructed': reconstructed,
            'anomaly_scores': anomaly_scores,
            'drift_signal': drift_signal,
            'hidden': hidden
        }

        if return_gates:
            output['gates'] = gates

        return output

    def compute_spatial_correlations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial correlations between features.

        Args:
            x: Input tensor [batch_size, window_size, input_dim]

        Returns:
            correlations: Correlation scores for each feature [batch_size, input_dim]
        """
        batch_size, window_size, input_dim = x.shape

        # Compute pairwise correlations for each sample in batch
        correlations = torch.zeros(batch_size, input_dim, device=x.device)

        for b in range(batch_size):
            sample = x[b]  # [window_size, input_dim]

            # Compute correlation matrix
            sample_centered = sample - sample.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(sample_centered.T, sample_centered) / (window_size - 1)

            # Compute standard deviations
            std_devs = torch.sqrt(torch.diag(cov_matrix))

            # Correlation matrix
            corr_matrix = cov_matrix / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1) + 1e-8)

            # Sum of significant correlations for each feature
            threshold = self.config.get('correlation_threshold', 0.3)
            significant_corrs = (torch.abs(corr_matrix) > threshold).float()
            correlations[b] = significant_corrs.sum(dim=1) - 1  # Subtract self-correlation

        return correlations

    def compute_loss(self, x: torch.Tensor, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the unified loss function.

        Args:
            x: Original input tensor
            output: Output from forward pass

        Returns:
            Dictionary containing all loss components
        """
        x_norm = self.normalize_input(x)
        reconstructed = output['reconstructed']
        gates = output.get('gates', None)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x_norm)

        # EWC loss
        ewc_loss = self.continual_learner.compute_ewc_loss()

        # Consistency loss (gate smoothness)
        consistency_loss = torch.tensor(0.0, device=x.device)
        if gates is not None and self.prev_gates is not None:
            consistency_loss = F.mse_loss(gates, self.prev_gates)

        # Sparsity regularization
        sparsity_loss = torch.tensor(0.0, device=x.device)
        if gates is not None:
            l1_loss = torch.norm(gates, p=1, dim=-1).mean()
            entropy_loss = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
            sparsity_loss = l1_loss + self.config.get('lambda_entropy', 0.001) * entropy_loss

        # Total loss
        lambda_ewc = self.config.get('lambda_ewc', 0.01)
        lambda_cons = self.config.get('lambda_cons', 0.001)
        lambda_sparsity = self.config.get('lambda_sparsity', 0.0001)

        total_loss = (recon_loss +
                      lambda_ewc * ewc_loss +
                      lambda_cons * consistency_loss +
                      lambda_sparsity * sparsity_loss)

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'ewc_loss': ewc_loss,
            'consistency_loss': consistency_loss,
            'sparsity_loss': sparsity_loss
        }

    def adapt_to_drift(self, x: torch.Tensor) -> None:
        """
        Adapt model parameters based on detected drift.

        Args:
            x: Recent data samples for adaptation
        """
        self.adaptation_mode = True

        # Update Fisher information for EWC
        self.continual_learner.update_fisher_information(x)

        # Adapt hyperparameters based on drift magnitude
        drift_magnitude = self.drift_detector.get_current_drift_magnitude()
        self.adapt_hyperparameters(drift_magnitude)

        self.adaptation_mode = False

    def adapt_hyperparameters(self, drift_magnitude: float) -> None:
        """
        Adapt hyperparameters based on current drift magnitude.

        Args:
            drift_magnitude: Current estimated drift magnitude
        """
        # Adapt EWC weight (decrease during high drift for faster adaptation)
        base_ewc = self.config.get('lambda_ewc_base', 0.01)
        self.config['lambda_ewc'] = base_ewc * torch.exp(-torch.tensor(drift_magnitude))

        # Adapt consistency weight (increase during high drift for stability)
        base_cons = self.config.get('lambda_cons_base', 0.001)
        self.config['lambda_cons'] = base_cons * (1 + drift_magnitude)

        logger.info(f"Adapted hyperparameters: λ_EWC={self.config['lambda_ewc']:.6f}, "
                    f"λ_cons={self.config['lambda_cons']:.6f}")

    def reset_states(self) -> None:
        """Reset all temporal states (for new sequences)."""
        self.prev_hidden = None
        self.prev_gates = None
        self.drift_detector.reset()
        self.continual_learner.reset()

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get current feature importance scores.

        Returns:
            importance: Feature importance scores [input_dim]
        """
        if self.prev_gates is not None:
            return self.prev_gates
        else:
            return torch.ones(self.input_dim) / self.input_dim

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint including all states."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'num_samples': self.num_samples,
            'continual_learner_state': self.continual_learner.get_state(),
            'drift_detector_state': self.drift_detector.get_state()
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint including all states."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint['config'])
        self.running_mean = checkpoint['running_mean']
        self.running_var = checkpoint['running_var']
        self.num_samples = checkpoint['num_samples']
        self.continual_learner.load_state(checkpoint['continual_learner_state'])
        self.drift_detector.load_state(checkpoint['drift_detector_state'])
        logger.info(f"Checkpoint loaded from {path}")


def create_stream_dad_model(input_dim: int, config: Dict) -> StreamDAD:
    """
    Factory function to create Stream-DAD model with default configurations.

    Args:
        input_dim: Number of input features
        config: Configuration dictionary

    Returns:
        model: Initialized Stream-DAD model
    """
    default_config = {
        'hidden_dim': 64,
        'window_size': 10,
        'num_layers': 2,
        'dropout': 0.1,
        'lambda_ewc': 0.01,
        'lambda_cons': 0.001,
        'lambda_sparsity': 0.0001,
        'lambda_entropy': 0.001,
        'correlation_threshold': 0.3,
        'gating': {},
        'drift': {},
        'continual': {}
    }

    # Update with user config
    default_config.update(config)

    model = StreamDAD(
        input_dim=input_dim,
        hidden_dim=default_config['hidden_dim'],
        window_size=default_config['window_size'],
        config=default_config
    )

    return model