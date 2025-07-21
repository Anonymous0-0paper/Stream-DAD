"""
Training Script for Stream-DAD.

This script implements the complete training procedure including
streaming simulation, drift adaptation, and comprehensive evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import wandb

# Stream-DAD imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from stream_dad.core.models import StreamDAD, create_stream_dad_model
from stream_dad.utils.data_loading import create_data_loader, inject_synthetic_drift
from stream_dad.utils.evaluation import ComprehensiveEvaluator, evaluate_anomaly_detection
from stream_dad.core.gating import GradientBasedImportance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamDADTrainer:
    """
    Trainer class for Stream-DAD with streaming simulation and drift adaptation.

    Handles the complete training loop including online adaptation,
    drift detection, and continuous evaluation.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_step = 0
        self.epoch = 0

        # Evaluation
        self.evaluator = ComprehensiveEvaluator(config.get('evaluation', {}))
        self.best_f1_score = 0.0
        self.best_model_path = None

        # Gradient-based importance tracker
        self.importance_tracker = None

        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project=config.get('wandb_project', 'stream-dad'), config=config)

    def setup_model(self, input_dim: int):
        """Initialize model, optimizer, and related components."""
        logger.info(f"Setting up Stream-DAD model with {input_dim} input features")

        # Create model
        self.model = create_stream_dad_model(input_dim, self.config['model'])
        self.model.to(self.device)

        # Initialize importance tracker
        self.importance_tracker = GradientBasedImportance(
            input_dim,
            ema_alpha=self.config['model'].get('importance_ema_alpha', 0.9)
        )

        # Setup optimizer
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')

        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.01),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Setup learning rate scheduler
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 1000),
                gamma=scheduler_config.get('gamma', 0.95)
            )

        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train_streaming(self, train_loader, val_loader=None):
        """
        Main streaming training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        logger.info("Starting streaming training...")

        # Training configuration
        max_steps = self.config['training'].get('max_steps', 10000)
        eval_interval = self.config['training'].get('eval_interval', 100)
        save_interval = self.config['training'].get('save_interval', 1000)
        adaptation_threshold = self.config['training'].get('adaptation_threshold', 0.3)

        self.model.train()

        # Streaming training loop
        step = 0
        running_loss = 0.0
        recent_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if step >= max_steps:
                break

            start_time = time.time()

            # Prepare batch
            data = batch['data'].to(self.device)
            labels = batch.get('labels', torch.zeros(data.shape[0], 1)).to(self.device)

            # Enable gradient computation for importance tracking
            data.requires_grad_(True)

            # Forward pass
            output = self.model(data, return_gates=True)

            # Compute losses
            loss_dict = self.model.compute_loss(data, output)
            total_loss = loss_dict['total_loss']

            # Compute gradient-based importance
            if self.importance_tracker is not None:
                try:
                    importance = self.importance_tracker.compute_importance(
                        data, total_loss, normalize=True
                    )

                    # Update gating network with importance
                    if hasattr(self.model.gating_network, 'importance_ema'):
                        self.model.gating_network.importance_ema.copy_(importance)

                except Exception as e:
                    logger.warning(f"Failed to compute importance: {e}")
                    importance = None
            else:
                importance = None

            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            grad_clip = self.config['training'].get('grad_clip', 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update continual learning components
            self.model.continual_learner.step()

            # Track training time
            training_time = time.time() - start_time

            # Update running statistics
            running_loss += total_loss.item()
            recent_losses.append(total_loss.item())
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            # Check for drift and adaptation
            current_drift = self.model.drift_detector.get_current_drift_magnitude()
            if current_drift > adaptation_threshold:
                logger.info(f"High drift detected ({current_drift:.4f}), triggering adaptation")
                self._adapt_to_drift(data, current_drift)

            #