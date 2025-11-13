"""
GNN Model Trainer

This module implements the training loop for the GNN surrogate model with:
- MLflow experiment tracking
- Early stopping
- Learning rate scheduling
- Mixed precision training (AMP)
- Gradient clipping
- Model checkpointing

The trainer follows MLOps best practices for reproducibility and monitoring.

Author: Materials Discovery Team
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from tqdm import tqdm

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Experiment tracking disabled.")

# Weights & Biases (optional, alternative to MLflow)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .gnn_surrogate import GraphSAGESurrogate, MultiTaskLoss, count_parameters

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if it doesn't improve
    for a specified number of epochs (patience).

    This saves compute and prevents the model from memorizing
    training data patterns that don't generalize.

    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     val_loss = train_epoch(...)
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" (lower is better) or "max" (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        # First epoch
        if self.best_score is None:
            self.best_score = score
            return False

        # Check if improved
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                self.early_stop = True
                return True

        return False


class GNNTrainer:
    """
    Trainer for GNN surrogate model with full MLOps integration.

    Features:
    - Automatic mixed precision (2× speedup on modern GPUs)
    - Gradient clipping (prevents exploding gradients)
    - Learning rate scheduling (improves convergence)
    - MLflow tracking (logs everything automatically)
    - Model checkpointing (save best models)
    - Early stopping (prevent overfitting)

    Example:
        >>> trainer = GNNTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     device="cuda",
        ...     use_mlflow=True,
        ... )
        >>> trainer.train(num_epochs=100)
        >>> trainer.save_model("models/best_model.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        test_loader: Optional[PyGDataLoader] = None,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        loss_weights: Optional[Dict[str, float]] = None,
        gradient_clip: float = 1.0,
        use_amp: bool = True,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        checkpoint_dir: str = "models/checkpoints",
        experiment_name: str = "gnn_training",
    ):
        """
        Initialize the trainer.

        Args:
            model: GNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            device: Device to train on ("auto", "cpu", "cuda", "mps")
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            loss_weights: Weights for multi-task loss
            gradient_clip: Gradient clipping threshold (None to disable)
            use_amp: Use automatic mixed precision
            use_mlflow: Track with MLflow
            use_wandb: Track with Weights & Biases
            checkpoint_dir: Directory for model checkpoints
            experiment_name: Name for experiment tracking
        """
        # Set up device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp and device == "cuda"  # AMP only works on CUDA

        # Loss function
        if loss_weights is None:
            loss_weights = {
                'energy': 1.0,
                'bandgap': 0.5,
                'stability': 2.0,
            }
        self.loss_fn = MultiTaskLoss(
            energy_weight=loss_weights.get('energy', 1.0),
            bandgap_weight=loss_weights.get('bandgap', 0.5),
            stability_weight=loss_weights.get('stability', 2.0),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        # AMP scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using automatic mixed precision (AMP)")

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Experiment tracking
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.experiment_name = experiment_name

        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        if self.use_wandb:
            wandb.init(project=experiment_name, config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'model_params': count_parameters(model),
            })
            logger.info(f"W&B project: {experiment_name}")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_energy_loss = 0.0
        total_bandgap_loss = 0.0
        total_stability_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.batch,
                        batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    )
                    loss, loss_dict = self.loss_fn(predictions, batch.y)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard forward pass (no AMP)
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                )
                loss, loss_dict = self.loss_fn(predictions, batch.y)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss_dict['total_loss']
            total_energy_loss += loss_dict['energy_loss']
            total_bandgap_loss += loss_dict['bandgap_loss']
            total_stability_loss += loss_dict['stability_loss']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        # Compute epoch averages
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_energy_loss': total_energy_loss / num_batches,
            'train_bandgap_loss': total_bandgap_loss / num_batches,
            'train_stability_loss': total_stability_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_energy_loss = 0.0
        total_bandgap_loss = 0.0
        total_stability_loss = 0.0
        num_batches = 0

        # Collect predictions for additional metrics
        all_predictions = []
        all_targets = []

        for batch in self.val_loader:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            )

            # Compute loss
            loss, loss_dict = self.loss_fn(predictions, batch.y)

            # Accumulate
            total_loss += loss_dict['total_loss']
            total_energy_loss += loss_dict['energy_loss']
            total_bandgap_loss += loss_dict['bandgap_loss']
            total_stability_loss += loss_dict['stability_loss']
            num_batches += 1

            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

        # Concatenate predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute additional metrics (MAE for regression tasks)
        energy_mae = np.abs(all_predictions[:, 0] - all_targets[:, 0]).mean()
        bandgap_mae = np.abs(all_predictions[:, 1] - all_targets[:, 1]).mean()

        # Stability accuracy (classification)
        stability_pred = (all_predictions[:, 2] > 0.5).astype(float)
        stability_acc = (stability_pred == all_targets[:, 2]).mean()

        metrics = {
            'val_loss': total_loss / num_batches,
            'val_energy_loss': total_energy_loss / num_batches,
            'val_bandgap_loss': total_bandgap_loss / num_batches,
            'val_stability_loss': total_stability_loss / num_batches,
            'val_energy_mae': energy_mae,
            'val_bandgap_mae': bandgap_mae,
            'val_stability_acc': stability_acc,
        }

        return metrics

    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 20,
        save_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs

        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            mode='min',
        )

        # Start MLflow run
        if self.use_mlflow:
            mlflow.start_run()
            mlflow.log_params({
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'gradient_clip': self.gradient_clip,
                'use_amp': self.use_amp,
                'model_params': count_parameters(self.model),
            })

        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            if self.use_mlflow:
                mlflow.log_metrics(all_metrics, step=epoch)

            if self.use_wandb:
                wandb.log(all_metrics, step=epoch)

            # Print summary
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"energy_mae={val_metrics['val_energy_mae']:.4f}, "
                f"bandgap_mae={val_metrics['val_bandgap_mae']:.4f}"
            )

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_model("best_model.pt")
                logger.info(f"New best model saved (val_loss={self.best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pt")

            # Early stopping
            if early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds")

        # Log model to MLflow
        if self.use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

        return {
            'train': self.train_history,
            'val': self.val_history,
        }

    def save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename

        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, filepath)

        logger.debug(f"Model saved to {filepath}")

    def load_model(self, filename: str):
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage (requires actual data)
    logging.basicConfig(level=logging.INFO)

    print("Trainer module loaded successfully")
    print("To use, create a model and data loaders, then:")
    print()
    print("  trainer = GNNTrainer(")
    print("      model=model,")
    print("      train_loader=train_loader,")
    print("      val_loader=val_loader,")
    print("  )")
    print("  history = trainer.train(num_epochs=100)")
