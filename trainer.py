"""
trainer.py

Model-Agnostic Training Infrastructure for UV-Stabilized Flow Matching
=======================================================================

Implements:
- Document Algorithm 1: Training loop with backbone-subtracted loss
- EMA (Exponential Moving Average) for stable sampling
- Mixed precision training (AMP)
- Checkpointing and logging
- Learning rate scheduling

Document References
-------------------
- Algorithm 1: Training procedure
- Section 7: IR prior sampling
- Section 5.1: UV lift
- eq (fm-loss): Loss computation
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Dataset

from ads_cft.config import TrainerConfig

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Used for stable sampling as per common practice in flow matching
    and diffusion models.

    Update rule: θ_ema = decay * θ_ema + (1 - decay) * θ

    Attributes
    ----------
    decay : float
        EMA decay rate
    shadow_params : Dict[str, Tensor]
        EMA shadow parameters
    device : Optional[torch.device]
        Target device for shadow parameters

    Example
    -------
    >>> ema = EMAModel(model, decay=0.9999)
    >>> for step in range(training_steps):
    ...     loss = train_step(model, batch)
    ...     if step >= 1000:
    ...         ema.update(model)
    >>> # For evaluation:
    >>> original = ema.apply_shadow(model)
    >>> samples = model.sample(...)
    >>> ema.restore(model, original)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize EMA tracker.

        Args:
            model: Model to track
            decay: EMA decay rate
            device: Target device for shadow parameters
        """
        self.decay = decay
        self.device = device

        # Create shadow parameters
        self.shadow_params: Dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
                if device is not None:
                    self.shadow_params[name] = self.shadow_params[name].to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with EMA.

        Args:
            model: Model with current parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> Dict[str, Tensor]:
        """
        Apply shadow parameters to model, returning original parameters.

        Args:
            model: Model to apply shadow parameters to

        Returns:
            Dictionary of original parameters for restoration
        """
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                original_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
        return original_params

    def restore(self, model: nn.Module, original_params: Dict[str, Tensor]) -> None:
        """
        Restore original parameters to model.

        Args:
            model: Model to restore
            original_params: Original parameters from apply_shadow
        """
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA state dictionary for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        for name, param in state_dict["shadow_params"].items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param)


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


def get_lr_scheduler(
    optimizer: Optimizer,
    config: TrainerConfig,
    total_steps: int,
) -> LRScheduler:
    """
    Create learning rate scheduler based on config.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        total_steps: Total number of training steps

    Returns:
        Learning rate scheduler

    Raises:
        ValueError: If scheduler type is unknown
    """
    if config.lr_scheduler == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)

    elif config.lr_scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate * config.min_lr_ratio,
        )

    elif config.lr_scheduler == "warmup_cosine":

        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, config.warmup_steps))
            else:
                # Cosine decay
                progress = float(step - config.warmup_steps) / float(
                    max(1, total_steps - config.warmup_steps)
                )
                return config.min_lr_ratio + 0.5 * (1.0 - config.min_lr_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )

        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")


# =============================================================================
# Checkpoint Management
# =============================================================================


@dataclass
class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Attributes
    ----------
    checkpoint_dir : Path
        Directory for saving checkpoints
    keep_last_n : int
        Number of checkpoints to keep

    Example
    -------
    >>> manager = CheckpointManager(Path("./checkpoints"), keep_last_n=5)
    >>> manager.save(step=1000, model=model, optimizer=optimizer, ...)
    >>> checkpoint = manager.load(path, model=model, optimizer=optimizer)
    """

    checkpoint_dir: Path
    keep_last_n: int = 5

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        ema: Optional[EMAModel],
        scaler: Optional[GradScaler],
        metrics: Dict[str, float],
        config: TrainerConfig,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional scheduler to save
            ema: Optional EMA to save
            scaler: Optional gradient scaler to save
            metrics: Current metrics
            config: Training configuration

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(config),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return path

    def load(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        ema: Optional[EMAModel] = None,
        scaler: Optional[GradScaler] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load into
            scheduler: Optional scheduler to load into
            ema: Optional EMA to load into
            scaler: Optional gradient scaler to load into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[: -self.keep_last_n]:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")


# =============================================================================
# Metrics Tracking
# =============================================================================


class MetricsTracker:
    """
    Track and aggregate training metrics.

    Provides running averages over a sliding window for smooth logging.

    Attributes
    ----------
    window_size : int
        Window size for computing running averages
    history : Dict[str, List[float]]
        Full history of all metrics
    step_history : List[int]
        Steps at which metrics were recorded

    Example
    -------
    >>> tracker = MetricsTracker(window_size=100)
    >>> for step in range(1000):
    ...     tracker.update(step, {"loss": loss_value})
    ...     if step % 100 == 0:
    ...         print(tracker.get_smoothed("loss"))
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize metrics tracker.

        Args:
            window_size: Window size for running averages
        """
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}
        self.step_history: List[int] = []

    def update(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Add metrics for a step.

        Non-numeric values are skipped.

        Args:
            step: Current training step
            metrics: Dictionary of metric values
        """
        for name, value in metrics.items():
            # Skip non-numeric values (like file paths)
            if isinstance(value, str):
                continue
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                continue

            if name not in self.history:
                self.history[name] = []
            self.history[name].append(float_val)
        self.step_history.append(step)

    def get_smoothed(self, name: str) -> float:
        """
        Get smoothed metric value (average over window).

        Args:
            name: Metric name

        Returns:
            Smoothed value, or NaN if metric not found
        """
        if name not in self.history or len(self.history[name]) == 0:
            return float("nan")
        values = self.history[name][-self.window_size :]
        return sum(values) / len(values)

    def get_latest(self, name: str) -> float:
        """
        Get most recent metric value.

        Args:
            name: Metric name

        Returns:
            Latest value, or NaN if metric not found
        """
        if name not in self.history or len(self.history[name]) == 0:
            return float("nan")
        return self.history[name][-1]

    def get_all_smoothed(self) -> Dict[str, float]:
        """
        Get all smoothed metrics.

        Returns:
            Dictionary of smoothed metric values
        """
        return {name: self.get_smoothed(name) for name in self.history}

    def to_dict(self) -> Dict[str, Any]:
        """
        Export all history.

        Returns:
            Dictionary with history and steps
        """
        return {
            "history": self.history,
            "steps": self.step_history,
        }


# =============================================================================
# Main Trainer Class
# =============================================================================


class Trainer:
    """
    Training harness for UV-stabilized flow matching models.

    Implements Document Algorithm 1:
    1. Sample data y ~ p_data (from DataLoader)
    2. Sample IR state S_0 ~ μ_IR (via model.sample_ir_prior)
    3. Set UV target S_1 = Lift(y) (via model.lift_data)
    4. Sample t ~ Uniform[0,1]
    5. Compute loss (backbone-subtracted bulk FM)
    6. Update parameters

    Attributes
    ----------
    model : nn.Module
        The flow matching model
    config : TrainerConfig
        Training configuration
    device : torch.device
        Training device
    train_loader : DataLoader
        Training data loader
    val_loader : Optional[DataLoader]
        Validation data loader
    optimizer : Optimizer
        Adam optimizer
    scheduler : LRScheduler
        Learning rate scheduler
    ema : Optional[EMAModel]
        EMA tracker
    scaler : Optional[GradScaler]
        Gradient scaler for mixed precision
    checkpoint_manager : CheckpointManager
        Checkpoint manager
    metrics_tracker : MetricsTracker
        Metrics tracker
    global_step : int
        Current training step
    epoch : int
        Current epoch
    best_metric : float
        Best metric value seen
    best_metric_name : Optional[str]
        Name of best metric

    Example
    -------
    >>> trainer = Trainer(
    ...     model=flow_model,
    ...     train_dataset=train_data,
    ...     config=TrainerConfig(max_epochs=100),
    ... )
    >>> results = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: TrainerConfig,
        val_dataset: Optional[Dataset] = None,
        eval_fn: Optional[Callable[[nn.Module, int], Dict[str, float]]] = None,
        epoch_callback: Optional[Callable[[nn.Module, int, int], None]] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: The flow matching model (must have compute_losses method)
            train_dataset: Training dataset
            config: Training configuration
            val_dataset: Optional validation dataset
            eval_fn: Optional evaluation function for custom metrics
            epoch_callback: Optional callback called at end of each epoch
        """
        self.model = model
        self.config = config
        self.eval_fn = eval_fn
        self.epoch_callback = epoch_callback

        # Set device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and config.device != "cpu",
            drop_last=True,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )

        self.val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory and config.device != "cpu",
            )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        # Estimate total steps
        steps_per_epoch = len(self.train_loader)
        if config.max_steps is not None:
            self.total_steps = config.max_steps
        else:
            self.total_steps = config.max_epochs * steps_per_epoch

        # LR scheduler
        self.scheduler = get_lr_scheduler(self.optimizer, config, self.total_steps)

        # EMA
        self.ema: Optional[EMAModel] = None
        if config.use_ema:
            self.ema = EMAModel(self.model, decay=config.ema_decay, device=self.device)

        # Mixed precision
        self.scaler: Optional[GradScaler] = None
        self.amp_dtype = torch.float16
        if config.use_amp:
            self.scaler = GradScaler(init_scale=config.grad_scaler_init_scale)
            if config.amp_dtype == "bfloat16":
                self.amp_dtype = torch.bfloat16

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.checkpoint_dir),
            keep_last_n=config.keep_last_n_checkpoints,
        )

        # Metrics
        self.metrics_tracker = MetricsTracker()

        # Residual norm history for tracking convergence
        self.residual_norm_history: List[Dict[str, Any]] = []

        # State
        self.global_step = 0
        self.epoch = 0

        # Best model tracking
        self.best_metric = float("inf")
        self.best_metric_name: Optional[str] = None

        # Set seed
        if config.seed is not None:
            self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_amp_context(self):
        """Get autocast context for mixed precision."""
        if self.config.use_amp:
            return autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def train_step(self, batch: Tensor) -> Dict[str, float]:
        """
        Single training step.

        Document Algorithm 1: Sample, compute loss, update.

        Args:
            batch: Training batch

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Handle tuple batches (e.g., (image, label) from image datasets)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]  # Extract data, ignore labels

        batch = batch.to(self.device)

        # Forward pass with optional AMP
        with self._get_amp_context():
            losses = self.model.compute_losses(
                batch,
                step=self.global_step,
                epoch=self.epoch,
            )
            loss = losses["total_loss"]

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.config.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

        # LR scheduler step
        self.scheduler.step()

        # EMA update
        if self.ema is not None and self.global_step >= self.config.ema_start_step:
            if self.global_step % self.config.ema_update_every == 0:
                self.ema.update(self.model)

        # Convert losses to float
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics: Dict[str, float] = {}

        # Validation loss
        if self.val_loader is not None:
            val_losses = []
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                with self._get_amp_context():
                    losses = self.model.compute_losses(batch)
                val_losses.append(losses["total_loss"].item())
            metrics["val_loss"] = sum(val_losses) / len(val_losses)

        # Custom evaluation with EMA model
        if self.eval_fn is not None:
            if self.ema is not None:
                original = self.ema.apply_shadow(self.model)
                eval_metrics = self.eval_fn(self.model, self.global_step)
                self.ema.restore(self.model, original)
            else:
                eval_metrics = self.eval_fn(self.model, self.global_step)
            metrics.update(eval_metrics)

        return metrics

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            resume_from: Optional checkpoint path to resume from

        Returns:
            Final metrics dictionary
        """
        # Resume from checkpoint if specified
        if resume_from is not None:
            checkpoint = self.checkpoint_manager.load(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.ema,
                self.scaler,
                device=self.device,
            )
            self.global_step = checkpoint["step"]
            self.epoch = self.global_step // len(self.train_loader)

        # Training loop
        logger.info(f"Starting training for {self.total_steps} steps")
        start_time = time.time()

        while self.global_step < self.total_steps:
            epoch_start = time.time()
            epoch_residual_norms: List[float] = []
            epoch_residual_phi_norms: List[float] = []
            epoch_residual_pi_norms: List[float] = []

            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                # Training step
                step_metrics = self.train_step(batch)
                self.metrics_tracker.update(self.global_step, step_metrics)

                # Track residual norms if available
                if "residual_norm" in step_metrics:
                    epoch_residual_norms.append(step_metrics["residual_norm"])
                if "residual_phi_norm" in step_metrics:
                    epoch_residual_phi_norms.append(step_metrics["residual_phi_norm"])
                if "residual_pi_norm" in step_metrics:
                    epoch_residual_pi_norms.append(step_metrics["residual_pi_norm"])

                # Logging
                if self.global_step % self.config.log_every == 0:
                    smoothed = self.metrics_tracker.get_all_smoothed()
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0

                    log_str = (
                        f"Step {self.global_step}/{self.total_steps} | "
                        f"Loss: {smoothed.get('total_loss', float('nan')):.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.1f}"
                    )
                    logger.info(log_str)

                # Evaluation
                if (
                    self.global_step % self.config.eval_every == 0
                    and self.global_step > 0
                ):
                    eval_metrics = self.evaluate()
                    self.metrics_tracker.update(self.global_step, eval_metrics)
                    logger.info(f"Eval metrics: {eval_metrics}")

                    # Track best model
                    metric_candidates = [
                        "swd",
                        "wasserstein_2d",
                        "wasserstein_x",
                        "mmd_rbf",
                        "total_loss",
                    ]
                    current_metric = None
                    metric_name = None

                    for candidate in metric_candidates:
                        if candidate in eval_metrics:
                            current_metric = eval_metrics[candidate]
                            metric_name = candidate
                            break

                    if current_metric is None:
                        smoothed = self.metrics_tracker.get_all_smoothed()
                        if "total_loss" in smoothed:
                            current_metric = smoothed["total_loss"]
                            metric_name = "total_loss"

                    # Save best model
                    if current_metric is not None and current_metric < self.best_metric:
                        self.best_metric = current_metric
                        self.best_metric_name = metric_name
                        best_model_path = (
                            self.checkpoint_manager.checkpoint_dir / "best_model.pt"
                        )
                        best_state = self.model.state_dict()
                        if (
                            hasattr(self.model, "data_mean")
                            and self.model.data_mean is not None
                        ):
                            best_state["data_mean"] = self.model.data_mean
                            best_state["data_std"] = self.model.data_std
                        torch.save(best_state, best_model_path)
                        logger.info(
                            f"Saved best model ({metric_name}={current_metric:.4f})"
                        )

                # Checkpointing
                if (
                    self.global_step % self.config.checkpoint_every == 0
                    and self.global_step > 0
                ):
                    self.checkpoint_manager.save(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        ema=self.ema,
                        scaler=self.scaler,
                        metrics=self.metrics_tracker.get_all_smoothed(),
                        config=self.config,
                    )

                self.global_step += 1

            # End of epoch
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {self.epoch} completed in {epoch_time:.1f}s")

            # Record epoch-level residual norm history
            if epoch_residual_norms:
                avg_residual_norm = sum(epoch_residual_norms) / len(epoch_residual_norms)
                history_entry = {
                    "epoch": self.epoch,
                    "residual_norm": avg_residual_norm,
                }
                # Add phi and pi norms if available
                if epoch_residual_phi_norms:
                    history_entry["residual_phi_norm"] = sum(epoch_residual_phi_norms) / len(epoch_residual_phi_norms)
                if epoch_residual_pi_norms:
                    history_entry["residual_pi_norm"] = sum(epoch_residual_pi_norms) / len(epoch_residual_pi_norms)
                self.residual_norm_history.append(history_entry)

            # Call epoch callback if provided
            if self.epoch_callback is not None:
                self.epoch_callback(self.model, self.epoch, self.global_step)

            self.epoch += 1

        # Final checkpoint
        self.checkpoint_manager.save(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            scaler=self.scaler,
            metrics=self.metrics_tracker.get_all_smoothed(),
            config=self.config,
        )

        # Save final model
        final_model_path = self.checkpoint_manager.checkpoint_dir / "final_model.pt"
        final_state = self.model.state_dict()
        if hasattr(self.model, "data_mean") and self.model.data_mean is not None:
            final_state["data_mean"] = self.model.data_mean
            final_state["data_std"] = self.model.data_std
        torch.save(final_state, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Save training history
        history_path = self.checkpoint_manager.checkpoint_dir / "training_history.pt"
        history_data = {
            "residual_norm_history": self.residual_norm_history,
            "metrics_history": self.metrics_tracker.to_dict(),
            "final_step": self.global_step,
            "final_epoch": self.epoch,
        }
        torch.save(history_data, history_path)
        logger.info(f"Saved training history to {history_path}")

        # Save best model if not saved during training
        best_model_path = self.checkpoint_manager.checkpoint_dir / "best_model.pt"
        if not best_model_path.exists():
            best_state = self.model.state_dict()
            if hasattr(self.model, "data_mean") and self.model.data_mean is not None:
                best_state["data_mean"] = self.model.data_mean
                best_state["data_std"] = self.model.data_std
            torch.save(best_state, best_model_path)
            logger.info(f"Saved best model (final weights) to {best_model_path}")

        # Save EMA model separately if available
        if self.ema is not None:
            ema_model_path = self.checkpoint_manager.checkpoint_dir / "ema_model.pt"
            original = self.ema.apply_shadow(self.model)
            ema_state = self.model.state_dict()
            if hasattr(self.model, "data_mean") and self.model.data_mean is not None:
                ema_state["data_mean"] = self.model.data_mean
                ema_state["data_std"] = self.model.data_std
            torch.save(ema_state, ema_model_path)
            self.ema.restore(self.model, original)
            logger.info(f"Saved EMA model to {ema_model_path}")

        # Log best model info
        if self.best_metric_name is not None:
            logger.info(f"Best model: {self.best_metric_name}={self.best_metric:.4f}")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        return {
            "final_step": self.global_step,
            "final_metrics": self.metrics_tracker.get_all_smoothed(),
            "total_time": total_time,
        }

    def get_ema_model(self) -> nn.Module:
        """
        Get model with EMA parameters applied.

        Returns a copy of the model with EMA weights for evaluation/sampling.

        Returns:
            Model with EMA parameters (or original model if EMA not enabled)
        """
        if self.ema is None:
            return self.model

        model_copy = copy.deepcopy(self.model)
        for name, param in model_copy.named_parameters():
            if name in self.ema.shadow_params:
                param.data.copy_(self.ema.shadow_params[name])
        return model_copy


# =============================================================================
# Utility Functions
# =============================================================================


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for training.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file to log to
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )