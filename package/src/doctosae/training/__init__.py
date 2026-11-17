"""Training of DoctoSAE models with the objective, schedule and checkpoint selection of the paper."""

from doctosae.training.config import TrainingConfig
from doctosae.training.trainer import Trainer

__all__ = ["Trainer", "TrainingConfig"]
