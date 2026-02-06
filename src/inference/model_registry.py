"""
ModelRegistry: discover, load, and manage trained RAFT-DVC models.

Design:
    - Convention over configuration: each model is a pair of files
      (checkpoints/foo.pth + configs/models/foo.yaml).
    - The registry auto-discovers models by scanning directories.
    - Loaded models are cached to avoid redundant disk reads.
    - This is the ONLY module that calls RAFTDVC.load_checkpoint().
      All other modules receive an already-loaded nn.Module.

Adding a new model:
    1. Train with desired (pyramid_levels, radius) combination.
    2. Save .pth checkpoint.
    3. Write a YAML model card in configs/models/.
    4. Done.  No Python code changes required.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass
class ModelCard:
    """
    Metadata about one trained model.
    Stored as a YAML sidecar alongside the .pth checkpoint.
    """

    # Identity
    name: str                               # e.g. "fine_p2_r4"
    description: str = ""

    # Architecture (must match the trained checkpoint)
    pyramid_levels: int = 4
    radius: int = 4
    feature_dim: int = 128
    hidden_dim: int = 96
    context_dim: int = 64
    iters: int = 12

    # File path
    checkpoint_path: str = ""

    # Informational metadata
    recommended_role: str = "fine"          # "coarse" or "fine"
    training_date: Optional[str] = None
    training_dataset: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_displacement_feature_space(self) -> float:
        """Max displacement in feature-space voxels."""
        return self.radius * (2 ** (self.pyramid_levels - 1))

    @property
    def gru_corr_input_dim(self) -> int:
        """Input dimension of correlation features to the GRU."""
        return self.pyramid_levels * (2 * self.radius + 1) ** 3


class ModelRegistry:
    """
    Discover, load, and cache trained models.

    Usage:
        registry = ModelRegistry(
            model_dir="./checkpoints",
            config_dir="./configs/models",
        )
        registry.scan()
        model, card = registry.get_model("fine_p2_r4", device)
    """

    def __init__(
        self,
        model_dir: str = "./checkpoints",
        config_dir: str = "./configs/models",
    ):
        self.model_dir = model_dir
        self.config_dir = config_dir
        self._catalog: Dict[str, ModelCard] = {}
        self._cache: Dict[str, nn.Module] = {}

    def scan(self) -> None:
        """
        Scan config_dir for .yaml model cards and populate the catalog.
        Does NOT load model weights (deferred to get_model).
        """
        raise NotImplementedError

    def list_models(self) -> List[ModelCard]:
        """Return all discovered ModelCards, sorted by name."""
        raise NotImplementedError

    def get_model(
        self,
        name: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[nn.Module, ModelCard]:
        """
        Load a model by name (with caching).

        Args:
            name: Model name (must exist in catalog after scan()).
            device: Target device.

        Returns:
            (loaded_model, model_card)

        Raises:
            KeyError: if name not found in catalog.
        """
        raise NotImplementedError

    def register_model(
        self,
        card: ModelCard,
    ) -> None:
        """
        Register a new model (e.g. after training).
        Writes the YAML sidecar to config_dir.
        """
        raise NotImplementedError
