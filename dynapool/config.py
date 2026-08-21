from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Every value that can affect a training run.

    The serialized configuration is hashed and stored with each checkpoint. A
    checkpoint is never resumed when this hash differs, which prevents an old
    Colab run from silently contaminating a revised experiment.
    """

    experiment_name: str = "reviewer_revision_v1"
    data_root: str = "data/tiny-imagenet-200"
    output_root: str = "outputs"
    num_classes: int = 200
    image_size: int = 64
    epochs: int = 50
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    crop_padding: int = 4
    horizontal_flip_probability: float = 0.5
    normalization_mean: tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
    normalization_std: tuple[float, float, float] = (0.2302, 0.2265, 0.2262)
    gate_hidden_dim: int = 256
    gate_dropout: float = 0.1
    gate_temperature: float = 1.0
    entropy_lambda: float = 0.05
    gem_p_init: float = 3.0
    gem_p_min: float = 1e-3
    gem_epsilon: float = 1e-6
    amp: bool = True
    deterministic: bool = True
    checkpoint_every_epochs: int = 1
    fake_data: bool = False
    fake_train_size: int = 512
    fake_val_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_name
