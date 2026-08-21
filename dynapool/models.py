from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


BRANCH_NAMES = ("avg", "max", "gem", "att")


class AvgPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool2d(x, 1).flatten(1)


class GeMPooling(nn.Module):
    """Generalized mean pooling with a strictly positive learned exponent."""

    def __init__(self, p_init: float = 3.0, p_min: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        if p_init <= p_min:
            raise ValueError("p_init must be greater than p_min")
        raw_init = math.log(math.expm1(p_init - p_min))
        self.raw_p = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.p_min = float(p_min)
        self.eps = float(eps)

    @property
    def p(self) -> torch.Tensor:
        return self.p_min + F.softplus(self.raw_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet layer4 ends in ReLU, but clamp also makes the domain explicit.
        safe_x = x.clamp_min(self.eps)
        pooled = safe_x.pow(self.p).mean(dim=(-2, -1))
        return pooled.clamp_min(self.eps).pow(1.0 / self.p)


class AttentionPooling(nn.Module):
    """Global attention aggregation using a learned 1x1 spatial scoring layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        scores = self.score(x).reshape(n, 1, h * w)
        attention = scores.softmax(dim=-1)
        features = x.reshape(n, c, h * w)
        return torch.bmm(features, attention.transpose(1, 2)).squeeze(-1)


class SmallImageResNet18(nn.Module):
    """ResNet-18 with a 3x3/stride-1 stem and no initial max-pool.

    A 64x64 input produces an 8x8 feature map immediately before global
    pooling. This avoids the 2x2 map produced by the ImageNet stem.
    """

    out_channels = 512

    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def _make_branch(name: str, channels: int, gem_kwargs: dict[str, float]) -> nn.Module:
    if name == "avg":
        return AvgPooling()
    if name == "max":
        return MaxPooling()
    if name == "gem":
        return GeMPooling(**gem_kwargs)
    if name == "att":
        return AttentionPooling(channels)
    raise ValueError(f"Unknown pooling branch: {name}")


class SinglePoolHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, branch: str, gem_kwargs: dict[str, float]):
        super().__init__()
        self.branch = branch
        self.pool = _make_branch(branch, channels, gem_kwargs)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, fmap: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pool(fmap)
        return {"logits": self.classifier(features), "features": features}


class MixturePoolHead(nn.Module):
    """Dense mixture head used by all DynaPool and ablation variants."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        active_branches: Sequence[str],
        gate_mode: str,
        hidden_dim: int,
        dropout: float,
        temperature: float,
        gem_kwargs: dict[str, float],
        fixed_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        if not active_branches:
            raise ValueError("At least one branch must be active")
        if len(set(active_branches)) != len(active_branches):
            raise ValueError("active_branches contains duplicates")
        self.active_branches = tuple(active_branches)
        self.gate_mode = gate_mode
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.branches = nn.ModuleDict(
            {name: _make_branch(name, channels, gem_kwargs) for name in self.active_branches}
        )
        if gate_mode == "dynamic":
            self.gate = nn.Sequential(
                nn.Linear(channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.active_branches)),
            )
            self.register_buffer("fixed_weights", torch.empty(0), persistent=False)
        elif gate_mode == "fixed":
            if fixed_weights is None or len(fixed_weights) != len(self.active_branches):
                raise ValueError("fixed_weights must match active_branches")
            weights = torch.as_tensor(fixed_weights, dtype=torch.float32)
            if (weights < 0).any() or not torch.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("fixed_weights must be finite, non-negative, and non-zero")
            self.register_buffer("fixed_weights", weights / weights.sum())
            self.gate = None
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, fmap: torch.Tensor) -> dict[str, torch.Tensor]:
        branch_features = torch.stack(
            [self.branches[name](fmap) for name in self.active_branches], dim=1
        )
        if self.gate_mode == "dynamic":
            gate_input = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            alpha = (self.gate(gate_input) / self.temperature).softmax(dim=-1)
        else:
            alpha = self.fixed_weights.unsqueeze(0).expand(fmap.shape[0], -1)
        features = (alpha.unsqueeze(-1) * branch_features).sum(dim=1)
        return {
            "logits": self.classifier(features),
            "alpha": alpha,
            "features": features,
        }


class PoolingClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, method: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.method = method

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(x))


def method_spec(method: str) -> tuple[tuple[str, ...], str]:
    specs = {
        "dyna": (BRANCH_NAMES, "dynamic"),
        "dyna_entropy": (BRANCH_NAMES, "dynamic"),
        "equal": (BRANCH_NAMES, "fixed"),
        "static_mean": (BRANCH_NAMES, "fixed"),
        "dyna_att_only": (("att",), "fixed"),
        "dyna_drop_avg": (("max", "gem", "att"), "dynamic"),
        "dyna_drop_max": (("avg", "gem", "att"), "dynamic"),
        "dyna_drop_gem": (("avg", "max", "att"), "dynamic"),
        "dyna_drop_att": (("avg", "max", "gem"), "dynamic"),
        "att_avg": (("avg", "att"), "dynamic"),
        "att_max": (("max", "att"), "dynamic"),
        "att_gem": (("gem", "att"), "dynamic"),
    }
    if method not in specs:
        raise ValueError(f"Unsupported mixture method: {method}")
    return specs[method]


def build_model(
    method: str,
    num_classes: int = 200,
    gate_hidden_dim: int = 256,
    gate_dropout: float = 0.1,
    gate_temperature: float = 1.0,
    gem_p_init: float = 3.0,
    gem_p_min: float = 1e-3,
    gem_epsilon: float = 1e-6,
    static_weights: Sequence[float] | None = None,
) -> PoolingClassifier:
    backbone = SmallImageResNet18()
    gem_kwargs = {"p_init": gem_p_init, "p_min": gem_p_min, "eps": gem_epsilon}
    if method in BRANCH_NAMES:
        head: nn.Module = SinglePoolHead(backbone.out_channels, num_classes, method, gem_kwargs)
    else:
        active, gate_mode = method_spec(method)
        fixed_weights: Iterable[float] | None = None
        if method == "equal":
            fixed_weights = [1.0 / len(active)] * len(active)
        elif method == "dyna_att_only":
            fixed_weights = [1.0]
        elif method == "static_mean":
            fixed_weights = static_weights
        head = MixturePoolHead(
            channels=backbone.out_channels,
            num_classes=num_classes,
            active_branches=active,
            gate_mode=gate_mode,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout,
            temperature=gate_temperature,
            gem_kwargs=gem_kwargs,
            fixed_weights=fixed_weights,
        )
    return PoolingClassifier(backbone, head, method)


def learned_gem_p(model: nn.Module) -> float | None:
    for module in model.modules():
        if isinstance(module, GeMPooling):
            return float(module.p.detach().cpu())
    return None
