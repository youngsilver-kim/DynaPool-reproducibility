from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import DatasetBundle, evaluation_loader, train_loader_for_epoch
from .models import BRANCH_NAMES, build_model, learned_gem_p
from .utils import (
    atomic_json_dump,
    atomic_torch_save,
    capture_rng_state,
    environment_metadata,
    restore_rng_state,
    set_global_seed,
    git_commit,
    write_csv,
)


def _run_dir(config: ExperimentConfig, method: str, seed: int) -> Path:
    return config.experiment_dir / method / f"seed_{seed:05d}"


def _run_hash(config: ExperimentConfig, method: str, seed: int, weights: Sequence[float] | None) -> str:
    payload = {
        "config_hash": config.stable_hash(),
        "git_commit": git_commit(),
        "method": method,
        "seed": seed,
        "static_weights": list(weights) if weights is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def learning_rate_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * float(epoch + 1) / max(1, config.warmup_epochs)
    progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
    return 0.5 * config.learning_rate * (1.0 + math.cos(math.pi * progress))


def _entropy_terms(alpha: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha is None or alpha.shape[1] <= 1:
        zero = alpha.new_zeros(()) if alpha is not None else torch.tensor(0.0)
        return zero, zero
    entropy = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(dim=1).mean()
    normalized = entropy / math.log(alpha.shape[1])
    uniform_penalty = 1.0 - normalized
    return entropy, uniform_penalty


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    config: ExperimentConfig,
    method: str,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    total_loss = total_ce = total_correct = total_entropy = total_samples = 0.0
    use_amp = config.amp and device.type == "cuda"
    entropy_lambda = config.entropy_lambda if method == "dyna_entropy" else 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            output = model(images)
            ce = criterion(output["logits"], targets)
            entropy, uniform_penalty = _entropy_terms(output.get("alpha"))
            loss = ce + entropy_lambda * uniform_penalty.to(ce.device)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = targets.shape[0]
        total_samples += batch
        total_loss += float(loss.detach()) * batch
        total_ce += float(ce.detach()) * batch
        total_correct += int((output["logits"].argmax(dim=1) == targets).sum())
        total_entropy += float(entropy.detach()) * batch
    return {
        "train_loss": total_loss / total_samples,
        "train_ce": total_ce / total_samples,
        "train_accuracy": total_correct / total_samples,
        "train_gate_entropy": total_entropy / total_samples,
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, collect: bool = False) -> dict[str, Any]:
    model.eval()
    total_loss = total_correct = total_samples = 0.0
    criterion = nn.CrossEntropyLoss()
    predictions: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        logits = output["logits"]
        batch = targets.shape[0]
        total_samples += batch
        total_loss += float(criterion(logits, targets)) * batch
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        if collect:
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.int16))
            targets_all.append(targets.cpu().numpy().astype(np.int16))
            if "alpha" in output:
                alphas.append(output["alpha"].float().cpu().numpy())
    result: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "n": int(total_samples),
    }
    if collect:
        result["predictions"] = np.concatenate(predictions)
        result["targets"] = np.concatenate(targets_all)
        if alphas:
            result["alphas"] = np.concatenate(alphas)
    return result


def summarize_alphas(alphas: np.ndarray, branch_names: Sequence[str]) -> dict[str, Any]:
    entropy = -(alphas * np.log(np.clip(alphas, 1e-12, None))).sum(axis=1)
    summary: dict[str, Any] = {
        "branches": list(branch_names),
        "sample_count": int(alphas.shape[0]),
        "entropy": {
            "mean": float(entropy.mean()),
            "std": float(entropy.std(ddof=1)),
            "min": float(entropy.min()),
            "q25": float(np.quantile(entropy, 0.25)),
            "median": float(np.median(entropy)),
            "q75": float(np.quantile(entropy, 0.75)),
            "max": float(entropy.max()),
        },
        "coefficients": {},
    }
    for index, name in enumerate(branch_names):
        values = alphas[:, index]
        summary["coefficients"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
        }
    return summary


def _active_branches(model: nn.Module) -> tuple[str, ...] | None:
    branches = getattr(model.head, "active_branches", None)
    return tuple(branches) if branches is not None else None


def _save_collected_outputs(
    run_dir: Path,
    split_name: str,
    result: dict[str, Any],
    class_names: Sequence[str],
    branch_names: Sequence[str] | None,
) -> dict[str, Any] | None:
    np.savez_compressed(
        run_dir / f"predictions_{split_name}.npz",
        targets=result["targets"],
        predictions=result["predictions"],
    )
    if "alphas" not in result or branch_names is None:
        return None
    alphas = result["alphas"]
    np.savez_compressed(
        run_dir / f"alphas_{split_name}.npz",
        alphas=alphas,
        targets=result["targets"],
        branches=np.asarray(branch_names),
    )
    summary = summarize_alphas(alphas, branch_names)
    atomic_json_dump(summary, run_dir / f"alpha_summary_{split_name}.json")
    class_rows: list[dict[str, Any]] = []
    targets = result["targets"]
    for class_index, class_name in enumerate(class_names):
        mask = targets == class_index
        if not mask.any():
            continue
        row: dict[str, Any] = {"class_index": class_index, "class_name": class_name, "n": int(mask.sum())}
        for index, branch in enumerate(branch_names):
            row[f"{branch}_mean"] = float(alphas[mask, index].mean())
            row[f"{branch}_std"] = float(alphas[mask, index].std(ddof=1)) if mask.sum() > 1 else 0.0
        class_rows.append(row)
    write_csv(class_rows, run_dir / f"alpha_by_class_{split_name}.csv")
    return summary


def resolve_static_weights(config: ExperimentConfig, seed: int) -> list[float]:
    source = _run_dir(config, "dyna", seed) / "alpha_summary_train.json"
    if not source.exists():
        raise FileNotFoundError(
            f"static_mean requires the completed dyna run for seed {seed}: {source}"
        )
    with source.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return [float(summary["coefficients"][name]["mean"]) for name in BRANCH_NAMES]


def run_training(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    method: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    static_weights = resolve_static_weights(config, seed) if method == "static_mean" else None
    run_dir = _run_dir(config, method, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_hash = _run_hash(config, method, seed, static_weights)
    result_path = run_dir / "result.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("run_hash") == run_hash and existing.get("status") == "completed":
            print(f"[skip] {method} seed={seed}: completed result found")
            return existing
        raise RuntimeError(
            f"Existing result has a different configuration: {result_path}. "
            "Use a new --experiment-name instead of overwriting evidence."
        )

    set_global_seed(seed, config.deterministic)
    model = build_model(
        method=method,
        num_classes=config.num_classes,
        gate_hidden_dim=config.gate_hidden_dim,
        gate_dropout=config.gate_dropout,
        gate_temperature=config.gate_temperature,
        gem_p_init=config.gem_p_init,
        gem_p_min=config.gem_p_min,
        gem_epsilon=config.gem_epsilon,
        static_weights=static_weights,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    scaler = _make_scaler(config.amp and device.type == "cuda")
    checkpoint_path = run_dir / "checkpoint_last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    elapsed_seconds = 0.0

    run_metadata = {
        "run_hash": run_hash,
        "config_hash": config.stable_hash(),
        "git_commit": git_commit(),
        "method": method,
        "seed": seed,
        "static_weights": static_weights,
        "reported_checkpoint_rule": "final epoch (no best-validation selection)",
        "architecture": {
            "backbone": (
                "ResNet-18, weights=None; torchvision ResNet initialization, with the replacement "
                "stem convolution explicitly initialized by Kaiming normal (fan_out, ReLU)"
            ),
            "head_initialization": "PyTorch default Linear/Conv2d initialization; GeM exponent initialized separately",
            "stem": "Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)+BN+ReLU; no initial max-pool",
            "feature_map_before_global_pooling": "512x8x8 for a 3x64x64 input",
            "attention_scoring": "Conv2d(512,1,kernel_size=1,bias=True) followed by spatial softmax",
            "gate_input": "512-D global-average-pooled backbone feature",
            "gate_mlp": (
                f"Linear(512,{config.gate_hidden_dim}) -> ReLU -> "
                f"Dropout({config.gate_dropout}) -> Linear({config.gate_hidden_dim},active_branch_count)"
            ),
            "gate_temperature": config.gate_temperature,
            "gem": (
                f"p=p_min+softplus(raw_p), p_init={config.gem_p_init}, "
                f"p_min={config.gem_p_min}, epsilon={config.gem_epsilon}"
            ),
        },
        "config": config.to_dict(),
    }
    atomic_json_dump(run_metadata, run_dir / "run_config.json")
    atomic_json_dump(environment_metadata(device), run_dir / "environment.json")

    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, device)
        if checkpoint.get("run_hash") != run_hash:
            raise RuntimeError(
                f"Checkpoint configuration mismatch: {checkpoint_path}. Use a new experiment name."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"[resume] {method} seed={seed}: epoch {start_epoch}/{config.epochs}")

    validation_loader = evaluation_loader(
        bundle.validation, config, seed=seed + 900_000, pin_memory=device.type == "cuda"
    )
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()
        lr = learning_rate_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Resetting the epoch seed ensures identical augmentation/order after resume.
        set_global_seed(seed * 100_003 + epoch, config.deterministic)
        train_loader = train_loader_for_epoch(
            bundle, config, run_seed=seed, epoch=epoch, pin_memory=device.type == "cuda"
        )
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, config, method)
        val_metrics = evaluate(model, validation_loader, device, collect=False)
        elapsed_seconds += time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            **train_metrics,
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(row)
        write_csv(history, run_dir / "history.csv")
        print(
            f"[{method} seed={seed}] epoch {epoch + 1:03d}/{config.epochs} "
            f"train={train_metrics['train_accuracy']:.4f} val={val_metrics['accuracy']:.4f}"
        )
        if (epoch + 1) % config.checkpoint_every_epochs == 0 or epoch + 1 == config.epochs:
            atomic_torch_save(
                {
                    "run_hash": run_hash,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "history": history,
                    "elapsed_seconds": elapsed_seconds,
                },
                checkpoint_path,
            )

    final_validation = evaluate(model, validation_loader, device, collect=True)
    train_eval_loader = evaluation_loader(
        bundle.train_eval, config, seed=seed + 800_000, pin_memory=device.type == "cuda"
    )
    final_train = evaluate(model, train_eval_loader, device, collect=True)
    branches = _active_branches(model)
    val_alpha_summary = _save_collected_outputs(
        run_dir, "validation", final_validation, bundle.class_names, branches
    )
    train_alpha_summary = _save_collected_outputs(
        run_dir, "train", final_train, bundle.class_names, branches
    )
    atomic_torch_save(
        {
            "run_hash": run_hash,
            "method": method,
            "seed": seed,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "static_weights": static_weights,
        },
        run_dir / "model_final.pt",
    )
    result = {
        "status": "completed",
        "run_hash": run_hash,
        "config_hash": config.stable_hash(),
        "method": method,
        "seed": seed,
        "epochs": config.epochs,
        "reported_checkpoint_rule": "final epoch (no best-validation selection)",
        "validation_accuracy": final_validation["accuracy"],
        "validation_loss": final_validation["loss"],
        "validation_samples": final_validation["n"],
        "training_accuracy_deterministic_view": final_train["accuracy"],
        "elapsed_seconds": elapsed_seconds,
        "learned_gem_p": learned_gem_p(model),
        "active_branches": list(branches) if branches is not None else [method],
        "gate_temperature": config.gate_temperature if branches is not None else None,
        "gate_hidden_dim": config.gate_hidden_dim if branches is not None else None,
        "gate_dropout": config.gate_dropout if branches is not None else None,
        "entropy_lambda": config.entropy_lambda if method == "dyna_entropy" else 0.0,
        "static_weights": static_weights,
        "validation_alpha_summary": val_alpha_summary,
        "train_alpha_summary": train_alpha_summary,
    }
    atomic_json_dump(result, result_path)
    checkpoint_path.unlink(missing_ok=True)  # compact final model replaces optimizer-heavy checkpoint
    print(f"[done] {method} seed={seed}: val={final_validation['accuracy']:.4f}")
    return result
