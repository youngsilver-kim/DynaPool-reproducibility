#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import torch

from dynapool.config import ExperimentConfig
from dynapool.data import build_datasets
from dynapool.engine import run_training


BASELINE_METHODS = ["avg", "max", "gem", "att", "dyna"]
REVIEWER_ABLATIONS = [
    "dyna_entropy",
    "dyna_att_only",
    "equal",
    "static_mean",
    "dyna_drop_avg",
    "dyna_drop_max",
    "dyna_drop_gem",
    "dyna_drop_att",
    "att_avg",
    "att_max",
    "att_gem",
]
ALL_METHODS = BASELINE_METHODS + REVIEWER_ABLATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible Tiny-ImageNet pooling benchmark with Colab-safe auto-resume"
    )
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--experiment-name", default="reviewer_revision_v1")
    parser.add_argument("--suite", choices=["baselines", "reviewer", "all"], default="baselines")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--entropy-lambda", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="Fast pipeline smoke test; not paper evidence")
    parser.add_argument("--fake-train-size", type=int, default=512)
    parser.add_argument("--fake-val-size", type=int, default=256)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("epochs and batch sizes must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise SystemExit("warmup-epochs must satisfy 0 <= warmup < epochs")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Provide one or more unique seeds")
    if args.entropy_lambda < 0:
        raise SystemExit("entropy-lambda must be non-negative")


def selected_methods(args: argparse.Namespace) -> list[str]:
    if args.methods:
        return args.methods
    if args.suite == "baselines":
        return BASELINE_METHODS
    if args.suite == "reviewer":
        return REVIEWER_ABLATIONS
    return ALL_METHODS


def main() -> int:
    args = parse_args()
    validate_args(args)
    methods = selected_methods(args)
    if "static_mean" in methods and "dyna" in methods and methods.index("static_mean") < methods.index("dyna"):
        raise SystemExit("static_mean must run after dyna because its fixed weights are derived from dyna")

    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        data_root=args.data_root,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        gate_temperature=args.gate_temperature,
        entropy_lambda=args.entropy_lambda,
        checkpoint_every_epochs=args.checkpoint_every,
        amp=not args.no_amp,
        deterministic=not args.non_deterministic,
        fake_data=args.fake_data,
        fake_train_size=args.fake_train_size,
        fake_val_size=args.fake_val_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} experiment={config.experiment_name} config_hash={config.stable_hash()}")
    print(f"methods={methods} seeds={args.seeds}")
    if args.fake_data:
        print("WARNING: --fake-data is only a software smoke test and must never be reported in the paper.")
    bundle = build_datasets(config)

    failures: list[str] = []
    for method in methods:
        for seed in args.seeds:
            try:
                run_training(config, bundle, method, seed, device)
            except KeyboardInterrupt:
                print("Interrupted. The last completed epoch is safely checkpointed.", file=sys.stderr)
                return 130
            except Exception as exc:
                failure = f"{method} seed={seed}: {type(exc).__name__}: {exc}"
                print(f"[failure] {failure}", file=sys.stderr)
                failures.append(failure)
                # static_mean cannot run for a missing dyna seed, but other independent runs should continue.
    if failures:
        print("\nFailed runs:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
