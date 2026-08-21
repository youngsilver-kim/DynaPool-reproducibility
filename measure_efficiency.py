#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from thop import profile

from dynapool.models import build_model
from dynapool.utils import environment_metadata, set_global_seed
from train import ALL_METHODS


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    repetitions: int,
    trials: int,
) -> list[float]:
    model.eval()
    for _ in range(warmup):
        model(sample)
    if sample.device.type == "cuda":
        torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        if sample.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize()
        values.append(1_000.0 * (time.perf_counter() - start) / repetitions)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Transparent MAC/parameter/latency measurement")
    parser.add_argument("--output-dir", default="outputs/efficiency")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.image_size, args.warmup, args.repetitions, args.trials) <= 0:
        raise SystemExit("All numeric measurement arguments must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(2026, deterministic=True)
    rows = []
    raw_rows = []
    for method in args.methods:
        static = [0.25] * 4 if method == "static_mean" else None
        model = build_model(method, static_weights=static).to(device).eval()
        sample = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        # THOP reports multiply-accumulate pairs (MACs). We additionally report
        # 2*MACs as a clearly labelled FLOP convention; the paper must not mix them.
        macs, parameters = profile(model, inputs=(sample,), verbose=False)
        latency_values = measure_latency(
            model, sample, args.warmup, args.repetitions, args.trials
        )
        rows.append(
            {
                "method": method,
                "batch_size": args.batch_size,
                "input_shape": f"{args.batch_size}x3x{args.image_size}x{args.image_size}",
                "parameters_million": parameters / 1e6,
                "macs_giga_per_batch": macs / 1e9,
                "flops_giga_per_batch_2_per_mac": 2.0 * macs / 1e9,
                "latency_ms_mean": float(np.mean(latency_values)),
                "latency_ms_std": float(np.std(latency_values, ddof=1)),
                "latency_ms_median": float(np.median(latency_values)),
                "warmup_iterations": args.warmup,
                "timed_repetitions_per_trial": args.repetitions,
                "trials": args.trials,
                "precision": "FP32",
                "eval_mode": True,
                "inference_mode": True,
                "cuda_synchronize": device.type == "cuda",
                "counting_convention": "THOP MACs; FLOPs column is explicitly 2 FLOPs per MAC",
            }
        )
        for index, latency in enumerate(latency_values, start=1):
            raw_rows.append({"method": method, "trial": index, "latency_ms": latency})
        print(f"{method}: {np.mean(latency_values):.3f} ± {np.std(latency_values, ddof=1):.3f} ms")
    pd.DataFrame(rows).to_csv(output_dir / "efficiency_summary.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(output_dir / "latency_raw_trials.csv", index=False)
    metadata = environment_metadata(device)
    metadata.update(
        {
            "measurement_batch_size": args.batch_size,
            "input_image_size": args.image_size,
            "warmup_iterations": args.warmup,
            "timed_repetitions_per_trial": args.repetitions,
            "trials": args.trials,
            "precision": "FP32",
            "counting_convention": "THOP reports MACs; an additional column reports 2 FLOPs per MAC",
            "note": "THOP hooks supported modules; functional elementwise operations may not be included in MAC totals.",
        }
    )
    with (output_dir / "measurement_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
