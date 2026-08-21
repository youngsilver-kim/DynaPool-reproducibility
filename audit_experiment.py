#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train import ALL_METHODS


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast audit of reviewer-revision evidence")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--require-efficiency", action="store_true")
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    config_hashes: set[str] = set()
    target_reference: dict[int, np.ndarray] = {}
    for method in args.methods:
        for seed in args.seeds:
            run_dir = root / method / f"seed_{seed:05d}"
            result_path = run_dir / "result.json"
            check(f"completed:{method}:{seed}", result_path.exists(), str(result_path))
            if not result_path.exists():
                continue
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            check(
                f"status:{method}:{seed}",
                result.get("status") == "completed",
                f"status={result.get('status')}",
            )
            config_hashes.add(str(result.get("config_hash")))
            history_path = run_dir / "history.csv"
            history_rows = len(pd.read_csv(history_path)) if history_path.exists() else 0
            check(
                f"history:{method}:{seed}",
                history_rows == int(result.get("epochs", -1)),
                f"rows={history_rows}, epochs={result.get('epochs')}",
            )
            check(
                f"validation_count:{method}:{seed}",
                int(result.get("validation_samples", -1)) == 10_000,
                f"n={result.get('validation_samples')}",
            )
            prediction_path = run_dir / "predictions_validation.npz"
            check(f"predictions:{method}:{seed}", prediction_path.exists(), str(prediction_path))
            if prediction_path.exists():
                targets = np.load(prediction_path)["targets"]
                if seed not in target_reference:
                    target_reference[seed] = targets
                check(
                    f"target_order:{method}:{seed}",
                    np.array_equal(target_reference[seed], targets),
                    "paired predictions share identical validation target order",
                )
            active = result.get("active_branches", [])
            if method not in {"avg", "max", "gem", "att"}:
                alpha_path = run_dir / "alphas_validation.npz"
                summary_path = run_dir / "alpha_summary_validation.json"
                check(f"alphas:{method}:{seed}", alpha_path.exists(), str(alpha_path))
                check(f"alpha_summary:{method}:{seed}", summary_path.exists(), str(summary_path))
            if "gem" in active:
                learned_p = result.get("learned_gem_p")
                check(
                    f"gem_p:{method}:{seed}",
                    learned_p is not None and float(learned_p) > 0,
                    f"learned_p={learned_p}",
                )

    check("single_config_hash", len(config_hashes) == 1, f"config_hashes={sorted(config_hashes)}")
    analysis_required = [
        root / "analysis" / "accuracy_summary.csv",
        root / "analysis" / "paired_seed_tests.csv",
        root / "analysis" / "mcnemar_auxiliary.csv",
        root / "analysis" / "gate_summary_by_run.csv",
        root / "analysis" / "learning_curves.png",
    ]
    for path in analysis_required:
        check(f"analysis:{path.name}", path.exists(), str(path))
    if args.require_efficiency:
        for name in ["efficiency_summary.csv", "latency_raw_trials.csv", "measurement_environment.json"]:
            path = root / "efficiency" / name
            check(f"efficiency:{name}", path.exists(), str(path))

    passed = sum(item["passed"] for item in checks)
    failed = len(checks) - passed
    report = {"status": "passed" if failed == 0 else "failed", "passed": passed, "failed": failed, "checks": checks}
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Audit: {passed} passed, {failed} failed")
    for item in checks:
        if not item["passed"]:
            print(f"FAIL {item['check']}: {item['detail']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
