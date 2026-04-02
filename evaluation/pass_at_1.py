from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.compile_and_run import evaluate_candidate


def discover_attempts(model_dir: Path, task_name: str) -> list[Path]:
    attempts = sorted(model_dir.glob(f"{task_name}_attempt*.cpp"))
    if attempts:
        return attempts
    fallback = model_dir / f"{task_name}.cpp"
    return [fallback] if fallback.exists() else []


def compute_pass_at_1(task_name: str, attempts: Iterable[Path]) -> dict:
    attempt_list = list(attempts)
    if not attempt_list:
        return {
            "task_name": task_name,
            "attempts": 0,
            "passes": 0,
            "pass_at_1": 0.0,
            "details": [],
        }

    details = []
    passes = 0
    for attempt_path in attempt_list:
        result = evaluate_candidate(task_name, attempt_path)
        details.append(result)
        if result["passed"]:
            passes += 1

    return {
        "task_name": task_name,
        "attempts": len(attempt_list),
        "passes": passes,
        "pass_at_1": passes / len(attempt_list),
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Pass@1 for a model-task directory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compute_pass_at_1(args.task, discover_attempts(Path(args.model_dir), args.task))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
