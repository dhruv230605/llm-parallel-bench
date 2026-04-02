from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.codeblue import score_codeblue
from evaluation.pass_at_1 import compute_pass_at_1, discover_attempts
from scripts.llm_client import sanitize_name


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "prompts" / "all_prompts.json"


def load_task_catalog() -> dict:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {entry["task_name"]: entry for entry in raw["tasks"]}


def evaluate_directory(
    models: Iterable[str],
    *,
    method: str,
    generated_root: Path,
    tasks: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    catalog = load_task_catalog()
    selected_tasks = list(tasks) if tasks else list(catalog.keys())
    rows: list[dict[str, object]] = []

    for model in models:
        primary_dir = generated_root / sanitize_name(model)
        model_dir = primary_dir if primary_dir.exists() else generated_root
        for task_name in selected_tasks:
            attempts = discover_attempts(model_dir, task_name)
            pass_result = compute_pass_at_1(task_name, attempts)
            reference_path = REPO_ROOT / catalog[task_name]["reference"]
            scores = [score_codeblue(attempt, reference_path)["final_score"] for attempt in attempts if attempt.exists()]
            rows.append(
                {
                    "Task Name": task_name,
                    "Model Name": model,
                    "Method": method,
                    "Pass@1": round(pass_result["pass_at_1"], 4),
                    "CodeBLUE Score": round(statistics.mean(scores), 4) if scores else 0.0,
                }
            )
    return rows


def write_results(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Task Name", "Model Name", "Method", "Pass@1", "CodeBLUE Score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def combine_results(*tables: list[dict[str, object]]) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    for table in tables:
        combined.extend(table)
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated code for Pass@1 and CodeBLUE.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = evaluate_directory(
        args.models,
        method=args.method,
        generated_root=Path(args.generated_root),
        tasks=args.tasks,
    )
    write_results(rows, Path(args.output))


if __name__ == "__main__":
    main()
