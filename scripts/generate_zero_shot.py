from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.llm_client import LLMClient, sanitize_name


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "prompts" / "all_prompts.json"
GENERATED_ROOT = REPO_ROOT / "generated_code"


def load_task_catalog(catalog_path: Path = CATALOG_PATH) -> dict:
    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {entry["task_name"]: entry for entry in raw["tasks"]}


def select_tasks(catalog: dict, requested_tasks: Sequence[str] | None) -> list[str]:
    if not requested_tasks:
        return list(catalog.keys())
    unknown = [task for task in requested_tasks if task not in catalog]
    if unknown:
        raise KeyError(f"Unknown tasks requested: {', '.join(unknown)}")
    return list(requested_tasks)


def generate_zero_shot(
    models: Iterable[str],
    *,
    attempts: int = 10,
    backend: str = "openai_compatible",
    tasks: Sequence[str] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    output_root: Path = GENERATED_ROOT,
) -> None:
    catalog = load_task_catalog()
    selected_tasks = select_tasks(catalog, tasks)
    client = LLMClient(backend=backend, api_base=api_base, api_key=api_key, temperature=temperature)
    print(
        f"[zero-shot] backend={backend} models={list(models)} tasks={len(selected_tasks)} attempts={attempts}",
        flush=True,
    )

    for model in models:
        model_dir = output_root / sanitize_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[zero-shot] model={model} start", flush=True)
        for task_name in selected_tasks:
            entry = catalog[task_name]
            prompt = entry["prompt"]
            reference_path = REPO_ROOT / entry["reference"]
            latest_attempt_path = None
            print(f"[zero-shot] model={model} task={task_name} start", flush=True)
            for attempt_index in range(1, attempts + 1):
                print(
                    f"[zero-shot] model={model} task={task_name} attempt={attempt_index}/{attempts}",
                    flush=True,
                )
                code = client.generate_code(model, prompt, reference_path=reference_path)
                attempt_path = model_dir / f"{task_name}_attempt{attempt_index:02d}.cpp"
                attempt_path.write_text(code + "\n", encoding="utf-8")
                latest_attempt_path = attempt_path

            if latest_attempt_path is not None:
                shutil.copyfile(latest_attempt_path, model_dir / f"{task_name}.cpp")
            print(f"[zero-shot] model={model} task={task_name} done", flush=True)
        print(f"[zero-shot] model={model} done", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate zero-shot C++ solutions for the benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1-mini", "meta/llama-3.1-8b-instruct", "qwen/qwen2.5-coder-32b-instruct"],
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--backend", choices=["openai_compatible", "reference"], default="openai_compatible")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_zero_shot(
        args.models,
        attempts=args.attempts,
        backend=args.backend,
        tasks=args.tasks,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
