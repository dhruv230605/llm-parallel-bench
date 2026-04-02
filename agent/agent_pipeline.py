from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from agent.react_loop import run_react_loop
from scripts.generate_zero_shot import load_task_catalog, select_tasks
from scripts.llm_client import LLMClient, sanitize_name


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = REPO_ROOT / "generated_code" / "agent"


def run_agent_benchmark(
    models: Sequence[str],
    *,
    tasks: Sequence[str] | None = None,
    backend: str = "openai_compatible",
    api_base: str | None = None,
    api_key: str | None = None,
    max_rounds: int = 3,
    temperature: float = 0.2,
    output_root: Path = GENERATED_ROOT,
) -> None:
    catalog = load_task_catalog()
    selected_tasks = select_tasks(catalog, tasks)
    client = LLMClient(backend=backend, api_base=api_base, api_key=api_key, temperature=temperature)

    for model in models:
        model_dir = output_root / sanitize_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        for task_name in selected_tasks:
            entry = catalog[task_name]
            reference_path = REPO_ROOT / entry["reference"]
            result = run_react_loop(
                task_name=task_name,
                prompt=entry["prompt"],
                tags=entry.get("tags", []),
                reference_path=reference_path,
                model=model,
                client=client,
                max_rounds=max_rounds,
            )
            (model_dir / f"{task_name}.cpp").write_text(result.final_code + "\n", encoding="utf-8")
            (model_dir / f"{task_name}_react.json").write_text(
                json.dumps(
                    {
                        "task_name": task_name,
                        "model": model,
                        "passed": result.passed,
                        "iterations": result.iterations,
                        "plan": result.plan,
                        "history": result.history,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
