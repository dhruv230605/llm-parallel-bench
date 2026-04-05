from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.agent_pipeline import run_agent_benchmark
from evaluation.evaluate_all import combine_results, evaluate_directory, write_results
from scripts.generate_zero_shot import generate_zero_shot
from scripts.llm_client import load_model_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def _model_preflight_rows(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    rows: list[str] = []
    missing: list[str] = []
    registry = load_model_registry()
    providers = registry.get("providers", {})
    models_cfg = registry.get("models", {})
    all_models = sorted(set(args.zero_shot_models + args.agent_models))

    for model in all_models:
        model_cfg = models_cfg.get(model, {})
        provider_name = model_cfg.get("provider", "unmapped")
        provider_cfg = providers.get(provider_name, {})
        key_env = provider_cfg.get("api_key_env")

        if args.backend == "reference":
            base = "reference-backend"
            key_status = "n/a"
            rows.append(
                f"[preflight] model={model} provider={provider_name} base={base} key={key_status}"
            )
            continue

        base = args.api_base or provider_cfg.get("api_base") or os.environ.get("LLM_API_BASE") or "<missing>"
        if args.api_key:
            key_status = "set(cli)"
        elif key_env and os.environ.get(key_env):
            key_status = f"set(env:{key_env})"
        elif os.environ.get("LLM_API_KEY"):
            key_status = "set(env:LLM_API_KEY)"
        else:
            key_status = "missing"
            missing.append(model)

        rows.append(f"[preflight] model={model} provider={provider_name} base={base} key={key_status}")

    return rows, missing


def log_preflight(args: argparse.Namespace) -> None:
    print("[benchmark] started", flush=True)
    print(f"[preflight] timestamp={datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"[preflight] repo_root={REPO_ROOT}", flush=True)
    print(f"[preflight] python={sys.executable}", flush=True)
    print(f"[preflight] backend={args.backend}", flush=True)
    print(f"[preflight] tasks={args.tasks if args.tasks else 'all'}", flush=True)
    print(f"[preflight] attempts={args.attempts}", flush=True)
    print(f"[preflight] max_rounds={args.max_rounds}", flush=True)
    print(f"[preflight] temperature={args.temperature}", flush=True)
    print(f"[preflight] zero_shot_models={args.zero_shot_models}", flush=True)
    print(f"[preflight] agent_models={args.agent_models}", flush=True)
    print(f"[preflight] skip_zero_shot={args.skip_zero_shot}", flush=True)
    print(f"[preflight] skip_agent={args.skip_agent}", flush=True)
    print(f"[preflight] api_base_override={'set' if args.api_base else 'unset'}", flush=True)
    print(f"[preflight] api_key_override={'set' if args.api_key else 'unset'}", flush=True)

    rows, missing = _model_preflight_rows(args)
    for row in rows:
        print(row, flush=True)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing API key configuration for model(s): "
            f"{joined}. Set provider-specific keys in environment or pass --api-key."
        )

    print("[preflight] checks=ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full LLM benchmark pipeline.")
    parser.add_argument(
        "--zero-shot-models",
        nargs="+",
        default=["gpt-4.1-mini", "meta/llama-3.1-8b-instruct", "qwen/qwen2.5-coder-32b-instruct"],
    )
    parser.add_argument(
        "--agent-models",
        nargs="+",
        default=["gpt-4.1-mini", "meta/llama-3.1-8b-instruct", "qwen/qwen2.5-coder-32b-instruct"],
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--backend", choices=["openai_compatible", "reference"], default="openai_compatible")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    zero_rows = []
    agent_rows = []
    log_preflight(args)

    if not args.skip_zero_shot:
        print(
            f"[benchmark] zero-shot start models={args.zero_shot_models} attempts={args.attempts}",
            flush=True,
        )
        generate_zero_shot(
            args.zero_shot_models,
            attempts=args.attempts,
            backend=args.backend,
            tasks=args.tasks,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
        )
        zero_rows = evaluate_directory(
            args.zero_shot_models,
            method="Zero-shot",
            generated_root=REPO_ROOT / "generated_code",
            tasks=args.tasks,
        )
        write_results(zero_rows, REPO_ROOT / "results" / "zero_shot_scores.csv")
        print("[benchmark] zero-shot done", flush=True)

    if not args.skip_agent:
        print(
            f"[benchmark] agent start models={args.agent_models} max_rounds={args.max_rounds} attempts={args.attempts}",
            flush=True,
        )
        run_agent_benchmark(
            args.agent_models,
            tasks=args.tasks,
            backend=args.backend,
            api_base=args.api_base,
            api_key=args.api_key,
            max_rounds=args.max_rounds,
            attempts=args.attempts,
            temperature=args.temperature,
        )
        agent_rows = evaluate_directory(
            args.agent_models,
            method="Agent",
            generated_root=REPO_ROOT / "generated_code" / "agent",
            tasks=args.tasks,
        )
        write_results(agent_rows, REPO_ROOT / "results" / "agent_scores.csv")
        print("[benchmark] agent done", flush=True)

    combined = combine_results(zero_rows, agent_rows)
    write_results(combined, REPO_ROOT / "results" / "comparison.csv")
    print("[benchmark] completed", flush=True)


if __name__ == "__main__":
    main()
