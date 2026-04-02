from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.agent_pipeline import run_agent_benchmark
from evaluation.evaluate_all import combine_results, evaluate_directory, write_results
from scripts.generate_zero_shot import generate_zero_shot


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full LLM benchmark pipeline.")
    parser.add_argument("--zero-shot-models", nargs="+", default=["gemini", "model2", "model3"])
    parser.add_argument("--agent-models", nargs="+", default=["gemini"])
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

    if not args.skip_zero_shot:
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

    if not args.skip_agent:
        run_agent_benchmark(
            args.agent_models,
            tasks=args.tasks,
            backend=args.backend,
            api_base=args.api_base,
            api_key=args.api_key,
            max_rounds=args.max_rounds,
            temperature=args.temperature,
        )
        agent_rows = evaluate_directory(
            args.agent_models,
            method="Agent",
            generated_root=REPO_ROOT / "generated_code" / "agent",
            tasks=args.tasks,
        )
        write_results(agent_rows, REPO_ROOT / "results" / "agent_scores.csv")

    combined = combine_results(zero_rows, agent_rows)
    write_results(combined, REPO_ROOT / "results" / "comparison.csv")


if __name__ == "__main__":
    main()
