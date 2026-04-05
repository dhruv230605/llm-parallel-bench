from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.agent_pipeline import run_agent_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-agent ReAct generation pipeline.")
    parser.add_argument("--models", nargs="+", default=["gemini"])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--backend", choices=["openai_compatible", "reference"], default="openai_compatible")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_agent_benchmark(
        args.models,
        tasks=args.tasks,
        backend=args.backend,
        api_base=args.api_base,
        api_key=args.api_key,
        max_rounds=args.max_rounds,
        attempts=args.attempts,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
