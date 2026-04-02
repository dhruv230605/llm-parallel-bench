from __future__ import annotations

from dataclasses import dataclass

from scripts.llm_client import LLMClient


@dataclass
class PlannerAgent:
    client: LLMClient
    model: str

    def heuristic_plan(self, prompt: str, tags: list[str]) -> list[str]:
        steps = [
            "Identify the exact function or class signature required by the task prompt.",
            "Choose concurrency primitives and data structures that fit the task constraints.",
            "Implement the core parallel algorithm with deterministic synchronization.",
            "Handle shutdown, edge cases, and invalid inputs explicitly.",
            "Verify the solution against the provided tests and refine performance-sensitive sections.",
        ]
        if "openmp" in tags:
            steps.insert(2, "Place OpenMP pragmas around the dominant loop nest and preserve correctness on non-OpenMP builds.")
        if "mpi" in tags:
            steps.insert(2, "Initialize communicator-aware reduction logic and keep the collective operation consistent across all ranks.")
        if "lock_free" in tags:
            steps.insert(2, "Use atomics with compare-exchange loops and avoid mutexes in the queue hot path.")
        return steps

    def plan(self, prompt: str, tags: list[str]) -> list[str]:
        if self.client.backend == "reference":
            return self.heuristic_plan(prompt, tags)

        message = (
            "Create a concise numbered implementation plan for this C++ parallel programming task. "
            "Focus on interface requirements, synchronization choices, edge cases, and testing considerations.\n\n"
            f"Tags: {', '.join(tags)}\n\nTask:\n{prompt}"
        )
        try:
            response = self.client.complete(
                self.model,
                [
                    {"role": "system", "content": "You are a planning agent for parallel C++ code generation."},
                    {"role": "user", "content": message},
                ],
                temperature=0.1,
            )
        except Exception:
            return self.heuristic_plan(prompt, tags)

        lines = []
        for raw_line in response.splitlines():
            line = raw_line.strip().lstrip("-").strip()
            if not line:
                continue
            if line[0].isdigit() and "." in line:
                line = line.split(".", 1)[1].strip()
            lines.append(line)
        return lines or self.heuristic_plan(prompt, tags)
