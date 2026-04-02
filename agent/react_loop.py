from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.coder_agent import CoderAgent
from agent.debugger_agent import DebuggerAgent
from agent.planner_agent import PlannerAgent
from agent.reviewer_agent import ReviewerAgent
from agent.tester_agent import TesterAgent
from scripts.llm_client import LLMClient


@dataclass
class ReactResult:
    final_code: str
    passed: bool
    iterations: int
    plan: list[str]
    history: list[dict]


def run_react_loop(
    *,
    task_name: str,
    prompt: str,
    tags: list[str],
    reference_path: Path,
    model: str,
    client: LLMClient,
    max_rounds: int = 3,
) -> ReactResult:
    planner = PlannerAgent(client=client, model=model)
    coder = CoderAgent(client=client, model=model)
    tester = TesterAgent()
    debugger = DebuggerAgent(client=client, model=model)
    reviewer = ReviewerAgent(client=client, model=model)

    plan = planner.plan(prompt, tags)
    current_code = coder.generate(prompt, plan, reference_path)
    history: list[dict] = []

    for iteration in range(1, max_rounds + 1):
        test_result = tester.test_code(task_name, current_code)
        history.append(
            {
                "iteration": iteration,
                "stage": "test",
                "passed": test_result["passed"],
                "status": test_result["status"],
            }
        )
        if test_result["passed"]:
            reviewed_code = reviewer.improve(prompt, current_code, tags, reference_path)
            reviewed_result = tester.test_code(task_name, reviewed_code)
            history.append(
                {
                    "iteration": iteration,
                    "stage": "review",
                    "passed": reviewed_result["passed"],
                    "status": reviewed_result["status"],
                }
            )
            if reviewed_result["passed"]:
                return ReactResult(
                    final_code=reviewed_code,
                    passed=True,
                    iterations=iteration,
                    plan=plan,
                    history=history,
                )
            return ReactResult(
                final_code=current_code,
                passed=True,
                iterations=iteration,
                plan=plan,
                history=history,
            )

        current_code = debugger.repair(prompt, current_code, test_result, reference_path)
        history.append(
            {
                "iteration": iteration,
                "stage": "debug",
                "passed": False,
                "status": "repaired_candidate_generated",
            }
        )

    final_result = tester.test_code(task_name, current_code)
    return ReactResult(
        final_code=current_code,
        passed=final_result["passed"],
        iterations=max_rounds,
        plan=plan,
        history=history,
    )
