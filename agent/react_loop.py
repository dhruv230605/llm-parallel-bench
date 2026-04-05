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
    print(f"[react] task={task_name} model={model} start", flush=True)
    planner = PlannerAgent(client=client, model=model)
    coder = CoderAgent(client=client, model=model)
    tester = TesterAgent()
    debugger = DebuggerAgent(client=client, model=model)
    reviewer = ReviewerAgent(client=client, model=model)

    print(f"[react] task={task_name} stage=plan start", flush=True)
    plan = planner.plan(prompt, tags)
    print(f"[react] task={task_name} stage=plan done steps={len(plan)}", flush=True)

    print(f"[react] task={task_name} stage=codegen start", flush=True)
    current_code = coder.generate(prompt, plan, reference_path)
    print(f"[react] task={task_name} stage=codegen done", flush=True)
    history: list[dict] = []

    for iteration in range(1, max_rounds + 1):
        print(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=test start", flush=True)
        test_result = tester.test_code(task_name, current_code)
        print(
            f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=test done passed={test_result['passed']} status={test_result['status']}",
            flush=True,
        )
        history.append(
            {
                "iteration": iteration,
                "stage": "test",
                "passed": test_result["passed"],
                "status": test_result["status"],
            }
        )
        if test_result["passed"]:
            print(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=review start", flush=True)
            reviewed_code = reviewer.improve(prompt, current_code, tags, reference_path)
            reviewed_result = tester.test_code(task_name, reviewed_code)
            print(
                f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=review done passed={reviewed_result['passed']} status={reviewed_result['status']}",
                flush=True,
            )
            history.append(
                {
                    "iteration": iteration,
                    "stage": "review",
                    "passed": reviewed_result["passed"],
                    "status": reviewed_result["status"],
                }
            )
            if reviewed_result["passed"]:
                print(f"[react] task={task_name} final=reviewed_pass iteration={iteration}", flush=True)
                return ReactResult(
                    final_code=reviewed_code,
                    passed=True,
                    iterations=iteration,
                    plan=plan,
                    history=history,
                )
            print(f"[react] task={task_name} final=base_pass iteration={iteration}", flush=True)
            return ReactResult(
                final_code=current_code,
                passed=True,
                iterations=iteration,
                plan=plan,
                history=history,
            )

        print(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=debug start", flush=True)
        current_code = debugger.repair(prompt, current_code, test_result, reference_path)
        print(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=debug done", flush=True)
        history.append(
            {
                "iteration": iteration,
                "stage": "debug",
                "passed": False,
                "status": "repaired_candidate_generated",
            }
        )

    print(f"[react] task={task_name} stage=final_test start", flush=True)
    final_result = tester.test_code(task_name, current_code)
    print(
        f"[react] task={task_name} stage=final_test done passed={final_result['passed']} status={final_result['status']}",
        flush=True,
    )
    return ReactResult(
        final_code=current_code,
        passed=final_result["passed"],
        iterations=max_rounds,
        plan=plan,
        history=history,
    )
