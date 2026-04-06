from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


def log_event(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _log_code_snippet(label: str, code: str) -> None:
    """Log first 50 lines and last 10 lines of code for visibility."""
    lines = code.split('\n')
    if len(lines) <= 65:
        log_event(f"[react] {label} ({len(lines)} lines):\n{code}")
    else:
        first_50 = '\n'.join(lines[:50])
        last_10 = '\n'.join(lines[-10:])
        log_event(f"[react] {label} ({len(lines)} lines) [first 50]:\n{first_50}\n... ({len(lines)-60} lines omitted) ...\n[last 10]:\n{last_10}")


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
    log_event(f"[react] task={task_name} model={model} start")
    planner = PlannerAgent(client=client, model=model)
    coder = CoderAgent(client=client, model=model)
    tester = TesterAgent()
    debugger = DebuggerAgent(client=client, model=model)
    reviewer = ReviewerAgent(client=client, model=model)

    log_event(f"[react] task={task_name} stage=plan start")
    plan = planner.plan(prompt, tags)
    log_event(f"[react] task={task_name} stage=plan done steps={len(plan)}")
    log_event(f"[react] PLAN_OUTPUT:\n{chr(10).join(f'  {i+1}. {step}' for i, step in enumerate(plan))}")

    log_event(f"[react] task={task_name} stage=codegen start")
    current_code = coder.generate(prompt, plan, reference_path)
    log_event(f"[react] task={task_name} stage=codegen done")
    _log_code_snippet("INITIAL_CODE", current_code)
    history: list[dict] = []

    for iteration in range(1, max_rounds + 1):
        log_event(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=test start")
        test_result = tester.test_code(task_name, current_code)
        log_event(
            f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=test done passed={test_result['passed']} status={test_result['status']}"
        )
        # Log diagnostics from test result
        if 'tests' in test_result and test_result['tests']:
            for test in test_result['tests']:
                log_event(f"[react] TEST_DIAGNOSTIC: {test.get('test_file', 'unknown')} status={test.get('status', 'unknown')}")
                if test.get('stderr'):
                    log_event(f"[react] TEST_STDERR:\n{test['stderr']}")
                if test.get('stdout'):
                    log_event(f"[react] TEST_STDOUT:\n{test['stdout']}")
        history.append(
            {
                "iteration": iteration,
                "stage": "test",
                "passed": test_result["passed"],
                "status": test_result["status"],
            }
        )
        if test_result["passed"]:
            log_event(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=review start")
            reviewed_code = reviewer.improve(prompt, current_code, tags, reference_path)
            reviewed_result = tester.test_code(task_name, reviewed_code)
            log_event(
                f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=review done passed={reviewed_result['passed']} status={reviewed_result['status']}"
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
                log_event(f"[react] task={task_name} final=reviewed_pass iteration={iteration}")
                return ReactResult(
                    final_code=reviewed_code,
                    passed=True,
                    iterations=iteration,
                    plan=plan,
                    history=history,
                )
            log_event(f"[react] task={task_name} final=base_pass iteration={iteration}")
            return ReactResult(
                final_code=current_code,
                passed=True,
                iterations=iteration,
                plan=plan,
                history=history,
            )

        log_event(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=debug start")
        current_code = debugger.repair(prompt, current_code, test_result, reference_path)
        _log_code_snippet(f"DEBUGGED_CODE_ITER{iteration}", current_code)
        log_event(f"[react] task={task_name} iteration={iteration}/{max_rounds} stage=debug done")
        history.append(
            {
                "iteration": iteration,
                "stage": "debug",
                "passed": False,
                "status": "repaired_candidate_generated",
            }
        )

    log_event(f"[react] task={task_name} stage=final_test start")
    final_result = tester.test_code(task_name, current_code)
    log_event(
        f"[react] task={task_name} stage=final_test done passed={final_result['passed']} status={final_result['status']}"
    )
    return ReactResult(
        final_code=current_code,
        passed=final_result["passed"],
        iterations=max_rounds,
        plan=plan,
        history=history,
    )
