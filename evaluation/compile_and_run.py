from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "prompts" / "all_prompts.json"


@dataclass
class TestExecutionResult:
    test_file: str
    passed: bool
    status: str
    compiler: str
    compile_command: list[str]
    run_command: list[str]
    stdout: str
    stderr: str


def load_task_catalog() -> dict[str, dict[str, Any]]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {entry["task_name"]: entry for entry in raw["tasks"]}


def find_executable(candidates: list[str]) -> str | None:
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def compiler_for_task(entry: dict[str, Any]) -> str | None:
    compile_cfg = entry.get("compile", {})
    if compile_cfg.get("mpi"):
        return find_executable(["mpic++", "mpicxx"])
    return find_executable(["g++", "clang++", "c++"])


def mpi_runner() -> str | None:
    return find_executable(["mpiexec", "mpirun"])


def compile_flags(entry: dict[str, Any]) -> list[str]:
    flags = ["-std=c++17", "-O2", "-Wall", "-Wextra"]
    compile_cfg = entry.get("compile", {})
    if compile_cfg.get("openmp"):
        flags.append("-fopenmp")
    if not compile_cfg.get("mpi"):
        flags.append("-pthread")
    return flags


def fallback_compile_flags(entry: dict[str, Any], flags: list[str]) -> list[str] | None:
    compile_cfg = entry.get("compile", {})
    if compile_cfg.get("openmp") and "-fopenmp" in flags:
        return [flag for flag in flags if flag != "-fopenmp"]
    return None


def run_command_for_task(executable: Path, entry: dict[str, Any]) -> list[str]:
    compile_cfg = entry.get("compile", {})
    if compile_cfg.get("mpi"):
        runner = mpi_runner()
        if runner is None:
            return []
        processes = str(compile_cfg.get("mpi_processes", 2))
        return [runner, "-n", processes, str(executable)]
    return [str(executable)]


def evaluate_candidate(task_name: str, candidate_path: Path) -> dict[str, Any]:
    catalog = load_task_catalog()
    if task_name not in catalog:
        raise KeyError(f"Task '{task_name}' does not exist in the prompt catalog.")

    entry = catalog[task_name]
    compiler = compiler_for_task(entry)
    if compiler is None:
        return {
            "task_name": task_name,
            "candidate": str(candidate_path),
            "passed": False,
            "status": "environment_error",
            "tests": [],
            "error": "No suitable C++ compiler found for this task.",
        }

    test_results: list[TestExecutionResult] = []
    for relative_test_path in entry["tests"]:
        test_path = REPO_ROOT / relative_test_path
        with tempfile.TemporaryDirectory(prefix=f"{task_name}_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            shutil.copy2(candidate_path, tmp_dir / "solution.cpp")
            shutil.copy2(test_path, tmp_dir / test_path.name)

            executable = tmp_dir / ("test_exec.exe" if sys.platform.startswith("win") else "test_exec")
            flags = compile_flags(entry)
            compile_cmd = [compiler, *flags, test_path.name, "-o", str(executable)]
            compile_proc = subprocess.run(
                compile_cmd,
                cwd=tmp_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            if compile_proc.returncode != 0:
                fallback_flags = fallback_compile_flags(entry, flags)
                if fallback_flags is not None:
                    fallback_cmd = [compiler, *fallback_flags, test_path.name, "-o", str(executable)]
                    fallback_proc = subprocess.run(
                        fallback_cmd,
                        cwd=tmp_dir,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if fallback_proc.returncode == 0:
                        compile_cmd = fallback_cmd
                        compile_proc = fallback_proc

            if compile_proc.returncode != 0:
                test_results.append(
                    TestExecutionResult(
                        test_file=relative_test_path,
                        passed=False,
                        status="compile_error",
                        compiler=compiler,
                        compile_command=compile_cmd,
                        run_command=[],
                        stdout=compile_proc.stdout,
                        stderr=compile_proc.stderr,
                    )
                )
                continue

            run_cmd = run_command_for_task(executable, entry)
            if not run_cmd:
                test_results.append(
                    TestExecutionResult(
                        test_file=relative_test_path,
                        passed=False,
                        status="environment_error",
                        compiler=compiler,
                        compile_command=compile_cmd,
                        run_command=[],
                        stdout="",
                        stderr="MPI runner not found.",
                    )
                )
                continue

            run_proc = subprocess.run(
                run_cmd,
                cwd=tmp_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            status = "passed" if run_proc.returncode == 0 else "runtime_error"
            test_results.append(
                TestExecutionResult(
                    test_file=relative_test_path,
                    passed=run_proc.returncode == 0,
                    status=status,
                    compiler=compiler,
                    compile_command=compile_cmd,
                    run_command=run_cmd,
                    stdout=run_proc.stdout,
                    stderr=run_proc.stderr,
                )
            )

    passed = bool(test_results) and all(result.passed for result in test_results)
    return {
        "task_name": task_name,
        "candidate": str(candidate_path),
        "passed": passed,
        "status": "passed" if passed else "failed",
        "tests": [asdict(result) for result in test_results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and run a solution against a benchmark task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_candidate(args.task, Path(args.candidate).resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
