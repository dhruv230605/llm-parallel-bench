from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.compile_and_run import evaluate_candidate


@dataclass
class TesterAgent:
    def test_code(self, task_name: str, code: str) -> dict:
        with tempfile.TemporaryDirectory(prefix=f"{task_name}_candidate_") as tmp_dir_name:
            candidate_path = Path(tmp_dir_name) / "candidate.cpp"
            candidate_path.write_text(code + "\n", encoding="utf-8")
            return evaluate_candidate(task_name, candidate_path)
