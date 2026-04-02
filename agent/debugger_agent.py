from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.llm_client import DEFAULT_SYSTEM_PROMPT, LLMClient


@dataclass
class DebuggerAgent:
    client: LLMClient
    model: str

    def repair(self, prompt: str, current_code: str, diagnostics: dict, reference_path: Path) -> str:
        if self.client.backend == "reference":
            return reference_path.read_text(encoding="utf-8")

        diagnostic_text = []
        for test in diagnostics.get("tests", []):
            diagnostic_text.append(f"Test: {test['test_file']}")
            diagnostic_text.append(f"Status: {test['status']}")
            if test["stderr"]:
                diagnostic_text.append("stderr:")
                diagnostic_text.append(test["stderr"])
            if test["stdout"]:
                diagnostic_text.append("stdout:")
                diagnostic_text.append(test["stdout"])
        joined_diagnostics = "\n".join(diagnostic_text)

        user_prompt = (
            f"{prompt}\n\n"
            "The following C++17 solution failed during testing. Fix the code while preserving the required interface.\n\n"
            "Current code:\n"
            f"{current_code}\n\n"
            "Diagnostics:\n"
            f"{joined_diagnostics}\n\n"
            "Return only the corrected complete C++ source file."
        )
        return self.client.generate_code(
            self.model,
            user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            reference_path=reference_path,
            temperature=0.1,
        )
