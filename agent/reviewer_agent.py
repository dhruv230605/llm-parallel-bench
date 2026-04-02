from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.llm_client import DEFAULT_SYSTEM_PROMPT, LLMClient


@dataclass
class ReviewerAgent:
    client: LLMClient
    model: str

    def improve(self, prompt: str, current_code: str, tags: list[str], reference_path: Path) -> str:
        if self.client.backend == "reference":
            return current_code

        user_prompt = (
            f"{prompt}\n\n"
            "Review the following C++17 solution for performance, synchronization safety, and parallel correctness. "
            "Preserve the public interface and return a full improved source file only.\n\n"
            f"Tags: {', '.join(tags)}\n\n"
            "Code:\n"
            f"{current_code}"
        )
        return self.client.generate_code(
            self.model,
            user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            reference_path=reference_path,
            temperature=0.1,
        )
