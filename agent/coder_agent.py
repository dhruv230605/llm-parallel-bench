from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.llm_client import DEFAULT_SYSTEM_PROMPT, LLMClient


@dataclass
class CoderAgent:
    client: LLMClient
    model: str

    def generate(self, prompt: str, plan: list[str], reference_path: Path) -> str:
        plan_block = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(plan))
        user_prompt = (
            f"{prompt}\n\n"
            "Follow this plan while coding:\n"
            f"{plan_block}\n\n"
            "Return only one complete C++17 source file. Do not include explanations."
        )
        return self.client.generate_code(
            self.model,
            user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            reference_path=reference_path,
            temperature=0.2,
        )
