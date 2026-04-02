from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert C++ parallel programming assistant. "
    "Return exactly one compilable C++17 source file and no Markdown fences."
)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "model"


def strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class LLMClient:
    backend: str = "openai_compatible"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    timeout: int = 120

    def __post_init__(self) -> None:
        self.api_base = self.api_base or os.environ.get("LLM_API_BASE")
        self.api_key = self.api_key or os.environ.get("LLM_API_KEY")

    def complete(self, model: str, messages: Iterable[dict], temperature: Optional[float] = None) -> str:
        if self.backend == "reference":
            raise RuntimeError("Text completion is unavailable for the reference backend.")
        if self.backend != "openai_compatible":
            raise ValueError(f"Unsupported backend: {self.backend}")
        if not self.api_base or not self.api_key:
            raise RuntimeError(
                "Missing LLM_API_BASE or LLM_API_KEY. "
                "Set them in the environment or use --backend reference for a smoke test."
            )

        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        request = urllib.request.Request(
            url=self.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")
        return choices[0]["message"]["content"].strip()

    def generate_code(
        self,
        model: str,
        user_prompt: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        reference_path: Optional[Path] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.backend == "reference":
            if reference_path is None:
                raise ValueError("reference_path is required for the reference backend.")
            return Path(reference_path).read_text(encoding="utf-8")
        text = self.complete(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return strip_code_fences(text)
