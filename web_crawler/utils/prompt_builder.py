from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests


OPENAI_API_BASE = "https://api.openai.com/v1"


@dataclass
class OpenAIHelper:
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: int = 90
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str | None:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = self.session.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0]["message"]["content"].strip()

    def _json_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict[str, Any] | None:
        text = self._chat(system_prompt, user_prompt, temperature=temperature)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def analyze_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        result = self._json_chat(
            "You evaluate whether GitHub-derived tasks are difficult for zero-shot LLM coding in parallel C++. Respond with JSON only.",
            (
                "Return JSON with keys `summary`, `difficulty_score`, `difficulty_reason`, and `hard_tags`.\n\n"
                f"Source type: {candidate.get('source_type', '')}\n"
                f"Repo: {candidate.get('repo', '')}\n"
                f"Title: {candidate.get('title', '')}\n"
                f"Summary: {candidate.get('summary', '')}\n"
                f"Keywords: {', '.join(candidate.get('keywords', []))}\n"
                f"Repo keywords: {', '.join(candidate.get('repo_keywords', []))}\n"
                f"Excerpt: {candidate.get('source_excerpt', '')[:3500]}"
            ),
        )
        if result:
            return result
        return {
            "summary": candidate.get("summary", ""),
            "difficulty_score": None,
            "difficulty_reason": "",
            "hard_tags": [],
        }

    def standardize_prompt(self, candidate: dict[str, Any]) -> dict[str, str]:
        result = self._json_chat(
            "Rewrite benchmark prompts for research datasets. Keep them one line and implementation-oriented. Respond with JSON only.",
            (
                "Return JSON with keys `prompt` and `difficulty_reason`.\n\n"
                f"Title: {candidate.get('title', '')}\n"
                f"Summary: {candidate.get('llm_summary') or candidate.get('summary', '')}\n"
                f"Tags: {', '.join(candidate.get('hard_tags', []) or candidate.get('keywords', []))}\n"
                f"Source type: {candidate.get('source_type', '')}"
            ),
        )
        if result and result.get("prompt"):
            return {
                "prompt": " ".join(result["prompt"].splitlines()).strip(),
                "difficulty_reason": result.get("difficulty_reason", "").strip(),
            }
        fallback_prompt = candidate.get("title") or candidate.get("summary") or "Implement a hard parallel C++ task mined from GitHub."
        return {
            "prompt": " ".join(fallback_prompt.splitlines())[:180].strip(),
            "difficulty_reason": candidate.get("difficulty_reason", ""),
        }

    def generate_task_package(self, candidate: dict[str, Any], grounding: dict[str, Any]) -> dict[str, str] | None:
        result = self._json_chat(
            (
                "You synthesize grounded benchmark tasks for hard parallel C++ research datasets. "
                "Use the GitHub artifact only as inspiration and grounding, not as text to copy. "
                "Return JSON only."
            ),
            (
                "Create a benchmark package with these JSON keys exactly: "
                "`title`, `prompt`, `difficulty_reason`, `code_cpp`, `code_h`, `test_cpp`, `run_tests_sh`.\n"
                "Rules:\n"
                "- `prompt` must be one line.\n"
                "- `code_cpp` must be a compilable C++17 reference implementation.\n"
                "- `code_h` may be empty if unnecessary.\n"
                "- `test_cpp` must compile as a single test file including ../code.cpp or ../code.h.\n"
                "- `run_tests_sh` must compile and run the test with g++ unless MPI/OpenMP is clearly required.\n"
                "- Keep the task difficult and concurrency-specific.\n\n"
                f"Candidate metadata:\n{json.dumps(candidate, ensure_ascii=True)[:4500]}\n\n"
                f"Grounding details:\n{json.dumps(grounding, ensure_ascii=True)[:6000]}"
            ),
            temperature=0.15,
        )
        if result and result.get("prompt") and result.get("code_cpp") and result.get("test_cpp"):
            return {
                "title": result.get("title", candidate.get("title", "GitHub-derived parallel task")).strip(),
                "prompt": " ".join(result["prompt"].splitlines()).strip(),
                "difficulty_reason": result.get("difficulty_reason", "").strip(),
                "code_cpp": result["code_cpp"],
                "code_h": result.get("code_h", ""),
                "test_cpp": result["test_cpp"],
                "run_tests_sh": result.get("run_tests_sh", "").strip(),
            }
        return None
