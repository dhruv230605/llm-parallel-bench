from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert C++ parallel programming assistant. "
    "Return exactly one compilable C++17 source file and no Markdown fences."
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "model_registry.json"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "model"


def strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_model_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


@dataclass
class LLMClient:
    backend: str = "openai_compatible"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    timeout: int = 180
    max_retries: int = 8
    retry_backoff_base: float = 2.0
    retry_backoff_cap: float = 120.0
    _registry: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.api_base = self.api_base or os.environ.get("LLM_API_BASE")
        self.api_key = self.api_key or os.environ.get("LLM_API_KEY")
        if not self._registry:
            self._registry = load_model_registry()

    def _resolve_endpoint(self, model: str) -> tuple[str, str]:
        """Return (api_base, api_key) for the given model.

        Uses the model registry when available; falls back to default fields.
        """
        registry = self._registry
        providers = registry.get("providers", {})
        models = registry.get("models", {})

        if model in models:
            provider_name = models[model].get("provider")
            provider = providers.get(provider_name, {})
            base = provider.get("api_base")
            key_env = provider.get("api_key_env")
            key = os.environ.get(key_env, "") if key_env else ""
            if base and key:
                return base, key

        # Fall back to instance defaults
        if self.api_base and self.api_key:
            return self.api_base, self.api_key

        raise RuntimeError(
            f"No API endpoint resolved for model '{model}'. "
            "Set the provider key in the environment, update model_registry.json, "
            "or pass --api-base / --api-key explicitly."
        )

    def complete(self, model: str, messages: Iterable[dict], temperature: Optional[float] = None) -> str:
        if self.backend == "reference":
            raise RuntimeError("Text completion is unavailable for the reference backend.")
        if self.backend != "openai_compatible":
            raise ValueError(f"Unsupported backend: {self.backend}")

        api_base, api_key = self._resolve_endpoint(model)

        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        payload_data = json.dumps(payload).encode("utf-8")
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            request = urllib.request.Request(
                url=api_base.rstrip("/") + "/chat/completions",
                data=payload_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < total_attempts:
                    wait_seconds = self._retry_after_seconds(exc, attempt)
                    _log(
                        "[rate-limit] model="
                        f"{model} provider_call={api_base.rstrip('/')}/chat/completions "
                        f"attempt={attempt}/{total_attempts} wait={wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                    continue
                if exc.code == 429:
                    raise RuntimeError(
                        "LLM request hit rate limit repeatedly and exhausted retries: "
                        f"model={model} attempts={total_attempts} last_response={body}"
                    ) from exc
                raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < total_attempts:
                    wait_seconds = min(self.retry_backoff_base ** (attempt - 1), self.retry_backoff_cap)
                    _log(
                        f"[transient-error] model={model} attempt={attempt}/{total_attempts} "
                        f"error={exc} wait={wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"LLM request failed: {exc}") from exc
            except (TimeoutError, OSError) as exc:
                if attempt < total_attempts:
                    wait_seconds = min(self.retry_backoff_base ** (attempt - 1), self.retry_backoff_cap)
                    _log(
                        f"[timeout] model={model} attempt={attempt}/{total_attempts} "
                        f"error={exc} wait={wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"LLM request timed out after {total_attempts} attempts: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")
        return choices[0]["message"]["content"].strip()

    def _retry_after_seconds(self, exc: urllib.error.HTTPError, attempt: int) -> float:
        retry_after = None
        if exc.headers:
            retry_after = exc.headers.get("Retry-After")

        if retry_after:
            retry_after = retry_after.strip()
            if retry_after.isdigit():
                return min(float(retry_after), self.retry_backoff_cap)
            try:
                retry_time = parsedate_to_datetime(retry_after)
                delta = (retry_time - datetime.now(retry_time.tzinfo)).total_seconds()
                if delta > 0:
                    return min(delta, self.retry_backoff_cap)
            except (TypeError, ValueError, OverflowError):
                pass

        return min(self.retry_backoff_base ** (attempt - 1), self.retry_backoff_cap)

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
