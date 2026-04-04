from __future__ import annotations

import re


COMMENT_PATTERN = re.compile(r"(?://\s*(TODO|FIXME|BUG|HACK)[:\s-]*(.*))", flags=re.IGNORECASE)


class CodeExtractor:
    def extract(self, repo_full_name: str, code_hits: list[dict], interesting_files: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for hit in code_hits:
            path = hit.get("path", "")
            candidates.append(
                {
                    "source_type": "code_search",
                    "title": f"Parallel code hotspot in {path}",
                    "summary": f"Relevant parallel C++ file discovered in {repo_full_name}: {path}",
                    "url": hit.get("html_url", ""),
                    "repo": repo_full_name,
                    "keywords": self._keywords_from_text(path),
                    "source_excerpt": hit.get("text_matches", ""),
                    "reference_snippet": "",
                }
            )

        for file_entry in interesting_files:
            path = file_entry["path"]
            content = file_entry["content"]
            comments = COMMENT_PATTERN.findall(content)
            if comments:
                for _, message in comments[:3]:
                    candidates.append(
                        {
                            "source_type": "todo_comment",
                            "title": f"TODO/FIXME from {path}",
                            "summary": message.strip() or f"Concurrency-related TODO in {path}",
                            "url": "",
                            "repo": repo_full_name,
                            "keywords": self._keywords_from_text(path + " " + message),
                            "source_excerpt": content[:2500],
                            "reference_snippet": self._truncate_cpp(content),
                        }
                    )
                continue

            lower_path = path.lower()
            if "example" in lower_path or "benchmark" in lower_path:
                candidates.append(
                    {
                        "source_type": "example_file",
                        "title": f"Parallel example from {path}",
                        "summary": f"Example or benchmark file in {repo_full_name} capturing non-trivial parallel logic.",
                        "url": "",
                        "repo": repo_full_name,
                        "keywords": self._keywords_from_text(path + " " + content[:800]),
                        "source_excerpt": content[:2500],
                        "reference_snippet": self._truncate_cpp(content),
                    }
                )
        return candidates

    def _truncate_cpp(self, content: str) -> str:
        lines = content.splitlines()
        return "\n".join(lines[:80]).strip()

    def _keywords_from_text(self, text: str) -> list[str]:
        lower = text.lower()
        keywords = []
        for candidate in [
            "thread",
            "mutex",
            "atomic",
            "openmp",
            "mpi",
            "parallel",
            "scheduler",
            "queue",
            "future",
            "async",
            "lock",
            "barrier",
            "concurrent",
        ]:
            if candidate in lower:
                keywords.append(candidate)
        return keywords
