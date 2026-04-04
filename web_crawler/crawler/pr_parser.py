from __future__ import annotations


class PRParser:
    def parse(self, repo: dict, pulls: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for pull in pulls:
            title = pull.get("title", "")
            body = pull.get("body", "") or ""
            labels = [label.get("name", "") for label in pull.get("labels", []) if label.get("name")]
            candidates.append(
                {
                    "source_type": "pull_request",
                    "title": title,
                    "summary": self._summary(title, body),
                    "url": pull.get("html_url", ""),
                    "repo": repo.get("full_name", ""),
                    "repo_url": repo.get("html_url", ""),
                    "repo_keywords": repo.get("matched_keywords", []),
                    "keywords": labels,
                    "source_excerpt": body[:4000],
                    "artifact_id": pull.get("number"),
                    "artifact_number": pull.get("number"),
                    "artifact_state": pull.get("state", ""),
                    "labels": labels,
                    "comments": int(pull.get("comments", 0) or 0),
                    "created_at": pull.get("created_at", ""),
                    "updated_at": pull.get("updated_at", ""),
                }
            )
        return candidates

    def _summary(self, title: str, body: str) -> str:
        plain = " ".join(part.strip() for part in [title, body] if part.strip())
        return plain[:700]
