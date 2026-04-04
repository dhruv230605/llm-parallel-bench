from __future__ import annotations


class IssueParser:
    def parse(self, repo: dict, issues: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for issue in issues:
            title = issue.get("title", "")
            body = issue.get("body", "") or ""
            labels = [label.get("name", "") for label in issue.get("labels", []) if label.get("name")]
            candidates.append(
                {
                    "source_type": "issue",
                    "title": title,
                    "summary": self._summary(title, body),
                    "url": issue.get("html_url", ""),
                    "repo": repo.get("full_name", ""),
                    "repo_url": repo.get("html_url", ""),
                    "repo_keywords": repo.get("matched_keywords", []),
                    "keywords": labels,
                    "source_excerpt": body[:4000],
                    "artifact_id": issue.get("number"),
                    "artifact_number": issue.get("number"),
                    "artifact_state": issue.get("state", ""),
                    "labels": labels,
                    "comments": int(issue.get("comments", 0) or 0),
                    "created_at": issue.get("created_at", ""),
                    "updated_at": issue.get("updated_at", ""),
                }
            )
        return candidates

    def _summary(self, title: str, body: str) -> str:
        plain = " ".join(part.strip() for part in [title, body] if part.strip())
        return plain[:700]
