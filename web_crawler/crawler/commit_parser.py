from __future__ import annotations


class CommitParser:
    def parse(self, repo: dict, commits: list[dict]) -> list[dict]:
        candidates: list[dict] = []
        for commit in commits:
            message = commit.get("commit", {}).get("message", "")
            title = message.splitlines()[0][:180]
            candidates.append(
                {
                    "source_type": "commit",
                    "title": title,
                    "summary": message[:700],
                    "url": commit.get("html_url", ""),
                    "repo": repo.get("full_name", ""),
                    "repo_url": repo.get("html_url", ""),
                    "repo_keywords": repo.get("matched_keywords", []),
                    "keywords": [],
                    "source_excerpt": message[:4000],
                    "artifact_id": commit.get("sha", ""),
                    "artifact_sha": commit.get("sha", ""),
                    "artifact_state": "merged",
                    "labels": [],
                    "comments": 0,
                    "created_at": commit.get("commit", {}).get("author", {}).get("date", ""),
                    "updated_at": commit.get("commit", {}).get("author", {}).get("date", ""),
                }
            )
        return candidates
