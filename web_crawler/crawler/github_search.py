from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Iterable

from utils.file_utils import load_json_if_exists, write_json
from utils.github_api import GitHubAPI


class GitHubSearch:
    def __init__(self, github_api: GitHubAPI, domain_keywords: list[str], keyword_weights: dict[str, float]) -> None:
        self.github_api = github_api
        self.domain_keywords = [keyword.lower() for keyword in domain_keywords]
        self.keyword_weights = {keyword.lower(): float(weight) for keyword, weight in keyword_weights.items()}
        self.logger = logging.getLogger("web_crawler.search")

    def search_repositories(
        self,
        queries: Iterable[str],
        cache_path: Path,
        per_query: int = 6,
        overall_limit: int = 20,
        refresh: bool = False,
    ) -> list[dict]:
        if not refresh:
            cached = load_json_if_exists(cache_path, default={})
            repositories = cached.get("repositories", []) if isinstance(cached, dict) else []
            if repositories:
                self.logger.info("Loaded %s repositories from cache %s", len(repositories[:overall_limit]), cache_path)
                return repositories[:overall_limit]

        seen: dict[str, dict] = {}
        for query in queries:
            self.logger.info("Searching GitHub repositories for query: %s", query)
            try:
                items = self.github_api.search_repositories(query, per_page=per_query)
            except Exception:
                items = []
                self.logger.warning("GitHub repository search failed for query: %s", query)
            for repo in items:
                normalized = self._normalize_repo(repo)
                if normalized["repo_score"] <= 0:
                    continue
                full_name = normalized["full_name"]
                previous = seen.get(full_name)
                if previous is None or normalized["repo_score"] > previous["repo_score"]:
                    seen[full_name] = normalized

        repositories = sorted(
            seen.values(),
            key=lambda item: (item["repo_score"], item["stars"], item["updated_at"]),
            reverse=True,
        )[:overall_limit]
        write_json(
            cache_path,
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "repositories": repositories,
            },
        )
        self.logger.info("Cached %s repositories to %s", len(repositories), cache_path)
        return repositories

    def _normalize_repo(self, repo: dict) -> dict:
        full_name = repo.get("full_name", "")
        description = repo.get("description", "") or ""
        topics = repo.get("topics", []) or []
        text = " ".join(
            [
                repo.get("name", ""),
                full_name,
                description,
                " ".join(topics),
            ]
        ).lower()

        matched_keywords = sorted({keyword for keyword in self.domain_keywords if keyword in text})
        weighted_score = sum(self.keyword_weights.get(keyword, 4.0) for keyword in matched_keywords)
        if any(token in text for token in ("parallel", "thread", "concurrent", "async", "scheduler", "mpi", "openmp")):
            weighted_score += 8.0

        return {
            "full_name": full_name,
            "html_url": repo.get("html_url", ""),
            "description": description,
            "topics": topics,
            "stars": int(repo.get("stargazers_count", 0) or 0),
            "updated_at": repo.get("updated_at", ""),
            "language": repo.get("language", ""),
            "default_branch": repo.get("default_branch", ""),
            "repo_score": round(weighted_score, 2),
            "matched_keywords": matched_keywords,
        }
