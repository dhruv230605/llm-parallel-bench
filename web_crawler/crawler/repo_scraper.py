from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

from utils.file_utils import load_json_if_exists, slugify, write_json
from utils.github_api import GitHubAPI


class RepoScraper:
    def __init__(self, github_api: GitHubAPI, cache_dir: Path) -> None:
        self.github_api = github_api
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("web_crawler.scraper")

    def scrape_repository(
        self,
        repo: dict,
        issue_limit: int = 10,
        pr_limit: int = 10,
        commit_limit: int = 10,
        refresh: bool = False,
    ) -> dict:
        cache_path = self.cache_dir / f"{slugify(repo['full_name'])}.json"
        if not refresh:
            cached = load_json_if_exists(cache_path, default={})
            if isinstance(cached, dict) and cached.get("repo", {}).get("full_name") == repo["full_name"]:
                self.logger.info("Loaded cached artifacts for %s", repo["full_name"])
                return cached

        full_name = repo["full_name"]
        repo_query = f"repo:{full_name}"
        self.logger.info(
            "Fetching metadata for %s (issues=%s, prs=%s, commits=%s)",
            full_name,
            issue_limit,
            pr_limit,
            commit_limit,
        )
        issues = self._safe_search(repo_query, "issue", issue_limit)
        pulls = self._safe_search(repo_query, "pr", pr_limit)
        commits = self._safe_commits(full_name, commit_limit)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo": repo,
            "issues": issues,
            "pulls": pulls,
            "commits": commits,
        }
        write_json(cache_path, payload)
        self.logger.info(
            "Cached artifacts for %s: %s issues, %s PRs, %s commits",
            full_name,
            len(issues),
            len(pulls),
            len(commits),
        )
        return payload

    def _safe_search(self, repo_query: str, kind: str, limit: int) -> list[dict]:
        try:
            return self.github_api.search_issues(repo_query, kind=kind, per_page=limit)
        except Exception:
            self.logger.warning("GitHub %s fetch failed for %s", kind, repo_query)
            return []

    def _safe_commits(self, full_name: str, limit: int) -> list[dict]:
        try:
            return self.github_api.list_commits(full_name, per_page=limit)
        except Exception:
            self.logger.warning("GitHub commit fetch failed for %s", full_name)
            return []
