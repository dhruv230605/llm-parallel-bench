from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any

import requests


GITHUB_API_BASE = "https://api.github.com"


@dataclass
class GitHubAPI:
    token: str | None = None
    timeout: int = 30
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.token = self.token or os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "parallel-cpp-benchmark-crawler",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.session.headers.update(headers)

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        response = self.session.get(
            f"{GITHUB_API_BASE}{path}",
            params={key: value for key, value in params.items() if value is not None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def search_repositories(self, query: str, per_page: int = 5) -> list[dict[str, Any]]:
        data = self.get("/search/repositories", q=f"{query} language:C++", sort="stars", order="desc", per_page=per_page)
        return data.get("items", [])

    def search_issues(self, query: str, kind: str, per_page: int = 10) -> list[dict[str, Any]]:
        issue_query = f"{query} is:{kind}"
        data = self.get("/search/issues", q=issue_query, sort="updated", order="desc", per_page=per_page)
        return data.get("items", [])

    def list_commits(self, repo_full_name: str, per_page: int = 10) -> list[dict[str, Any]]:
        data = self.get(f"/repos/{repo_full_name}/commits", per_page=per_page)
        return data if isinstance(data, list) else []

    def search_code(self, query: str, per_page: int = 10) -> list[dict[str, Any]]:
        data = self.get("/search/code", q=query, per_page=per_page)
        return data.get("items", [])

    def get_repository(self, repo_full_name: str) -> dict[str, Any]:
        return self.get(f"/repos/{repo_full_name}")

    def get_pull(self, repo_full_name: str, pull_number: int) -> dict[str, Any]:
        return self.get(f"/repos/{repo_full_name}/pulls/{pull_number}")

    def get_pull_files(self, repo_full_name: str, pull_number: int, per_page: int = 30) -> list[dict[str, Any]]:
        data = self.get(f"/repos/{repo_full_name}/pulls/{pull_number}/files", per_page=per_page)
        return data if isinstance(data, list) else []

    def get_commit(self, repo_full_name: str, sha: str) -> dict[str, Any]:
        return self.get(f"/repos/{repo_full_name}/commits/{sha}")

    def get_tree(self, repo_full_name: str, tree_sha: str) -> list[dict[str, Any]]:
        data = self.get(f"/repos/{repo_full_name}/git/trees/{tree_sha}", recursive=1)
        return data.get("tree", [])

    def get_file_content(self, repo_full_name: str, path: str, ref: str | None = None) -> str:
        data = self.get(f"/repos/{repo_full_name}/contents/{path}", ref=ref)
        if data.get("encoding") == "base64":
            encoded = data.get("content", "")
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
        download_url = data.get("download_url")
        if not download_url:
            return ""
        response = self.session.get(download_url, timeout=self.timeout)
        response.raise_for_status()
        return response.text
