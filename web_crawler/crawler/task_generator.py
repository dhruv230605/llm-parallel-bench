from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from utils.file_utils import load_json_if_exists, reset_directory, slugify, write_json, write_text
from utils.github_api import GitHubAPI
from utils.prompt_builder import OpenAIHelper


class TaskGenerator:
    def __init__(
        self,
        base_dir: Path,
        github_api: GitHubAPI,
        openai_helper: OpenAIHelper,
        domain_keywords: list[str],
    ) -> None:
        self.base_dir = base_dir
        self.github_api = github_api
        self.openai_helper = openai_helper
        self.domain_keywords = [keyword.lower() for keyword in domain_keywords]
        self.details_cache_dir = self.base_dir / "cache" / "details"
        self.logger = logging.getLogger("web_crawler.tasks")

    def build_tasks(self, ranked_candidates: list[dict], count: int = 25) -> list[dict]:
        tasks: list[dict] = []
        self.logger.info("Beginning task synthesis from %s ranked candidates", len(ranked_candidates))
        for index, candidate in enumerate(ranked_candidates, start=1):
            if len(tasks) >= count:
                break
            self.logger.info(
                "Grounding candidate %s/%s: %s [%s]",
                index,
                len(ranked_candidates),
                candidate.get("title", ""),
                candidate.get("source_type", ""),
            )
            grounding = self._ground_candidate(candidate)
            if not self._grounding_is_sufficient(candidate, grounding):
                self.logger.info(
                    "Skipping candidate due to insufficient grounding: %s",
                    candidate.get("url", "") or candidate.get("title", ""),
                )
                continue

            standardized = self.openai_helper.standardize_prompt(candidate)
            candidate = dict(candidate)
            candidate["prompt"] = standardized["prompt"]
            if standardized["difficulty_reason"]:
                candidate["difficulty_reason"] = standardized["difficulty_reason"]

            package = self.openai_helper.generate_task_package(candidate, grounding)
            if not package:
                self.logger.warning(
                    "Task package generation failed for candidate: %s",
                    candidate.get("url", "") or candidate.get("title", ""),
                )
                continue

            tasks.append(
                {
                    "title": package["title"],
                    "prompt": package["prompt"],
                    "difficulty_reason": package["difficulty_reason"] or candidate.get("difficulty_reason", ""),
                    "code_cpp": package["code_cpp"],
                    "code_h": package.get("code_h", ""),
                    "test_cpp": package["test_cpp"],
                    "run_tests_sh": package.get("run_tests_sh") or self._default_run_script(candidate, package),
                    "keywords": candidate.get("hard_tags", []) or candidate.get("keywords", []),
                    "source_metadata": {
                        "source_type": candidate.get("source_type", ""),
                        "repo": candidate.get("repo", ""),
                        "repo_url": candidate.get("repo_url", ""),
                        "url": candidate.get("url", ""),
                        "artifact_id": candidate.get("artifact_id", ""),
                        "difficulty_score": candidate.get("difficulty_score"),
                        "heuristic_score": candidate.get("heuristic_score"),
                        "hard_tags": candidate.get("hard_tags", []),
                        "grounding_cache": grounding.get("cache_path", ""),
                    },
                }
            )
            self.logger.info("Accepted task %s/%s: %s", len(tasks), count, package["title"])
        return tasks[:count]

    def write_dataset(self, tasks: list[dict]) -> None:
        dataset_dir = self.base_dir / "dataset"
        reset_directory(dataset_dir)
        self.logger.info("Writing %s tasks to dataset directory %s", len(tasks), dataset_dir)

        for index, task in enumerate(tasks, start=1):
            task_dir = dataset_dir / f"task{index:02d}"
            tests_dir = task_dir / "tests"
            prompt = " ".join(task["prompt"].splitlines()).strip()
            write_text(task_dir / "prompt.txt", prompt + "\n")
            write_text(task_dir / "code.cpp", task["code_cpp"].rstrip() + "\n")
            if task.get("code_h"):
                write_text(task_dir / "code.h", task["code_h"].rstrip() + "\n")
            write_text(tests_dir / "test1.cpp", task["test_cpp"].rstrip() + "\n")
            write_text(tests_dir / "run_tests.sh", task["run_tests_sh"].rstrip() + "\n")
            write_json(
                task_dir / "metadata.json",
                {
                    "title": task["title"],
                    "difficulty_reason": task["difficulty_reason"],
                    "keywords": task["keywords"],
                    "source_metadata": task.get("source_metadata", {}),
                },
            )
        self.logger.info("Dataset write complete")

    def write_readme(self, repositories: list[dict], ranked_candidates: list[dict], selected_tasks: list[dict]) -> None:
        readme = self.base_dir / "README.md"
        lines = [
            "# Parallel C++ Web Crawler",
            "",
            "This standalone module mines GitHub for hard parallel-programming tasks in C++, caches repository and artifact metadata locally, filters candidates with keyword- and complexity-based scoring, and exports a grounded benchmark dataset.",
            "",
            "## How The Crawler Works",
            "",
            "1. Load search queries and domain keywords from `config/`.",
            "2. Search GitHub repositories and keep only domain-matching repositories, caching their URLs and scores in `cache/repositories.json`.",
            "3. For each selected repository, fetch up to 10 issues, 10 pull requests, and 10 commits, caching each repo artifact dump under `cache/repos/`.",
            "4. Convert GitHub metadata into candidate tasks and score them using keyword weights, complexity bonuses, and easy-task rejection.",
            "5. Keep roughly 5-7 candidates per repository, then use `gpt-4o-mini` only on shortlisted candidates for normalization and final grounded task synthesis.",
            "6. Fetch deeper grounding details only for final selected artifacts and cache them under `cache/details/`.",
            "7. Export `prompt.txt`, `code.cpp`, optional `code.h`, tests, and provenance metadata under `dataset/task01` onward.",
            "",
            "## Why These Tasks Are Hard For Zero-Shot Prompting",
            "",
            "- They center on synchronization, shared-state coordination, and non-trivial parallel decomposition.",
            "- Metadata scoring prioritizes race conditions, deadlocks, lock-free patterns, work stealing, OpenMP, MPI, and distributed coordination.",
            "- Easy maintenance-only tasks such as docs, lint, typo, and formatting changes are rejected before LLM usage.",
            "- Pull requests and commits with stronger grounding are preferred because they provide implementation evidence and patch context.",
            "",
            "## Cache Layout",
            "",
            "```text",
            "cache/",
            "  repositories.json",
            "  repos/<repo>.json",
            "  details/<artifact>.json",
            "```",
            "",
            "This prevents reusing the GitHub API for the same repositories and shortlisted artifacts on every run.",
            "",
            "## Running The Crawler",
            "",
            "```bash",
            "python main.py",
            "```",
            "",
            "Environment variables:",
            "",
            "- `GITHUB_TOKEN`: recommended for higher GitHub API limits.",
            "- `OPENAI_API_KEY`: required for grounded task synthesis with `gpt-4o-mini`.",
            "",
            "## Dataset Structure",
            "",
            "```text",
            "dataset/",
            "  task01/",
            "    prompt.txt",
            "    code.cpp",
            "    tests/test1.cpp",
            "    tests/run_tests.sh",
            "  ...",
            "```",
            "",
            f"Repositories selected in the latest run: `{len(repositories)}`.",
            f"Ranked candidates kept after filtering: `{len(ranked_candidates)}`.",
            f"Tasks written in the latest run: `{len(selected_tasks)}`.",
            "",
            "## Selected Repositories",
            "",
        ]

        if repositories:
            for repo in repositories[:20]:
                lines.append(
                    f"- `{repo.get('full_name', '')}` - {repo.get('html_url', '')} - score {repo.get('repo_score', 0)}"
                )
        else:
            lines.append("- No repositories were available from cache or live GitHub in the latest run.")

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- This version intentionally does not use any seed task library.",
                "- If live GitHub data or OpenAI synthesis is unavailable, the crawler may export fewer than 25 grounded tasks rather than silently fabricating fallback tasks.",
            ]
        )
        write_text(readme, "\n".join(lines) + "\n")
        self.logger.info("README updated at %s", readme)

    def _ground_candidate(self, candidate: dict) -> dict[str, Any]:
        cache_key = "::".join(
            [
                candidate.get("repo", ""),
                candidate.get("source_type", ""),
                str(candidate.get("artifact_id", "")),
            ]
        )
        cache_path = self.details_cache_dir / f"{slugify(cache_key)}.json"
        cached = load_json_if_exists(cache_path, default={})
        if isinstance(cached, dict) and cached:
            cached["cache_path"] = str(cache_path)
            self.logger.info("Loaded cached grounding for %s", cache_key)
            return cached

        grounding: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo": candidate.get("repo", ""),
            "repo_url": candidate.get("repo_url", ""),
            "artifact_url": candidate.get("url", ""),
            "source_type": candidate.get("source_type", ""),
            "title": candidate.get("title", ""),
            "summary": candidate.get("summary", ""),
            "excerpt": candidate.get("source_excerpt", ""),
            "hard_tags": candidate.get("hard_tags", []),
        }

        source_type = candidate.get("source_type")
        repo = candidate.get("repo", "")
        if source_type == "pull_request" and repo and candidate.get("artifact_number"):
            try:
                pull = self.github_api.get_pull(repo, int(candidate["artifact_number"]))
            except Exception:
                pull = {}
            try:
                files = self.github_api.get_pull_files(repo, int(candidate["artifact_number"]), per_page=20)
            except Exception:
                files = []
            grounding["pull"] = {
                "title": pull.get("title", candidate.get("title", "")),
                "body": pull.get("body", "") or candidate.get("source_excerpt", ""),
                "changed_files": [
                    {
                        "filename": file_item.get("filename", ""),
                        "status": file_item.get("status", ""),
                        "additions": file_item.get("additions", 0),
                        "deletions": file_item.get("deletions", 0),
                        "patch": (file_item.get("patch") or "")[:4000],
                    }
                    for file_item in files[:8]
                ],
            }
        elif source_type == "commit" and repo and candidate.get("artifact_sha"):
            try:
                commit = self.github_api.get_commit(repo, candidate["artifact_sha"])
            except Exception:
                commit = {}
            grounding["commit"] = {
                "message": commit.get("commit", {}).get("message", candidate.get("summary", "")),
                "files": [
                    {
                        "filename": file_item.get("filename", ""),
                        "status": file_item.get("status", ""),
                        "patch": (file_item.get("patch") or "")[:4000],
                    }
                    for file_item in commit.get("files", [])[:8]
                ],
            }
        else:
            grounding["issue"] = {
                "body": candidate.get("source_excerpt", ""),
                "comments": candidate.get("comments", 0),
                "labels": candidate.get("labels", []),
            }

        write_json(cache_path, grounding)
        grounding["cache_path"] = str(cache_path)
        self.logger.info("Cached grounding details for %s", cache_key)
        return grounding

    def _grounding_is_sufficient(self, candidate: dict, grounding: dict[str, Any]) -> bool:
        source_type = candidate.get("source_type")
        if source_type == "pull_request":
            changed_files = grounding.get("pull", {}).get("changed_files", [])
            return bool(changed_files)
        if source_type == "commit":
            files = grounding.get("commit", {}).get("files", [])
            return bool(files) or len(grounding.get("commit", {}).get("message", "")) >= 100
        return len(grounding.get("issue", {}).get("body", "")) >= 250 and len(candidate.get("hard_tags", [])) >= 2

    def _default_run_script(self, candidate: dict, package: dict[str, str]) -> str:
        text = " ".join(
            [
                candidate.get("prompt", ""),
                candidate.get("title", ""),
                package.get("prompt", ""),
                " ".join(candidate.get("hard_tags", [])),
            ]
        ).lower()
        if "mpi" in text:
            return "#!/usr/bin/env bash\nset -euo pipefail\nmpic++ -std=c++17 -O2 test1.cpp -o test1 && mpiexec -n 2 ./test1"
        if "openmp" in text:
            return "#!/usr/bin/env bash\nset -euo pipefail\ng++ -std=c++17 -O2 -fopenmp -pthread test1.cpp -o test1 && ./test1"
        return "#!/usr/bin/env bash\nset -euo pipefail\ng++ -std=c++17 -O2 -pthread test1.cpp -o test1 && ./test1"
