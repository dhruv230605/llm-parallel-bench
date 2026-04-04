from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any

from utils.prompt_builder import OpenAIHelper


EASY_PATTERNS = (
    "readme",
    "docs",
    "documentation",
    "typo",
    "spell",
    "format",
    "formatting",
    "lint",
    "style",
    "whitespace",
    "rename",
    "chore",
    "ci",
    "release",
    "version bump",
    "dependency update",
)


class DifficultyFilter:
    def __init__(self, difficulty_keywords: dict[str, Any], domain_keywords: list[str], openai_helper: OpenAIHelper) -> None:
        self.difficulty_keywords = difficulty_keywords
        self.domain_keywords = [keyword.lower() for keyword in domain_keywords]
        self.keyword_weights = {
            keyword.lower(): float(weight)
            for keyword, weight in difficulty_keywords.get("keyword_weights", {}).items()
        }
        self.openai_helper = openai_helper
        self.logger = logging.getLogger("web_crawler.filter")

    def rank(
        self,
        candidates: list[dict],
        per_repo_limit: int = 6,
        llm_per_repo_limit: int = 3,
        global_limit: int = 25,
    ) -> list[dict]:
        self.logger.info("Filtering %s candidates with threshold %s", len(candidates), self.difficulty_keywords.get("threshold", 45))
        prefiltered: list[dict] = []
        rejected_easy = 0
        for candidate in candidates:
            if self._reject_easy(candidate):
                candidate["rejected_reason"] = "easy_task_pattern"
                rejected_easy += 1
                continue
            heuristic = self.heuristic_score(candidate)
            candidate["heuristic_score"] = round(heuristic, 2)
            candidate["difficulty_score"] = round(heuristic, 2)
            candidate["hard_tags"] = self._matched_keywords(candidate)
            if heuristic >= self.difficulty_keywords.get("threshold", 45):
                prefiltered.append(candidate)
        self.logger.info(
            "Heuristic pass complete: %s retained, %s rejected as easy",
            len(prefiltered),
            rejected_easy,
        )

        grouped: dict[str, list[dict]] = defaultdict(list)
        for candidate in prefiltered:
            grouped[candidate.get("repo", "")].append(candidate)

        shortlisted: list[dict] = []
        for repo, repo_candidates in grouped.items():
            repo_candidates.sort(key=lambda item: (item["heuristic_score"], item.get("comments", 0)), reverse=True)
            repo_candidates = self._deduplicate(repo_candidates)
            llm_targets = repo_candidates[:llm_per_repo_limit]
            self.logger.info(
                "Repo %s: %s heuristic candidates, %s sent to LLM, %s kept for shortlist",
                repo,
                len(repo_candidates),
                len(llm_targets),
                min(len(repo_candidates), per_repo_limit),
            )
            for candidate in llm_targets:
                analysis = self.openai_helper.analyze_candidate(candidate)
                llm_score = analysis.get("difficulty_score")
                if isinstance(llm_score, (int, float)):
                    candidate["difficulty_score"] = round(0.7 * candidate["heuristic_score"] + 0.3 * float(llm_score), 2)
                candidate["difficulty_reason"] = analysis.get("difficulty_reason", "") or candidate.get("difficulty_reason", "")
                candidate["llm_summary"] = analysis.get("summary", "") or candidate.get("summary", "")
                candidate["hard_tags"] = sorted(set(candidate.get("hard_tags", []) + (analysis.get("hard_tags") or [])))
            shortlisted.extend(repo_candidates[:per_repo_limit])

        shortlisted.sort(key=lambda item: (item["difficulty_score"], item["heuristic_score"]), reverse=True)
        deduped = self._deduplicate(shortlisted)[:global_limit]
        self.logger.info("Final shortlist contains %s candidates", len(deduped))
        return deduped

    def heuristic_score(self, candidate: dict) -> float:
        text = " ".join(
            [
                candidate.get("title", ""),
                candidate.get("summary", ""),
                candidate.get("source_excerpt", ""),
                " ".join(candidate.get("keywords", [])),
                " ".join(candidate.get("repo_keywords", [])),
            ]
        ).lower()

        keyword_score = sum(weight for keyword, weight in self.keyword_weights.items() if keyword in text)
        complexity_bonus = self._complexity_bonus(candidate, text)
        source_bonus = float(
            self.difficulty_keywords.get("source_type_bonus", {}).get(candidate.get("source_type", ""), 0)
        )
        return min(keyword_score + complexity_bonus + source_bonus, 100.0)

    def _complexity_bonus(self, candidate: dict, text: str) -> float:
        bonus = 0.0
        structural = self.difficulty_keywords.get("structural_bonus", {})

        if any(token in text for token in ("deadlock", "race", "memory model", "ordering", "linearizable", "aba")):
            bonus += float(structural.get("synchronization_hazard", 0))
        if any(token in text for token in ("openmp", "mpi", "distributed", "collective", "barrier", "shard")):
            bonus += float(structural.get("framework", 0))
        if any(token in text for token in ("scheduler", "thread pool", "work stealing", "dag", "graph", "queue", "hash map")):
            bonus += float(structural.get("multi_file", 0))
        if any(token in text for token in ("performance", "benchmark", "latency", "throughput", "scalability")):
            bonus += float(structural.get("performance", 0))
        if candidate.get("comments", 0) >= 4:
            bonus += 4.0
        if len(candidate.get("source_excerpt", "")) >= 800:
            bonus += 3.0
        if candidate.get("source_type") == "pull_request":
            bonus += 3.0
        return bonus

    def _reject_easy(self, candidate: dict) -> bool:
        text = " ".join(
            [
                candidate.get("title", ""),
                candidate.get("summary", ""),
                candidate.get("source_excerpt", ""),
            ]
        ).lower()
        hard_match = any(keyword in text for keyword in self.domain_keywords)
        easy_match = any(pattern in text for pattern in EASY_PATTERNS)
        return easy_match and not hard_match

    def _matched_keywords(self, candidate: dict) -> list[str]:
        text = " ".join(
            [
                candidate.get("title", ""),
                candidate.get("summary", ""),
                candidate.get("source_excerpt", ""),
                " ".join(candidate.get("keywords", [])),
                " ".join(candidate.get("repo_keywords", [])),
            ]
        ).lower()
        return sorted({keyword for keyword in self.domain_keywords if keyword in text})

    def _deduplicate(self, ranked: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for candidate in ranked:
            key = "::".join(
                [
                    candidate.get("repo", ""),
                    candidate.get("source_type", ""),
                    str(candidate.get("artifact_id", "")),
                    candidate.get("title", "").strip().lower(),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique
