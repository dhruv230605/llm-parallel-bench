from __future__ import annotations

import logging
from pathlib import Path

from crawler.commit_parser import CommitParser
from crawler.difficulty_filter import DifficultyFilter
from crawler.github_search import GitHubSearch
from crawler.issue_parser import IssueParser
from crawler.pr_parser import PRParser
from crawler.repo_scraper import RepoScraper
from crawler.task_generator import TaskGenerator
from utils.file_utils import load_json
from utils.github_api import GitHubAPI
from utils.logging_utils import configure_logging
from utils.prompt_builder import OpenAIHelper


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "cache"
    log_path = base_dir / "logs" / "crawler.log"
    configure_logging(log_path)
    logger = logging.getLogger("web_crawler.main")

    logger.info("Crawler run started")
    logger.info("Loading configuration files from %s", base_dir / "config")
    search_queries = load_json(base_dir / "config" / "search_queries.json")["queries"]
    difficulty_keywords = load_json(base_dir / "config" / "difficulty_keywords.json")
    domain_keywords = load_json(base_dir / "config" / "domain_keywords.json")["keywords"]

    logger.info("Initializing GitHub and OpenAI clients")
    github_api = GitHubAPI()
    openai_helper = OpenAIHelper()

    searcher = GitHubSearch(
        github_api,
        domain_keywords=domain_keywords,
        keyword_weights=difficulty_keywords.get("keyword_weights", {}),
    )

    scraper = RepoScraper(github_api, cache_dir=cache_dir / "repos")
    issue_parser = IssueParser()
    pr_parser = PRParser()
    commit_parser = CommitParser()
    difficulty_filter = DifficultyFilter(difficulty_keywords, domain_keywords, openai_helper)
    task_generator = TaskGenerator(base_dir, github_api, openai_helper, domain_keywords)

    logger.info("Searching repositories using %s queries", len(search_queries))
    repositories = searcher.search_repositories(
        search_queries,
        cache_path=cache_dir / "repositories.json",
        per_query=6,
        overall_limit=20,
    )
    
    logger.info("Repository selection complete: %s repositories", len(repositories))

    candidates: list[dict] = []
    for index, repo in enumerate(repositories, start=1):
        logger.info("Scraping repository %s/%s: %s", index, len(repositories), repo.get("full_name", ""))
        scraped = scraper.scrape_repository(repo, issue_limit=10, pr_limit=10, commit_limit=10)
        issue_candidates = issue_parser.parse(repo, scraped.get("issues", []))
        pr_candidates = pr_parser.parse(repo, scraped.get("pulls", []))
        commit_candidates = commit_parser.parse(repo, scraped.get("commits", []))
        candidates.extend(issue_candidates)
        candidates.extend(pr_candidates)
        candidates.extend(commit_candidates)
        logger.info(
            "Repo %s yielded %s issues, %s PRs, %s commits, %s total candidates so far",
            repo.get("full_name", ""),
            len(issue_candidates),
            len(pr_candidates),
            len(commit_candidates),
            len(candidates),
        )

    logger.info("Starting difficulty ranking for %s raw candidates", len(candidates))
    ranked = difficulty_filter.rank(
        candidates,
        per_repo_limit=6,
        llm_per_repo_limit=3,
        global_limit=80,
    )
    logger.info("Difficulty ranking complete: %s ranked candidates", len(ranked))

    logger.info("Starting grounded task synthesis")
    selected_tasks = task_generator.build_tasks(ranked, count=25)
    logger.info("Task synthesis complete: %s tasks selected", len(selected_tasks))

    logger.info("Writing dataset and README")
    task_generator.write_dataset(selected_tasks)
    task_generator.write_readme(repositories, ranked, selected_tasks)
    logger.info("Crawler run finished. Log file: %s", log_path)

    print(f"Repositories selected: {len(repositories)}")
    print(f"Raw candidates collected: {len(candidates)}")
    print(f"Ranked candidates retained: {len(ranked)}")
    print(f"Dataset tasks written: {len(selected_tasks)}")
    print(f"Repository cache: {cache_dir / 'repositories.json'}")
    print(f"Repo artifact cache directory: {cache_dir / 'repos'}")
    print(f"Grounding cache directory: {cache_dir / 'details'}")
    print(f"Log file: {log_path}")
    if not openai_helper.enabled:
        print("OpenAI API not configured; task synthesis is disabled, so the dataset may contain zero grounded tasks.")
    if not repositories:
        print("No repositories were available from live GitHub or cache.")


if __name__ == "__main__":
    main()
